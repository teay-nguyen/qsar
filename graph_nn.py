import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
from mordred import Calculator, descriptors
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import trange

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device {DEVICE}')

#    TODO: rewrite in tinygrad
#    TODO: write accuracy testing schemes

"""

1. train the graph neural netwowrk to output graph (molecular) embeddings (which are vectors)
2. concatenate it with preprocessed Mordred descriptors
3. train the regressor on the concatenated tensor data
4. the final scalar logit is whatever the target is

export embeddings: E ∈ R^{N×H}, H = embedding size
compute mordred descriptors: D ∈ R^{N×K}, K = >1000 the last time I checked
concatenate X = [E | D] ∈ R^{N×(H+K)} then train RF/XGBoost/SVM to predict the label (pChEMBL)

"""

HYB_CHOICES = [
  rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
  rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
  rdchem.HybridizationType.SP3D2
]

#    Graph Convolutional Network

def atom_features(atom: rdchem.Atom):
  # One-hot hybridization (compact)
    hyb = [int(atom.GetHybridization() == h) for h in HYB_CHOICES]
    feats = [atom.GetAtomicNum(), atom.GetTotalDegree(), atom.GetFormalCharge(),
             int(atom.GetIsAromatic()), atom.GetTotalNumHs(includeNeighbors=True),
             int(atom.HasProp('_ChiralityPossible'))] + hyb
    return feats

def bond_order(bond: rdchem.Bond):
    # Map to a scalar weight usable by GCNConv (edge_weight)
    types = {rdchem.BondType.SINGLE:1.0, rdchem.BondType.DOUBLE:2.0,
             rdchem.BondType.TRIPLE:3.0, rdchem.BondType.AROMATIC:1.5}
    return types.get(bond.GetBondType(), 1.0)

def convert_to_pyg(mol):
  Chem.SanitizeMol(mol)
  x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
  # Edges: undirected graph → add both directions
  src, dst, w = [], [], []
  for bond in mol.GetBonds():
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    src += [i, j]
    dst += [j, i]
    ww = bond_order(bond)
    w  += [ww, ww]

  edge_index = torch.tensor([src, dst], dtype=torch.long)
  #edge_weight = torch.tensor(w, dtype=torch.float) if w else torch.tensor([], dtype=torch.float)
  #data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
  data = Data(x=x, edge_index=edge_index)
  return data

class GCN(nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels):
    super().__init__()
    self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True)
    self.conv2 = GCNConv(hidden_channels, out_channels, normalize=True)
    self.dropout = .1
    self.head = nn.Sequential(
      nn.Linear(out_channels, out_channels//2), nn.ReLU(),
      nn.Linear(out_channels//2, 1)
    )

  def forward(self, data, return_emb=False):
    x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_weight
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.conv1(x, edge_index, edge_weight).relu()
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.conv2(x, edge_index, edge_weight)
    g = global_mean_pool(x, batch) #   graph embeddings
    if return_emb: return g
    return self.head(g).squeeze(1)






class MLPRegressor(nn.Module):
  def __init__(self, d_in, d_hidden, d_out=1, drop=.1):
    super().__init__()
    self.layers = []
    last = d_in
    for h in d_hidden:
      self.layers.extend([nn.Linear(last,h), nn.ReLU(), nn.Dropout(drop)])
      last = h
    self.layers.append(nn.Linear(last, d_out))
    self.layers = nn.Sequential(*self.layers)

  def forward(self, x):
    return self.layers(x)

"""
heteroscedastic MLP (predict mean and aleatoric variation)
- instead of assuming noise in regression is const. (homoscedastic), the model learns input-dependent noise (heteroscedastic)
- assay variability, dataset curation, outliers
- aleatoric means uncertainty due to noise in data, not model ignorance

get predicted mean and predicted log variance, train using the nll gaussian.

"""

class MLP(nn.Module):
  def __init__(self, d_in, d_hidden, min_logvar=-3.0, max_logvar=5.0):
    super().__init__()
    self.body = MLPRegressor(d_in, d_hidden, d_out=2)
    self.min_lv, self.max_lv = min_logvar, max_logvar

  def forward(self, x):
    out = self.body(x)
    mu, log_var = out[:,:1], out[:,1:2].clamp(self.min_lv, self.max_lv)
    return mu, log_var

def train_hetero(model, loader, epochs=100, lr=1e-3, wd=1e-4):
  model.to(DEVICE)
  model.train()
  opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
  for epoch in (t:=trange(epochs)):
    running,n = 0,0
    for data in loader:
      data = data.to(DEVICE, non_blocking=True)
      x, y = data.x, data.y.view(-1,1)
      opt.zero_grad()
      mu,log_var = model(x)
      nll = 0.5*(log_var + (y-mu)**2 / torch.exp(log_var)) + (.5*np.log(2*np.pi))
      loss = nll.mean()
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 5.0)
      opt.step()
      bs = y.size(0)
      running += loss.item() * bs
      n += bs
    t.set_description(f'loss {running/max(n,1):.4f} epoch {epoch+1}')

def set_seed(seed=1337):
  rng = np.random.default_rng(seed)
  torch.manual_seed(seed)
  return rng

def export_emb(model, loader):
  model.eval()
  outs = []
  for data in loader:
    data = data.to(DEVICE, non_blocking=True)
    outs.append(model(data, return_emb=True).detach().cpu())
  return torch.cat(outs, dim=0).numpy()

"""
what works:
- get rid of edge_weights
- reduce dropout
- scale y to standardized y
- sanity check (this is new to me)

used ChEMBL dataset for CHEMBL203 (Epidermal growth factor receptor erbB1) 

"""

if __name__ == "__main__":
  set_seed()
  fp = 'data/DOWNLOAD-gU8RPQ5Wut7KaKJdHzr2fUYYJcpIjb0ClUND2cUakNk_eq_.csv'
  df = pd.read_csv(fp, delimiter=';')
  df = df.dropna(subset=['Smiles', 'pChEMBL Value'])
  df['mol'] = df['Smiles'].apply(Chem.MolFromSmiles)

  Y = df['pChEMBL Value'].astype(float).values
  Y_mean, Y_std = float(np.mean(Y)), float(np.std(Y)+1e-8)
  df['y_std'] = (Y-Y_mean)/Y_std

  os.makedirs('cache', exist_ok=True)
  calc = Calculator(descriptors, ignore_3D=True) #    take 2d descriptors only
  if not os.path.exists('cache/descriptors.pkl'):
    print('did not find cache/descriptors.pkl, generating descriptors...')
    desc_df = calc.pandas(df['mol']) #    concatenate descriptors with GNN learned embeddings
    desc_df.to_pickle('cache/descriptors.pkl')
  else:
    print('found pickle: cache/descriptors.pkl')
    desc_df = pd.read_pickle('cache/descriptors.pkl')

  imputer = SimpleImputer(strategy="median")
  scaler = StandardScaler()
  desc_df = desc_df.apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan).astype(np.float32)
  mordred_descriptors = scaler.fit_transform(imputer.fit_transform(desc_df.drop(columns=desc_df.columns[desc_df.isna().all()])))
  graphs = []
  for mol, target in zip(df['mol'], df['y_std']):
    g = convert_to_pyg(mol)
    g.y = torch.tensor([float(target)], dtype=torch.float)
    graphs.append(g)

  loader = DataLoader(graphs, batch_size=32, shuffle=True)
  sloader = DataLoader(graphs[:32], batch_size=32, shuffle=True)
  model = GCN(in_channels=graphs[0].x.shape[1], hidden_channels=128, out_channels=128).to(DEVICE)
  opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
  if not os.path.exists('cache/chkpnt.pth'):
    def train_step(ld):
      model.train()
      running = 0.0
      for batch_idx,data in enumerate(ld):
        data = data.to(DEVICE, non_blocking=True)
        opt.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        running += loss.item()*data.num_graphs
      return running/len(ld.dataset)

    for _ in range(50): loss = train_step(sloader)
    print(f'sanity, small subset loss {loss:.3f}')

    accs = []
    for epoch in (t:=trange(1,21)):
      loss = train_step(loader)
      accs.append(loss)
      t.set_description(f'loss {loss:.3f} epoch {epoch}')

    torch.save(model.state_dict(), 'cache/chkpnt.pth')
    print('saved model to cache/chkpnt.pth')
  else:
    print('found checkpoint cache/chkpnt.pth')
    model.load_state_dict(torch.load('cache/chkpnt.pth'))
    model.eval()

  E = export_emb(model, loader)
  X_stack = np.hstack([E,mordred_descriptors])
  y = df['y_std'].to_numpy(dtype=float)

  feats = [Data(x=torch.tensor(x, dtype=torch.float32).unsqueeze(0), y=torch.tensor([target], dtype=torch.float32)) for x,target in zip(X_stack,y)]
  feat_loader = DataLoader(feats, batch_size=512, shuffle=True)
  mlp = MLP(X_stack.shape[-1], (256, 128))
  train_hetero(mlp, feat_loader)
