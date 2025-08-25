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
import matplotlib.pyplot as plt
from tqdm import trange

from sklearn.ensemble import RandomForestRegressor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device {DEVICE}')

#    TODO: rewrite in tinygrad

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
    feats = [atom.GetAtomicNum(), atom.GetTotalDegree(), atom.GetFormalCharge(), int(atom.GetIsAromatic()), atom.GetTotalNumHs(includeNeighbors=True), int(atom.HasProp('_ChiralityPossible'))] + hyb
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
  edge_weight = torch.tensor(w, dtype=torch.float) if w else torch.tensor([], dtype=torch.float)
  data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
  return data

class GCN(nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels):
    super().__init__()
    self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True)
    self.conv2 = GCNConv(hidden_channels, out_channels, normalize=True)
    self.head = nn.Linear(out_channels, 1)

  def forward(self, data, return_emb=False):
    x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_weight
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.conv1(x, edge_index, edge_weight).relu()
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.conv2(x, edge_index, edge_weight)
    g = global_mean_pool(x, batch) #   graph embeddings
    if return_emb: return g
    return self.head(g).squeeze(1)

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

#    NOTE: pretrain the encoder first
#    TODO: concatenate mordred descriptors with learned embeddings

if __name__ == "__main__":
  set_seed()
  fp = 'data/DOWNLOAD-gU8RPQ5Wut7KaKJdHzr2fUYYJcpIjb0ClUND2cUakNk_eq_.csv'
  df = pd.read_csv(fp, delimiter=';')
  df = df.dropna(subset=['Smiles', 'pChEMBL Value'])
  df['mol'] = df['Smiles'].apply(Chem.MolFromSmiles)

  os.makedirs('cache', exist_ok=True)
  calc = Calculator(descriptors, ignore_3D=True)
  if not os.path.exists('cache/descriptors.pkl'):
    desc_df = calc.pandas(df['mol']) #    concatenate descriptors with GNN learned embeddings
    desc_df.to_pickle('cache/descriptors.pkl')
  else:
    print('found pickle: cache/descriptors.pkl')
    desc_df = pd.read_pickle('cache/descriptors.pkl')

  graphs = []
  for mol, target in zip(df['mol'], df['pChEMBL Value']):
    g = convert_to_pyg(mol)
    g.y = torch.tensor([float(target)], dtype=torch.float)
    graphs.append(g)

  loader = DataLoader(graphs, batch_size=32, shuffle=True)
  model = GCN(in_channels=graphs[0].x.shape[1], hidden_channels=128, out_channels=128).to(DEVICE) #  embedding size = out_channels
  opt = torch.optim.Adam(model.parameters())

  accs = []
  for epoch in (t := trange(1,21)):
    model.train()
    for batch_idx, data in enumerate(loader):
      data = data.to(DEVICE, non_blocking=True)
      opt.zero_grad()
      out = model(data)
      loss = F.mse_loss(out, data.y.view(-1))
      loss.backward()
      opt.step()
      li = loss.item()
      accs.append(li)
      t.set_description(f'loss {li:.3f} epoch {epoch}')

  E = export_emb(model, loader)
  print(f'exported embedding {E.shape}')

  plt.plot(accs)
  plt.show()