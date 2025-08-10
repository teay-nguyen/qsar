import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
from mordred import Calculator, descriptors
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch

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
    bt = bond.GetBondType()
    if bt == rdchem.BondType.SINGLE: return 1.0
    if bt == rdchem.BondType.DOUBLE: return 2.0
    if bt == rdchem.BondType.TRIPLE: return 3.0
    if bt == rdchem.BondType.AROMATIC: return 1.5
    return 1.0

def convert_to_pyg(mol):
  Chem.SanitizeMol(mol)
  x = [atom_features(a) for a in mol.GetAtoms()]
  x = torch.tensor(x, dtype=torch.float)

  # Edges: undirected graph → add both directions
  src, dst, w = [], [], []
  for bond in mol.GetBonds():
    i = bond.GetBeginAtomIdx()
    j = bond.GetEndAtomIdx()
    src += [i, j]
    dst += [j, i]
    ww = bond_order(bond)
    w  += [ww, ww]

  edge_index = torch.tensor([src, dst], dtype=torch.long)
  edge_weight = torch.tensor(w, dtype=torch.float) if w else torch.tensor([], dtype=torch.float)

  data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
  return data

class GCN(torch.nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels):
    super().__init__()
    self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True)
    self.conv2 = GCNConv(hidden_channels, out_channels, normalize=True)

  def forward(self, data):
    x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_weight
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.conv1(x, edge_index, edge_weight).relu()
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.conv2(x, edge_index, edge_weight)

    return global_mean_pool(x, batch) #   graph embeddings

#    NOTE: pretrain the encoder first
#    TODO: concatenate mordred descriptors with learned embeddings

if __name__ == "__main__":
  fp = 'data/DOWNLOAD-gU8RPQ5Wut7KaKJdHzr2fUYYJcpIjb0ClUND2cUakNk_eq_.csv'
  df = pd.read_csv(fp, delimiter=';')
  df = df.dropna(subset=['Smiles', 'pChEMBL Value'])
  df['mol'] = df['Smiles'].apply(Chem.MolFromSmiles)

  calc = Calculator(descriptors, ignore_3D=True)
  # desc_df = calc.pandas(df['mol']) #    these descriptors from Mordred are to be combined with GNN 

  graphs = []
  for mol, target in zip(df['mol'], df['pChEMBL Value']):
    g = convert_to_pyg(mol)
    g.y = torch.tensor([float(target)], dtype=torch.float)
    graphs.append(g)

  loader = DataLoader(graphs, batch_size=32, shuffle=True)
  model = GCN(in_channels=graphs[0].x.shape[1], hidden_channels=128, out_channels=128) #  embedding size = out_channels

  for batch_idx, data in enumerate(loader):
    with torch.no_grad():
      out = model(data)
