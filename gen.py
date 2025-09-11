from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
from rdkit.Chem.rdchem import ChiralType
from typing import Optional, Union, List, Dict, Any, Iterable
from dataclasses import dataclass
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
import itertools
import copy

AROMATIC_VALENCES = {
  "B": (3,), "Al": (3,),
  "C": (4,), "Si": (4,),
  "N": (3, 5), "P": (3, 5), "As": (3, 5),
  "O": (2, 4), "S": (2, 4), "Se": (2, 4), "Te": (2, 4)
}

def set_custom_on_rdkit(a: Chem.Atom, element:str, is_aromatic:bool, isotope:int|None, chirality:str|None, h_count:int|None, charge:int):
    Z = Chem.GetPeriodicTable().GetAtomicNumber(element)
    a.SetAtomicNum(Z)

    a.SetFormalCharge(int(charge))
    if isotope is not None: a.SetIsotope(int(isotope))
    a.SetIsAromatic(bool(is_aromatic))

    # set hydrogens
    if h_count is None:
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    else:
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(int(h_count))

    if chirality in ("@", "@@"): a.SetChiralTag(ChiralType.CHI_TETRAHEDRAL_CW if chirality == "@" else ChiralType.CHI_TETRAHEDRAL_CCW)
    a.UpdatePropertyCache(strict=False)

def get_custom_from_rdkit(a: Chem.Atom) -> dict:
    ch, chirality = a.GetChiralTag(), None
    if ch == ChiralType.CHI_TETRAHEDRAL_CW: chirality = "@"
    elif ch == ChiralType.CHI_TETRAHEDRAL_CCW: chirality = "@@"
    return {"index": a.GetIdx(),
            "element": a.GetSymbol(),
            "is_aromatic": a.GetIsAromatic(),
            "isotope": (a.GetIsotope() or None),
            "chirality": chirality,
            "h_count": (None if not a.GetNoImplicit() else a.GetTotalNumHs()),
            "charge": a.GetFormalCharge()}

@dataclass
class Attribute:
  index:int
  token:str

class DirectedBond:
  __slots__ = 'src', 'dst', 'order', 'stereo', 'ring_bond'
  def __init__(self, src:int, dst:int, order:Union[int,float], stereo:Optional[str], ring_bond:bool):
    self.src:int = src
    self.dst:int = dst
    self.order:Union[int,float] = order
    self.stereo:str = stereo
    self.ring_bond:bool = ring_bond

class MolecularGraph:
  def __init__(self, attributable:bool=False):
    self._roots:List[int] = []
    self._atoms:List[rdchem.Atom] = []
    self._bonds:Dict[tuple[int, int], DirectedBond] = {}
    self._adj:List[List[Optional[DirectedBond]]]= []
    self._bond_cnts:List[float] = []
    self._delocal:Dict[int, List[int]] = {} #    delocalization subgraph
    self._attr:Dict[Any, List[Attribute]] = {}
    self._attributable = attributable

  def __len__(self) -> int: return len(self._atoms)
  def __iter__(self) -> Iterable[rdchem.Atom]: return iter(self._atoms)
  def has_bond(self, a:int, b:int) -> bool:
    if a > b: a, b = b, a
    return (a,b) in self._bond_dict

  def has_out_ring_bond(self, src:int) -> bool: return any(e is not None and e.ring for e in self._adj[src])
  def get_attribution(self, o:Union[DirectedBond, rdchem.Atom]) -> Optional[List[Attribute]]: return self._attr.get(o) if self._attributable else None
  def get_directed_bond(self, src:int, dst:int) -> DirectedBond: return self._bonds[(src,dst)]
  def get_outward_directed_bonds(self, src:int) -> List[Optional[DirectedBond]]: return self._adj[src]
  def get_bond_count(self, idx:int) -> int: return int(self._bond_cnts[idx])

  def add_atom(self, atom:rdchem.Atom, mark_root:bool=False) -> rdchem.Atom:
    gidx = len(self._atoms)
    atom.SetIntProp('_gidx', gidx)
    if mark_root: self._roots.append(gidx)
    self._atoms.append(atom)
    self._adj.append([])
    self._bond_cnts.append(0)
    if atom.GetIsAromatic(): self._delocal[gidx] = []
    return atom

  def add_attribution(self, o:Union[DirectedBond,rdchem.Atom], attr:List[Attribute]) -> None:
    if not self._attributable: return
    self._attr.setdefault(o, []).extend(attr)

  def add_bond(self, src:int, dst:int, order:Union[int,float], stereo:Optional[str]) -> DirectedBond:
    assert src < dst
    bond = DirectedBond(src, dst, float(order), stereo, ring_bond=False)
    self._insert_edge(bond, pos=-1)
    self._bond_cnts[src] += order
    self._bond_cnts[dst] += order
    if order == 1.5:
      self._delocal.setdefault(src, []).append(dst)
      self._delocal.setdefault(dst, []).append(src)
    return bond

  def add_placeholder_bond(self, src:int) -> int:
    self._adj[src].append(None)
    return len(self._adj[src]) - 1

  def add_ring_bond(self, a:int, b:int, order:Union[int,float], a_stereo:Optional[str], b_stereo:Optional[str], a_pos:int=-1, b_pos:int=-1) -> None:
    a_bond = DirectedBond(a, b, float(order), a_stereo, ring_bond=True)
    b_bond = DirectedBond(b, a, float(order), b_stereo, ring_bond=True)
    self._insert_edge(a_bond, a_pos)
    self._insert_edge(b_bond, b_pos)
    self._bond_cnts[a] += order
    self._bond_cnts[b] += order
    if order == 1.5:
      self._delocal.setdefault(a, []).append(b)
      self._delocal.setdefault(b, []).append(a)

  def update_bond_order(self, a:int, b:int, new_order:Union[int,float]): #    atom index
    assert 1 <= new_order <= 3
    if a > b: a,b = b,a
    fwd = self._bonds[(a,b)]
    if new_order == fwd.order: return
    bonds = (fwd,) if not fwd.ring_bond else (fwd, self._bonds[(b, a)])
    old = bonds[0].order
    for e in bonds: e.order = float(new_order)
    d = float(new_order)-float(old)
    self._bond_cnts[a] += delta
    self._bond_cnts[b] += delta

  def _insert_edge(self, bond:DirectedBond, pos:int) -> None:
    self._bonds[(bond.src, bond.dst)] = bond
    out = self._adj[bond.src]
    if pos == -1 or pos == len(out): out.append(bond)
    elif out[pos] is None: out[pos] = bond
    else: out.insert(pos, bond)

"""

the delocalization subgraph is the part of the graph that contains only the atoms and bonds participating in
the aromatic (delocalized pi) systems.
* contains a mapping built only from the bonds marked with order 1.5
* isolates the aromatic network so can be kekulized without touching the rest of the molecule
* prune nodes that can't take part

"""

  #    aromaticity / kekulization
  def is_kekulized(self) -> bool: return not self._delocal
  def kekulize(self) -> bool:
    # Based on Apodaca's article: https://depth-first.com/articles/2020/02/10/a-comprehensive-treatment-of-aromaticity-in-the-smiles-language/
    if self.is_kekulized(): return True
    ds = self._delocal
    kept = set(itertools.filterfalse(self.))
    #    TODO: write the rest

  def _prune_from_ds(self, node:int) -> bool:
    adj = self._delocal[node]
    if not adj: return True #    aromatic atom with no aromatic bonds
    atom = get_custom_from_rdkit(self._atoms[node])
    valences = AROMATIC_VALENCES[atom["element"]]
    used = int(self._bond_cnts[node] - .5 * len(adj)) #    treat each ds bond (order 1.5) as a single when counting
    #    TODO: write the rest

class Copy:
  def __call__(self, dat):
    data = copy.copy(dat)
    data.y = data.y[:,0]
    return data

if __name__ == '__main__':
  graph = MolecularGraph()
  transform = T.Compose([Copy()])
  dataset = QM9('cache/QM9', transform=transform).shuffle()
  m = Chem.MolFromSmiles('Cc1ccccc1')
  for a in m.GetAtoms():
    pass
