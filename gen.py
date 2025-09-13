from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
from rdkit.Chem.rdchem import ChiralType
from typing import Optional, Union, List, Dict, Any, Iterable
from dataclasses import dataclass, field
from torch_geometric.datasets import QM9
from collections import deque
import torch_geometric.transforms as T
import itertools, copy, heapq

"""

the delocalization subgraph is the part of the graph that contains only the atoms and bonds participating in
the aromatic (delocalized pi) systems.
* contains a mapping built only from the bonds marked with order 1.5
* isolates the aromatic network so can be kekulized without touching the rest of the molecule
* prune nodes that can't take part

during kekulize():
    (a) prune DS nodes that are chemically ineligible;
    (b) compute a perfect matching over the pruned DS (if any);
    (c) "de-aromatize" (set all DS bonds to single and clear aromatic flags);
    (d) assign double bonds according to the matching.

matching algorithm:
warm start with a greedy matching and then iteratively search for augmenting
paths using a modified BFS. If not augmenting path exists from a
remaining free vertex, the graph has no perfect matching

"""

AROMATIC_VALENCES = {
  "B": (3,), "Al": (3,),
  "C": (4,), "Si": (4,),
  "N": (3, 5), "P": (3, 5), "As": (3, 5),
  "O": (2, 4), "S": (2, 4), "Se": (2, 4), "Te": (2, 4)}

VALENCE_ELECTRONS = {
  "B": 3, "Al": 3,
  "C": 4, "Si": 4,
  "N": 5, "P": 5, "As": 5,
  "O": 6, "S": 6, "Se": 6, "Te": 6}

def set_custom_on_rdkit(a: Chem.Atom, element:str, is_aromatic:bool, isotope:int|None, chirality:str|None, h_count:int|None, charge:int):
    Z = Chem.GetPeriodicTable().GetAtomicNumber(element)
    a.SetAtomicNum(Z)

    a.SetFormalCharge(int(charge))
    if isotope is not None: a.SetIsotope(int(isotope))
    a.SetIsAromatic(bool(is_aromatic))

    impl = h_count is not None #    set hydrogens
    hs = 0 if h_count is None else int(h_count)
    a.SetNoImplicit(impl)
    a.SetNumExplicitHs(hs)

    if chirality in ("@", "@@"): a.SetChiralTag(ChiralType.CHI_TETRAHEDRAL_CW if chirality == "@" else ChiralType.CHI_TETRAHEDRAL_CCW)
    a.UpdatePropertyCache(strict=False)

def get_custom_from_rdkit(a: Chem.Atom) -> dict:
    ch, chirality = a.GetChiralTag(), None
    if ch == ChiralType.CHI_TETRAHEDRAL_CW: chirality = "@"
    elif ch == ChiralType.CHI_TETRAHEDRAL_CCW: chirality = "@@"
    return {"index": a.GetIdx(), "element": a.GetSymbol(), "is_aromatic": a.GetIsAromatic(), "isotope": (a.GetIsotope() or None), "chirality": chirality, "h_count": (None if not a.GetNoImplicit() else a.GetTotalNumHs()), "charge": a.GetFormalCharge()}

"""
find an augmenting pth starting at the free vertex 'root'
* returns vertices (u0,v0,v1,u1,u2,v2,...) that will be paired in order.

bfs over vertices u and scan neighbors v
* if v is free: an augmenting endpoint is found
* if v is matched to m, we enqueue m as the next outer vertex
"""

def find_augmenting_path(graph:List[List[int]], root:int, matching:List[Optional[int]]) -> Optional[List[int]]:
  assert matching[root] is None
  other_end, node_queue = None, deque([root]) #    run modified BFS to find path from root to unmatched node
  parents = [None]*len(graph) #    parent BFS tree - None indicates an unvisited node
  parents[root] = [None, None]

  while node_queue:
    node = node_queue.popleft()
    for adj in graph[node]:
      if matching[adj] is None: #    unmatched node
        if adj != root: #    augmenting path found
          parents[adj] = [node,adj]
          other_end = adj
          break
      else:
        adj_mate = matching[adj]
        if parents[adj_mate] is None:
          parents[adj_mate] = [node,adj]
          node_queue.append(adj_mate)
    if other_end is not None: break
  if other_end is None: return None
  path, node = [], other_end
  while node != root:
    path.append(parents[node][1])
    path.append(parents[node][0])
    node = parents[node][0]
  return path

def flip_augmenting_path(matching, path):
  for i in range(0,len(path),2):
    a,b = path[i], path[i+1]
    matching[a] = b
    matching[b] = a

"""

build a maximal matching by always pairing a node with its first available free neighbor,
prioritizing nodes that currently have fewer free neighbors

"""

def greedy_matching(graph):
  n = len(graph)
  matching = [None]*n
  free_degrees = [len(adj) for adj in graph] #    free_degrees[i] = number of unmatched neighbors for node i
  #    prioritize nodes with fewer unmatched neighbors
  node_pqueue = [(free_degrees[i], i) for i in range(n)]
  heapq.heapify(node_pqueue)
  while node_pqueue:
    _,node = heapq.heappop(node_pqueue)
    if matching[node] is not None or free_degrees[node] == 0: continue #    node cannot be matched
    mate = next((i for i in graph[node] if matching[i] is None), None) #    match node with 1st unmatched neighbor
    if mate is None: continue
    matching[node] = mate
    matching[mate] = node

    #    Update free-degree estimates and push neighbors that remain candidates
    for adj in itertools.chain(graph[node], graph[mate]):
      free_degrees[adj] -= 1
      if matching[adj] is None and free_degrees[adj] > 0: heapq.heappush(node_pqueue, (free_degrees[adj], adj))
  return matching

def find_perfect_matching(graph:List[List[int]]) -> Optional[List[int]]:
  matching = greedy_matching(graph) #    start with a maximal matching for efficiency
  unmatched = set(i for i,m in enumerate(matching) if m is None)
  while unmatched:
    #    find augmenting path which starts at root
    root = unmatched.pop()
    path = find_augmenting_path(graph,root,matching)
    if path is None: return None
    flip_augmenting_path(matching,path)
    unmatched.discard(path[0])
    unmatched.discard(path[-1])
  return [int(x) for x in matching]

@dataclass
class Attribute:
  index:int
  token:str

@dataclass
class AttributeMap:
  index:int
  token:str
  attribution:List[Attribute] = field(default_factory=list)

class DirectedBond:
  __slots__ = 'src', 'dst', 'order', 'stereo', 'ring_bond'
  def __init__(self, src:int, dst:int, order:Union[int,float], stereo:Optional[str], ring_bond:bool):
    self.src:int = src
    self.dst:int = dst
    self.order:Union[int,float] = order
    self.stereo:Optional[str] = stereo
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
    return (a,b) in self._bonds

  def has_out_ring_bond(self, src:int) -> bool: return any(e is not None and e.ring_bond for e in self._adj[src])
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
    self._bond_cnts[a] += d
    self._bond_cnts[b] += d

  def _insert_edge(self, bond:DirectedBond, pos:int) -> None:
    self._bonds[(bond.src, bond.dst)] = bond
    out = self._adj[bond.src]
    if pos == -1 or pos == len(out): out.append(bond)
    elif out[pos] is None: out[pos] = bond
    else: out.insert(pos, bond)

  #    aromaticity / kekulization
  def is_kekulized(self) -> bool: return not self._delocal
  def kekulize(self) -> bool:
    #    based on Apodaca's article: https://depth-first.com/articles/2020/02/10/a-comprehensive-treatment-of-aromaticity-in-the-smiles-language/
    if self.is_kekulized(): return True
    ds = self._delocal
    kept = set(itertools.filterfalse(self._prune_from_ds, ds))
    label_to_node = sorted(kept)
    node_to_label = {v:i for i,v in enumerate(label_to_node)}
    pruned = [[] for _ in range(len(kept))]

    for u in kept:
      lu = node_to_label[u]
      pruned[lu].extend(node_to_label[v] for v in ds[u] if v in kept)

    matching = find_perfect_matching(pruned)
    if matching is None: return False

    for u,nbrs in ds.items(): #    de-aromatize and reset singles
      for v in nbrs: self.update_bond_order(u,v,new_order=1)
      self._atoms[u].SetIsAromatic(False)
      self._atoms[u].UpdatePropertyCache(strict=False)
      self._bond_cnts[u] = float(int(self._bond_cnts[u]))

    #    add double bonds according to matching (only use u < v once)
    for u,v in enumerate(matching):
      if u < v:
        a, b = label_to_node[u], label_to_node[v]
        self.update_bond_order(a, b, new_order=2)

    self._delocal = {}
    return True

  def _prune_from_ds(self, node:int) -> bool:
    adj = self._delocal[node]
    if not adj: return True #    aromatic atom with no aromatic bonds
    atom = get_custom_from_rdkit(self._atoms[node])
    valences = AROMATIC_VALENCES[atom["element"]]
    used = int(self._bond_cnts[node] - .5 * len(adj)) #    treat each ds bond (order 1.5) as a single when counting
    if atom['h_count'] is None: #    account for implicit Hs
      assert atom['charge'] == 0
      return any(used == v for v in valences)

    valence = valences[-1] - atom['charge']
    used += atom['h_count']
    bound_e = max(0, atom['charge']) + atom['h_count'] + int(self._bond_cnts[node]) + int(2*(self._bond_cnts[node]%1)) #    count the total number of bound electrons of each atom
    radical_e = max(0, VALENCE_ELECTRONS[atom['element']]-bound_e)%2 #    number of unpaired electrons of each atom
    free_e = valence - used - radical_e
    if any(used == v - atom['charge'] for v in valences): return True
    return not (free_e >= 0 and (free_e%2 != 0))

class Copy:
  def __call__(self, dat):
    data = copy.copy(dat)
    data.y = data.y[:,0]
    return data

if __name__ == '__main__':
  #    a) Matching: 4-cycle has a perfect matching
  G = [[1,3],[0,2],[1,3],[0,2]]
  assert find_perfect_matching(G) is not None

  #    b) Matching: triangle (odd cycle) has none
  H = [[1,2],[0,2],[0,1]]
  assert find_perfect_matching(H) is None

  #    c) Ring flag derived correctly
  db = DirectedBond(0,1,1.0,None,True)
  assert db.ring_bond is True


  graph = MolecularGraph()
  transform = T.Compose([Copy()])
  dataset = QM9('cache/QM9', transform=transform).shuffle()
  m = Chem.MolFromSmiles('Cc1ccccc1')
  for a in m.GetAtoms():
    pass