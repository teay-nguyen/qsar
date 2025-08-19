import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tinygrad import Tensor, nn
from tinygrad.device import Device

print(f'using {Device.DEFAULT}')

class FeatureLayer:
  def __init__(self, feat_dropout):
    self.l = [nn.Linear(2048, 1024), Tensor.dropout]

  def __call__(self, x:Tensor):
    x = x.sequential(self.l)
    return x

class Tree:
  def __init__(self, depth, n_in_feature, used_feature_rate, n_class, jointly_training=True):
    self.depth = depth
    self.n_leaf = 2 ** depth
    self.n_class = n_class
    self.jointly_training = jointly_training

    n_used_feature = int(n_in_feature*used_feature_rate)
    onehot = Tensor(np.eye(n_in_feature), dtype='float32')
    using_idx = Tensor(np.random.choice(np.arange(n_in_feature), n_used_feature, replace=False), dtype='int')
    self.feature_mask = onehot[using_idx].T

    self.pi = Tensor.rand(self.n_leaf, n_class) if jointly_training else (Tensor.ones(self.n_leaf, n_class) / n_class)
    self.decision = [nn.Linear(n_used_feature, self.n_leaf), Tensor.sigmoid]

  def __call__(self, x:Tensor):
    feats = x.dot(self.feature_mask)
    decision = feats.sequential(self.decision)

    decision = Tensor.unsqueeze(decision, dim=2)
    decision_comp = 1 - decision
    decision = Tensor.cat(decision, decision_comp, dim=2)

    bs = x.shape[0]
    _mu = Tensor.ones(bs,1,1)
    begin_idx, end_idx = 1, 2
    for n_layer in range(self.depth):
      _mu = _mu.view(bs,-1,1).repeat(1,1,2)
      _decision = decision[:, begin_idx:end_idx, :]
      _mu = _mu*_decision
      begin_idx = end_idx
      end_idx = begin_idx + 2**(n_layer+1)

    mu = _mu.view(bs, self.n_leaf)
    return mu

  def get_pi(self):
    return Tensor.softmax(self.pi) if self.jointly_training else self.pi

class Forest:
  def __init__(self, n_tree, tree_depth, n_in_feature, tree_feature_rate, n_class, jointly_training):
    self.trees = []
    self.n_tree = n_tree
    for _ in range(n_tree):
      tree = Tree(tree_depth, n_in_feature, tree_feature_rate, n_class, jointly_training)
      self.trees.append(tree)

  def __call__(self, x):
    probs = []
    for tree in self.trees:
      mu = tree(x)
      p = mu.dot(tree.get_pi())
      probs.append(p.unsqueeze(2))
    probs = Tensor.cat(*probs, dim=2)
    prob = Tensor.sum(probs, axis=2) / self.n_tree
    return prob

class NDF:
  def __init__(self, feature_layer, forest):
    self.feature_layer = feature_layer
    self.forest = forest

  def __call__(self, x):
    out = self.feature_layer(x)
    out = out.view(x.shape[0], -1)
    out = self.forest(out)
    return out

#    this isn't training properly

if __name__ == '__main__':
  fp = 'data/DOWNLOAD-gU8RPQ5Wut7KaKJdHzr2fUYYJcpIjb0ClUND2cUakNk_eq_.csv'
  df = pd.read_csv(fp, delimiter=';')

  df = df.dropna(subset=['Smiles', 'pChEMBL Value'])
  df['mol'] = df['Smiles'].apply(Chem.MolFromSmiles)

  fpgen = AllChem.GetRDKitFPGenerator()
  df['fp'] = df['mol'].apply(fpgen.GetFingerprint)

  X_train = Tensor([list(fp) for fp in df['fp']], dtype='float32') 
  Y_train = Tensor(df['pChEMBL Value'].values, dtype='float32')

  bs = 128
  dropout = .3
  n_tree = 5
  n_class = 1
  tree_depth = 3

  tree_feature_rate = .5

  lr = .001
  jointly_training = False
  epochs = 10

  feat_layer = FeatureLayer(dropout)
  forest = Forest(n_tree, tree_depth, 1024, tree_feature_rate, n_class, jointly_training)
  model = NDF(feat_layer, forest)
  optim = nn.optim.Adam(nn.state.get_parameters(model), lr=lr, eps=1e-5)
  lossfn = lambda x,y: (x-y)**2

  def get_batch():
    i = 0
    while True:
      x = X_train[i:i+bs]
      y = Y_train[i:i+bs]
      yield x, y
      i += bs
      if (i+bs+1)>=X_train.shape[0]:
        i = 0

  for epoch in range(1,epochs+1):
    feat_batches, target_batches = [], []
    with Tensor.train(False):
      for batch_idx, (dat, target) in enumerate(get_batch()):
        feats = model.feature_layer(dat)
        feats = feats.view(feats.shape[0], -1)
        feat_batches.append(feats)

      for tree in model.forest.trees:
        mu_batches = []
        for feats in feat_batches:
          mu = tree(feats)
          mu_batches.append(mu)
        for _ in range(20):
          new_pi = Tensor.zeros((tree.n_leaf, tree.n_class))  # Tensor [n_leaf,n_class]
          for mu, target in zip(mu_batches, target_batches):
            pi = tree.get_pi()  # [n_leaf,n_class]
            prob = mu.dot(pi)  # [batch_size,n_class]

            # Variable to Tensor
            pi = pi.data
            prob = prob.data
            mu = mu.data

            _target = target.unsqueeze(1)  # [batch_size,1,n_class]
            _pi = pi.unsqueeze(0)  # [1,n_leaf,n_class]
            _mu = mu.unsqueeze(2)  # [batch_size,n_leaf,1]
            _prob = Tensor.clamp(prob.unsqueeze(1), min=1e-6, max=1.)  # [batch_size,1,n_class]

            _new_pi = Tensor.mul(Tensor.mul(_target, _pi), _mu) / _prob  # [batch_size,n_leaf,n_class]
            new_pi += Tensor.sum(_new_pi, axis=0)
          new_pi = Tensor.softmax(new_pi, axis=1).data
          tree.pi.data = new_pi

    with Tensor.train():
      for batch_idx, (dat, target) in enumerate(get_batch()):
        optim.zero_grad()
        out = model(dat).flatten()
        loss = lossfn(out, target).mean()
        loss.backward()
        optim.step()

        print(f'loss {loss.item()} batch_idx {batch_idx}')
