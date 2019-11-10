from CoEVOL import CoEVOL
import networkx as nx
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set()

T = 1
s = 1
A = np.empty((T, s), dtype=object)

c1 = nx.erdos_renyi_graph(2242, 0.01)
label1 = [0] * 2242

new_indices = 2242 + np.arange(3262)
c2 = nx.erdos_renyi_graph(3262, 0.01)
label2 = [1] * 3262

mapping = dict(zip(np.arange(3262), new_indices))
c2 = nx.relabel_nodes(c2, mapping)
g = nx.compose(c1, c2)

colors = np. concatenate( (np.array(['red'] * 2242), np.array(['blue'] * 3262)), axis=None)
labels = np.concatenate((np.array(label1), np.array(label2)), axis=None)

sparse = nx.to_scipy_sparse_matrix(g)
# A[0,0] = sparse.dot(sparse.transpose())
A[0,0] = sparse

coevol = CoEVOL(A, k=2)
U, X, Y = coevol.factorize()


plt.scatter(U[:,0], U[:,1], label=labels, c=colors)
plt.show()
