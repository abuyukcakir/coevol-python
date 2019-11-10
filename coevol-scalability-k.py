from CoEVOL import CoEVOL
import networkx as nx
import time
import numpy as np
import matplotlib.pyplot as plt

T = 1
s = 1
A = np.empty((T, s), dtype=object)

nodes= 10000
k = [3, 6, 9, 12, 15]

timings = []
g = nx.erdos_renyi_graph(nodes, 0.01)

for i in range(len(k)):

    A[0, 0] = nx.to_scipy_sparse_matrix(g)

    # Start timing
    start_time = time.time()

    coevol = CoEVOL(A, k=k[i])
    coevol.factorize()
    end_time = time.time()
    timings.append(end_time - start_time)

plt.plot(k, timings)

plt.title('CoEVOL on Erdos-Renyi with $G(100, p=0.01)$, changing $k$')
plt.xlabel('Latent dimension $k$')
plt.ylabel('Execution Time (seconds)')

locs, labs = plt.xticks()
plt.xticks(locs[1:10])

plt.savefig('scalability-k.pdf')
plt.show()
