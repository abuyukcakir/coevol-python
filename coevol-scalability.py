from CoEVOL import CoEVOL
import networkx as nx
import time
import numpy as np
import matplotlib.pyplot as plt

T = 1
s = 1
A = np.empty((T, s), dtype=object)

num = [500, 1000, 5000, 10000, 20000, 40000]
num_edges = []

for i in range(len(num)):
    num_edges.append(num[i] * (num[i] - 1) / 200)

timings = []

for i in range(len(num)):
    g = nx.erdos_renyi_graph(num[i], 0.01)

    A[0, 0] = nx.to_scipy_sparse_matrix(g)

    # Start timing
    start_time = time.time()

    coevol = CoEVOL(A, k=2)
    coevol.factorize()
    end_time = time.time()
    timings.append(end_time - start_time)

plt.loglog(num_edges, timings)

plt.title('Scalability of CoEVOL on Erdos-Renyi with $G(n, p=0.01)$')
plt.xlabel('Number of Edges in Erdos-Renyi Graph')
plt.ylabel('Execution Time (seconds)')

locs, labs = plt.xticks()
plt.xticks(locs[1:7])

plt.savefig('scalability.pdf')
plt.show()
