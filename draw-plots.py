import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

x = np.arange(5)
y = np.arange(5)
x_, y_ = np.meshgrid(x, y)

ks = ['2', '3', '5', '7', '10', '15']
thetas = ['0.1', '0.3', '0.5', '0.7', '0.9']

ks_, thetas_ = np.meshgrid(ks, thetas)

# ks, thetas = np.meshgrid(ks, thetas)
print(x)
print(y)
# Draw surface plot

errors = np.array( [
            [3.577, 3.564, 3.495, 3.489, 3.493], #k=2
            [, 4.256, 4.296, 4.058, 4.110], #k=3
            [5.320, 5.475, 5.464, 5.383, 5.208],  # k=5
            [, 6.416, , , ], #k=7
            [7.327, 7.655, 7.360, 7.498, 7.540],  # k=10
            [8.684, 8.915, 8.31, 8.23, 8.0]    # k = 15
            ] )


surf = ax.plot_surface(x_, y_, errors, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xticks(x)
ax.set_xticklabels(ks)
ax.set_xlabel('k (Latent Dimension)')


ax.set_yticks(y)
ax.set_yticklabels(thetas)
ax.set_ylabel('Theta (Decay Factor)')

ax.set_zlabel('RMSE')

plt.savefig('teste.pdf')
plt.show()
