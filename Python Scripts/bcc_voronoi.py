import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Voronoi

points = np.array( [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [1, 1, 0],
                        [0, 0, 1],
                        [1, 0, 1],
                        [0, 1, 1],
                        [1, 1, 1],
                        [0.5, 0.5, 0.5],
                        [0.5, 0.5, 1.5],
                        [0.5, 0.5, -0.5],
                        [-0.5, 0.5, 0.5],
                        [1.5, 0.5, 0.5],
                        [0.5, -0.5, 0.5],
                        [0.5, 1.5, 0.5]

])

vor = Voronoi(points)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot Voronoi cells
for simplex in vor.ridge_vertices:
    simplex = np.asarray(simplex)
    if np.all(simplex >= 0):
        poly = Poly3DCollection([vor.vertices[simplex]], alpha=0.25)
        ax.add_collection3d(poly)

# Customize the plot (if needed)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.scatter3D(points[:,0], points[:,1], points[:,2])
ax.scatter3D(vor.vertices[:,0], vor.vertices[:,1], vor.vertices[:,2])

plt.show()

print(vor.vertices)
