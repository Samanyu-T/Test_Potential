import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

def create_cell_vac(n):

    x = []

    if n % np.floor(n) == 0: 
        n = int(n)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    x.append([i,j,k])
                    if i < n-1 and j < n-1 and k < n-1:
                        x.append([i + 0.5,j + 0.5, k + 0.5])
        x = np.array(x)

    else:
        n_even = int(np.floor(n))
        for i in range(n_even):
            for j in range(n_even):
                for k in range(n_even):
                    x.append([i,j,k])

        x = np.array(x)
        x = np.vstack([x, x + 0.5])


    for i in range(int(n*2)-5):
            mask = np.all(x == i/2 + np.array([1, 1, 1]), axis = 1)
            mask = np.invert(mask)
            x = x[mask]
    return x

def cost_function(x, n_vac):

    x = x.reshape(len(x)//3,3)
    unit_cube = create_cell_vac(n_vac/2 + 2.5)

    cost = 0
    for i, xi in enumerate(x):

        for j, xj in enumerate(x):

            if i != j:

                cost += 2/np.linalg.norm(xi - xj)**2
            
        cost += np.sum(np.linalg.norm(unit_cube - xi, axis = 1)**2)

    return cost

n_vac = 2
n_atoms = 4

x0 =  np.random.uniform(low = 0.5, high = 1.5 + (n_vac -1), size = (n_atoms*3,))

x0 = x0.flatten()
bnds = []
for i in range(x0.shape[0]):
    bnds.append([0.5, 1.5 + (n_vac -1)])

x_opt = minimize(cost_function, x0=x0,args=(n_vac), method = 'Nelder-Mead' , bounds= bnds)

x_opt = minimize(cost_function, x0=x_opt.x, args = (n_vac), method = 'Nelder-Mead', bounds= bnds)


unit_cube = create_cell_vac(n_vac/2 + 2.5)

x_max = x_opt.x.reshape(len(x0)//3, 3)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter3D(unit_cube[:,0], unit_cube[:,1], unit_cube[:,2])
ax.scatter3D(x_max[:,0], x_max[:,1], x_max[:,2])
plt.show()

print(np.linalg.norm(x_max - x_max[0], axis = 1))
print(x_max - 0.5)
print(cost_function(x_opt.x, n_vac))