
import os
import time
import numpy as np
from lammps import lammps
from mpi4py import MPI
from itertools import combinations_with_replacement
from scipy.optimize import minimize, Bounds

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
    unit_cube = create_cell_vac(n_vac)

    cost = 0
    for i, xi in enumerate(x):

        for j, xj in enumerate(x):

            if i != j:

                cost += 0.5/np.linalg.norm(xi - xj)**2
            
        cost += np.sum(np.linalg.norm(unit_cube - xi, axis = 1)**2)

    return cost

def initial_guess(n_vac, n_atoms):

    x0 =  np.random.uniform(low = 0.5, high = 1.5 + (n_vac -1), size = (n_atoms*3,))

    x0 = x0.flatten()
    bnds = []

    for i in range(x0.shape[0]):
        bnds.append([0.5, 1.5 + (n_vac -1)])

    x_opt = minimize(cost_function, x0=x0,args=(n_vac), method = 'Nelder-Mead' , bounds= bnds)

    x_opt = minimize(cost_function, x0=x_opt.x, args = (n_vac), method = 'Nelder-Mead', bounds= bnds)

    return x_opt.x.reshape(len(x0)//3, 3) - 0.5

# template to replace MPI functionality for single threaded use
class MPI_to_serial():

    def bcast(self, *args, **kwargs):

        return args[0]

    def barrier(self):

        return 0



class Lammps_Vacancy():

    def __init__(self):

        # try running in parallel, otherwise single thread
        try:

            self.comm = MPI.COMM_WORLD

            self.me = self.comm.Get_rank()

            self.nprocs = self.comm.Get_size()

            self.mode = 'MPI'

        except:

            self.me = 0

            self.nprocs = 1

            self.comm = MPI_to_serial()

            self.mode = 'serial'

        self.comm.barrier()

        
    def Build_Vacancy(self, size, n_he, n_h, n_vac):

        potfolder = 'Potentials/Tungsten_Hydrogen_Helium/'

        potfile = potfolder + 'WHHe_final.eam.alloy'

        lmp = lammps()

        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('boundary p p p')

        lmp.command('lattice bcc %f orient x 1 0 0 orient y 0 1 0 orient z 0 0 1' % 3.14484257)

        lmp.command('region r_simbox block 0 %d 0 %d 0 %d units lattice' % (size, size, size))
        
        lmp.command('create_box 3 r_simbox')
        
        lmp.command('create_atoms 1 box')

        for i in range(n_vac):

            lmp.command('region r_vac_%d sphere %f %f %f 0.1 units lattice' 
                        % (i, size//2 + (i+1)/2, size//2 + (i+1)/2, size//2 + (i+1)/2))
            
            lmp.command('delete_atoms region r_vac_%d ' % i)
         


        #lmp.command('create_atoms 3 single 2.4 2.4 2.4 units lattice')
            x_init = initial_guess(n_vac, 3)

            print(x_init)
            
            lmp.command('create_atoms 3 single %f %f %f units lattice' 
                        % ( size//2 + x_init[0,0], size//2 + + x_init[0,1], size//2 + + x_init[0,2]))
            lmp.command('create_atoms 3 single %f %f %f units lattice' 
                        % ( size//2 + + x_init[1,0], size//2 + + x_init[1,1], size//2 + + x_init[1,2]))
            lmp.command('create_atoms 3 single %f %f %f units lattice' 
                        % ( size//2 + + x_init[2,0], size//2 + + x_init[2,1], size//2 + + x_init[2,2]))
            
        #lmp.command('create_atoms 3 single 2.6 2.4 2.7 units lattice')


        lmp.command('mass 1 183.84')

        lmp.command('mass 2 1.00784')

        lmp.command('mass 3 4.002602')

        lmp.command('pair_style eam/alloy' )

        lmp.command('pair_coeff * * %s W H He' % potfile)

        lmp.command('run 0')

        lmp.command('timestep %f' % 2e-3)

        lmp.command('thermo 50')

        lmp.command('thermo_style custom step temp pe press') 

        lmp.command('run 200')

        lmp.command('minimize 1e-5 1e-8 100 1000')

        lmp.command('compute potential all pe/atom')

        lmp.command('run 0')
        
        pe = lmp.get_thermo('pe')

        lmp.close()

        return pe
    
Instance = Lammps_Vacancy()

size  = 7
n_vac = 1
n_he  = 0
n_h   = 0

b_energy = np.array([-8.949, -4.25/2, 0])

perfect = -8.95*2*size**3 
perfect = Instance.Build_Vacancy(size, 0, 0, 0)

vacancy = Instance.Build_Vacancy(size, n_h, n_he, n_vac)

if Instance.me == 0:
    print(vacancy - perfect + n_vac*b_energy[0] - b_energy[1]*n_h - b_energy[2]*n_he)

MPI.Finalize()