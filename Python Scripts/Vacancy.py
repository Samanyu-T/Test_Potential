
import os
import time
import numpy as np
from lammps import lammps
from mpi4py import MPI
from itertools import combinations_with_replacement
from scipy.optimize import minimize, Bounds
from scipy.spatial import cKDTree


def create_BCC(n_vac):

    n = n_vac + 1 

    x  = []

    for i in range(n):
        for j in range(n):
            for k in range(n):

                x.append([i,j,k])
                x.append([i+0.5, j+0.5, k+0.5])

    x = np.array(x)

    del_idx = []

    for vac in range(n_vac):
        matches = np.all(x == (vac + 1)*np.array([0.5, 0.5, 0.5]), axis=1)
        del_idx.append(np.where(matches)[0][0])
    
    x = np.delete(x, del_idx, axis = 0)

    return x

def search_space(n_vac):
    
    step_size = 0.25
    box_size  = (1 + n_vac)/2
    n_samples = int(box_size/step_size) + 1 

    x = np.linspace(0, box_size, n_samples)
    xx,yy,zz = np.meshgrid(x,x,x)

    xyz = np.hstack([xx.reshape(n_samples**3,1), yy.reshape(n_samples**3,1), zz.reshape(n_samples**3,1)])

    #Ensure that there are no overlaps between Tungsten sites and Search Space
    bcc = create_BCC(n_vac)

    del_idx  = []

    for i, _xyz in enumerate(xyz):
        check = np.all(bcc==_xyz, axis = 1)

        if True in check:
            del_idx.append(i)

    xyz = np.delete(xyz, del_idx, axis = 0)

    return xyz


def find_neighbors(xyz):

    kdtree = cKDTree(xyz)

    neighbors = kdtree.query_ball_point(xyz , 1/4)

    return neighbors

def acceptance_probability(delta, T):

    if delta < 0:
        return 1
    else:
        return np.exp(-delta/T)


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


    def Build_Vacancy(self, size, n_h, n_he, n_vac, x_init):

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
                        % (i, size//2 + (i + 1)/2, size//2 + (i+1)/2, size//2 + (i+1)/2))
            
            lmp.command('delete_atoms region r_vac_%d ' % i)
        
        if n_he + n_h > 0:

            for i in range(n_h):
                lmp.command('create_atoms 2 single %f %f %f units lattice' 
                            % ( size//2 + x_init[i, 0], size//2 + x_init[i,1], size//2 + x_init[i, 2]))   

            for i in range(n_he):
                lmp.command('create_atoms 3 single %f %f %f units lattice' 
                            % ( size//2 + x_init[i + n_h,0], size//2 + x_init[i + n_h,1], size//2 + x_init[i + n_h,2]))         

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

        lmp.command('minimize 1e-5 1e-8 10 10')

        lmp.command('minimize 1e-5 1e-8 100 100')

        lmp.command('minimize 1e-5 1e-8 100 1000')

        lmp.command('compute potential all pe/atom')

        lmp.command('run 0')
        
        pe = lmp.get_thermo('pe')

        lmp.command('write_dump all atom dump.atom')
        lmp.close()

        return pe
    
    def simulated_annealing(self, size, n_h, n_he, n_vac, k_max, T0):

        state_space = search_space(n_vac)

        neighbors = find_neighbors(state_space)

        state_idx = np.random.randint(0, len(state_space), n_h + n_he)

        state  = state_space[state_idx]

        temp_state_idx = state_idx

        temp_state  = state

        state_pe = self.Build_Vacancy(size=size, n_h=n_h, n_he=n_he, n_vac=n_vac, x_init=state)

        if self.me == 0:
            with open('Test_Anneal.txt', 'w') as f:
                f.write('')

        for k in range(k_max):

            if self.me == 0:
                with open('Test_Anneal.txt', 'a') as f:
                    f.write(np.array2string(state) + '\t' + str(state_pe) + '\n')

            T = T0*(1 - (k - 1)/k_max)

            for i_state, s_idx in enumerate(state_idx):

                rand_neighbour = np.random.randint(0, len(neighbors[s_idx]))

                temp_state_idx[i_state] = neighbors[s_idx][rand_neighbour]

                temp_state[i_state] = state_space[temp_state_idx[i_state]]
            
            temp_state_pe = self.Build_Vacancy(size=size, n_h=n_h, n_he=n_he, n_vac=n_vac, x_init=temp_state)

            acceptance = acceptance_probability(temp_state_pe - state_pe, T)

            rand_num = np.random.rand(1)
            
            if rand_num <= acceptance:

                state = temp_state
                state_idx = temp_state_idx
                state_pe = temp_state_pe

        return state, state_pe
    

size  = 7
n_vac = 1
n_he  = 0
n_h   = 1
n_atoms = 2*size**3

Instance = Lammps_Vacancy()

b_energy = np.array([-8.949, -4.25/2, 0])

k_max = 10
T0 = 4

perfect = Instance.Build_Vacancy(size = size, n_h=0, n_he=0, n_vac=0, x_init=np.array([[1,0.25,0.5]]))
#vac_pos, vacancy = Instance.simulated_annealing(size =size, n_h=n_h, n_he=n_he, n_vac=n_vac, k_max=k_max, T0=T0)
pure_vacancy = Instance.Build_Vacancy(size = size, n_h=0, n_he=0, n_vac=n_vac, x_init=np.array([[0.5,0.5,0.25]]))
#perfect = n_atoms*b_energy[0]
hydrogen = Instance.Build_Vacancy(size = size, n_h=n_h, n_he=n_he, n_vac=n_vac, x_init=np.array([[0.5,   0.5 ,0.5]]))

if Instance.me == 0:
    print(hydrogen - perfect + n_vac*b_energy[0] - b_energy[1]*n_h - b_energy[2]*n_he)
    print(hydrogen - pure_vacancy - b_energy[1]*n_h - b_energy[2]*n_he)
    #print(vacancy - perfect + n_vac*b_energy[0])

MPI.Finalize()
