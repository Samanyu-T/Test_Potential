import os
import time
import numpy as np
from lammps import lammps
from mpi4py import MPI
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import copy 

# template to replace MPI functionality for single threaded use
class MPI_to_serial():

    def bcast(self, *args, **kwargs):

        return args[0]

    def barrier(self):

        return 0



class Lammps_Point_Defect():

    def __init__(self, size, n_vac, n_inter):

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

        self.size  = int(size)
        self.n_vac = int(n_vac)
        self.n_inter = n_inter

        potfolder = 'Potentials/Tungsten_Hydrogen_Helium/'

        self.potfile = potfolder + 'W_H_He.eam.alloy'

    def Build_Defect(self, xyz_inter = [[], [], []]):

        ''' xyz_inter gives a list of the intersitial atoms for each species i,e W H He in that order
            they are in lattice units and are consistent with the Lammps co-ords of the cell'''

        lmp = lammps()

        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('boundary p p p')

        lmp.command('lattice bcc %f orient x 1 0 0 orient y 0 1 0 orient z 0 0 1' % 3.14484257)

        lmp.command('region r_simbox block 0 %d 0 %d 0 %d units lattice' % (self.size, self.size, self.size))
                    
        lmp.command('create_box 3 r_simbox')
        
        lmp.command('create_atoms 1 box')

        #Create a Vacancy of n-atoms along the <1,1,1> direction the vacancy will be at the centre of the cell

        for i in range(self.n_vac):
            lmp.command('region r_vac_%d sphere %f %f %f 0.1 units lattice' 
                        % (i, self.size//2 + (i + 1)/2, self.size//2 + (i+1)/2, self.size//2 + (i+1)/2))
            
            lmp.command('delete_atoms region r_vac_%d ' % i)

        #Create the set of intersitials

        for element, xyz_element in enumerate(xyz_inter):
            for xyz in xyz_element:
                lmp.command('create_atoms %d single %f %f %f units lattice' % (element + 1, xyz[0], xyz[1], xyz[2]))



        rng_num = np.random.randint(low = 0, high = 10000)

        lmp.command('mass 1 183.84')

        lmp.command('mass 2 1.00784')

        lmp.command('mass 3 4.002602')

        mass = 1.03499e-4* np.array([183.84, 10.00784, 4.002602])

        lmp.command('pair_style eam/alloy' )

        lmp.command('pair_coeff * * %s W H He' % self.potfile)

        N_anneal = 2

        E0 = 2

        EN = 1

        constant = (2/(3*8.617333262e-5))

        T0 = constant*E0

        TN = constant*EN

        decay_constant = (1/N_anneal)*np.log(T0/TN)

        for i in range(N_anneal):

            for element in range(len(xyz_inter)):
                
                lmp.command('group int_%d type %d' % (element+1, element+1))

                if len(xyz_inter[element]) > 1:

                    T = T0*np.exp(-decay_constant*i)

                    lmp.command('velocity int_%d create %f %d dist gaussian mom yes rot no units box' % (element+1 ,T, rng_num) )
                
                elif len(xyz_inter[element]) == 1:

                    E = E0*np.exp(-decay_constant*i)

                    speed = np.sqrt(2*E/mass[element])

                    unit_vel = np.hstack( [np.random.randn(1), np.random.randn(1), np.random.randn(1)] )

                    unit_vel /= np.linalg.norm(unit_vel)

                    vel = speed*unit_vel

                    lmp.command('velocity int_%d set %f %f %f sum yes units box' % (element+1,vel[0], vel[1], vel[2]))

            lmp.command('run 0')

            lmp.command('timestep %f' % 2e-3)

            lmp.command('thermo 50')

            lmp.command('thermo_style custom step temp pe press') 

            lmp.command('run 100')

            lmp.command('minimize 1e-5 1e-8 10 10')

            lmp.command('minimize 1e-5 1e-8 100 100')

            lmp.command('minimize 1e-5 1e-8 100 1000')

            lmp.command('run 0')

        pe = lmp.get_thermo('pe')

        lmp.close()

        return pe
    
    def get_octahedral_sites(self):

        oct_sites_0 = np.zeros((3,3))

        k = 0

        for [i,j] in itertools.combinations([0, 1, 2],2):
            oct_sites_0[k,[i,j]] = 0.5
            k += 1
            
        oct_sites_1 = np.ones((3,3))
        k = 0

        for [i,j] in itertools.combinations([0, 1, 2],2):
            oct_sites_1[k,[i,j]] = 0.5
            k += 1

        oct_sites_unit = np.vstack([oct_sites_0, oct_sites_1])

        n_iter = np.clip(self.n_vac, a_min = 1, a_max = None)

        oct_sites = np.vstack([oct_sites_unit + i*0.5 for i in range(n_iter)])

        return np.unique(oct_sites, axis = 0)

    def get_tetrahedral_sites(self):

        tet_sites_0 = np.zeros((12,3))
        k = 0

        for [i,j] in itertools.combinations([0, 1, 2],2):
            tet_sites_0[4*k:4*k+4,[i,j]] = np.array( [[0.5 , 0.25],
                                                [0.25, 0.5],
                                                [0.5 , 0.75],
                                                [0.75, 0.5] ])

            k += 1

        tet_sites_1 = np.ones((12,3))
        k = 0

        for [i,j] in itertools.combinations([0, 1, 2],2):
            tet_sites_1[4*k:4*k+4,[i,j]] = np.array( [[0.5 , 0.25],
                                                [0.25, 0.5],
                                                [0.5 , 0.75],
                                                [0.75, 0.5] ])

            k += 1

        tet_sites_unit = np.vstack([tet_sites_0, tet_sites_1])

        n_iter = np.clip(self.n_vac, a_min = 1, a_max = None)

        tet_sites = np.vstack([tet_sites_unit + i*0.5 for i in range(n_iter)])

        return np.unique(tet_sites, axis = 0)
    
    def get_diagonal_sites(self):

        diag_sites_0 = np.array([ 
                                [0.25, 0.25, 0.25],
                                [0.75, 0.75, 0.75],
                                [0.25, 0.75, 0.75],
                                [0.75, 0.25, 0.25],
                                [0.75, 0.25, 0.75],
                                [0.25, 0.75, 0.25],
                                [0.75, 0.75, 0.25],
                                [0.25, 0.25, 0.75]
                            ])    


        n_iter = np.clip(self.n_vac, a_min = 1, a_max = None)

        diag_sites_unit = np.vstack([diag_sites_0 + i*0.5 for i in range(n_iter)])

        return np.unique(diag_sites_unit, axis = 0)
    
    def get_central_sites(self):

        central_sites = [ (i+1)*[0.5, 0.5, 0.5] for i in range(self.n_vac)]

        return np.array(central_sites)


def optimize_sites(Instance):

    #Find the sites of interest in a BCC crystal with a vacancy

    oct =  size//2 + Instance.get_octahedral_sites()

    tet = size//2 + Instance.get_tetrahedral_sites()

    diag = size//2 + Instance.get_diagonal_sites()

    central = size//2 + Instance.get_central_sites()

    available_sites = [oct, tet, diag, central]

    store_sites = []

    store_sites.append([[], [], []])

    previous_sites = copy.deepcopy(store_sites[0])

    energies = []

    for n_vac in range(1,2):

        Instance.n_vac = n_vac

        for n_h in range(10):

            new_site_idx = [0, 0, 0, 0]

            new_site_pe = np.zeros(shape=(len(available_sites) ,))

            for site_type_idx in range(len(available_sites)):

                if len(available_sites[site_type_idx]) > 0:

                    distance = np.zeros(shape= (len(available_sites[site_type_idx]),) )
                    test_sites = []

                    for prev_site_idx in range(len(previous_sites[1])):

                        prev_site_element = np.array(previous_sites[1][prev_site_idx])

                        distance += np.linalg.norm(available_sites[site_type_idx] - prev_site_element, axis = 1)
                    
                    new_site_idx[site_type_idx] = int(np.argmax(distance))

                    test_sites.append(copy.deepcopy(previous_sites))

                    test_sites[0][1].append(available_sites[site_type_idx][new_site_idx[site_type_idx]])

                    new_site_pe[site_type_idx] = Instance.Build_Defect(test_sites[0])
                
                else:

                    new_site_pe[site_type_idx] = np.inf

            min_pe_idx = np.argmin(new_site_pe)

            previous_sites[1].append(available_sites[min_pe_idx][new_site_idx[min_pe_idx]])

            available_sites[min_pe_idx] = np.delete(available_sites[min_pe_idx], new_site_idx[min_pe_idx], axis= 0)

            store_sites.append(copy.deepcopy(previous_sites))

            energies.append(new_site_pe[min_pe_idx])

    return store_sites, energies


size = 7

b_energy = np.array([-8.949, -4.25/2, 0])

Instance = Lammps_Point_Defect(size, 0, 0)


oct =  size//2 + Instance.get_octahedral_sites()

tet = size//2 + Instance.get_tetrahedral_sites()

diag = size//2 + Instance.get_diagonal_sites()

central = size//2 + Instance.get_central_sites()

perfect = Instance.Build_Defect()

h_int   = Instance.Build_Defect([[],[tet[0]], []])

Instance.n_vac = 1

vacancy = Instance.Build_Defect()

oct =  size//2 + Instance.get_octahedral_sites()

tet = size//2 + Instance.get_tetrahedral_sites()

diag = size//2 + Instance.get_diagonal_sites()

central = size//2 + Instance.get_central_sites()

intersitial_sites = [[], [oct[0]], []]

Instance.n_inter = np.array([len(element) for element in intersitial_sites])

defect = Instance.Build_Defect(intersitial_sites)

N_atoms = 2*size**3

sites, defect_energies = optimize_sites(Instance)

if Instance.me == 0:
    #print(defect -  perfect*((N_atoms-1)/N_atoms) - np.dot(Instance.n_inter  ,b_energy))
    print(defect - vacancy - h_int + perfect)
    print('H + Vacancy',defect, 'H Intersitial' , h_int,'Vacancy' ,vacancy,'Perfect', perfect,'Gas in Vacuum', np.dot(Instance.n_inter  ,b_energy))
    print('Vacancy',vacancy - perfect - 8.949)
    print('H-Intersitial',h_int - perfect - np.dot(Instance.n_inter  ,b_energy))
    print(sites)
    print(defect_energies)

