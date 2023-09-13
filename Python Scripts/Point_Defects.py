import os
import time
import numpy as np
from lammps import lammps
from mpi4py import MPI
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import copy 
import json

# template to replace MPI functionality for single threaded use
class MPI_to_serial():

    def bcast(self, *args, **kwargs):

        return args[0]

    def barrier(self):

        return 0



class Lammps_Point_Defect():

    def __init__(self, size, n_vac, n_inter, potential_type):

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
        self.pot_type = potential_type

        if potential_type == 'Wang':
            
            self.potfolder = 'Potentials/Wang_Potential/'

            self.potfile_WH = self.potfolder + 'WHff.eam.alloy'

            self.potfile_He = self.potfolder + 'He-Beck1968_modified.table'

            self.potfile_WHe = self.potfolder + 'W-He-Juslin.table'

        else:

            self.potfolder = 'Potentials/Tungsten_Hydrogen_Helium/'

            self.potfile = self.potfolder + 'WHHe_final.eam.alloy'

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

        if self.pot_type == 'Wang':

            lmp.command('pair_style hybrid/overlay eam/alloy zbl 1.4 1.8 zbl 0.9 1.5 zbl 0.2 0.35 lj/cut 7.913 table spline 10000 table spline 10000')
            lmp.command('pair_coeff      * *  eam/alloy  %s W H  NULL' % self.potfile_WH)
            lmp.command('pair_coeff 1 1 zbl 1 74.0 74.0')
            lmp.command('pair_coeff 2 2 zbl 3 1.0 1.0')
            lmp.command('pair_coeff 1 2 zbl 2 74.0 1.0')
            lmp.command('pair_coeff      2 3 lj/cut 5.9225e-4 1.333')
            lmp.command('pair_coeff      3 3 table 1 %s HeHe' % self.potfile_He)
            lmp.command('pair_coeff      1 3 table 2 %s WHe'  % self.potfile_WHe)

        else:

            lmp.command('pair_style eam/alloy' )

            lmp.command('pair_coeff * * %s W H He' % self.potfile)

        N_anneal = int(2)

        t_anneal = int(1e2)

        E0 = 5

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

                lmp.command('run %d' % t_anneal)
                
                lmp.command('minimize 1e-5 1e-8 10 10')

                lmp.command('minimize 1e-5 1e-8 10 100')
                
                #lmp.command('fix free all box/relax aniso 0.0')

                lmp.command('minimize 1e-5 1e-8 100 1000')

                lmp.command('run 0')

                lmp.command('write_dump all atom Lammps_Dump/(Vac:%d)(H:%d)(He:%d).atom' % 
                            (self.n_vac, len(xyz_inter[1]), len(xyz_inter[2])))


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

        central_sites = [ (i+1)*np.array([0.5, 0.5, 0.5]) for i in range(self.n_vac)]

        return np.array(central_sites)
    
    def get_all_sites(self):
        oct =  Instance.size//2 + Instance.get_octahedral_sites()

        tet = Instance.size//2 + Instance.get_tetrahedral_sites()

        diag = Instance.size//2 + Instance.get_diagonal_sites()

        central = Instance.size//2 + Instance.get_central_sites()

        return [oct, tet, diag, central]


def add_intersitial(Instance, element_idx, previous_config, available_sites):

    #Initialize lists to store the energies and sites of a test configuration
    test_config_idx = [0 for i in range(len(available_sites))]

    test_config_pe = np.zeros(shape=(len(available_sites) ,))

    for site_type_idx in range(len(available_sites)):

        if len(available_sites[site_type_idx]) > 0:
            
            #Initialize an array to store the distance between previous sites and the potentially available sites
            distance = np.zeros(shape= (len(available_sites[site_type_idx]),) )

            #Loop through each occupied site and find the distance between the occupied site and an available ite
            for prev_site_idx in range(len(previous_config[element_idx])):

                prev_site_element = np.array(previous_config[element_idx][prev_site_idx])

                distance += np.linalg.norm(available_sites[site_type_idx] - prev_site_element, axis = 1)
            
            #Test the site which maximally distant from previously occupied sites
            test_config_idx[site_type_idx] = int(np.argmax(distance))

            test_config = copy.deepcopy(previous_config)

            test_config[element_idx].append(available_sites[site_type_idx][int(np.argmax(distance))])

            #Use LAMMPs to determine the Potential of this configuration
            test_config_pe[site_type_idx] = Instance.Build_Defect(test_config)
        
        else:

            test_config_pe[site_type_idx] = np.inf
            
    #Choose the Config which minimizes the energy of the system
    min_pe_idx = int(np.argmin(test_config_pe))

    new_config = copy.deepcopy(previous_config)

    #Add the new site to the config
    new_config[element_idx].append(available_sites[min_pe_idx][test_config_idx[min_pe_idx]].tolist())

    new_config_pe = np.min(test_config_pe)

    #Remove the chosen site from the available sites
    available_sites[min_pe_idx] = np.delete(available_sites[min_pe_idx], test_config_idx[min_pe_idx], axis= 0)

    return new_config_pe, new_config, available_sites

def sequential_clustering(Instance, n_vac, element_idx, max_occupancy, init_config = [[], [], []]):

    Instance.n_vac = 0

    perfect = Instance.Build_Defect()

    available_sites = Instance.get_all_sites()

    intersitial, _, __ = add_intersitial(Instance, element_idx, [[], [], []], available_sites)

    defect_energies =[]
    defect_config   = []

    Instance.n_vac = n_vac

    available_sites = Instance.get_all_sites()

    defect_energies.append(Instance.Build_Defect(init_config))


    for i in range(max_occupancy):

        defect_energy, init_config, available_sites = add_intersitial(Instance, element_idx, init_config, available_sites)

        defect_config.append(init_config)

        defect_energies.append(defect_energy)

    defect_energies = np.array(defect_energies)

    binding_energies = intersitial - perfect - np.diff(defect_energies) 

    return binding_energies, defect_config


size = 7

b_energy = np.array([-8.949, -4.25/2, 0])

pot_type = 'Daniel'

Instance = Lammps_Point_Defect(size, 0, 0, pot_type)

available_sites = Instance.get_all_sites()

perfect = Instance.Build_Defect()

h_int   = Instance.Build_Defect([[],[available_sites[1][0]], []])

Instance.n_vac = 2

vacancy = Instance.Build_Defect()

available_sites = Instance.get_all_sites()

intersitial_sites = [[], [available_sites[1][0]], []]

Instance.n_inter = np.array([len(element) for element in intersitial_sites])

defect = Instance.Build_Defect(intersitial_sites)

N_atoms = 2*size**3

data = {}

data['energy'] = {}

data['config'] = {}

# data = Element_in_Vacancy(Instance, element_idx=2, max_intersitials=6)
data['energy']['He_x + He'], data['config']['He_x + He'],  = sequential_clustering(Instance,n_vac = 0,
                                                       element_idx=2, max_occupancy=6, init_config=[[], [],[]])

data['energy']['VHe_x + He'], data['config']['VHe_x + He'] = sequential_clustering(Instance,n_vac = 1, 
                                                                                 element_idx=2, max_occupancy=7, init_config=[[],[],[]])

data['energy']['V_2He_x + He'], data['config']['V_2He_x + He'] = sequential_clustering(Instance,n_vac = 2,
                                                                                  element_idx=2, max_occupancy=7, init_config=[[],[],[]])

data['energy']['H_x + H'], data['config']['H_x + H'],  = sequential_clustering(Instance,n_vac = 0,
                                                       element_idx=1, max_occupancy=6, init_config=[[], [],[]])

data['energy']['VH_x + H'], data['config']['VH_x + H'],  = sequential_clustering(Instance,n_vac = 1,
                                                       element_idx=1, max_occupancy=6, init_config=[[], [],[]])



if Instance.me == 0:

    # #print(defect -  perfect*((N_atoms-1)/N_atoms) - np.dot(Instance.n_inter  ,b_energy))
    # print(defect - vacancy - h_int + perfect)
    # print('H + Vacancy',defect, 'H Intersitial' , h_int,'Vacancy' ,vacancy,'Perfect', perfect,'Gas in Vacuum', np.dot(Instance.n_inter  ,b_energy))
    # print('Vacancy',vacancy - perfect - 8.949)
    # print('H-Intersitial',h_int - perfect - np.dot(Instance.n_inter  ,b_energy))

    # file_path = 'Data/Defect Analysis/Helium_in_Vacancy_%s.json' % pot_type
    # with open(file_path, "w") as json_file:
    #     json.dump(data, json_file, indent=4)  # indent for pretty formatting (optional)
    print(data['energy'])


