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

    def __init__(self, size, n_vac, potential_type):

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
        self.pot_type = potential_type

        if potential_type == 'Wang':
            
            self.potfolder = 'Potentials/Wang_Potential/'

            self.potfile_WH = self.potfolder + 'WHff.eam.alloy'

            self.potfile_He = self.potfolder + 'He-Beck1968_modified.table'

            self.potfile_WHe = self.potfolder + 'W-He-Juslin.table'

        else:

            self.potfolder = 'Potentials/Tungsten_Hydrogen_Helium/'

            self.potfile = self.potfolder + 'WHHe_test.eam.alloy'
        
        self.savename = 'rand'

    def Build_Defect(self, xyz_inter = [[], [], []], alattice = 3.14484257):

        ''' xyz_inter gives a list of the intersitial atoms for each species i,e W H He in that order
            they are in lattice units and are consistent with the Lammps co-ords of the cell'''
        

        folder_name = "Animation/%s" % self.savename

        if Instance.me == 0:
            # Check if the folder exists
            if not os.path.exists(folder_name):
                # If it doesn't exist, create the folder
                os.makedirs(folder_name)
                print(f"Folder '{folder_name}' created.")
            else:
                print(f"Folder '{folder_name}' already exists.")

            
            # List all files in the folder
            files = os.listdir(folder_name)

            try:
                # Loop through the files and delete each one
                for file in files:
                    file_path = os.path.join(folder_name, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            except:
                print('all deleted')

        lmp = lammps()

        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('boundary p p p')

        lmp.command('lattice bcc %f orient x 1 0 0 orient y 0 1 0 orient z 0 0 1' % alattice)

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

        lmp.command('mass 1 183.84')

        lmp.command('mass 2 1.00784')

        lmp.command('mass 3 4.002602')

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

        lmp.command('run 0')

        lmp.command('thermo 50')

        lmp.command('compute pot all pe/atom')

        lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz')

        lmp.command('dump myDump all custom 1 Animation/%s/animation.*.dump id type x y z c_pot' % self.savename)

        lmp.command('timestep 1e-3')

        lmp.command('min_style quickmin')

        lmp.command('minimize 1e-15 1e-18 10 10')

        lmp.command('minimize 1e-15 1e-18 10 100')

        lmp.command('minimize 1e-15 1e-18 100 1000')
        
        pe = lmp.get_thermo('pe')

        pxx = lmp.get_thermo('pxx')
        pyy = lmp.get_thermo('pyy')
        pzz = lmp.get_thermo('pzz')
        pxy = lmp.get_thermo('pxy')
        pxz = lmp.get_thermo('pxz')
        pyz = lmp.get_thermo('pyz')

        xyz_system = np.array(lmp.gather_atoms('x',1,3))

        xyz_system = xyz_system.reshape(len(xyz_system)//3,3)

        xyz_inter_relaxed = [[],[],[]]

        N_atoms = 2*self.size**3 - self.n_vac

        idx = 0

        for element, xyz_element in enumerate(xyz_inter):
            for i in range(len(xyz_element)):
                vec = (xyz_system[N_atoms + idx]/alattice)
                xyz_inter_relaxed[element].append(vec.tolist())
                idx += 1


        lmp.close()

        stress = np.array([pxx, pyy, pzz, pxy, pxz, pyz])

        self.savename = 'rand'

        return pe, xyz_inter_relaxed, stress.tolist()

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

        sites = {}

        sites['oct'] =  self.size//2 + self.get_octahedral_sites()

        sites['tet'] = self.size//2 + self.get_tetrahedral_sites()

        sites['diag'] = self.size//2 + self.get_diagonal_sites()

        sites['central'] = self.size//2 + self.get_central_sites()

        return sites
    
    def find_strain(self, stress):

        C11 = 3.201
        C12 = 1.257
        C44 = 1.020

        C = np.array( [
            [C11, C12, C12, 0, 0, 0],
            [C12, C11, C12, 0, 0, 0],
            [C12, C12, C11, 0, 0, 0],
            [0, 0, 0, C44, 0, 0],
            [0, 0, 0, 0, C44, 0],
            [0, 0, 0, 0, 0, C44]
        ])

        conversion = 1.602177e2

        C = conversion*C

        stress = np.array(stress)*1e-4

        strain = np.linalg.solve(C, stress)

        return strain

    def create_animation(self,xyz_inter = [[], [], []], alattice =  3.14484257):

        folder_path = "Animation"

        # List all files in the folder
        files = os.listdir(folder_path)

        try:
            # Loop through the files and delete each one
            for file in files:
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except:
            print('all deleted')

        ''' xyz_inter gives a list of the intersitial atoms for each species i,e W H He in that order
            they are in lattice units and are consistent with the Lammps co-ords of the cell'''

        lmp = lammps()

        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('boundary p p p')

        lmp.command('lattice bcc %f orient x 1 0 0 orient y 0 1 0 orient z 0 0 1' % alattice)

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

        lmp.command('mass 1 183.84')

        lmp.command('mass 2 1.00784')

        lmp.command('mass 3 4.002602')

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

        lmp.command('run 0')

        lmp.command('thermo 50')

        lmp.command('compute pot all pe/atom')

        lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz')

        lmp.command('dump myDump all custom 1 Animation/animation.*.dump id type x y z c_pot')

        lmp.command('timestep 1e-3')

        lmp.command('min_style quickmin')

        lmp.command('minimize 1e-15 1e-18 10 10')

        lmp.command('minimize 1e-15 1e-18 10 100')

        lmp.command('minimize 1e-15 1e-18 100 1000')
        lmp.close()

def minimize_single_intersitial(Instance, element_idx):

    energy_lst = []

    type_lst = []

    relaxed_lst = []

    available_sites = Instance.get_all_sites()

    for site_type in available_sites:

        test_site = [[],[],[]]
        
        if len(available_sites[site_type]) > 0:
            test_site[element_idx].append(available_sites[site_type][0])

            energy, relaxed, _ = Instance.Build_Defect(test_site)

            energy_lst.append(energy)

            type_lst.append(site_type)

            relaxed_lst.append(relaxed)
        
        else:
            energy_lst.append(np.inf)

            type_lst.append(site_type)
        
    energy_arr = np.array(energy_lst)

    min_energy_idx = np.argmin(energy_arr)

    xyz_init = [[],[],[]]

    xyz_init[element_idx].append(available_sites[type_lst[min_energy_idx]][0])    

    return energy_lst[min_energy_idx], xyz_init, relaxed_lst[min_energy_idx]

def check_proximity(xyz_init, test_site):

    for xyz_element in xyz_init:
        for vec in xyz_element:
            distance = np.linalg.norm(test_site - vec)
            if distance < 0.1:
                return False
    
    return True

def minimize_add_intersitial(Instance, element_idx, xyz_init):

    energy_lst = []

    type_lst = []

    idx_lst = []

    relaxed_lst = []

    available_sites = Instance.get_all_sites()

    for site_type in available_sites:

        for idx, site in enumerate(available_sites[site_type]):

            valid = check_proximity(xyz_init, site)

            test_site = copy.deepcopy(xyz_init)

            test_site[element_idx].append(site)

            if valid:

                energy, relaxed, _ = Instance.Build_Defect(test_site)

            else:

                energy = np.inf

                relaxed = copy.deepcopy(test_site)

            energy_lst.append(energy)

            type_lst.append(site_type)

            idx_lst.append(idx)

            relaxed_lst.append(relaxed)

    energy_arr = np.array(energy_lst)

    energy_arr = np.nan_to_num(energy_arr)

    min_energy_idx = np.argmin(energy_arr)

    xyz_init_new = copy.deepcopy(xyz_init)

    xyz_init_new[element_idx].append(available_sites[type_lst[min_energy_idx]][idx_lst[min_energy_idx]].tolist())    

    return energy_lst[min_energy_idx], xyz_init_new, relaxed_lst[min_energy_idx]

size = 7

b_energy = np.array([-8.949, -4.25/2, 0])

pot_type = 'Daniel'

Instance = Lammps_Point_Defect(size, 0, pot_type)

perfect, _, stress0 = Instance.Build_Defect()

Instance.n_vac = 0

sites = [
            [3.25, 3.5 , 3   ],
            [3   , 3.5 , 3.25],
            [3.75, 3.5 , 3   ],
            [3.5 , 4.0 , 3.25],
            [3   , 3.5 , 3.5 ],
            [3   , 3.50, 3.75],
            [3.50, 4.00, 3.75],
            [3.25, 3.50, 4   ],
            [2.75, 3.50, 4   ]
]
abc = ['A','B','C','D','E','F','G','H','I']

data = {}

data['H-H'] = {}

data['H-H']['binding'] = []
data['H-H']['relaxed'] = []
data['H-H']['b_length'] = []
data['H-H']['stress'] = []

tet, _, __ = Instance.Build_Defect( [[],[sites[0]],[]])
oct, _, __ = Instance.Build_Defect( [[],[sites[4]],[]])

for idx, test_site in enumerate(sites[1:]):
    
    Instance.savename = 'H(LE)H(%s)' % abc[idx]

    pe, relaxed, stress = Instance.Build_Defect( [ [],[test_site, sites[0]], [] ])

    test_pe, _, __ = Instance.Build_Defect( [ [],[test_site], [] ])

    data['H-H']['binding'].append( tet + tet - pe - perfect)

    data['H-H']['relaxed'].append([relaxed[1][0], relaxed[1][1]])

    data['H-H']['b_length'].append(np.linalg.norm( np.array(relaxed[1][0]) - np.array(relaxed[1][1]) ) )

    data['H-H']['stress'].append(stress)


data['He-H'] = {}

data['He-H']['binding'] = []
data['He-H']['relaxed'] = []
data['He-H']['b_length'] = []
data['He-H']['stress'] = []

tet, _, __ = Instance.Build_Defect( [[],[],[sites[0]]])
oct, _, __ = Instance.Build_Defect( [[],[],[sites[4]]])

for idx, test_site in enumerate(sites[1:]):

    Instance.savename = 'He(LE)H(%s)' % abc[idx]

    pe, relaxed, stress = Instance.Build_Defect( [ [],[test_site], [sites[0]] ])

    test_pe, _, __ = Instance.Build_Defect( [ [],[sites[0]], [] ])

    data['He-H']['binding'].append( test_pe + tet - pe - perfect)

    data['He-H']['relaxed'].append([relaxed[1][0], relaxed[2][0]])

    data['He-H']['b_length'].append(np.linalg.norm( np.array(relaxed[1][0]) - np.array(relaxed[2][0]) ) )

    data['He-H']['stress'].append(stress)


data['He-He'] = {}

data['He-He']['binding'] = []
data['He-He']['relaxed'] = []
data['He-He']['b_length'] = []
data['He-He']['stress'] = []

tet, _, __ = Instance.Build_Defect( [[],[],[sites[0]]])
oct, _, __ = Instance.Build_Defect( [[],[],[sites[4]]])

for idx, test_site in enumerate(sites[1:]):

    Instance.savename = 'He(LE)He(%s)' % abc[idx]

    pe, relaxed, stress = Instance.Build_Defect( [ [],[], [test_site, sites[0]] ])

    test_pe, _, __ = Instance.Build_Defect( [ [],[], [test_site] ])

    data['He-He']['binding'].append( tet + tet - pe - perfect)

    data['He-He']['relaxed'].append([relaxed[2][0], relaxed[2][1]])

    data['He-He']['b_length'].append(np.linalg.norm( np.array(relaxed[2][0]) - np.array(relaxed[2][1]) ) )

    data['He-He']['stress'].append(stress)


with open('Data/My Data/my_bonding.json','w') as file:
    json.dump(data, file, indent=4)

animation = [
            [],
            [

            ],
            [
                [
                    3.25,
                    3.50,
                    4.00
                ]
            ]
        ]
# Instance.create_animation(animation)
