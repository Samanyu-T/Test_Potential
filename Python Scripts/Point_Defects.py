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
        
        self.Perfect_Crystal()


    def Perfect_Crystal(self, alattice = 3.14484257):

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

        lmp.command('fix 3 all box/relax  aniso 0.0')

        lmp.command('run 0')

        lmp.command('thermo 5')

        lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')
        
        lmp.command('minimize 1e-9 1e-12 100 1000')

        lmp.command('write_data Elastic_Data/perfect.data')
        
        lmp.command('write_dump all atom Elastic_Data/disp.0.dump')

        pxx = lmp.get_thermo('pxx')
        pyy = lmp.get_thermo('pyy')
        pzz = lmp.get_thermo('pzz')
        pxy = lmp.get_thermo('pxy')
        pxz = lmp.get_thermo('pxz')
        pyz = lmp.get_thermo('pyz')

        self.stress0 = np.array([pxx, pyy, pzz, pxy, pxz, pyz]) 
        
        self.alattice = (lmp.get_thermo('xhi') - lmp.get_thermo('xlo'))/self.size

        self.pe0 = lmp.get_thermo('pe')

        self.vol0 = lmp.get_thermo('vol')

        lmp.close()

    def Build_Defect(self, xyz_inter = [[], [], []], alattice = 3.14484257):

        ''' xyz_inter gives a list of the intersitial atoms for each species i,e W H He in that order
            they are in lattice units and are consistent with the Lammps co-ords of the cell'''

        lmp = lammps()

        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('boundary p p p')

        lmp.command('lattice bcc %f orient x 1 0 0 orient y 0 1 0 orient z 0 0 1' % self.alattice)

        lmp.command('region r_simbox block 0 %d 0 %d 0 %d units lattice' % (self.size, self.size, self.size))
                    
        lmp.command('create_box 3 r_simbox')
        
        lmp.command('create_atoms 1 box')

        #Create a Vacancy of n-atoms along the <1,1,1> direction the vacancy will be at the centre of the cell

        for i in range(self.n_vac):
            lmp.command('region r_vac_%d sphere %f %f %f 0.2 units lattice' 
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

        lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

        # lmp.command('fix 3 all box/relax  aniso 0.0')

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

        self.vol = lmp.get_thermo('vol')

        self.stress_voigt = np.array([pxx, pyy, pzz, pxy, pxz, pyz]) - self.stress0

        self.strain_tensor = self.find_strain()

        self.relaxation_volume = 2*np.trace(self.strain_tensor)*self.vol/self.alattice**3
        # self.relaxation_volume = 2*(self.vol - self.vol0)/self.alattice**3

        lmp.command('write_dump all atom Lammps_Dump/(Vac:%d)(H:%d)(He:%d).atom' % 
                    (self.n_vac, len(xyz_inter[1]), len(xyz_inter[2])))


        pe = lmp.get_thermo('pe')

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

        e0 = self.pe0/(2*self.size**3)

        return pe - self.pe0 + self.n_vac*e0 + len(xyz_inter[1])*2.121, xyz_inter_relaxed
    
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
    
    def find_strain(self):

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

        stress = self.stress_voigt*1e-4

        self.strain_voigt = np.linalg.solve(C, stress)

        strain_tensor = np.array( [ 
            [self.strain_voigt[0], self.strain_voigt[3]/2, self.strain_voigt[4]/2],
            [self.strain_voigt[3]/2, self.strain_voigt[1], self.strain_voigt[5]/2],
            [self.strain_voigt[4]/2, self.strain_voigt[5]/2, self.strain_voigt[2]]
        ])

        return strain_tensor

    def minimize_single_intersitial(self, element_idx):

        energy_lst = []

        type_lst = []

        relaxed_lst = []

        available_sites = self.get_all_sites()

        for site_type in available_sites:

            test_site = [[],[],[]]
            
            if len(available_sites[site_type]) > 0:
                test_site[element_idx].append(available_sites[site_type][0])

                energy, relaxed = self.Build_Defect(test_site)

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

    def check_proximity(self,xyz_init, test_site):

        for xyz_element in xyz_init:
            for vec in xyz_element:
                distance = np.linalg.norm(test_site - vec)
                if distance < 0.1:
                    return False
        
        return True
        
    def minimize_add_intersitial(self, element_idx, xyz_init):

        energy_lst = []

        type_lst = []

        idx_lst = []

        relaxed_lst = []

        available_sites = self.get_all_sites()

        for site_type in available_sites:

            for idx, site in enumerate(available_sites[site_type]):

                valid = self.check_proximity(xyz_init, site)

                test_site = copy.deepcopy(xyz_init)

                test_site[element_idx].append(site)

                if valid:

                    energy, relaxed = self.Build_Defect(test_site)

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

pot_type = 'Daniel'

Instance = Lammps_Point_Defect(size, 0, pot_type)

Instance.n_vac = 1

vac, _ = Instance.Build_Defect()


data = {}

for n_vac in range(3):

    Instance.n_vac = n_vac
    
    vac_energy, _ = Instance.Build_Defect()

    data['V%dH%dHe%d' % (n_vac, 0, 0)] = {}

    data['V%dH%dHe%d' % (n_vac, 0, 0)]['energy_opt'] = vac_energy

    data['V%dH%dHe%d' % (n_vac, 0, 0)]['xyz_opt']    = [[],[],[]]

    data['V%dH%dHe%d' % (n_vac, 0, 0)]['strain_tensor']    = Instance.strain_tensor.tolist()

    data['V%dH%dHe%d' % (n_vac, 0, 0)]['relaxation_volume']    = Instance.relaxation_volume

    for element_idx in [1,2]:

        for i in range(1,7):

            if element_idx == 1:
                h_key = i
                he_key = 0

            elif element_idx == 2:
                h_key = 0
                he_key = i

            energy_opt, xyz_init, xyz_opt = Instance.minimize_add_intersitial( element_idx,
                                            data['V%dH%dHe%d' % (
                                                                    n_vac, 
                                                                    np.clip(h_key - 1, a_min= 0, a_max=None),
                                                                    np.clip(he_key - 1, a_min= 0, a_max=None)
                                                                   )
                                                ]
                                                ['xyz_opt']   


                                                
                                                                    )
            

            data['V%dH%dHe%d' % (n_vac, h_key, he_key)] = {}

            data['V%dH%dHe%d' % (n_vac, h_key, he_key)]['energy_opt'] = energy_opt

            data['V%dH%dHe%d' % (n_vac, h_key, he_key)]['xyz_opt']    = copy.deepcopy(xyz_opt)

            data['V%dH%dHe%d' % (n_vac, h_key, he_key)]['strain_tensor']       = Instance.strain_tensor.tolist()

            data['V%dH%dHe%d' % (n_vac, h_key, he_key)]['relaxation_volume']   = Instance.relaxation_volume

n_vac = 0

for n_vac in range(2):
    Instance.n_vac = n_vac

    for h_key in range(1,7):
        for he_key in range(1,7):


            energy_opt, xyz_init, xyz_opt = Instance.minimize_add_intersitial( 1,
                                            data['V%dH%dHe%d' % (
                                                                    n_vac, 
                                                                    np.clip(h_key - 1, a_min= 0, a_max=None),
                                                                    he_key
                                                                    )
                                                ]
                                                ['xyz_opt']
                                                                    )

            data['V%dH%dHe%d' % (n_vac, h_key, he_key)] = {}

            data['V%dH%dHe%d' % (n_vac, h_key, he_key)]['energy_opt'] = energy_opt

            data['V%dH%dHe%d' % (n_vac, h_key, he_key)]['xyz_opt']    = copy.deepcopy(xyz_opt)

            data['V%dH%dHe%d' % (n_vac, h_key, he_key)]['strain_tensor']       = Instance.strain_tensor.tolist()

            data['V%dH%dHe%d' % (n_vac, h_key, he_key)]['relaxation_volume']   = Instance.relaxation_volume


if Instance.me == 0:

    with open('Data/My Data/point_defects_formations.json', 'w') as file:
        json.dump(data, file, indent=4)
