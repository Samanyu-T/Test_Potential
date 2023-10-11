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
import ctypes

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

    def mpiprint(self, string):
        if self.me == 0:
            print(string)

    def Build_Defect(self, save_name, xyz_inter = [[], [], []], alattice = 3.14484257):

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

        lmp.command('thermo_style custom step temp pe press')

        lmp.command('minimize 1e-5 1e-8 10 10')

        lmp.command('minimize 1e-5 1e-8 10 100')

        lmp.command('minimize 1e-9 1e-12 100 1000')
        
        lmp.command('write_data Migration_Data/%s.data' % save_name)
        
        lmp.command('write_dump all custom Migration_Data/%s.dump id x y z' % save_name)

        pe = lmp.get_thermo('pe')

        xyz_system = np.array(lmp.gather_atoms('x',1,3))

        xyz_system = xyz_system.reshape(len(xyz_system)//3,3)

        xyz_inter_relaxed = [[],[],[]]

        N_atoms = 2*self.size**3 - self.n_vac

        idx = 0

        for element, xyz_element in enumerate(xyz_inter):
            for i in range(len(xyz_element)):
                vec = (xyz_system[N_atoms + idx])
                xyz_inter_relaxed[element].append(vec.tolist())
                idx += 1


        lmp.close()

        return pe, xyz_inter_relaxed
    
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
    
    def Interpolate(self,file_name, init_pos, final_pos, N = 10):

        init_pos = np.array(init_pos)

        final_pos = np.array(final_pos)

        lmp = lammps()

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('read_data Migration_Data/%s.data' % file_name)

        lmp.command('group hydrogen type %d' % 3)
    
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

        xyz = np.array(lmp.gather_atoms('x', 1, 3))

        pe  = lmp.get_thermo('pe')

        if self.me == 0:
            with open('Migration_Data/interpolation_energies.txt', 'w') as f:
                f.write('%f %f %f %f \n' % (xyz[-3], xyz[-2], xyz[-1], pe))

        for i in range(N):

            disp_pos = (final_pos - init_pos)/N

            lmp.command('displace_atoms hydrogen move %f %f %f units box' %
                        (disp_pos[0], disp_pos[1], disp_pos[2]))
            
            lmp.command('run 0')

            lmp.command('write_dump all custom Migration_Data/interpolate.%d.dump id x y z' % i)

            pe = lmp.get_thermo('pe')

            xyz = np.array(lmp.gather_atoms('x', 1, 3))

            if self.me == 0:
                with open('Migration_Data/interpolation_energies.txt', 'a') as f:
                    f.write('%d %f %f %f %f \n' % (i, xyz[-3], xyz[-2], xyz[-1], pe))
        lmp.close()

size = 7

n_vac = 0

potential_type = 'Daniel'

Instance = Lammps_Point_Defect(size, n_vac, potential_type)

tet = Instance.get_tetrahedral_sites() + size//2

oct = Instance.get_octahedral_sites() + size//2

pe_init, relaxed_init = Instance.Build_Defect('init', xyz_inter=[[],[], [tet[0]]])

pe_final, relaxed_final = Instance.Build_Defect('final', xyz_inter=[[],[], [tet[3]]])

# N = 6

# Instance.Interpolate('init', relaxed_init[1][0], relaxed_final[1][0], N)

# for i in range(N):
#     with open('Migration_Data/interpolate.%d.dump' % i, 'r') as src:
#         src_lines = src.readlines()
    
#     with open('Migration_Data/interpolate_neb.%d.dump' % i, 'w') as dest:
#         dest.writelines(src_lines[9:])

with open('Migration_Data/final.dump', 'r') as src:
    src_lines = src.readlines()

with open('Migration_Data/final_neb.dump', 'w') as dest:
    dest.write('%d \n' % int(2*size**3+1))
    dest.writelines(src_lines[9:])

if Instance.me == 0:
    print(pe_init,relaxed_init)
    print(pe_final, relaxed_final)
