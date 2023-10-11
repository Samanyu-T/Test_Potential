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



class Lammps_Elastic():

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

        lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz')
        
        lmp.command('minimize 1e-9 1e-12 100 1000')

        lmp.command('write_data Elastic_Data/perfect.data')
        
        lmp.command('write_dump all atom Elastic_Data/disp.0.dump')

        lmp.close()

    
    def Strain_Crystal(self, delta = np.array([1e-3, 0, 0])):
        
        lmp = lammps()

        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')
        
        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('read_data Elastic_Data/perfect.data')

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

        lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz')

        lmp.command('thermo 10')
        
        lmp.command('run 0')

        # xyz = lmp.gather_atoms('x', 1, 3)

        # xyz_new = self.find_strained_cords(xyz, strain)

        # lmp.scatter_atoms('x', 1 , 3, xyz_new)

        lmp.command('change_box all x scale %f y scale %f z scale %f remap' % (1 + delta[0], 1 + delta[1], 1 + delta[2]) )

        lmp.command('minimize 1e-9 1e-12 1000 10000')
        
        pxx = lmp.get_thermo('pxx')
        pyy = lmp.get_thermo('pyy')
        pzz = lmp.get_thermo('pzz')
        pxy = lmp.get_thermo('pxy')
        pxz = lmp.get_thermo('pxz')
        pyz = lmp.get_thermo('pyz')


        lmp.command('write_dump all atom Elastic_Data/disp.1.dump')

        lmp.close()

        strain = np.array([ 
    
    [delta[0], 0.5*(delta[0] + delta[1]), 0.5*(delta[0] + delta[2])],

    [0.5*(delta[0] + delta[1]), delta[1], 0.5*(delta[1] + delta[2])],

    [0.5*(delta[0] + delta[2]),  0.5*(delta[1] + delta[2]), delta[2]]
])
        stress = -np.array([ 
            [pxx, pxy, pxy],
            [pxy, pyy, pyz],
            [pxz, pyz, pzz]
        ])


        return stress, strain


    def find_strained_cords(self, xyz, strain):

        xyz = np.array(xyz)
        
        N3 = len(xyz)
                
        xyz = xyz.reshape(N3//3,3)

        xyz_new = np.zeros(xyz.shape)

        for i in range(len(xyz)):
            
            if i == 0:
                xyz_new[i] = xyz[i]
            
            else:
                
                a = xyz[i] - xyz[i-1]

                xyz_new[i] = xyz_new[i-1] + a + strain
        

        double_array = (N3*ctypes.c_double)()

        xyz_new = xyz_new.flatten()

        double_array[:] = xyz_new[:]

        return double_array



size = 7
Instance = Lammps_Elastic(size = size, n_vac= 0, potential_type='Daniel')

Instance.Perfect_Crystal()
stress_lst = []
strain_lst = []

stress, strain = Instance.Strain_Crystal([1e-1, 0, 0])

# stress_lst.append(stress)
# strain_lst.append(strain)

# stress, strain = Instance.Strain_Crystal([1e-1, 1e-1, 0])
# stress_lst.append(stress)
# strain_lst.append(strain)

# stress, strain = Instance.Strain_Crystal([1e-1, 1e-1, 1e-1])
# stress_lst.append(stress)
# strain_lst.append(strain)

# stress_lst = np.array(stress_lst)
# strain_lst = np.array(strain_lst)

if Instance.me == 0:
    print(stress, strain)