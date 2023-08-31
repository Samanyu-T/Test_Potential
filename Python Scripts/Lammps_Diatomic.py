# initialise

import os, sys, shutil, json, glob
import time
import numpy as np
from scipy.spatial import cKDTree 
from lammps import lammps
from ctypes import *
from mpi4py import MPI
from itertools import combinations_with_replacement


# template to replace MPI functionality for single threaded use
class MPI_to_serial():

    def bcast(self, *args, **kwargs):

        return args[0]

    def barrier(self):

        return 0



class Lammps_Diatomic():

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

        
    def Build_Diatomic(self, type1, type2, a, idx):

        lmp = lammps()

        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('boundary f f f')

        #lmp.command('lattice bcc %f' % a)

        lmp.command('region r_simbox block 0 10 0 10 0 10 units box')
                    
        lmp.command('create_box 3 r_simbox')

        x  = a/np.sqrt(3)

        lmp.command('create_atoms %d single 0 0 0 units box' % (type1) )

        lmp.command('create_atoms %d single %f %f %f units box' 
                    % (type2, x, x, x ) )

        lmp.command('mass 1 183.84')

        lmp.command('mass 2 1.00784')

        lmp.command('mass 3 4.002602')

        lmp.command('pair_style hybrid eam/alloy lj/cut 7.913 table spline 4999 table spline 325' )

        lmp.command('pair_coeff * * eam/alloy MNL6_WH.eam.alloy W H NULL')

        lmp.command('pair_coeff 2 3 lj/cut 5.9225e-4 1.333')

        lmp.command('pair_coeff 3 3 table 1 He-Beck1968_modified.table HeHe')

        lmp.command('pair_coeff 1 3 table 2 W-He-Juslin.table WHe')

        lmp.command('run 0')

        pe = lmp.get_thermo('pe')

        lmp.command('write_dump all custom Dump/diatomic.%d.dump id type x y z' % idx)
        lmp.close()

        return pe
    
    def Build_Diatomic_ZBL(self, type1, type2, a, idx):

        lmp = lammps()

        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('boundary f f f')

        #lmp.command('lattice bcc %f' % a)

        lmp.command('region r_simbox block 0 10 0 10 0 10 units box')
                    
        lmp.command('create_box 3 r_simbox')

        x  = a/np.sqrt(3)

        lmp.command('create_atoms %d single 0 0 0 units box' % (1) )

        lmp.command('create_atoms %d single %f %f %f units box' 
                    % (1, x, x, x ) )

        lmp.command('mass 1 183.84')

        lmp.command('mass 2 1.00784')

        lmp.command('mass 3 4.002602')

        lmp.command('pair_style zbl 0.01 0.02' )

        lmp.command('pair_coeff 1 1 74.0 74.0')

        lmp.command('pair_coeff 1 2 74.0 1.0')

        lmp.command('pair_coeff 1 3 74.0 2.0')

        lmp.command('pair_coeff 2 3 2.0  1.0')

        lmp.command('pair_coeff 2 2 1.0  1.0')

        lmp.command('pair_coeff 3 3 2.0  2.0')

        lmp.command('run 0')

        pe = lmp.get_thermo('pe')

        lmp.command('write_dump all custom Dump/diatomic.%d.dump id type x y z' % idx)
        lmp.close()

        return pe

    def Build_Diatomic_Overlay(self, type1, type2, a, idx):

        lmp = lammps()

        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('boundary f f f')

        #lmp.command('lattice bcc %f' % a)

        lmp.command('region r_simbox block 0 10 0 10 0 10 units box')
                    
        lmp.command('create_box 3 r_simbox')

        x  = a/np.sqrt(3)

        lmp.command('create_atoms %d single 0 0 0 units box' % (type1) )

        lmp.command('create_atoms %d single %f %f %f units box' 
                    % (type2, 0.0, 0.0, a ) )

        lmp.command('mass 1 183.84')

        lmp.command('mass 2 1.00784')

        lmp.command('mass 3 4.002602')

        lmp.command('pair_style hybrid/overlay eam/alloy zbl 1e-3 1e-2')

        lmp.command('pair_coeff * *  eam/alloy W_H_He_0001.eam.alloy W H He')
        lmp.command('pair_coeff 2 2 zbl 1.0  1.0')

        lmp.command('run 0')

        pe = lmp.get_thermo('pe')

        lmp.command('write_dump all custom Dump/diatomic.%d.dump id type x y z' % idx)
        lmp.close()

        return pe



Instance = Lammps_Diatomic()
N = 100
x = np.linspace(0.5,4,N)
#x = np.logspace(-6,1,N)
pe = np.zeros(N)

for [i,j] in combinations_with_replacement([1, 2, 3],2):

    for idx, _x in enumerate(x):

        #print(i, j, _x)
        pe[idx] = Instance.Build_Diatomic_Overlay(i, j, _x, idx)

    np.savetxt('Data/Distance_%d%d.txt' %(i, j), x)
    np.savetxt('Data/Overlay_%d%d.txt' %(i, j), pe)

MPI.Finalize()