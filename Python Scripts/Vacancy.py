
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

        
    def Build_Vacancy(self, size, He, H, V):

        lmp = lammps()

        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('boundary p p p')

        lmp.command('lattice bcc %f orient x 1 0 0 orient y 0 1 0 orient z 0 0 1' % 3.14484257)

        lmp.command('region r_simbox block 0 %d 0 %d 0 %d units lattice' % (size, size, size))
        
        lmp.command('create_box 2 r_simbox')
        
        lmp.command('create_atoms 1 box')

        for i in range(V):

            lmp.command('region r_vac_%d sphere %f %f %f 0.1 units lattice' 
                        % (i, size//2 + i/2, size//2 + i/2, size//2 + i/2))
            
            lmp.command('delete_atoms region r_vac_%d ' % i)
         


        #lmp.command('create_atoms 3 single 2.4 2.4 2.4 units lattice')
        #lmp.command('create_atoms 2 single 2.5 2.5 2.5 units lattice')

        #lmp.command('create_atoms 3 single 2.6 2.4 2.7 units lattice')


        lmp.command('mass 1 183.84')

        lmp.command('mass 2 1.00784')

        #lmp.command('mass 3 4.002602')

        lmp.command('pair_style eam/alloy' )

        lmp.command('pair_coeff * * MNL6_WH.eam.alloy W H')

        lmp.command('run 0')

        lmp.command('timestep %f' % 2e-3)

        lmp.command('thermo 50')

        lmp.command('thermo_style custom step temp pe press') 

        lmp.command('run 200')

        lmp.command('minimize 1e-5 1e-8 100 1000')

        lmp.command('compute potential all pe/atom')

        lmp.command('run 0')
        lmp.command('write_dump all custom Dump/vac_%d.dump id type x y z c_potential modify sort id' % V )
        
        pe = lmp.get_thermo('pe')

        lmp.close()

        return pe
    
Instance = Lammps_Vacancy()

perfect = Instance.Build_Vacancy(11, 0, 0, 0)

vacancy = Instance.Build_Vacancy(11, 0, 0, 1)

print(perfect - vacancy - -8.949)
MPI.Finalize()