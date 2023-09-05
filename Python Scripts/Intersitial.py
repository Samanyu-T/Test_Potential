
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



class Lammps_Intersitial():

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

        
    def Build_Intersitial(self, size, atype, pos):

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

        if pos == 'tet':
            lmp.command('create_atoms %d single %f %f %f units lattice' 
                        % (atype, size//2 + 0.25, size//2  + 0.5, size//2))
            
        elif pos == 'oct':
            lmp.command('create_atoms %d single %f %f %f units lattice' 
                        % (atype, size//2 + 0.5, size//2 + 0.5, size//2 + 0)) 
                   
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

        lmp.close()

        return pe

tic = time.perf_counter()
Instance = Lammps_Intersitial()
size = 8
atype = 2

perfect = Instance.Build_Intersitial(size, atype, 'crystalline')
oct = Instance.Build_Intersitial(size, atype, 'oct')
tet = Instance.Build_Intersitial(size, atype, 'tet')

b_energy = np.array([-8.949, -4.25/2, 0])
toc = time.perf_counter()



if Instance.me == 0:
    print('Perfect Crystal: %7.3f \n Octahedral Inersitial: %7.3f \n Tetrahedral Intersitial: %7.3f'
        % (perfect, oct, tet) )

    print('E_oct: %7.3f \n E_tet: %7.3f \n E_oct - E_tet %7.3f'
        % (oct - perfect - b_energy[atype-1], tet - perfect - b_energy[atype-1], oct - tet ))
    
    print(toc-tic)
MPI.Finalize()

