
import os, sys, shutil, json, glob
import time
import numpy as np
from scipy.spatial import cKDTree 
from lammps import lammps
from ctypes import *
from mpi4py import MPI
from itertools import combinations_with_replacement
import pandas as pd
from IPython.display import display, Latex
import matplotlib.pyplot as plt


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

        potfile = potfolder + 'W_H_He.eam.alloy'

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
                        % (atype, size//2 + 0.25, size//2  + 0.5, size//2 + 0))
            
        elif pos == 'oct':
            lmp.command('create_atoms %d single %f %f %f units lattice' 
                        % (atype, size//2 + 0.5, size//2 + 0.5, size//2 + 0)) 
        
        elif pos == '111':
            lmp.command('create_atoms %d single %f %f %f units lattice' 
                        % (atype, size//2 + 0.25, size//2 + 0.25, size//2 + 0.25)) 
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
size = 10

data = { '$E_{oct}$': [] ,
         '$E_{tet}$': [],
         '$E_{oct} - E_{tet}$': []}

data = {}

elements = ['W', 'H', 'He']

for i in elements:
    data['$E_{oct}^{%s}$' % i] = []
    data['$E_{tet}^{%s}$' % i] = []
    data['$E_{111}^{%s}$' % i] = []
    data['$E_{oct}^{%s} - E_{tet}^{%s}$' % (i, i)] = []

for atype in range(1,4):
    perfect = Instance.Build_Intersitial(size, atype, 'crystalline')
    oct = Instance.Build_Intersitial(size, atype, 'oct')
    tet = Instance.Build_Intersitial(size, atype, 'tet')
    _111 = Instance.Build_Intersitial(size, atype, '111')

    b_energy = np.array([-8.949, -4.25/2, 0])

    if Instance.me == 0:

        oct_int = oct - perfect - b_energy[atype-1]
        tet_int = tet - perfect - b_energy[atype-1]
        oct_tet = oct - tet
        int_111 = _111 - perfect - b_energy[atype-1]

        data['$E_{oct}^{%s}$' % elements[atype-1]].append(oct_int)
        data['$E_{tet}^{%s}$' % elements[atype-1]].append(tet_int)
        data['$E_{111}^{%s}$' % elements[atype-1]].append(int_111)
        data['$E_{oct}^{%s} - E_{tet}^{%s}$' % (elements[atype-1], elements[atype-1])].append(oct_tet)



if Instance.me == 0:
    df = pd.DataFrame(data).transpose()
    df.to_csv('Data/Defect Analysis/Intersitial_Data.csv')  
    toc = time.perf_counter()
    print(toc-tic)
    latex_table = df.to_latex(index=True, float_format="%.2f")

    with open('Data/Defect Analysis/Intersitial_Data.tex', 'w') as f:

        f.write(latex_table)

conv_111 = []
sizes = np.arange(4,15)
for s in sizes:
    perfect = Instance.Build_Intersitial(s, 1, 'crystalline')
    _111 = Instance.Build_Intersitial(s, 1, '111')
    int_111 = _111 - perfect - b_energy[0]
    conv_111.append(int_111)

if Instance.me == 0:
    plt.plot(sizes, conv_111)
    plt.title('Convergence of formation energy of <111> intersitial')
    plt.xlabel('Boxsize')
    plt.ylabel('Formation Energy')
    plt.show()

MPI.Finalize()

