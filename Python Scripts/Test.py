import numpy as np
from lammps import lammps
from mpi4py import MPI
import itertools
import copy 
import os

def test_config(potfile, input_path, output_path, n_delete):

    lmp = lammps()

    lmp.command('# Lammps input file')

    lmp.command('units metal')

    lmp.command('atom_style atomic')

    # lmp.command('atom_modify map array sort 0 0.0')

    lmp.command('read_data %s' % input_path)

    N = lmp.get_natoms()

    # if n_delete==0:
    #     pass
    
    # elif n_delete==1:

    #     lmp.command('group del_atom id %d' % (N))

    #     lmp.command('delete_atoms group del_atom')
    
    # else:
    #     lmp.command('group del_atom type 3')

    #     lmp.command('delete_atoms group del_atom')

    for i in range(n_delete):
        lmp.command('group del_atom%d id %d' % (i,N-i))

        lmp.command('delete_atoms group del_atom%d compress no' % i)

    lmp.command('pair_style eam/alloy')

    lmp.command('pair_coeff * * %s W H He' % potfile)

    lmp.command('run 0')

    lmp.command('thermo 10')

    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz')
    
    lmp.command('compute pot all pe/atom')

    lmp.command('run 0')

    lmp.command('write_dump all custom %s id type x y z c_pot' % (output_path))

    lmp.close()

n_vac = 0

input_folder = 'Defect_InitFiles'
input_name = '(Vac:%d)(H:1)(He:1).data' % n_vac
input_path = os.path.join(input_folder, input_name)

def main(input_path, component):

    potfile = 'Potentials/Tungsten_Hydrogen_Helium/WHHe_%s_only.eam.alloy' % component

    output_folder = 'Components'
    output_name = '%s_V%dH1He1.dump' % (component, n_vac)
    output_path = os.path.join(output_folder, output_name)

    test_config(potfile, input_path, output_path, n_delete=0)

    output_folder = 'Components'
    output_name = '%s_V%dH1He0.dump' % (component, n_vac)
    output_path = os.path.join(output_folder, output_name)

    test_config(potfile, input_path, output_path, n_delete=1)

main(input_path,'eam')
main(input_path,'pairwise')
main(input_path,'edensity')


input_folder = 'Defect_InitFiles'
input_name = '(Vac:%d)(H:1)(He:0).data' % n_vac
input_path = os.path.join(input_folder, input_name)

component = 'eam'
output_folder = 'Components'
output_name = '%s_V%dH1He0_relaxed.dump' % (component, n_vac)
output_path = os.path.join(output_folder, output_name)
potfile = 'Potentials/Tungsten_Hydrogen_Helium/WHHe_%s_only.eam.alloy' % component
test_config(potfile, input_path, output_path, n_delete=0)

component = 'pairwise'
output_folder = 'Components'
output_name = '%s_V%dH1He0_relaxed.dump' % (component, n_vac)
output_path = os.path.join(output_folder, output_name)
potfile = 'Potentials/Tungsten_Hydrogen_Helium/WHHe_%s_only.eam.alloy' % component
test_config(potfile, input_path, output_path, n_delete=0)

component = 'edensity'
output_folder = 'Components'
output_name = '%s_V%dH1He0_relaxed.dump' % (component, n_vac)
output_path = os.path.join(output_folder, output_name)
potfile = 'Potentials/Tungsten_Hydrogen_Helium/WHHe_%s_only.eam.alloy' % component
test_config(potfile, input_path, output_path, n_delete=0)

input_folder = 'Defect_InitFiles'
input_name = '(Vac:%d)(H:0)(He:0).data' % n_vac
input_path = os.path.join(input_folder, input_name)

component = 'eam'
output_folder = 'Components'
output_name = '%s_V%dH0He0_relaxed.dump' % (component, n_vac)
output_path = os.path.join(output_folder, output_name)
potfile = 'Potentials/Tungsten_Hydrogen_Helium/WHHe_%s_only.eam.alloy' % component
test_config(potfile, input_path, output_path, n_delete=0)

component = 'pairwise'
output_folder = 'Components'
output_name = '%s_V%dH0He0_relaxed.dump' % (component, n_vac)
output_path = os.path.join(output_folder, output_name)
potfile = 'Potentials/Tungsten_Hydrogen_Helium/WHHe_%s_only.eam.alloy' % component
test_config(potfile, input_path, output_path, n_delete=0)

component = 'edensity'
output_folder = 'Components'
output_name = '%s_V%dH0He0_relaxed.dump' % (component, n_vac)
output_path = os.path.join(output_folder, output_name)
potfile = 'Potentials/Tungsten_Hydrogen_Helium/WHHe_%s_only.eam.alloy' % component
test_config(potfile, input_path, output_path, n_delete=0)


input_folder = 'Defect_InitFiles'
input_name = '(Vac:%d)(H:0)(He:1).data' % n_vac
input_path = os.path.join(input_folder, input_name)

component = 'eam'
output_folder = 'Components'
output_name = '%s_V%dH0He1_relaxed.dump' % (component, n_vac)
output_path = os.path.join(output_folder, output_name)
potfile = 'Potentials/Tungsten_Hydrogen_Helium/WHHe_%s_only.eam.alloy' % component
test_config(potfile, input_path, output_path, n_delete=0)

component = 'pairwise'
output_folder = 'Components'
output_name = '%s_V%dH0He1_relaxed.dump' % (component, n_vac)
output_path = os.path.join(output_folder, output_name)
potfile = 'Potentials/Tungsten_Hydrogen_Helium/WHHe_%s_only.eam.alloy' % component
test_config(potfile, input_path, output_path, n_delete=0)

component = 'edensity'
output_folder = 'Components'
output_name = '%s_V%dH0He1_relaxed.dump' % (component, n_vac)
output_path = os.path.join(output_folder, output_name)
potfile = 'Potentials/Tungsten_Hydrogen_Helium/WHHe_%s_only.eam.alloy' % component
test_config(potfile, input_path, output_path, n_delete=0)