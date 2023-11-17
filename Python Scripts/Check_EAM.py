from Lammps_PDefect_Classes import test_config
import os

data = {}

potfile = 'Potentials/Tungsten_Hydrogen_Helium/WHHe_eam_only.eam.alloy'
folder_path = 'Defect_InitFiles'
output_folder = 'EAM_Only'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# List all files in the folder
files = os.listdir(folder_path)

for file in files:
    file_path = os.path.join(folder_path, file)
    pe = test_config(file_path, potfile, output_folder)
    print(pe)


# List all files in the folder

potfile = 'Potentials/Tungsten_Hydrogen_Helium/WHHe_pairwise_only.eam.alloy'

files = os.listdir(folder_path)
output_folder = 'Pairwise_Only'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for file in files:
    file_path = os.path.join(folder_path, file)
    pe = test_config(file_path, potfile, output_folder)
    print(pe)