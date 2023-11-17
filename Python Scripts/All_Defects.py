from Lammps_PDefect_Classes import Lammps_Point_Defect, test_config
import numpy as np
import copy
import json 
import os 

folder_path = "Lammps_Dump"

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


folder_path = "Defect_InitFiles"

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

size = 7

pot_type = 'Daniel'

Instance = Lammps_Point_Defect(size=size, n_vac=0, potential_type=pot_type, potfile='potential.eam.alloy', surface=False, depth=size//2)

Instance.n_vac = 1

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

