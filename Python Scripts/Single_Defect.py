from Lammps_PDefect_Classes import Lammps_Point_Defect
import numpy as np
import copy
import json 
import os 


size = 7

pot_type = 'Daniel'

Instance = Lammps_Point_Defect(size=size, n_vac=0, potential_type=pot_type, potfile='WHHe_He_edensity.eam.alloy', surface=False, depth=size//2)

sites = Instance.get_all_sites()

site = [
            [],
            [

            ],
            [
                [
                    3.238497068322839,
                    3.4996099600074877,
                    4.001234856080179
                ]
            ]
        ]

print(Instance.Build_Defect(site))
print(Instance.relaxation_volume)