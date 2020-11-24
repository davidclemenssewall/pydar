# -*- coding: utf-8 -*-
"""
test_ScanArea_registration_list.py test registration list functionality.

Created on Thu Sep 17 10:31:09 2020

@author: d34763s
"""

import os
from collections import namedtuple
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\lidar_processing\\')
import pydar

# %% Test named tuple

Registration = namedtuple('Registration', ['project_name_0', 'project_name_1',
                                           'reflector_list', 'mode', 
                                           'yaw_angle'], 
                          defaults=[None, None, None])

r = Registration('mosaic_01_102019.RiSCAN', 'mosaic_01_102019.RiSCAN')

registration_list = [Registration('mosaic_01_102019.RiSCAN', 
                                  'mosaic_01_102019.RiSCAN'),
                     Registration('mosaic_01_102019.RiSCAN', 
                                  'mosaic_01_102519.RiSCAN',
                                  ['r01', 'r02', 'r03', 'r09', 'r08'],
                                  'LS')]

project_names = ['mosaic_01_102019.RiSCAN', 'mosaic_01_102519.RiSCAN']

# %% Test init

snow1 = pydar.ScanArea('D:\\mosaic_lidar\\Snow1\\', project_names,
                       registration_list)

# %% Test register all

snow1.register_all()

# %% Look at 10-25 and see if it's registered correctly

z_min = -3
z_max = -1.5

snow1.project_dict['mosaic_01_102519.RiSCAN'].display_project(z_min, z_max)