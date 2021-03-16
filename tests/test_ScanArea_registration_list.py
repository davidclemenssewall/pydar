# -*- coding: utf-8 -*-
"""
test_ScanArea_registration_list.py test registration list functionality.

Created on Thu Sep 17 10:31:09 2020

@author: d34763s
"""

from collections import namedtuple
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% Test named tuple

Registration = namedtuple('Registration', ['project_name_0', 'project_name_1',
                                           'reflector_list', 'mode', 
                                           'yaw_angle'], 
                          defaults=[None, None, None])

project_names = ['mosaic_rov_190120.RiSCAN',
                 'mosaic_rov_250120.RiSCAN',
                 'mosaic_rov_040220.RiSCAN'
                 ]

registration_list = [Registration('mosaic_rov_190120.RiSCAN', 
                                  'mosaic_rov_190120.RiSCAN'),
                     Registration('mosaic_rov_190120.RiSCAN',
                                  'mosaic_rov_250120.RiSCAN',
                                  ['r28', 'r29', 'r30', 'r32', 'r34', 'r35',
                                   'r22'],
                                  'LS'),
                     Registration('mosaic_rov_250120.RiSCAN',
                                  'mosaic_rov_040220.RiSCAN',
                                  ['r28', 'r29', 'r30', 'r31', 'r32', 'r34', 
                                   'r35', 'r36'],
                                  'LS')
                     ]


# %% Test init
project_path = '/media/thayer/Data/mosaic_lidar/ROV/'

scan_area = pydar.ScanArea(project_path, project_names,
                       registration_list, import_mode='poly')

# %% Test register all

scan_area.register_all()

# %% examine transformed_history_dicts
import json

print(json.dumps(scan_area.project_dict[project_names[2]]
                 .scan_dict['ScanPos001'].filt_history_dict, indent=4))

# %% Investigate the transformed polydata
print(scan_area.project_dict[project_names[2]]
                 .scan_dict['ScanPos001'].currentFilter.GetOutput())

# %% Look at 10-25 and see if it's registered correctly

z_min = -3
z_max = -1.5

scan_area.project_dict[project_names[2]].display_project(z_min, z_max)