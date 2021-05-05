#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ScanArea_max_alignment_ss.py

Created on Tue May  4 10:44:02 2021

@author: thayer
"""

import json
from collections import namedtuple
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar


# %% Registration list

Registration = namedtuple('Registration', ['project_name_0', 'project_name_1',
                                           'reflector_list', 'mode', 
                                           'yaw_angle'], 
                          defaults=[None, None, None])

project_names = ['mosaic_rov_190120.RiSCAN',
                 'mosaic_rov_250120.RiSCAN'
                 ]

registration_list = [Registration('mosaic_rov_250120.RiSCAN', 
                                  'mosaic_rov_250120.RiSCAN'),
                      Registration('mosaic_rov_250120.RiSCAN',
                                   'mosaic_rov_190120.RiSCAN',
                                   ['r28', 'r29', 'r30', 'r32', 'r34', 'r35',
                                    'r22'],
                                   'LS')
                      ]

# %% Init
project_path = '/media/thayer/Data/mosaic_lidar/ROV/'
suffix = 'slfsnow'


scan_area = pydar.ScanArea(project_path, project_names,
                       registration_list, import_mode='read_scan',
                       create_id=False, las_fieldnames=['Points', 'PointId',
                                                        'Classification'],
                       class_list='all', suffix=suffix)


scan_area.register_all()

# %% Test max_alignment_ss

A, history_dict, count = scan_area.max_alignment_ss(project_names[1], 
                                             project_names[0], 'ScanPos001',
                                             return_count=True)

# %% examine output

print(A)
print(count)
