#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ScanArea_load_las_fast

Created on Fri Feb 26 12:15:44 2021

@author: thayer
"""

import time
from collections import namedtuple
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% test init project

project_path = "/media/thayer/Data/mosaic_lidar/ROV"
project_name = "mosaic_rov_190120.RiSCAN"


las_fieldnames = ['Points']

t0 = time.perf_counter()
project = pydar.Project(project_path, project_name,
                      import_las=True, 
                      las_fieldnames=las_fieldnames,
                      create_id=False)
t1 = time.perf_counter()

print(t1-t0)

# %% display

project.apply_transforms(['sop'])
project.display_project(-3, -1)

# %% Delete project and load scan_area

del project

# %% load scan area

Registration = namedtuple('Registration', ['project_name_0', 'project_name_1',
                                           'reflector_list', 'mode', 
                                           'yaw_angle'], 
                          defaults=[None, None, None])

project_names = ['mosaic_rov_040120.RiSCAN',
                 'mosaic_rov_110120.RiSCAN',
                 'mosaic_rov_190120.RiSCAN',
                 'mosaic_rov_250120.RiSCAN',
                 'mosaic_rov_040220.RiSCAN',
                 'mosaic_rov_220220.RiSCAN.RiSCAN',
                 'mosaic_02_040420.RiSCAN',
                 'mosaic_02_110420_rov.RiSCAN',
                 'mosaic_rov_170420.RiSCAN',
                 'mosaic_rov_220420.RiSCAN',
                 'mosaic_rov_290420.RiSCAN',
                 'mosaic_rov_02_090520.RiSCAN']

registration_list = [Registration('mosaic_rov_190120.RiSCAN', 
                                  'mosaic_rov_190120.RiSCAN'),
                     Registration('mosaic_rov_190120.RiSCAN',
                                  'mosaic_rov_110120.RiSCAN',
                                  ['r05', 'r28', 'r29', 'r31', 'r32', 'r33',
                                   'r34'],
                                  'LS'),
                     Registration('mosaic_rov_190120.RiSCAN',
                                  'mosaic_rov_040120.RiSCAN',
                                  ['r28', 'r29', 'r30', 'r31', 'r32', 'r33'],
                                  'LS'),
                     Registration('mosaic_rov_190120.RiSCAN',
                                  'mosaic_rov_250120.RiSCAN',
                                  ['r28', 'r29', 'r30', 'r32', 'r34', 'r35',
                                   'r22'],
                                  'LS'),
                     Registration('mosaic_rov_250120.RiSCAN',
                                  'mosaic_rov_040220.RiSCAN',
                                  ['r28', 'r29', 'r30', 'r31', 'r32', 'r34', 
                                   'r35', 'r36'],
                                  'LS'),
                     Registration('mosaic_rov_040220.RiSCAN',
                                  'mosaic_rov_220220.RiSCAN.RiSCAN',
                                  ['r28', 'r31', 'r32', 'r34'],
                                  'Yaw'),
                     Registration('mosaic_rov_220220.RiSCAN.RiSCAN',
                                  'mosaic_02_040420.RiSCAN',
                                  ['r29', 'r30', 'r33', 'r36'],
                                  'LS'),
                     Registration('mosaic_02_040420.RiSCAN',
                                  'mosaic_02_110420_rov.RiSCAN',
                                  ['r29', 'r30', 'r33', 'r35', 'r37'],
                                  'LS'),
                     Registration('mosaic_02_040420.RiSCAN',
                                  'mosaic_rov_170420.RiSCAN',
                                  ['r29', 'r30', 'r35', 'r36', 'r37'],
                                  'LS'),
                     Registration('mosaic_rov_170420.RiSCAN',
                                  'mosaic_rov_220420.RiSCAN',
                                  ['r29', 'r30', 'r35', 'r36', 'r37'],
                                  'LS'),
                     Registration('mosaic_rov_220420.RiSCAN',
                                  'mosaic_rov_290420.RiSCAN',
                                  ['r30', 'r33', 'r35', 'r36'],
                                  'LS'),
                     Registration('mosaic_rov_290420.RiSCAN',
                                  'mosaic_rov_02_090520.RiSCAN',
                                  ['r30', 'r33', 'r35', 'r36'],
                                  'LS')
                     ]

sub_list = ['mosaic_rov_190120.RiSCAN',
            'mosaic_rov_250120.RiSCAN']
 
# %% Init

project_path = '/media/thayer/Data/mosaic_lidar/ROV' #'D:\\mosaic_lidar\\ROV\\'

scan_area = pydar.ScanArea(project_path, project_names,
                       registration_list, load_scans=False, read_scans=False)

for project_name in sub_list:
    print(project_name)
    scan_area.add_project(project_name, load_scans=True, read_scans=False,
                          import_las=True, create_id=False,
                          las_fieldnames=['Points'])

scan_area.register_all()

# %% Display

scan_area.project_dict[sub_list[1]].display_project(-3, -1)