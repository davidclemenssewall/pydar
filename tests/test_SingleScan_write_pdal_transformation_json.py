# -*- coding: utf-8 -*-
"""
test_SingleScan_write_pdal_transformation_json.py

Created on Wed Dec  2 15:15:19 2020

@author: d34763s
"""

# %% Imports

import os
from collections import namedtuple
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\pydar\\')
import pydar

# %% Parameters to initialize project and scan_area
# This contains the info on which projects to load and how to align them in 
# space with each other.

Registration = namedtuple('Registration', ['project_name_0', 'project_name_1',
                                           'reflector_list', 'mode', 
                                           'yaw_angle'], 
                          defaults=[None, None, None])

project_names = ['mosaic_01_101819.RiSCAN',
                 'mosaic_01_102019.RiSCAN',
                 'mosaic_01_102519.RiSCAN',
                 'mosaic_01_110119.RiSCAN',
                 'mosaic_01_110819.RiSCAN',
                 'mosaic_01_111519.RiSCAN',
                 'mosaic_01b_061219.RiSCAN.RiSCAN.RiSCAN',
                 'mosaic_01_122719.RiSCAN',
                 'mosaic_01_040120.RiSCAN',
                 'mosaic_01_180120.RiSCAN',
                 'mosaic_01_290120.RiSCAN',
                 'mosaic_01_060220.RiSCAN',
                 'mosaic_01_150220.RiSCAN.RiSCAN',
                 'mosaic_01_280220.RiSCAN'
                 ]

registration_list = [Registration('mosaic_01_102019.RiSCAN', 
                                  'mosaic_01_102019.RiSCAN'),
                     Registration('mosaic_01_102019.RiSCAN',
                                  'mosaic_01_101819.RiSCAN',
                                  ['r04', 'r05', 'r07', 'r09'],
                                  'Yaw'),
                     Registration('mosaic_01_102019.RiSCAN', 
                                  'mosaic_01_102519.RiSCAN',
                                  ['r01', 'r02', 'r03', 'r09', 'r08'],
                                  'LS'),
                     Registration('mosaic_01_102519.RiSCAN',
                                  'mosaic_01_110119.RiSCAN',
                                  ['r01', 'r03', 'r04', 'r05', 'r06', 'r07'],
                                  'LS'),
                     Registration('mosaic_01_110119.RiSCAN',
                                  'mosaic_01_111519.RiSCAN',
                                  ['r02', 'r03', 'r04'],
                                  'Yaw'),
                     Registration('mosaic_01_111519.RiSCAN',
                                  'mosaic_01_110819.RiSCAN',
                                  ['r02', 'r05', 'r06', 'r07', 'r10'],
                                  'LS'),
                     Registration('mosaic_01_111519.RiSCAN',
                                  'mosaic_01b_061219.RiSCAN.RiSCAN.RiSCAN',
                                  ['r01', 'r11'],
                                  'Yaw'),
                     Registration('mosaic_01b_061219.RiSCAN.RiSCAN.RiSCAN',
                                  'mosaic_01_122719.RiSCAN',
                                  ['r02', 'r11'],
                                  'Yaw'),
                     Registration('mosaic_01_122719.RiSCAN',
                                  'mosaic_01_040120.RiSCAN',
                                  ['r01', 'r13', 'r14', 'r15'],
                                  'Yaw'),
                     Registration('mosaic_01_040120.RiSCAN',
                                  'mosaic_01_180120.RiSCAN',
                                  ['r03', 'r09', 'r10', 'r11', 'r24'],
                                  'LS'),
                     Registration('mosaic_01_180120.RiSCAN',
                                  'mosaic_01_290120.RiSCAN',
                                  ['r01', 'r02', 'r03', 'r09', 'r10', 
                                   'r12', 'r13', 'r14'],
                                  'LS'),
                     Registration('mosaic_01_290120.RiSCAN',
                                  'mosaic_01_060220.RiSCAN',
                                  ['r01', 'r03', 'r09', 'r12', 'r14', 'r23'],
                                  'LS'),
                     Registration('mosaic_01_060220.RiSCAN',
                                  'mosaic_01_150220.RiSCAN.RiSCAN',
                                  ['r03', 'r09', 'r23'],
                                  'Yaw'),
                     Registration('mosaic_01_150220.RiSCAN.RiSCAN',
                                  'mosaic_01_280220.RiSCAN',
                                  ['r10', 'r11', 'r24', 'r12'],
                                  'LS')
                     ]

# Sub list are the scans we actually want to load
sub_list = ['mosaic_01_110119.RiSCAN',
            'mosaic_01_040120.RiSCAN']

# %% Initialize project

# Initialize scan area object, setting the flags to false means we don't try
# to actually load any scan data that we don't have.
# First argument is path to wherever you put that file
scan_area = pydar.ScanArea('D:\\mosaic_lidar\\Snow1\\', 
                           project_names, registration_list, load_scans=False,
                           read_scans=False)

for project_name in sub_list:
    scan_area.add_project(project_name, load_scans=True, read_scans=False)

# %% Register

# This step roughly aligns the reference frames in all of the Snow1 scans
scan_area.register_all()

# %% test function

os.chdir('D:\\mosaic_lidar\\Snow1\\mosaic_01_040120.RiSCAN\\lasfiles\\')

for scan_name in scan_area.project_dict[sub_list[1]].scan_dict:
    (scan_area.project_dict[sub_list[1]].scan_dict[scan_name].
     write_pdal_transformation_json())

os.chdir('D:\\mosaic_lidar\\Snow1\\mosaic_01_110119.RiSCAN\\lasfiles\\')

for scan_name in scan_area.project_dict[sub_list[0]].scan_dict:
    (scan_area.project_dict[sub_list[0]].scan_dict[scan_name].
     write_pdal_transformation_json())

# Used command line to run pdal pipelines, plotted with paraview and the
# output looks correct.