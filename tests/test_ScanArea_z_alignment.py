#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ScanArea_z_alignment.py

Test functionality for aligning scans in successive projects by their gridded
z values.

Created on Mon Mar  8 16:09:57 2021

@author: thayer
"""

from collections import namedtuple
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar


# %% Registration list

Registration = namedtuple('Registration', ['project_name_0', 'project_name_1',
                                           'reflector_list', 'mode', 
                                           'yaw_angle'], 
                          defaults=[None, None, None])

project_names = [#'mosaic_rov_040120.RiSCAN',
                 #'mosaic_rov_110120.RiSCAN',
                 'mosaic_rov_190120.RiSCAN',
                 'mosaic_rov_250120.RiSCAN'#,
                 #'mosaic_rov_040220.RiSCAN',
                 #'mosaic_rov_220220.RiSCAN.RiSCAN'
                 ]

registration_list = [Registration('mosaic_rov_190120.RiSCAN', 
                                  'mosaic_rov_190120.RiSCAN'),
                     # Registration('mosaic_rov_190120.RiSCAN',
                     #              'mosaic_rov_110120.RiSCAN',
                     #              ['r05', 'r28', 'r29', 'r31', 'r32', 'r33',
                     #               'r34'],
                     #              'LS'),
                     # Registration('mosaic_rov_190120.RiSCAN',
                     #              'mosaic_rov_040120.RiSCAN',
                     #              ['r28', 'r29', 'r30', 'r31', 'r32', 'r33'],
                     #              'LS'),
                      Registration('mosaic_rov_190120.RiSCAN',
                                   'mosaic_rov_250120.RiSCAN',
                                   ['r28', 'r29', 'r30', 'r32', 'r34', 'r35',
                                    'r22'],
                                   'LS')#,
                      # Registration('mosaic_rov_250120.RiSCAN',
                      #              'mosaic_rov_040220.RiSCAN',
                      #              ['r28', 'r29', 'r30', 'r31', 'r32', 'r34', 
                      #               'r35', 'r36'],
                      #              'LS'),
                      # Registration('mosaic_rov_040220.RiSCAN',
                      #              'mosaic_rov_220220.RiSCAN.RiSCAN',
                      #              ['r28', 'r31', 'r32', 'r34'],
                      #              'Yaw')
                      ]

# %% Init
project_path = '/media/thayer/Data/mosaic_lidar/ROV/'

scan_area = pydar.ScanArea(project_path, project_names,
                       registration_list, import_mode='import_npy',
                       create_id=False, las_fieldnames=['Points'],
                       class_list='all')


scan_area.register_all()

# %% align two scans in sub_list

scan_area.z_align_all(frac_exceed_diff_cutoff=0.25,
                      bin_reduc_op='mean', diff_mode='mode')

# %% Let's look at what some of the z offsets look like

def ss_print_z_offset(ss):
    print(ss.scan_name + ': ' +
          str(ss.transform_dict["z_offset"].GetPosition()[2]))

for project_name in project_names:
    print(project_name)
    for scan_name in scan_area.project_dict[project_name].scan_dict:
        ss_print_z_offset(
            scan_area.project_dict[project_name].scan_dict[scan_name])
