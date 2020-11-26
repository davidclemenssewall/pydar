# -*- coding: utf-8 -*-
"""
example_havoc_20201126.py

Example showing how to read in, register, and display lidar data for Adam
Steer.

Created on Thu Nov 26 00:11:02 2020

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
scan_area = pydar.ScanArea('D:\\mosaic_lidar\\test_havoc\\Snow1\\', 
                           project_names, registration_list, load_scans=False,
                           read_scans=False)

# Load the one scan we do have.
for project_name in sub_list:
    scan_area.add_project(project_name, load_scans=True, read_scans=False)

# %% Register

# This step roughly aligns the reference frames in all of the Snow1 scans
scan_area.register_all()

# %% We can easily display all scans in a project

# the ScanArea object stores each project in a dict keyed on project_name
z_min = -3
z_max = -1

scan_area.project_dict[sub_list[1]].display_project(z_min, z_max)

# %% Convert intensity field to reflectance per riscan's instructions
# and apply our rudimentary reflectance range correction

# Parameters for reflectance range corrections
r_min = 5
r_max = 200
num = 35
base = 2

for project_name in sub_list:
    scan_area.project_dict[project_name].create_reflectance()
    scan_area.project_dict[project_name].correct_reflectance_radial('median',
        r_min=r_min, r_max=r_max, num=num, base=base)
    
# %% Display colored by reflectance

v_min = -20
v_max = -5

scan_area.project_dict[sub_list[1]].display_project(v_min, v_max,
                                                    field='reflectance')

# %% We can also display colored by our attempted corrected reflectance

v_min = -5
v_max = 5

scan_area.project_dict[sub_list[1]].display_project(v_min, v_max,
                                                    field='reflectance_radial')


# %% Accessing individual scan data

# Data for the underlying vtk objects can be accessed as numpy arrays
# For example, the x,y,z points in the scanner's coordinate system
print('x, y, z points in scanners coordinate system:')
print(scan_area.project_dict[sub_list[0]].scan_dict['ScanPos001'].dsa_raw.
      Points)

print('PointData arrays for specific scan')
print(scan_area.project_dict[sub_list[0]].scan_dict['ScanPos001'].dsa_raw.
      PointData.keys())

print('Intensity array ')
print(scan_area.project_dict[sub_list[0]].scan_dict['ScanPos001'].dsa_raw.
      PointData['Intensity'])


