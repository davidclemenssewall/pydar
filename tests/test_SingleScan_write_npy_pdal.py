# -*- coding: utf-8 -*-
"""
test_SingleScan_write_npy_pdal.py

Test our functionality for writing structured numpy arrays in a format PDAL
can read.

Created on Mon Nov 30 18:04:30 2020

@author: d34763s
"""

# %% Imports

import os
import numpy as np
from vtk.numpy_interface import dataset_adapter as dsa
import pdal
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

# location to put outputfiles
output_dir = 'D:\\mosaic_lidar\\Snow1\\temp\\'

# %% Initialize project

# Initialize scan area object, setting the flags to false means we don't try
# to actually load any scan data that we don't have.
# First argument is path to wherever you put that file
scan_area = pydar.ScanArea('D:\\mosaic_lidar\\Snow1\\', 
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

for project_name in sub_list:
    scan_area.project_dict[project_name].create_reflectance()

# %% Create numpy structured array
project_name = sub_list[1]
scan_name = 'ScanPos006'

pdata = (scan_area.project_dict[project_name].scan_dict[scan_name].
         transformFilter.GetOutput())

dsa_pdata = dsa.WrapDataObject(pdata)

n_pts = pdata.GetNumberOfPoints()

output_npy = np.zeros(n_pts, dtype={'names':('X', 'Y', 'Z', 'Reflectance'),
                                    'formats':(np.float32, np.float32, 
                                               np.float32, np.float32)})
output_npy['X'] = dsa_pdata.Points[:,0]
output_npy['Y'] = dsa_pdata.Points[:,1]
output_npy['Z'] = dsa_pdata.Points[:,2]
output_npy['Reflectance'] = dsa_pdata.PointData['reflectance']

np.save(output_dir + 'test', output_npy)

# %% test write_npy_pdal
project_name = sub_list[1]

for scan_name in scan_area.project_dict[project_name].scan_dict:
    scan_area.project_dict[project_name].scan_dict[scan_name].write_npy_pdal(
        output_dir)

# %% See if we can load numpy array with pdal

json = """
[
    {
        "filename": "D:/mosaic_lidar/Snow1/temp/mosaic_01_040120.RiSCAN_ScanPos005.npy",
        "type": "readers.numpy"
    }
]
"""

pipeline = pdal.Pipeline(json)
count = pipeline.execute()
arrays = pipeline.arrays
metadata = pipeline.metadata
log = pipeline.log