#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:51:02 2022

@author: thayer
"""

import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% Load ScanArea

project_names = ['mosaic_01_101819.RiSCAN',
                 'mosaic_01_102019.RiSCAN',
                 ]
project_path = '/media/thayer/Data/mosaic_lidar/Snow1/'

scan_area = pydar.ScanArea(project_path, project_names, import_mode='empty')

# %% Compare reflectors

scan_area.compare_reflectors(project_names[0], project_names[1],
                             delaunay=True)

# %% Named tuple provides specifications for alignment
# Unfortunately this is currently named Registration may change to alignment
# in future.

from collections import namedtuple

Registration = namedtuple('Registration', ['project_name_0', 'project_name_1',
                                           'reflector_list', 'mode', 
                                           'yaw_angle'], 
                          defaults=[None, None, None])

# A registration tuple that is just the same scan area indicates that this
# is the PRCS all other Projects will be aligned into
scan_area.add_registration_tuple(Registration('mosaic_01_101819.RiSCAN', 
                                              'mosaic_01_101819.RiSCAN'))
# This tuple specifies which scans to align and which reflectors to use
scan_area.add_registration_tuple(Registration('mosaic_01_101819.RiSCAN',
                                              'mosaic_01_102019.RiSCAN',
                                              ['r04', 'r05', 'r07', 'r09'],
                                              'Yaw'))
# register_all executes the reflector alignment specified by the tuples
scan_area.register_all()