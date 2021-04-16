#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_Project_create_z_sigma.py

Created on Wed Apr  7 13:17:11 2021

@author: thayer
"""

import sys
import platform
if platform.system()=='Windows':
    sys.path.append('C:/Users/d34763s/Desktop/DavidCS/PhD/code/pydar/')
else:
    sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# Load a singlescan
# Project path
if platform.system()=='Windows':
    project_path = 'D:\\mosaic_lidar\\ROV\\'
else:
    project_path = '/media/thayer/Data/mosaic_lidar/ROV/'
project_name = 'mosaic_rov_040120.RiSCAN'

project = pydar.Project(project_path, project_name, import_mode=
                      'read_scan', las_fieldnames=['Points', 'PointId',
                                                   'Classification'],
                      class_list=[0, 73])

project.apply_transforms(['sop'])

project.create_z_sigma()

v_min = 0
v_max = 0.015

project.display_project(v_min, v_max, field='z_sigma')

