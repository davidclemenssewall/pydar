#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_FlakeOut.py

A demonstration of using the FlakeOut filter for the README

Created on Tue Oct 11 09:21:38 2022

@author: thayer
"""

import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

project_name = 'mosaic_01_101819.RiSCAN'
project_path = '/media/thayer/Data/mosaic_lidar/Snow1/'

# Import project
project = pydar.Project(project_path, project_name, import_mode='import_las')

# Apply SOP transform to all SingleScans
project.apply_transforms(['sop'])

# Apply returnindex filter 
# (aka filter points that are visibly isolated in space)
radial_precision=0.005
project.apply_snowflake_filter_returnindex(radial_precision=radial_precision)

# By default we filter out snowflakes, so if we want to display them we
# need to reset the classes we filter out
for scan_name in project.scan_dict.keys():
    project.scan_dict[scan_name].update_current_filter('all')

# Display, note now we are displaying by classification and the z_* are
# superfluous
z_min = -3.5
z_max = -1.5
project.display_project(z_min, z_max, field='Classification')

# Apply elevation and local outlier filtering
z_max = 1.5 # max height in project reference frame
leafsize = 100
z_std_mult = 3.5

# Apply the filter to each SingleScan
for scan_name in project.scan_dict:
    ss = project.scan_dict[scan_name]
    ss.apply_elevation_filter(z_max)
    ss.apply_rmin_filter()
    ss.apply_snowflake_filter_3(z_std_mult, leafsize)

# Display
project.display_project(z_min, z_max, field='Classification')
