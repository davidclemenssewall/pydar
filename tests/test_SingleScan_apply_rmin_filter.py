#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 14:04:37 2021

@author: thayer
"""

import time
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# Load a Single Scan
project_path = '/media/thayer/Data/mosaic_lidar/ROV/'
project_name = 'mosaic_rov_02_090520.RiSCAN'
scan_name = 'ScanPos001'

ss = pydar.SingleScan(project_path, project_name, scan_name, import_mode=
                      'read_scan', las_fieldnames=['Points', 'Classification',
                                                   'PointId'])

ss.add_sop()
ss.apply_transforms(['sop'])

# %% First, a very basic elevation filter
z_max = 3
ss.apply_elevation_filter(z_max)

# %% Apply new filter

t0 = time.perf_counter()
ss.apply_rmin_filter()
t1 = time.perf_counter()

print(t1-t0)

# %% Just for interest, also apply snowflake filter 3
leafsize = 100
z_std_mult = 3.5

t0 = time.perf_counter()
ss.apply_snowflake_filter_3(z_std_mult, leafsize)
t1 = time.perf_counter()

print(t1-t0)


# %% Display classification

ss.create_filter_pipeline()

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderer.AddActor(ss.actor)

renderWindow.AddRenderer(renderer)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)

iren.Initialize()
renderWindow.Render()
iren.Start()
