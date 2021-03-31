#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_SingleScan_apply_manual_filter.py

Test using the manual selection loop based filter

Created on Tue Mar 30 16:48:05 2021

@author: thayer
"""

import numpy as np
import vtk
import sys
import platform
if platform.system()=='Windows':
    sys.path.append('C:/Users/d34763s/Desktop/DavidCS/PhD/code/pydar/')
else:
    sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# Load a Single Scan
project_path = '/media/thayer/Data/mosaic_lidar/ROV/'
project_name = 'mosaic_rov_250120.RiSCAN'
scan_name = 'ScanPos008'

ss = pydar.SingleScan(project_path, project_name, scan_name, import_mode=
                      'read_scan', las_fieldnames=['Points', 'Classification',
                                                   'PointId'])

ss.add_sop()
ss.apply_transforms(['sop'])

corner_coords = np.array([[232.0, 116.6, 0],
                          [164.6, 57.1, 0],
                          [143.3, 3.6, 0],
                          [195.5, -47.1, 0],
                          [288.1, -54.1, 0],
                          [337.0, -30.1, 0],
                          [390.7, -34.3, 0],
                          [451.3, 36.0, 0],
                          [297.5, 154.9, 0]])

ss.apply_manual_filter(corner_coords)


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
