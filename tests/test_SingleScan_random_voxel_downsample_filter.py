#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_SingleScan_random_voxel_downsample_filter.py

test random voxel downsampling

Created on Sun Mar 21 15:08:14 2021

@author: thayer
"""

import json
import numpy as np
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% Test init

project_path = '/media/thayer/Data/mosaic_lidar/ROV/'
project_name = 'mosaic_rov_190120.RiSCAN'
scan_name = 'ScanPos004'

ss = pydar.SingleScan(project_path, project_name, scan_name, import_mode=
                      'read_scan')

ss.add_sop()
ss.add_z_offset(3)

ss.apply_transforms(['sop', 'z_offset'])

# %% Test voxel downsampling

wx = 1
wy = 1
wz = None

ss.random_voxel_downsample_filter(wx, wy, wz)

# %% Examine output and plot

pdata, history_dict = ss.get_polydata(history_dict=True)

print(pdata)

print(json.dumps(history_dict, indent=4))

# %% display output
import vtk

z_min = 0
z_max = 1.5

ss.create_elevation_pipeline(z_min, z_max)

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()

renderer.AddActor(ss.actor)
    
scalarBar = vtk.vtkScalarBarActor()
scalarBar.SetLookupTable(pydar.mplcmap_to_vtkLUT(z_min, z_max))
renderer.AddActor2D(scalarBar)


renderWindow.AddRenderer(renderer)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)

iren.Initialize()
renderWindow.Render()
iren.Start()