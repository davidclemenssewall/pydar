# -*- coding: utf-8 -*-
"""
test_SingleScan_create_filter_pipeline.py

Test displaying different filters

Created on Mon Sep 28 15:31:59 2020

@author: d34763s
"""

import vtk
import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\lidar_processing\\')
import pydar

# %% init with a scan that has some snowflakes

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01_102519.RiSCAN'
scan_name = 'ScanPos003'

ss = pydar.SingleScan(project_path, project_name, scan_name)

ss.add_sop()
print(ss.transform_dict['sop'].GetPosition())
ss.apply_transforms(['sop'])

# %% Now let's try applying our filter

z_max = -1

ss.apply_elevation_filter(z_max)

# %% Now display

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