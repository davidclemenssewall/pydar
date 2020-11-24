# -*- coding: utf-8 -*-
"""
test_SingleScan_snowflake_filter_2.py

Created on Tue Sep 29 15:36:01 2020

@author: d34763s
"""

import vtk
import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\lidar_processing\\')
import pydar
import time

# %% init

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01_102519.RiSCAN'
scan_name = 'ScanPos003'

ss = pydar.SingleScan(project_path, project_name, scan_name)

ss.add_sop()
print(ss.transform_dict['sop'].GetPosition())
ss.apply_transforms(['sop'])

# %% Apply filter

z_diff = 0.05
r_min = 5
N = 5

t0 = time.process_time()
ss.apply_snowflake_filter_2(z_diff, N, r_min)
t1 = time.process_time()
print(t1-t0)

# %% Display filtered points

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

# %% Display by elevation

z_min = -3
z_max = -1.5

ss.create_elevation_pipeline(z_min, z_max)
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
