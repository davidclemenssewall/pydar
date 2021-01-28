# -*- coding: utf-8 -*-
"""
test_ClassFilter.py

Test new ClassFilter class

Created on Wed Jan 27 09:28:39 2021

@author: d34763s
"""

import vtk
import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\pydar\\')
import pydar

# %% Test init

project_path = 'D:\\mosaic_lidar\\ROV\\'
project_name = 'mosaic_rov_040120.RiSCAN'
scan_name = 'ScanPos001'

ss = pydar.SingleScan(project_path, project_name, scan_name, read_scan=True)

# %% Display

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

# %% Try replacing currentFilter with our new class filter

ss.currentFilter = pydar.ClassFilter(ss.transformFilter.GetOutputPort())

z_min = -5
z_max = 0

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