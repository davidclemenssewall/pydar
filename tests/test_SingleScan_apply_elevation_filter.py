# -*- coding: utf-8 -*-
"""
test_SingleScan_apply_elevation_filter.py

test the elevation filter functionality and in general the new filtering 
paradigm.

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
#ss.apply_transforms(['sop'])

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(ss.get_polydata())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderer.AddActor(actor)

renderWindow.AddRenderer(renderer)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)

iren.Initialize()
renderWindow.Render()
iren.Start()

# %% Let's print out some point values

print(ss.dsa_raw.Points[:10, :])

# %% Now let's try applying our filter

z_max = -1

ss.apply_elevation_filter(z_max)

print(ss.dsa_raw.PointData['flag_filter'][:10])

# Seems to work good, next we'll develop a way of displaying this.