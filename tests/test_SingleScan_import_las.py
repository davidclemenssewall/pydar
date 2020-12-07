# -*- coding: utf-8 -*-
"""
test_SingleScan_import_las.py

Test functionality for importing singlescan as a las file using PDAL
Created on Sat Dec  5 15:05:51 2020

@author: d34763s
"""

import vtk
import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\pydar\\')
import pydar

# %% Test init

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01_040120.RiSCAN'
scan_name = 'ScanPos006'

ss = pydar.SingleScan(project_path, project_name, scan_name, import_las=True)

# %% Test add SOP and add z offset

ss.add_sop()
ss.add_z_offset(4.5)

print(ss.transform_dict['sop'].GetOrientation())

# %% Apply transforms
ss.apply_transforms(['sop', 'z_offset'])

# %% Create elevation pipeline
z_min = 0
z_max = 2

ss.create_elevation_pipeline(z_min, z_max)

# %% Render actor

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

# %% Create reflectance pipeline

v_min = -20
v_max = 5

ss.create_reflectance_pipeline(v_min, v_max)

# %% Render actor

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderer.AddActor(ss.actor)

scalarBar = vtk.vtkScalarBarActor()
scalarBar.SetLookupTable(pydar.mplcmap_to_vtkLUT(v_min, v_max))
renderer.AddActor2D(scalarBar)


renderWindow.AddRenderer(renderer)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)

iren.Initialize()
renderWindow.Render()
iren.Start()