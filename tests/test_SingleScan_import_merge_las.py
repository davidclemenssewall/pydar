#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_SingleScan_import_merge_las.py

Test functionality for merging multiple LAS files into one scan in PDAL
Useful for when scanner only collects partial scans
Created on Fri May 20 17:07:35 2022

@author: thayer
"""


import vtk
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% Test init

project_path = '/media/thayer/Data/SALVO/tls_data/chk'
project_name = 'salvo_chk_190522.RiSCAN'
scan_name = 'ScanPos004'

ss = pydar.SingleScan(project_path, project_name, scan_name, import_mode='import_las')

# %% Test add SOP and add z offset

ss.add_sop()
ss.add_z_offset(2.5)

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