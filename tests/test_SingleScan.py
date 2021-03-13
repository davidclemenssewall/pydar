# -*- coding: utf-8 -*-
"""
Script to test SingleScan class in pydar module.

Created on Tue Sep  8 15:54:32 2020

@author: d34763s
"""

import json
import vtk
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% Test init

project_path = '/media/thayer/Data/mosaic_lidar/ROV/'
project_name = 'mosaic_rov_190120.RiSCAN'
scan_name = 'ScanPos004'

ss = pydar.SingleScan(project_path, project_name, scan_name)

# %% Check that we created a filter tree
print(json.dumps(ss.filt_history_dict, indent=4))

# %% Test init

project_path = '/media/thayer/Data/mosaic_lidar/ROV/'
project_name = 'mosaic_rov_190120.RiSCAN'
scan_name = 'ScanPos004'

ss = pydar.SingleScan(project_path, project_name, scan_name, import_mode=
                      'import_las')

# %% Check that we created a filter tree
print(json.dumps(ss.filt_history_dict, indent=4))


# %% Test add SOP and add z offset

ss.add_sop()
ss.add_z_offset(3)

print(ss.transform_dict['sop'].GetOrientation())


# %% Apply transforms
ss.apply_transforms(['sop', 'z_offset'])
print(json.dumps(ss.filt_history_dict, indent=4))

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