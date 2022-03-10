#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_Project_get_local_max.py

Created on Thu Mar 10 05:16:26 2022

@author: thayer
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# Let's load a project
project_path = '/media/thayer/Data/mosaic_lidar/ROV/'
project_name = 'mosaic_rov_040120.RiSCAN'

project = pydar.Project(project_path, project_name, import_mode=
                      'read_scan', las_fieldnames=['Points', 'PointId',
                                                   'Classification'])

project.read_transforms()
project.apply_transforms(['current_transform'])

# %% Get local maxima

z_threshold = -1.9
rmax = 0.5

local_max = project.get_local_max(z_threshold, rmax, return_dist=True,
                                  return_zs=True, closest_only=True)

# %% Display

zmin = -3
zmax = -1


points = vtk.vtkPoints()
points.SetData(numpy_to_vtk(local_max[0], array_type=vtk.VTK_DOUBLE))
pdata = vtk.vtkPolyData()
pdata.SetPoints(points)

vgf = vtk.vtkVertexGlyphFilter()
vgf.SetInputData(pdata)
vgf.Update()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(vgf.GetOutputPort())
mapper.SetScalarVisibility(0)

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(1.0, 1.0, 1.0)
actor.GetProperty().RenderPointsAsSpheresOn()
actor.GetProperty().SetPointSize(10)

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderer.AddActor(actor)

for scan_name in project.scan_dict:
    ss = project.scan_dict[scan_name]
    ss.create_elevation_pipeline(zmin, zmax)
    renderer.AddActor(ss.actor)
    
renderWindow.AddRenderer(renderer)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)

iren.Initialize()
renderWindow.Render()
iren.Start()

# %% As before, but now let's color by distance to check that worked

points = vtk.vtkPoints()
points.SetData(numpy_to_vtk(local_max[0], array_type=vtk.VTK_DOUBLE))
pdata = vtk.vtkPolyData()
pdata.SetPoints(points)
arr = numpy_to_vtk(local_max[1], array_type=vtk.VTK_DOUBLE)
arr.SetName('dist')
pdata.GetPointData().SetScalars(arr)

vgf = vtk.vtkVertexGlyphFilter()
vgf.SetInputData(pdata)
vgf.Update()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(vgf.GetOutputPort())
mapper.SetLookupTable(pydar.mplcmap_to_vtkLUT(0, 150,
                                              name='viridis'))
mapper.SetScalarRange(0, 150)
mapper.SetScalarVisibility(1)

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().RenderPointsAsSpheresOn()
actor.GetProperty().SetPointSize(10)

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderer.AddActor(actor)

for scan_name in project.scan_dict:
    ss = project.scan_dict[scan_name]
    ss.create_solid_pipeline(color='grey')
    renderer.AddActor(ss.actor)
    
renderWindow.AddRenderer(renderer)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)

iren.Initialize()
renderWindow.Render()
iren.Start()
