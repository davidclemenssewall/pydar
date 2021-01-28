# -*- coding: utf-8 -*-
"""
test_Project_merged_points_to_mesh

Created on Mon Sep 21 11:59:28 2020

@author: d34763s
"""

import vtk
import os
import sys
sys.path.append('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\pydar\\')
#os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\pydar\\')
#print(os.getcwd())
import pydar

# %% init

project_path = 'D:\\mosaic_lidar\\ROV\\'
project_name = 'mosaic_rov_040120.RiSCAN'

project = pydar.Project(project_path, project_name, load_scans=True,
                        read_scans=True)

project.apply_transforms(['sop'])

# %% test merged_points_to_mesh

subgrid_x = 25
subgrid_y = 25

project.merged_points_to_mesh(subgrid_x, subgrid_y)

# %% Display

z_min = -3
z_max = -1.5

elevFilter = vtk.vtkSimpleElevationFilter()
elevFilter.SetInputData(project.mesh)
elevFilter.Update()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(elevFilter.GetOutputPort())
mapper.SetLookupTable(pydar.mplcmap_to_vtkLUT(z_min, z_max))
mapper.SetScalarRange(z_min, z_max)

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderer.AddActor(actor)

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