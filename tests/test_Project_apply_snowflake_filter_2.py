# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:53:24 2020

@author: d34763s
"""

import time
import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\lidar_processing\\')
import pydar
import vtk

# %% init

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01_102519.RiSCAN'

project = pydar.Project(project_path, project_name)

project.apply_transforms(['sop'])

# %% Test snowflake_filter_2

z_diff = 0.05
r_min = 5
N = 5

t0 = time.process_time()
project.apply_snowflake_filter_2(z_diff, N, r_min)
t1 = time.process_time()
print(t1-t0)

# %% Display

z_min = -3
z_max = -1.5

project.display_project(z_min, z_max)

# %% Let's convert to mesh to convince ourselves that we really did catch the
# snowflakes
subgrid_x = 30
subgrid_y = 30
alpha = 0.2
overlap = 0.2

project.merged_points_to_mesh(subgrid_x, subgrid_y, alpha=alpha, 
                              overlap=overlap)

# %% And display mesh

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