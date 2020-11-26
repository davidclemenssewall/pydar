# -*- coding: utf-8 -*-
"""
test_Project_apply_snowflake_filter

Created on Wed Sep 23 10:22:08 2020

@author: d34763s
"""

import vtk
import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\pydar\\')
import pydar
from math import sqrt

# %% Test init

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01_102019.RiSCAN'

project = pydar.Project(project_path, project_name)
print(project.project_date)

# %% Add z offset and apply transforms

project.add_z_offset(3)
project.apply_transforms(['sop', 'z_offset'])

# %% Display

z_min = 0
z_max = 2

project.display_project(z_min, z_max)

# %% Apply snowflake filter

shells = [(0, 5, 0, 0), # remove all points within 5 m of scanner
          (5, 25, .09*sqrt(2), 5),
          (25, 40, .25*sqrt(2), 5),
          (40, 60, .50*sqrt(2), 5),
          (60, 80, .9*sqrt(2), 5),
          (80, 100, 1.2*sqrt(2), 5),
          (100, None, 1, 0)]

project.apply_snowflake_filter(shells)

# %% Display

z_min = 0
z_max = 2

project.display_project(z_min, z_max)

# %% Let's see if mesh creation can get rid of some of the snowflakes 
# we'll try to use a relatively high alpha value to make it work

subgrid_x = 50
subgrid_y = 50
alpha = 6

project.merged_points_to_mesh(subgrid_x, subgrid_y, alpha=alpha)

# %% display

elevFilter = vtk.vtkSimpleElevationFilter()
elevFilter.SetInputData(project.mesh)
elevFilter.Update()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(elevFilter.GetOutput())
mapper.SetLookupTable(pydar.mplcmap_to_vtkLUT(z_min, z_max))
mapper.SetScalarRange(z_min, z_max)
mapper.SetScalarVisibility(1)

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

# %% Let's write out the mesh for convenience

project.write_mesh()

# %% and test that we can read it back in

delattr(project, 'mesh')
project.read_mesh()

# %% Display

elevFilter = vtk.vtkSimpleElevationFilter()
elevFilter.SetInputData(project.mesh)
elevFilter.Update()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(elevFilter.GetOutput())
mapper.SetLookupTable(pydar.mplcmap_to_vtkLUT(z_min, z_max))
mapper.SetScalarRange(z_min, z_max)
mapper.SetScalarVisibility(1)

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