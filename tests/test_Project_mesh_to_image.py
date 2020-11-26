# -*- coding: utf-8 -*-
"""
test_Project_mesh_to_image

Created on Mon Sep 21 11:59:28 2020

@author: d34763s
"""

import vtk
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\pydar\\')
import pydar
from vtk.numpy_interface import dataset_adapter as dsa


# %% init

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01_180120.RiSCAN'

project = pydar.Project(project_path, project_name)

project.apply_transforms(['sop'])

load_mesh = True

# %% test merged_points_to_mesh

if load_mesh:
    project.read_mesh()
else:
    subgrid_x = 50
    subgrid_y = 50
    alpha = 5 # no gaps larger than 0.2
    overlap = 0.25
    
    project.merged_points_to_mesh(subgrid_x, subgrid_y, alpha=alpha,
                                  overlap=overlap)
    project.write_mesh()

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

# %% test mesh_to_image

z_min = -3
z_max = -1.5

nx = 7000
ny = 7000
dx = .1
dy = .1
x0 = -150.0
y0 = -350.0

project.mesh_to_image(nx, ny, dx, dy, x0, y0)


# Let's try to display
mapper = vtk.vtkDataSetMapper()
mapper.SetInputData(project.image)
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

# %% test plot_image

z_min = -3
z_max = -1.5

project.plot_image(z_min, z_max)

# %% Do some other testing

im = vtk.vtkImageData()
im.SetDimensions(project.image.
                 GetDimensions())
im.SetOrigin(project.image.
             GetOrigin())
im.SetSpacing(project.image.
              GetSpacing())
arr = vtk.vtkFloatArray()
arr.SetNumberOfValues(project.image.
                      GetNumberOfPoints())
arr.SetName('Difference')
im.GetPointData().SetScalars(arr)
dsa_temp = dsa.WrapDataObject(im)

# Try setting the array
dsa_temp.PointData['Difference'][:] = np.ravel(project.get_np_nan_image() - 1)

# Display the image
mapper = vtk.vtkDataSetMapper()
mapper.SetInputData(im)
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