# -*- coding: utf-8 -*-
"""
test_ScanArea_difference_projects

Created on Thu Sep 24 10:34:30 2020

@author: d34763s
"""

import os
import math
from math import sqrt
from collections import namedtuple
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\pydar\\')
import pydar
import vtk
import numpy as np
import matplotlib.pyplot as plt

# %% Registration list

Registration = namedtuple('Registration', ['project_name_0', 'project_name_1',
                                           'reflector_list', 'mode', 
                                           'yaw_angle'], 
                          defaults=[None, None, None])

project_names = ['mosaic_01_180120.RiSCAN',
                 'mosaic_01_290120.RiSCAN',
                 'mosaic_01_060220.RiSCAN']

registration_list = [Registration('mosaic_01_180120.RiSCAN', 
                                  'mosaic_01_180120.RiSCAN'),
                     Registration('mosaic_01_180120.RiSCAN',
                                  'mosaic_01_290120.RiSCAN',
                                  ['r01', 'r02', 'r03', 'r09', 'r10', 
                                   'r12', 'r13', 'r14'],
                                  'LS'),
                     Registration('mosaic_01_290120.RiSCAN',
                                  'mosaic_01_060220.RiSCAN',
                                  ['r01', 'r03', 'r09', 'r12', 'r14', 'r23'],
                                  'LS')
                     ]

# %% Init and register

scan_area = pydar.ScanArea('D:\\mosaic_lidar\\Snow1\\', project_names,
                       registration_list)


scan_area.register_all()
print('done registering')



# %% Now create meshes

load_mesh = True

if load_mesh:
    for project_name in scan_area.project_dict:
        scan_area.project_dict[project_name].read_mesh()
else:
    subgrid_x = 30
    subgrid_y = 30
    alpha = 6
    overlap = 0.2
    
    scan_area.merged_points_to_mesh(subgrid_x, subgrid_y, alpha=alpha, overlap=overlap)
    # And write to files
    for project_name in scan_area.project_dict:
        print(project_name)
        scan_area.project_dict[project_name].write_mesh()
        
# %% Display

project_name = 'mosaic_01_060220.RiSCAN'
project = scan_area.project_dict[project_name]

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
camera = renderer.GetActiveCamera()
print(camera.GetPosition())
iren.Start()

# %% Create Images

nx = 7000
ny = 7000
dx = .1
dy = .1
x0 = -150.0
y0 = -350.0

scan_area.mesh_to_image(nx, ny, dx, dy, x0, y0)

# %% Display

project_name = 'mosaic_01_290120.RiSCAN'
project = scan_area.project_dict[project_name]

z_min = -3
z_max = -1.5

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

# %% Test difference projects

project_name_0 = 'mosaic_01_180120.RiSCAN' #'mosaic_01_290120.RiSCAN' #
project_name_1 = 'mosaic_01_290120.RiSCAN' #'mosaic_01_060220.RiSCAN' #

scan_area.difference_projects(project_name_0, project_name_1)

# %% Display in VTK

z_min = -.3
z_max = 0.3

# Let's try to display
mapper = vtk.vtkDataSetMapper()
mapper.SetInputData(scan_area.difference_dict[
    (project_name_0, project_name_1)])
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

# %% Write to file

diff_window = 0.3

scan_area.write_plot_difference_projects(project_name_0, project_name_1,
                                         diff_window)

# %% Display

diff_window = .1
z_min = -3.9
z_max = -2.9

f, axs = plt.subplots(1, 3, figsize=(30, 10))

cf = axs[0].imshow(scan_area.project_dict[project_name_0].image_numpy, 
               vmin=z_min, vmax=z_max, cmap='cividis')
axs[0].axis('equal')
axs[0].set_title(project_name_0)
f.colorbar(cf, ax=axs[0])

cf = axs[1].imshow(scan_area.difference_dict[(project_name_0, project_name_1)], 
               vmin=-1*diff_window, vmax=diff_window, cmap='coolwarm')
axs[1].axis('equal')
axs[1].set_title('difference')
f.colorbar(cf, ax=axs[1])

cf = axs[2].imshow(scan_area.project_dict[project_name_1].image_numpy, 
               vmin=z_min, vmax=z_max, cmap='cividis')
axs[2].axis('equal')
axs[2].set_title(project_name_1)
f.colorbar(cf, ax=axs[2])

# %% Look at histogram of differences

plt.hist(scan_area.difference_dict[(project_name_0, project_name_1)].ravel(),
         10000, density=True)
plt.xlim([-.5, .5])


# %% Ask what proportion of the surface did not change

change = .02

pix_same = (np.abs(
            scan_area.difference_dict[(project_name_0, project_name_1)].
            ravel())<=change).sum()

pix_total = (scan_area.difference_dict[(project_name_0, project_name_1)].size 
             - np.isnan(scan_area.difference_dict[(project_name_0, project_name_1)].
                        ravel()).sum())

print(pix_same/pix_total)

print(np.nanmean(scan_area.difference_dict[(project_name_0, project_name_1)]))


