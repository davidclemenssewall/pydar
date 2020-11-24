# -*- coding: utf-8 -*-
"""
test_Project_create_im_nan_border.py

Created on Tue Nov  3 17:05:50 2020

@author: d34763s
"""

import vtk
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\lidar_processing\\')
import pydar
from vtk.numpy_interface import dataset_adapter as dsa
from scipy.special import gamma, kv
import time


# %% init

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01_102019.RiSCAN'


project = pydar.Project(project_path, project_name)

read_scans = True

if read_scans:
    project.read_scans()
    
else: 
    z_diff = 0.05
    r_min = 5
    N = 5
    project.apply_snowflake_filter_2(z_diff, N, r_min)
    project.write_scans()

project.apply_transforms(['sop'])

load_mesh = True

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
    
# %% Display to check

z_min = -3
z_max = -1.5

project.display_project(z_min, z_max)

# %% Now create_empirical cdf

# Let's make these the same bounds as the image we'll compare with

nx = 700
ny = 600
dx = .1
dy = .1
x0 = 90
y0 = 100

project.mesh_to_image(nx, ny, dx, dy, x0, y0)

project.create_empirical_cdf_image(z_min, z_max)

# %% Now test create_im_gaus

project.create_im_gaus()

# %% and Display

project.image.GetPointData().SetActiveScalars('im_gaus')
project.display_image(-4, 4)

# %% Create im_nan_border

buffer = 2
project.create_im_nan_border(buffer)

# %% Display

project.image.GetPointData().SetActiveScalars('im_nan_border')
mapper = vtk.vtkImageMapper()
mapper.SetInputData(project.image)
mapper.SetColorWindow(1.0)
mapper.SetColorLevel(0.5)

actor = vtk.vtkActor2D()
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

# %% Create theta1 and theta

# First we need to create our target ro and weights arrays
# Create array of distances for correlation
x = np.hstack((np.arange(nx//2)*dx, np.arange(-nx//2, 0)*dx))[np.newaxis, :]
y = np.hstack((np.arange(ny//2)*dx, np.arange(-ny//2, 0)*dx))[:, np.newaxis]
X = np.tile(x, (ny, 1))
Y = np.tile(y, (1, nx))
r = np.hypot(X, Y)
# For convenience we'll set r[0,0] equal to 1 so we don't have to deal with
# dividing by zero, we will always manually set the 0,0 component of ro
# and the weights array
r[0, 0] = 1

# Now create target ro
p = 15
nu = 0.4
k = np.sqrt(8)/p

ro = ((2**(1-nu))/gamma(nu)) * ((k*r)**nu) * kv(nu, k*r)
ro[0, 0] = 1

# Create weights array
wts = 1/r

# Plot just to be sure
#plt.imshow(ro)

(theta1, theta) = pydar.fit_theta_to_ro(ro, wts, p=15)

print(theta1)
print(theta)

q_min = pydar.block_circ_base(ny, nx, 2, theta) * theta1 
sigma_min = (1/(nx*ny))*np.fft.irfft2(np.fft.rfft2(q_min, norm='ortho')**(-1), 
                                   norm='ortho')

project.add_theta(theta1, theta)

# %% now test create_gmrf

t0 = time.perf_counter()
project.create_gmrf()
t1 = time.perf_counter()

# %% and fill missing pixels and display

project.create_im_gaus_mean_fill()

# %% display
project.image.GetPointData().SetActiveScalars('im_gaus_mean_fill')
project.display_image(-4, 4)

# %% now examine if we transform filled pixels back to orginal distribution
project.create_elevation_mean_fill()

# %% and display
project.image.GetPointData().SetActiveScalars('Elevation_mean_fill')
project.display_image(-2.75, -1.75)