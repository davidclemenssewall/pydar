# -*- coding: utf-8 -*-
"""
example_kaku_20201124.py

Basic script to introduce Robbie Mallett to the capabilities in pydar. This
script was written and is designed to be run in an ipython kernel in spyder
but should work otherwise.

Created on Tue Nov 24 18:33:45 2020

@author: d34763s
"""

# %% Imports

import numpy as np
import vtk
import matplotlib.pyplot as plt
import os
import copy
from collections import namedtuple
from scipy.linalg import lstsq
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\pydar\\')
import pydar

fig_write_path = ("C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\"
                  + "presentations\\figures\\surface_roughness_20201125\\")

# %% Parameters to initialize project and scan_area
# This is a bit overkill for just looking at a single scan, as we are here,
# but it's convenient to keep everything in the same reference frame as I use
# in other places. It will also make it easier to expand this script if we 
# want.

Registration = namedtuple('Registration', ['project_name_0', 'project_name_1',
                                           'reflector_list', 'mode', 
                                           'yaw_angle'], 
                          defaults=[None, None, None])

project_names = ['mosaic_01_101819.RiSCAN',
                 'mosaic_01_102019.RiSCAN',
                 'mosaic_01_102519.RiSCAN',
                 'mosaic_01_110119.RiSCAN',
                 'mosaic_01_110819.RiSCAN',
                 'mosaic_01_111519.RiSCAN',
                 'mosaic_01b_061219.RiSCAN.RiSCAN.RiSCAN',
                 'mosaic_01_122719.RiSCAN',
                 'mosaic_01_040120.RiSCAN',
                 'mosaic_01_180120.RiSCAN',
                 'mosaic_01_290120.RiSCAN',
                 'mosaic_01_060220.RiSCAN',
                 'mosaic_01_150220.RiSCAN.RiSCAN',
                 'mosaic_01_280220.RiSCAN'
                 ]

registration_list = [Registration('mosaic_01_102019.RiSCAN', 
                                  'mosaic_01_102019.RiSCAN'),
                     Registration('mosaic_01_102019.RiSCAN',
                                  'mosaic_01_101819.RiSCAN',
                                  ['r04', 'r05', 'r07', 'r09'],
                                  'Yaw'),
                     Registration('mosaic_01_102019.RiSCAN', 
                                  'mosaic_01_102519.RiSCAN',
                                  ['r01', 'r02', 'r03', 'r09', 'r08'],
                                  'LS'),
                     Registration('mosaic_01_102519.RiSCAN',
                                  'mosaic_01_110119.RiSCAN',
                                  ['r01', 'r03', 'r04', 'r05', 'r06', 'r07'],
                                  'LS'),
                     Registration('mosaic_01_110119.RiSCAN',
                                  'mosaic_01_111519.RiSCAN',
                                  ['r02', 'r03', 'r04'],
                                  'Yaw'),
                     Registration('mosaic_01_111519.RiSCAN',
                                  'mosaic_01_110819.RiSCAN',
                                  ['r02', 'r05', 'r06', 'r07', 'r10'],
                                  'LS'),
                     Registration('mosaic_01_111519.RiSCAN',
                                  'mosaic_01b_061219.RiSCAN.RiSCAN.RiSCAN',
                                  ['r01', 'r11'],
                                  'Yaw'),
                     Registration('mosaic_01b_061219.RiSCAN.RiSCAN.RiSCAN',
                                  'mosaic_01_122719.RiSCAN',
                                  ['r02', 'r11'],
                                  'Yaw'),
                     Registration('mosaic_01_122719.RiSCAN',
                                  'mosaic_01_040120.RiSCAN',
                                  ['r01', 'r13', 'r14', 'r15'],
                                  'Yaw'),
                     Registration('mosaic_01_040120.RiSCAN',
                                  'mosaic_01_180120.RiSCAN',
                                  ['r03', 'r09', 'r10', 'r11', 'r24'],
                                  'LS'),
                     Registration('mosaic_01_180120.RiSCAN',
                                  'mosaic_01_290120.RiSCAN',
                                  ['r01', 'r02', 'r03', 'r09', 'r10', 
                                   'r12', 'r13', 'r14'],
                                  'LS'),
                     Registration('mosaic_01_290120.RiSCAN',
                                  'mosaic_01_060220.RiSCAN',
                                  ['r01', 'r03', 'r09', 'r12', 'r14', 'r23'],
                                  'LS'),
                     Registration('mosaic_01_060220.RiSCAN',
                                  'mosaic_01_150220.RiSCAN.RiSCAN',
                                  ['r03', 'r09', 'r23'],
                                  'Yaw'),
                     Registration('mosaic_01_150220.RiSCAN.RiSCAN',
                                  'mosaic_01_280220.RiSCAN',
                                  ['r10', 'r11', 'r24', 'r12'],
                                  'LS')
                     ]

sub_list = ['mosaic_01_060220.RiSCAN']

poly = 'all_within_16m'

# %% Initialize project

# Initialize scan area object, setting the flags to false means we don't try
# to actually load any scan data that we don't have.
# First argument is path to wherever you put that file
scan_area = pydar.ScanArea('D:\\mosaic_lidar\\test_for_robbie\\Snow1\\', 
                           project_names, registration_list, load_scans=False,
                           read_scans=False)

# Load the one scan we do have.
for project_name in sub_list:
    scan_area.add_project(project_name, load_scans=True, read_scans=False,
                          poly=poly)

# %% Register

# This step roughly aligns the reference frames in all of the Snow1 scans
scan_area.register_all()

# %% Ref for simplicity

# We won't use any more of the functionality from ScanArea here
project = scan_area.project_dict[sub_list[0]]

# %% Display

# Let's begin by just displaying the project in an interactive vtk window
# The default behavior of Project.display_project is to display points colored
# by height, although this can be changed if desired

# Set z_min and z_max for colorbar
z_min = -2.65
z_max = -2.15

project.display_project(z_min, z_max)

# %% Orthogonal projection

# We can also display using an orthogonal projection, convenient for figures
# Note that if you tilt the viewport the scale bars (and the projection
# generally) are no longer valid.

# Try pressing 'u' while in the viewport. Once you close the vtk window
# camera info should print to the terminal
project.display_project(z_min, z_max, mapview=True)

# %% Show example of using implicit functions to select a region of data

# Lower left corner of box of data we want
x0 = -106.40873943777369 #-106.13593789207866
y0 = -155.87079302216304 #-157.30306794316678
# Dimensions of box (in m)
wx = 0.6 #1.2
wy = 0.6 # 1.2
# Rotation angle of box (in degrees CCW from + x-axis)
yaw = -105 #-60 # 45

# Create box object
box = vtk.vtkBox()
box.SetBounds((0, wx, 0, wy, -10, 10))
# We need a transform to put the data in the desired location relative to our
# box
transform = vtk.vtkTransform()
transform.PostMultiply()
transform.RotateZ(yaw)
transform.Translate(x0, y0, 0)
# That transform moves the box relative to the data, so the box takes its
# inverse
transform.Inverse()
box.SetTransform(transform)

# vtkExtractPoints does the actual filtering
extractPoints = vtk.vtkExtractPoints()
extractPoints.SetImplicitFunction(box)
extractPoints.SetInputData(project.get_merged_points())
extractPoints.GenerateVerticesOn()

# Elevation filter copies z values to scalars for displaying by color
elevFilter = vtk.vtkSimpleElevationFilter()
elevFilter.SetInputConnection(extractPoints.GetOutputPort())


# This is a standard vtk visualization pipeline
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(elevFilter.GetOutputPort())
mapper.SetLookupTable(pydar.mplcmap_to_vtkLUT(z_min, z_max))
mapper.SetScalarRange(z_min, z_max)
actor = vtk.vtkActor()
actor.SetMapper(mapper)
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)

renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)
iren.Initialize()
renderWindow.Render()
iren.Start()

print("Number of points: " + str(mapper.GetInput().GetNumberOfPoints()))
print("Point Density (/m**2): " + str(mapper.GetInput().GetNumberOfPoints()/
                                      (wx*wy)))

# %% Create Mesh

# 40,000 points is too many to compute a pairwise semivariogram for. Instead
# we can exploit the FFT if we can work with gridded data. The first step in
# creating a gridded product is to create a mesh that then we will interpolate
# on a grid
# Lower left corner of box of data we want
x0 = -106.40873943777369 #-106.13593789207866
y0 = -155.87079302216304 #-157.30306794316678
# Dimensions of box (in m)
wx = 0.6 #1.2
wy = 0.6 # 1.2
# Rotation angle of box (in degrees CCW from + x-axis)
yaw = -105 # 45

# Alpha determines the maximum gap between points that we'll allow the
# triangulation to fill. Let's set it at 5 mm
alpha = 0.004#0.005

project.merged_points_to_mesh(wx*3, wy*3, alpha=alpha, x0=x0, y0=y0, wx=wx,
                              wy=wy, yaw=yaw)

# %% Display mesh

z_min = -2.49
z_max = -2.42

# Elevation filter copies z values to scalars for displaying by color
elevFilter = vtk.vtkSimpleElevationFilter()
elevFilter.SetInputData(project.mesh)


# This is a standard vtk visualization pipeline works for mesh as well.
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(elevFilter.GetOutputPort())
mapper.SetLookupTable(pydar.mplcmap_to_vtkLUT(z_min, z_max))
mapper.SetScalarRange(z_min, z_max)
actor = vtk.vtkActor()
actor.SetMapper(mapper)
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)

# Let's try overlaying on points to make sure we have the transform right
renderer.AddActor(project.scan_dict['ScanPos011'].actor)

renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)
iren.Initialize()
renderWindow.Render()
iren.Start()

# %% Now that we have a mesh we can create a gridded version

x0 = -106.40873943777369 + .04 #-106.13593789207866
y0 = -155.87079302216304 - .04 # -157.30306794316678 + .15
nx = 128 # 200
ny = 128 # 200
dx = .004 # 0.005
dy = .004 # 0.005
yaw = -105 # 45

project.mesh_to_image(nx, ny, dx, dy, x0, y0, yaw=yaw)

# %% We can also display the image

project.display_image(z_min, z_max)

# %% fit slope
# This region is sloped (not an issue we typically deal with on complete scans)

# Get image as numpy array for easy manipulation
im = project.get_np_nan_image()

X, Y = np.meshgrid(dx*np.arange(nx), dy*np.arange(ny))

data_mask = np.logical_not(np.isnan(im.ravel()))

A = np.hstack((np.ones((data_mask.sum(), 1)), 
                  X.ravel()[data_mask][:, np.newaxis],
                  Y.ravel()[data_mask][:, np.newaxis],
                  (X**2).ravel()[data_mask][:, np.newaxis],
                  (X*Y).ravel()[data_mask][:, np.newaxis],
                  (Y**2).ravel()[data_mask][:, np.newaxis],
                  (X**3).ravel()[data_mask][:, np.newaxis],
                  (Y*X**2).ravel()[data_mask][:, np.newaxis],
                  (X*Y**2).ravel()[data_mask][:, np.newaxis],
                  (Y**3).ravel()[data_mask][:, np.newaxis]))

coef, _, _, _ = lstsq(A, im.ravel()[data_mask])

im_slope = (coef[0] + coef[1]*X + coef[2]*Y + coef[3]*(X**2) + coef[4]*(X*Y) +
            coef[5]*(Y**2) + coef[6]*(X**3) + coef[7]*(Y*X**2) + coef[8]*
            (X*Y**2) + coef[9]*(Y**3))

im_detrend = im - im_slope

f, axs = plt.subplots(1, 3, figsize=(20, 8))

h0 = axs[0].imshow(im, origin='lower', extent=[0, nx*dx, 0, ny*dy])
axs[0].axis('equal')
f.colorbar(h0, ax=axs[0], orientation='horizontal', shrink=0.8)
axs[0].set_title('Surface')
axs[0].set_xlabel('(m)')

h1 = axs[1].imshow(im_slope, origin='lower', extent=[0, nx*dx, 0, ny*dy])
axs[1].axis('equal')
f.colorbar(h1, ax=axs[1], orientation='horizontal', shrink=0.8)
axs[1].set_title('Cubic Trend')
axs[1].set_xlabel('(m)')

h2 = axs[2].imshow(im_detrend, origin='lower', extent=[0, nx*dx, 0, ny*dy])
axs[2].axis('equal')
f.colorbar(h2, ax=axs[2], orientation='horizontal', shrink=0.8)
axs[2].set_title('Detrended Surface')
axs[2].set_xlabel('(m)')

f.savefig(fig_write_path + "fit_slope_CT.png", transparent=True)

# %% look at power spectrum of signal and also show histogram of detrended
# surface

# zero fill, to use FFT
im_detrend_0 = copy.deepcopy(im_detrend)
im_detrend_0[np.isnan(im_detrend)] = 0
spec = np.fft.fft2(im_detrend_0)

pwr_spec = np.abs(spec)**2

dnu_x = 1/(nx*dx)
dnu_y = 1/(ny*dy)

f, axs = plt.subplots(1, 3, figsize=(20, 8))

h0 = axs[0].imshow(im_detrend, origin='lower', extent=[0, nx*dx, 0, ny*dy])
axs[0].axis('equal')
f.colorbar(h0, ax=axs[0], orientation='horizontal', shrink=0.8, 
           label='Height (m)')
axs[0].set_title('Detrended Surface')
axs[0].set_xlabel('(m)')

h1 = axs[1].imshow(np.fft.fftshift(10*np.log10(pwr_spec)), vmin=-40, vmax=20,
                   extent=[-dnu_x*nx/2, dnu_x*nx/2, -dnu_y*ny/2, dnu_y*ny/2])
axs[1].axis('equal')
axs[1].set_title('Power Spectrum')
f.colorbar(h1, ax=axs[1], orientation='horizontal', shrink=0.8, label=
           'Power (dB)')

axs[2].hist(im_detrend.ravel()[data_mask], bins=100, density=True)
axs[2].set_xlabel('Detrended Height (m)')
axs[2].set_ylabel('Density')
axs[2].set_title('Height Distribution')

f.savefig(fig_write_path + "pwr_spec_CT.png", transparent=True)

# %% Compare with a smooth surface
# Above we are looking at a region of crag and tail surface, let's compare
# that with a smooth wind-packed surface (the other dominant surface type,
# at least from what I saw on leg 3)

# %% Create Mesh

# 40,000 points is too many to compute a pairwise semivariogram for. Instead
# we can exploit the FFT if we can work with gridded data. The first step in
# creating a gridded product is to create a mesh that then we will interpolate
# on a grid
# Lower left corner of box of data we want
x0 = -106.2199977990745 
y0 = -147.83927995978277 
# Dimensions of box (in m)
wx = 0.6 
wy = 0.6 
# Rotation angle of box (in degrees CCW from + x-axis)
yaw = -30 

# Alpha determines the maximum gap between points that we'll allow the
# triangulation to fill. Let's set it at 5 mm
alpha = 0.004

project.merged_points_to_mesh(wx*3, wy*3, alpha=alpha, x0=x0, y0=y0, wx=wx,
                              wy=wy, yaw=yaw)

# %% Display mesh

z_min = -2.4
z_max = -2.2

# Elevation filter copies z values to scalars for displaying by color
elevFilter = vtk.vtkSimpleElevationFilter()
elevFilter.SetInputData(project.mesh)


# This is a standard vtk visualization pipeline works for mesh as well.
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(elevFilter.GetOutputPort())
mapper.SetLookupTable(pydar.mplcmap_to_vtkLUT(z_min, z_max))
mapper.SetScalarRange(z_min, z_max)
actor = vtk.vtkActor()
actor.SetMapper(mapper)
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)

# Let's try overlaying on points to make sure we have the transform right
renderer.AddActor(project.scan_dict['ScanPos011'].actor)

renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)
iren.Initialize()
renderWindow.Render()
iren.Start()

# %% Now that we have a mesh we can create a gridded version

x0 = -106.2199977990745 + .04
y0 = -147.83927995978277 + .04
nx = 128 # 200
ny = 128 # 200
dx = .004 # 0.005
dy = .004 # 0.005
yaw = -30 

project.mesh_to_image(nx, ny, dx, dy, x0, y0, yaw=yaw)

# %% We can also display the image

project.display_image(z_min, z_max)

# %% fit slope
# This region is sloped (not an issue we typically deal with on complete scans)

# Get image as numpy array for easy manipulation
im2 = project.get_np_nan_image()

X, Y = np.meshgrid(dx*np.arange(nx), dy*np.arange(ny))

data_mask = np.logical_not(np.isnan(im2.ravel()))

A = np.hstack((np.ones((data_mask.sum(), 1)), 
                  X.ravel()[data_mask][:, np.newaxis],
                  Y.ravel()[data_mask][:, np.newaxis],
                  (X**2).ravel()[data_mask][:, np.newaxis],
                  (X*Y).ravel()[data_mask][:, np.newaxis],
                  (Y**2).ravel()[data_mask][:, np.newaxis],
                  (X**3).ravel()[data_mask][:, np.newaxis],
                  (Y*X**2).ravel()[data_mask][:, np.newaxis],
                  (X*Y**2).ravel()[data_mask][:, np.newaxis],
                  (Y**3).ravel()[data_mask][:, np.newaxis]))

coef, _, _, _ = lstsq(A, im2.ravel()[data_mask])

im2_slope = (coef[0] + coef[1]*X + coef[2]*Y + coef[3]*(X**2) + coef[4]*(X*Y) +
            coef[5]*(Y**2) + coef[6]*(X**3) + coef[7]*(Y*X**2) + coef[8]*
            (X*Y**2) + coef[9]*(Y**3))

im2_detrend = im2 - im2_slope

f, axs = plt.subplots(1, 3, figsize=(20, 8))

h0 = axs[0].imshow(im2, origin='lower', extent=[0, nx*dx, 0, ny*dy])
axs[0].axis('equal')
f.colorbar(h0, ax=axs[0], orientation='horizontal', shrink=0.8)
axs[0].set_title('Surface')
axs[0].set_xlabel('(m)')

h1 = axs[1].imshow(im2_slope, origin='lower', extent=[0, nx*dx, 0, ny*dy])
axs[1].axis('equal')
f.colorbar(h1, ax=axs[1], orientation='horizontal', shrink=0.8)
axs[1].set_title('Cubic Trend')
axs[1].set_xlabel('(m)')

h2 = axs[2].imshow(im2_detrend, origin='lower', extent=[0, nx*dx, 0, ny*dy])
axs[2].axis('equal')
f.colorbar(h2, ax=axs[2], orientation='horizontal', shrink=0.8)
axs[2].set_title('Detrended Surface')
axs[2].set_xlabel('(m)')

f.savefig(fig_write_path + "fit_slope_WP.png", transparent=True)

# %% look at power spectrum of signal and also show histogram of detrended
# surface

# zero fill, to use FFT
im2_detrend_0 = copy.deepcopy(im2_detrend)
im2_detrend_0[np.isnan(im2_detrend)] = 0
spec2 = np.fft.fft2(im2_detrend_0)

pwr_spec2 = np.abs(spec2)**2

dnu_x = 1/(nx*dx)
dnu_y = 1/(ny*dy)

f, axs = plt.subplots(1, 3, figsize=(20, 8))

h0 = axs[0].imshow(im2_detrend, origin='lower', extent=[0, nx*dx, 0, ny*dy])
axs[0].axis('equal')
f.colorbar(h0, ax=axs[0], orientation='horizontal', shrink=0.8, 
           label='Height (m)')
axs[0].set_title('Detrended Surface')
axs[0].set_xlabel('(m)')

h1 = axs[1].imshow(np.fft.fftshift(10*np.log10(pwr_spec2)), vmin=-40, vmax=20,
                   extent=[-dnu_x*nx/2, dnu_x*nx/2, -dnu_y*ny/2, dnu_y*ny/2])
axs[1].axis('equal')
axs[1].set_title('Power Spectrum')
axs[1].set_xlabel('Spatial Frequence (1/m)')
f.colorbar(h1, ax=axs[1], orientation='horizontal', shrink=0.8, label=
           'Power (dB)')

axs[2].hist(im2_detrend.ravel()[data_mask], bins=100, density=True)
axs[2].set_xlabel('Detrended Height (m)')
axs[2].set_ylabel('Density')
axs[2].set_title('Height Distribution')

f.savefig(fig_write_path + "pwr_spec_WP.png", transparent=True)

# %% Finally, let's make a comparison figure

z_max = .006
z_min = -.006

f = plt.figure(figsize=(20, 16), constrained_layout=False)
axs = np.array([[None, None], [None, None]])
axs[0, 0] = f.add_subplot(2, 2, 1)
axs[0, 1] = f.add_subplot(2, 2, 2, sharey=axs[0,0])
axs[1, 0] = f.add_subplot(2, 2, 3)
axs[1, 1] = f.add_subplot(2, 2, 4)

h0 = axs[0, 0].imshow(im_detrend, origin='lower', extent=[0, nx*dx, 0, ny*dy],
                      vmin=z_min, vmax=z_max)
f.colorbar(h0, ax=axs[0, 0], orientation='vertical', shrink=0.8, 
            label='Height (m)')

axs[0, 0].axis('equal')
axs[0, 0].set_title('Crag and Tail Detrended Surface')
axs[0, 0].set_xlabel('(m)')

h0 = axs[0, 1].imshow(im2_detrend, origin='lower', extent=[0, nx*dx, 0, ny*dy],
                      vmin=z_min, vmax=z_max)

f.colorbar(h0, ax=axs[0, 1], orientation='vertical', shrink=0.8, 
            label='Height (m)')
axs[0, 1].set_title('Windpacked Detrended Surface')
axs[0, 1].set_xlabel('(m)')
axs[0, 1].axis('equal')

# Plot histogram comparison
axs[1, 0].hist(im_detrend.ravel()[data_mask], bins=100, density=True, label=
               'Crag and Tail', alpha=0.5)
axs[1, 0].hist(im2_detrend.ravel()[data_mask], bins=100, density=True, label=
               'Windpacked', alpha=0.5)
axs[1, 0].set_xlabel('Detrended Height (m)')
axs[1, 0].set_ylabel('Density')
axs[1, 0].legend()

# Compare Power Spectra
axs[1, 1].plot(dnu_x*np.arange(nx//2), 10*np.log10(pwr_spec[0,:nx//2]), 'b-',
               label='Crag and Tail x direction')
axs[1, 1].plot(dnu_y*np.arange(ny//2), 10*np.log10(pwr_spec[:ny//2,0]), 'b:',
               label='Crag and Tail y direction')
axs[1, 1].plot(dnu_x*np.arange(nx//2), 10*np.log10(pwr_spec2[0,:nx//2]), 'r-',
               label='Windpacked x direction')
axs[1, 1].plot(dnu_y*np.arange(ny//2), 10*np.log10(pwr_spec2[:ny//2,0]), 'r:',
               label='Windpacked y direction')
axs[1, 1].set_ylim([-40, 20])
axs[1, 1].set_xlabel('Spatial Frequency (1/m)')
axs[1, 1].set_ylabel('Power (dB)')
axs[1, 1].legend()

f.savefig(fig_write_path + "compare_CT_WP.png", transparent=True)