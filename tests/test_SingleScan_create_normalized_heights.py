# -*- coding: utf-8 -*-
"""
test_SingleScan_create_normalized_heights.py

Test the functionality for creating normalized heights.

Created on Tue Oct 27 10:46:12 2020

@author: d34763s
"""

import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import sys
import platform
if platform.system()=='Windows':
    sys.path.append('C:/Users/d34763s/Desktop/DavidCS/PhD/code/pydar/')
else:
    sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% init

if platform.system()=='Windows':
    project_path = 'D:\\mosaic_lidar\\ROV\\'
else:
    project_path = '/media/thayer/Data/mosaic_lidar/ROV/'
project_name = 'mosaic_rov_250120.RiSCAN'
scan_name = 'ScanPos004'

ss = pydar.SingleScan(project_path, project_name, scan_name, las_fieldnames=
                      ['Points', 'PointId', 'Classification'], 
                      class_list='all', import_mode='read_scan')

ss.add_sop()
print(ss.transform_dict['sop'].GetPosition())
ss.apply_transforms(['sop'])
# Create z sigma
ss.create_z_sigma()

# %% Display to check

v_min = 0
v_max = 0.015

ss.create_reflectance_pipeline(v_min, v_max, field='z_sigma')

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderer.AddActor(ss.actor)

renderWindow.AddRenderer(renderer)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)

iren.Initialize()
renderWindow.Render()
iren.Start()

# %% We need the z values to create a histogram (in the future this step
#  will be part of the project class)

dsa_filter = dsa.WrapDataObject(ss.currentFilter.GetOutput())
z = dsa_filter.Points[:, 2].squeeze()

minh = np.min(z)
maxh = 0 # np.max(z) # Just set this to zero to make things more reasonable
nbins = int(1000*(maxh - minh))
pdf, bin_edges = np.histogram(z[z<0], # and clip here too
                          density=True, bins=nbins)
xdist = (bin_edges[:-1] + bin_edges[1:])/2
cdf = np.cumsum(pdf)/1000

plt.plot(xdist, cdf)

# %% now test method

ss.create_normalized_heights(xdist, cdf)

# plot histogram to check
plt.hist(ss.dsa_raw.PointData['norm_height'])

# That looks okay, the ship is clearly evident in the right side of dist

# %% look at norm_z_sigma

plt.hist(ss.dsa_raw.PointData['norm_z_sigma'], range=[0, 0.2], bins=100)

# a couple of outliers but that could just be very far away points.

# %% to be really certain let's display the result.
ss.create_reflectance_pipeline(-3, 3, field='norm_height')

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderer.AddActor(ss.actor)

renderWindow.AddRenderer(renderer)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)

iren.Initialize()
renderWindow.Render()
iren.Start()

# That works pretty much perfectly

# %% and display z_sigma

ss.create_reflectance_pipeline(0, 0.1, field='norm_z_sigma')

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderer.AddActor(ss.actor)

renderWindow.AddRenderer(renderer)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)

iren.Initialize()
renderWindow.Render()
iren.Start()


# %% Finally check that when we apply transforms it removes the norm_height
# array, this is desirable because we don't want norm_height to not correspond
# to the current z-values

ss.apply_transforms(['sop'])
print(ss.dsa_raw.PointData.keys())