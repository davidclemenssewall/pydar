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
import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\lidar_processing\\')
import pydar

# %% init

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01_102519.RiSCAN'
scan_name = 'ScanPos003'

ss = pydar.SingleScan(project_path, project_name, scan_name)

ss.read_scan()

ss.add_sop()
print(ss.transform_dict['sop'].GetPosition())
ss.apply_transforms(['sop'])

# %% Display to check

z_min = -3
z_max = -1.5

ss.create_elevation_pipeline(z_min, z_max)

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

# %% to be really certain let's display the result.
# this is a bit more complicated but we can make it work

pdata = ss.currentFilter.GetOutput()
pdata.GetPointData().SetActiveScalars('norm_height')

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(pdata)
mapper.SetLookupTable(pydar.mplcmap_to_vtkLUT(-3, 3))
mapper.SetScalarRange(-3, 3)
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

# That works pretty much perfectly

# %% Finally check that when we apply transforms it removes the norm_height
# array, this is desirable because we don't want norm_height to not correspond
# to the current z-values

ss.apply_transforms(['sop'])
print(ss.dsa_raw.PointData.keys())