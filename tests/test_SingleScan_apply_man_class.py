#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_singlescan_apply_man_class.py

Created on Fri Jun 11 08:41:08 2021

@author: thayer
"""

import vtk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% location

project_path = "/media/thayer/Data/mosaic_lidar/ROV"
project_name = "mosaic_rov_250120.RiSCAN"
scan_name = 'ScanPos001'

ss = pydar.SingleScan(project_path, project_name, scan_name, 
                      import_mode='read_scan', las_fieldnames=['Points',
                        'PointId', 'Classification'], class_list='all')

# %% Display

ss.create_filter_pipeline()

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

# %% load man class

ss.load_man_class()

ss.apply_man_class()

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

