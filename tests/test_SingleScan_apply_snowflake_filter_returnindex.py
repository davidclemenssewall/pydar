# -*- coding: utf-8 -*-
"""
test_SingleScan_apply_snowflake_filter_returnindex.py

Test our functionality for filtering snowflakes based upon their return index

Created on Mon Dec  7 16:48:49 2020

@author: d34763s
"""

import numpy as np
import vtk
import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\pydar\\')
import pydar

# %% Test init

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01_040120.RiSCAN'
scan_name = 'ScanPos006'

ss = pydar.SingleScan(project_path, project_name, scan_name, import_las=True)


# %% test filter

ss.clear_flag_filter()

ss.apply_snowflake_filter_returnindex()


# %% create filter pipelin

ss.create_filter_pipeline()

# %% Render actor

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