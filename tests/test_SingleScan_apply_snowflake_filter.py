# -*- coding: utf-8 -*-
"""
test_SingleScan_apply_snowflake_filter.py

Created on Mon Sep 21 18:28:57 2020

@author: d34763s
"""

import vtk
import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\lidar_processing\\')
import pydar
from math import sqrt

# %% Init

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01_102519.RiSCAN'
scan_name = 'ScanPos003'

ss = pydar.SingleScan(project_path, project_name, scan_name)

ss.add_sop()
print(ss.transform_dict['sop'].GetPosition())
ss.apply_transforms(['sop'])

# %% test snowflake_filter

shells = [(0, 5, 0, 0), # remove all points within 5 m of scanner
          (5, 25, .1*sqrt(2), 5),
          (25, 40, .25*sqrt(2), 5),
          (40, 60, .6*sqrt(2), 5),
          (60, 100, 1.4*sqrt(2), 5),
          (100, None, 1, 0)]

ss.apply_snowflake_filter(shells)

# %% Display

cellData = vtk.vtkUnsignedCharArray()
cellData.SetNumberOfComponents(3)
cellData.SetNumberOfTuples(ss.get_polydata().GetNumberOfPoints())
cellData.FillComponent(0, 0)
cellData.FillComponent(1, 255)
cellData.FillComponent(2, 0)
ss.get_polydata().GetPointData().SetScalars(cellData)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(ss.get_polydata())
mapper.SetScalarModeToUsePointData()

actor = vtk.vtkActor()
actor.SetMapper(mapper)

cutCellData = vtk.vtkUnsignedCharArray()
cutCellData.SetNumberOfComponents(3)
cutCellData.SetNumberOfTuples(ss.filteredPoints.GetOutput().GetNumberOfPoints())
cutCellData.FillComponent(0, 255)
cutCellData.FillComponent(1, 0)
cutCellData.FillComponent(2, 0)
ss.filteredPoints.GetOutput().GetPointData().SetScalars(cutCellData)

cutMapper = vtk.vtkPolyDataMapper()
cutMapper.SetInputData(ss.filteredPoints.GetOutput())
cutMapper.SetScalarModeToUsePointData()

cutActor = vtk.vtkActor()
cutActor.SetMapper(cutMapper)

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderer.AddActor(actor)
renderer.AddActor(cutActor)

renderWindow.AddRenderer(renderer)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)

iren.Initialize()
renderWindow.Render()
iren.Start()