# -*- coding: utf-8 -*-
"""
test_SingleScan_get_point_boxes

Created on Fri Sep 18 18:58:24 2020

@author: d34763s
"""

import vtk
import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\lidar_processing\\')
import pydar

# %% init

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01_102519.RiSCAN'
scan_name = 'ScanPos003'

ss = pydar.SingleScan(project_path, project_name, scan_name)

ss.add_sop()
print(ss.transform_dict['sop'].GetPosition())
ss.apply_transforms(['sop'])

# %% test out get_point_boxes

boxes_dict = {(-190, -90, 173, 373, -5, 0) : None,
              (-90, 10, 173, 373, -5, 0) : None}

ss.get_point_boxes(boxes_dict)

# %% Display

z_min = -2.75
z_max = -1.25

elevFilter = vtk.vtkSimpleElevationFilter()
elevFilter.SetInputData(boxes_dict[(-90, 10, 173, 373, -5, 0)])
#elevFilter.SetInputConnection(ss.transformFilter.GetOutputPort())
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