#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_Project_transect_points.py

Created on Wed Apr 14 21:23:09 2021

@author: thayer
"""

import vtk
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# Load a singlescan
# Project path
project_path = '/media/thayer/Data/mosaic_lidar/ROV/'
project_name = 'mosaic_rov_040120.RiSCAN'

project = pydar.Project(project_path, project_name, import_mode=
                      'read_scan', las_fieldnames=['Points', 'PointId',
                                                   'Classification'],
                      class_list='all')

project.apply_transforms(['sop'])

project.display_project(-4.5, 2.5)

# Pick p0 over the rov tent and p1 over the ship to make things easy
x0 = 17.3
y0 = -10.4
x1 = 317
y1 = -135
d = 3

pdata = project.transect_points(x0, y0, x1, y1, d)

vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
vertexGlyphFilter.SetInputData(pdata)
vertexGlyphFilter.Update()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(vertexGlyphFilter.GetOutputPort())
mapper.SetScalarVisibility(0)

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(1,0,0)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)

renderWindow = vtk.vtkRenderWindow()
renderWindow.SetSize(2000, 1500)
renderWindow.AddRenderer(renderer)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)

style = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(style)
    
iren.Initialize()
renderWindow.Render()
iren.Start()

# Needed to get window to close on linux
renderWindow.Finalize()
iren.TerminateApp()