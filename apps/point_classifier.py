# -*- coding: utf-8 -*-
"""

point_classifier.py

An application for manually classifying points in mosaic lidar scans to use
as training data for a classifier.

Created on Thu Jan  7 10:48:45 2021

@author: d34763s
"""

import sys
import vtk
import pandas as pd
import math
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from PyQt5 import QtCore, QtGui
from PyQt5 import Qt
from collections import namedtuple
import re
import os
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class MainWindow(Qt.QMainWindow):
    
    def __init__(self, parent=None):
        Qt.QMainWindow.__init__(self, parent)
        
        # Useful variables:
        # Dictionary mapping classification text to numbers and vice versa
        self.class_dict = {'Never Classified' : 0,
                           'Unclassified' : 1,
                           'Ground' : 2, 
                           'Building' : 6,
                           'Low Point (Noise)' : 7, 
                           'Snowflake' : 65, 
                           'Pole' : 67, 
                           'Human' : 68,
                           'Snowmobile' : 69,
                           'Road' : 70,
                           'Flag' : 71,
                           'Wire' : 72
                           }
              
        # Create the main layout
        self.frame = Qt.QFrame()
        main_layout = Qt.QHBoxLayout()
        
        # Create the visualization layout, will contain the renderwindow
        # and a toolbar with options
        vis_layout = Qt.QVBoxLayout()
        
        # Create the vis_tools_layout to sit beneath the renderwindow
        vis_tools_layout = Qt.QHBoxLayout()
        
        # Populate the vis_tools_layout
        self.field_selector = Qt.QComboBox()
        self.field_selector.setEnabled(0)
        self.field_selector.setSizeAdjustPolicy(0)
        self.v_min = Qt.QLineEdit('-5.0')
        self.v_min.setValidator(Qt.QDoubleValidator())
        self.v_min.setEnabled(0)
        self.v_max = Qt.QLineEdit('-1.0')
        self.v_max.setValidator(Qt.QDoubleValidator())
        self.v_max.setEnabled(0)
        self.near_label = Qt.QLineEdit('0.1')
        self.near_label.setValidator(Qt.QDoubleValidator())
        self.far_label = Qt.QLineEdit('1000.0')
        self.far_label.setValidator(Qt.QDoubleValidator())
        look_down_button = Qt.QPushButton('Look Down')
        self.show_class_checkbox = Qt.QCheckBox('Highlight Classified')
        self.show_class_checkbox.setChecked(0)
        self.show_class_checkbox.setEnabled(0)
        vis_tools_layout.addWidget(self.field_selector)
        vis_tools_layout.addWidget(self.v_min)
        vis_tools_layout.addWidget(self.v_max)
        vis_tools_layout.addWidget(self.near_label)
        vis_tools_layout.addWidget(self.far_label)
        vis_tools_layout.addWidget(look_down_button)
        vis_tools_layout.addWidget(self.show_class_checkbox)
        
        # Populate the vis_layout
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        
        vis_layout.addWidget(self.vtkWidget)
        vis_layout.addLayout(vis_tools_layout)
        
        # Create the Options layout, which will contain tools to select files
        # classify points, etc
        opt_layout = Qt.QVBoxLayout()
        
        # Populate the opt_layout
        
        # Scan Area button
        self.sel_scan_area_button = Qt.QPushButton("Select Scan Area")
        # Create the file dialog that we'll use
        self.proj_dialog = Qt.QFileDialog(self)
        self.proj_dialog.setFileMode(4) # set file mode to pick directories
        opt_layout.addWidget(self.sel_scan_area_button)
        
        # ComboBox containing available projects
        self.proj_combobox = Qt.QComboBox()
        self.proj_combobox.setEnabled(0)
        self.proj_combobox.setSizeAdjustPolicy(0)
        opt_layout.addWidget(self.proj_combobox)
        
        # Label
        read_lbl = Qt.QLabel("How to Read Scans:")
        opt_layout.addWidget(read_lbl)
        
        # ComboBox to indicate how to read scans
        self.read_scan_box = Qt.QComboBox()
        self.read_scan_box.addItems(['Read Saved', 'Read LAS', 
                                     'Read PolyData'])
        self.read_scan_box.setCurrentIndex(0)
        opt_layout.addWidget(self.read_scan_box)
        
        # Button to prompt us to select a project
        self.sel_proj_button = Qt.QPushButton("Select Project")
        self.sel_proj_button.setEnabled(0)
        opt_layout.addWidget(self.sel_proj_button)
        
        # Create Layout for checkboxes indicating which scans to display
        self.scan_layout = Qt.QVBoxLayout()
        scan_group_box = Qt.QGroupBox()
        self.scan_button_group = Qt.QButtonGroup()
        self.scan_button_group.setExclusive(0)
        # Dictionary object containing checkboxes keyed on 'scan_name's
        self.scan_check_dict = {}
        scan_group_box.setLayout(self.scan_layout)
        opt_layout.addWidget(scan_group_box)
        
        # Classification or labelling boxes
        clear_button = Qt.QPushButton("Clear Selected")
        opt_layout.addWidget(clear_button)
        self.class_combobox = Qt.QComboBox()
        self.class_combobox.addItems(['Ground', 'Building', 'Low Point (Noise)',
                                     'Snowflake', 'Pole', 'Human', 
                                     'Snowmobile', 'Road', 'Flag', 'Wire'])
        self.class_combobox.setCurrentText('Ground')
        opt_layout.addWidget(self.class_combobox)
        self.class_button = Qt.QPushButton("Set Class")
        self.class_button.setEnabled(0)
        opt_layout.addWidget(self.class_button)
        
        # Which fields to use for classifier
        feature_list = ['Linearity', 'Planarity', 'Scattering', 'Verticality',
                        'Density', 'HeightAboveGround', 'dist', 
                        'HorizontalClosestPoint', 'VerticalClosestPoint']
        self.feature_check_dict = {}
        feature_group_box = Qt.QGroupBox()
        feature_layout = Qt.QVBoxLayout()
        for feature in feature_list:
            self.feature_check_dict[feature] = Qt.QCheckBox(feature)
            self.feature_check_dict[feature].setChecked(0)
            feature_layout.addWidget(self.feature_check_dict[feature])
        feature_group_box.setLayout(feature_layout)
        opt_layout.addWidget(feature_group_box)
        
        # Train Classifier and Apply Classifier buttons
        self.train_combobox = Qt.QComboBox()
        self.train_combobox.addItems(['This Scan', 'Scan Area', 'All'])
        opt_layout.addWidget(self.train_combobox)
        train_button = Qt.QPushButton("Train Classifier")
        opt_layout.addWidget(train_button)
        apply_button = Qt.QPushButton("Classify")
        opt_layout.addWidget(apply_button)
        
        # Add checkboxes for which classes to render
        class_layout = Qt.QVBoxLayout()
        self.class_group_box = Qt.QGroupBox()
        self.class_button_group = Qt.QButtonGroup()
        self.class_button_group.setExclusive(0)
        self.class_check_dict = {}
        for category in self.class_dict:
            self.class_check_dict[category] = Qt.QCheckBox(category)
            self.class_check_dict[category].setChecked(0)
            self.class_button_group.addButton(self.class_check_dict[category])
            class_layout.addWidget(self.class_check_dict[category])
        self.class_group_box.setLayout(class_layout)
        self.class_group_box.setEnabled(0)
        #opt_layout.addWidget(self.class_group_box)
        
        # Add write scans button
        write_button = Qt.QPushButton("Write Scans")
        opt_layout.addWidget(write_button)
        
        # Populate the main layout
        main_layout.addLayout(vis_layout)
        main_layout.addLayout(opt_layout)
        
        # Set layout for the frame and set central widget
        self.frame.setLayout(main_layout)
        self.setCentralWidget(self.frame)
        
        # Signals and slots
        self.sel_scan_area_button.clicked.connect(
            self.on_sel_scan_area_button_click)
        self.sel_proj_button.clicked.connect(self.on_sel_proj_button_click)
        self.proj_dialog.fileSelected.connect(self.on_scan_area_selected)
        self.scan_button_group.buttonToggled.connect(
            self.on_scan_checkbox_changed)
        clear_button.clicked.connect(self.on_clear_button_click)
        self.class_button.clicked.connect(self.on_class_button_click)
        train_button.clicked.connect(self.on_train_button_click)
        apply_button.clicked.connect(self.on_apply_button_click)
        write_button.clicked.connect(self.on_write_button_click)
        self.field_selector.currentTextChanged.connect(
            self.on_field_selector_changed)
        self.v_min.editingFinished.connect(self.on_v_edit)
        self.v_max.editingFinished.connect(self.on_v_edit)
        self.near_label.editingFinished.connect(self.on_clip_changed)
        self.far_label.editingFinished.connect(self.on_clip_changed)
        look_down_button.clicked.connect(self.look_down)
        self.show_class_checkbox.toggled.connect(self.on_show_class_toggled)
        
        self.show()
        
        # VTK setup
        # Dicts to hold visualization pipeline objects
        self.elev_filt_dict = {}
        self.class_filt_dict = {}
        self.mapper_dict = {}
        self.actor_dict = {}
        
        # Selected dict, to hold points until we classify them
        self.selected_poly_dict = {}
        self.selected_append = vtk.vtkAppendPolyData()
        pdata = vtk.vtkPolyData()
        self.selected_append.AddInputData(pdata)
        self.selected_mapper = vtk.vtkPolyDataMapper()
        self.selected_mapper.SetInputConnection(
            self.selected_append.GetOutputPort())
        self.selected_mapper.ScalarVisibilityOff()
        self.selected_actor = vtk.vtkActor()
        self.selected_actor.SetMapper(self.selected_mapper)
        self.selected_actor.GetProperty().SetColor(0, 1, 0)
        self.selected_actor.GetProperty().SetPointSize(2.0)
        
        # Classified dict, to show manually classified points
        self.man_class_dict = {}
        self.man_class_append = vtk.vtkAppendPolyData()
        pdata = vtk.vtkPolyData()
        self.man_class_append.AddInputData(pdata)
        self.man_class_mapper = vtk.vtkPolyDataMapper()
        self.man_class_mapper.SetInputConnection(
            self.man_class_append.GetOutputPort())
        self.man_class_mapper.ScalarVisibilityOff()
        self.man_class_actor = vtk.vtkActor()
        self.man_class_actor.SetMapper(self.man_class_mapper)
        self.man_class_actor.GetProperty().SetColor(1, 1, 0)
        self.man_class_actor.GetProperty().SetPointSize(1.9)
        
        
        # Renderer and interactor
        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.selected_actor)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtkWidget.GetRenderWindow().AddObserver("ModifiedEvent", 
                                                     self.
                                                     on_modified_renderwindow)
        style = vtk.vtkInteractorStyleRubberBandPick()
        areaPicker = vtk.vtkAreaPicker()
        areaPicker.AddObserver("EndPickEvent", self.on_end_pick)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.SetPicker(areaPicker)
        self.iren.Initialize()
        self.iren.SetInteractorStyle(style)
        self.iren.Start()
    
    def on_sel_scan_area_button_click(self, s):
        """
        Open file dialog to select scan area directory

        Parameters
        ----------
        s : int
            Button status. Not used.

        Returns
        -------
        None.

        """
        
        self.proj_dialog.exec_()
        
    def on_scan_area_selected(self, dir_str):
        """
        Load the selected scan area and enable project selection

        Parameters
        ----------
        dir_str : str
            Path that the user selected.

        Returns
        -------
        None.

        """
        
        # Parse project path
        dir_list = str(dir_str).split('/')
        scan_area_name = dir_list[-1]
        project_path = dir_str
        #project_path = dir_str + '/'
        #project_path = project_path.replace('/', '\\')
        
        
        # Registration object
        Registration = namedtuple('Registration', ['project_name_0', 'project_name_1',
                                           'reflector_list', 'mode', 
                                           'yaw_angle'], 
                          defaults=[None, None, None])
        
        # Depending on the scan_area_name load and register project
        if scan_area_name=='Snow2':
            project_names = ['mosaic_02_110619.RiSCAN',
                             'mosaic_02_111319.RiSCAN']
            
            registration_list = [Registration('mosaic_02_111319.RiSCAN', 
                                              'mosaic_02_111319.RiSCAN'),
                                 Registration('mosaic_02_111319.RiSCAN',
                                              'mosaic_02_110619.RiSCAN',
                                              ['r17', 'r19', 'r20', 'r37', 
                                               'r38'],
                                              'LS')
                                 ]
        elif scan_area_name=='Snow1':
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
                             'mosaic_01_280220.RiSCAN',
                             'mosaic_01_220320.RiSCAN',
                             'mosaic_01_080420.RiSCAN',
                             'mosaic_01_080420b.RiSCAN',
                             'mosaic_01_250420.RiSCAN.RiSCAN',
                             'mosaic_01_260420.RiSCAN',
                             'mosaic_01_030520.RiSCAN']
            
            registration_list = [Registration('mosaic_01_102019.RiSCAN', 
                                              'mosaic_01_102019.RiSCAN'),
                                 Registration('mosaic_01_102019.RiSCAN',
                                              'mosaic_01_101819.RiSCAN',
                                              ['r04', 'r05', 'r07', 'r09'],
                                              'Yaw'),
                                 Registration('mosaic_01_102019.RiSCAN', 
                                              'mosaic_01_102519.RiSCAN',
                                              ['r01', 'r02', 'r03', 'r09', 
                                               'r08'],
                                              'LS'),
                                 Registration('mosaic_01_102519.RiSCAN',
                                              'mosaic_01_110119.RiSCAN',
                                              ['r01', 'r03', 'r04', 'r05', 
                                               'r06', 'r07'],
                                              'LS'),
                                 Registration('mosaic_01_110119.RiSCAN',
                                              'mosaic_01_111519.RiSCAN',
                                              ['r02', 'r03', 'r04'],
                                              'Yaw'),
                                 Registration('mosaic_01_111519.RiSCAN',
                                              'mosaic_01_110819.RiSCAN',
                                              ['r02', 'r05', 'r06', 'r07', 
                                               'r10'],
                                              'LS'),
                                 Registration('mosaic_01_111519.RiSCAN',
                                              'mosaic_01b_061219.RiSCAN.'+
                                              'RiSCAN.RiSCAN',
                                              ['r01', 'r11'],
                                              'Yaw'),
                                 Registration('mosaic_01b_061219.RiSCAN.'+
                                              'RiSCAN.RiSCAN',
                                              'mosaic_01_122719.RiSCAN',
                                              ['r02', 'r11'],
                                              'Yaw'),
                                 Registration('mosaic_01_122719.RiSCAN',
                                              'mosaic_01_040120.RiSCAN',
                                              ['r01', 'r13', 'r14', 'r15'],
                                              'Yaw'),
                                 Registration('mosaic_01_040120.RiSCAN',
                                              'mosaic_01_180120.RiSCAN',
                                              ['r03', 'r09', 'r10', 'r11', 
                                               'r24'],
                                              'LS'),
                                 Registration('mosaic_01_180120.RiSCAN',
                                              'mosaic_01_290120.RiSCAN',
                                              ['r01', 'r02', 'r03', 'r09', 
                                               'r10', 
                                               'r12', 'r13', 'r14'],
                                              'LS'),
                                 Registration('mosaic_01_290120.RiSCAN',
                                              'mosaic_01_060220.RiSCAN',
                                              ['r01', 'r03', 'r09', 'r12', 
                                               'r14', 'r23'],
                                              'LS'),
                                 Registration('mosaic_01_060220.RiSCAN',
                                              'mosaic_01_150220.RiSCAN.RiSCAN',
                                              ['r03', 'r09', 'r23'],
                                              'Yaw'),
                                 Registration('mosaic_01_150220.RiSCAN.RiSCAN',
                                              'mosaic_01_280220.RiSCAN',
                                              ['r10', 'r11', 'r24', 'r12'],
                                              'LS'),
                                 Registration('mosaic_01_280220.RiSCAN',
                                              'mosaic_01_220320.RiSCAN',
                                              ['r10', 'r11', 'r24'],
                                              'Yaw'),
                                 Registration('mosaic_01_220320.RiSCAN',
                                              'mosaic_01_080420.RiSCAN',
                                              ['r10', 'r11'],
                                              'Yaw'),
                                 Registration('mosaic_01_080420.RiSCAN',
                                              'mosaic_01_080420b.RiSCAN',
                                              ['r24', 'r26'],
                                              'Yaw'),
                                 Registration('mosaic_01_080420b.RiSCAN',
                                              'mosaic_01_250420.RiSCAN.RiSCAN',
                                              ['r24', 'r27'],
                                              'Yaw'),
                                 Registration('mosaic_01_250420.RiSCAN.RiSCAN',
                                              'mosaic_01_260420.RiSCAN',
                                              ['r24', 'r27'],
                                              'Yaw'),
                                 Registration('mosaic_01_260420.RiSCAN',
                                              'mosaic_01_030520.RiSCAN',
                                              ['r25'],
                                              'Trans',
                                              math.pi*9/8)
                                 ]
        elif scan_area_name=='ROV':
            project_names = ['mosaic_rov_040120.RiSCAN',
                             'mosaic_rov_110120.RiSCAN',
                             'mosaic_rov_190120.RiSCAN',
                             'mosaic_rov_250120.RiSCAN',
                             'mosaic_rov_040220.RiSCAN',
                             'mosaic_rov_220220.RiSCAN.RiSCAN',
                             'mosaic_02_040420.RiSCAN',
                             'mosaic_02_110420_rov.RiSCAN',
                             'mosaic_rov_170420.RiSCAN',
                             'mosaic_rov_220420.RiSCAN',
                             'mosaic_rov_290420.RiSCAN',
                             'mosaic_rov_02_090520.RiSCAN']
            
            registration_list = [Registration('mosaic_rov_250120.RiSCAN', 
                                    'mosaic_rov_250120.RiSCAN'),
                        Registration('mosaic_rov_250120.RiSCAN',
                                     'mosaic_rov_190120.RiSCAN',
                                     ['r28', 'r29', 'r30', 'r32', 'r34', 'r35',
                                      'r22'],
                                     'LS'),
                        Registration('mosaic_rov_190120.RiSCAN',
                                     'mosaic_rov_110120.RiSCAN',
                                     ['r05', 'r28', 'r29', 'r31', 'r32', 'r33',
                                      'r34'],
                                     'LS'),
                         Registration('mosaic_rov_190120.RiSCAN',
                                      'mosaic_rov_040120.RiSCAN',
                                      ['r28', 'r29', 'r30', 'r31', 'r32', 'r33'],
                                      'LS'),
                        Registration('mosaic_rov_250120.RiSCAN',
                                    'mosaic_rov_040220.RiSCAN',
                                    ['r28', 'r29', 'r30', 'r31', 'r32', 'r34', 
                                     'r35', 'r36'],
                                    'LS'),
                        Registration('mosaic_rov_040220.RiSCAN',
                                      'mosaic_rov_220220.RiSCAN.RiSCAN',
                                      ['r28', 'r31', 'r32', 'r34'],
                                      'Yaw'),
                        Registration('mosaic_rov_220220.RiSCAN.RiSCAN',
                                      'mosaic_02_040420.RiSCAN',
                                      ['r29', 'r30', 'r33', 'r36'],
                                      'LS'),
                         Registration('mosaic_02_040420.RiSCAN',
                                      'mosaic_02_110420_rov.RiSCAN',
                                      ['r29', 'r30', 'r33', 'r35', 'r37'],
                                      'LS'),
                         Registration('mosaic_02_040420.RiSCAN',
                                      'mosaic_rov_170420.RiSCAN',
                                      ['r29', 'r30', 'r35', 'r36', 'r37'],
                                      'LS'),
                         Registration('mosaic_rov_170420.RiSCAN',
                                      'mosaic_rov_220420.RiSCAN',
                                      ['r29', 'r30', 'r35', 'r36', 'r37'],
                                      'LS'),
                         Registration('mosaic_rov_220420.RiSCAN',
                                      'mosaic_rov_290420.RiSCAN',
                                      ['r30', 'r33', 'r35', 'r36'],
                                      'LS'),
                         Registration('mosaic_rov_290420.RiSCAN',
                                      'mosaic_rov_02_090520.RiSCAN',
                                      ['r30', 'r33', 'r35', 'r36'],
                                      'LS')
                       ]
        else:
            raise ValueError("You have selected a nonexistant scan area."
                             " please start again")
        
        # Init scan_area
        self.scan_area = pydar.ScanArea(project_path, project_names,
                                        registration_list, load_scans=False,
                                        read_scans=False, import_las=False)
        
        # Update proj_combobox with available scans
        self.proj_combobox.addItems(project_names)
        self.proj_combobox.setEnabled(1)
        
        # Enable sel_proj_button
        self.sel_proj_button.setEnabled(1)
        
        # Disable further scan area changes
        self.sel_scan_area_button.setEnabled(0)
    
    def on_sel_proj_button_click(self, s):
        
        # Clear selections if there are any
        self.clear_selection()
        
        # Clear the man_class_dict
        # uncheck the show class checkbox
        if self.show_class_checkbox.isChecked():
            self.show_class_checkbox.setChecked(0)
        for scan_name in self.man_class_dict:
            self.man_class_append.RemoveInputData(
                self.man_class_dict[scan_name])
        self.man_class_dict.clear()

        # Once we have selected a project we want to load that project into
        # memory
        # Get Scan loading parameters from read_scan_box
        if self.read_scan_box.currentIndex()==0:
            read_scans=True
            import_las=False
        elif self.read_scan_box.currentIndex()==1:
            read_scans=False
            import_las=True
        else:
            read_scans=False
            import_las=False
            
        # Parse project path and name and load project
        project_name = self.proj_combobox.currentText()
        self.scan_area.add_project(project_name, read_scans=read_scans,
                                   import_las=import_las)
        self.scan_area.register_all()
        self.project = self.scan_area.project_dict[project_name]
        
        # Load the man_class table
        for scan_name in self.project.scan_dict:
            self.project.scan_dict[scan_name].load_man_class()
        
        # Enable v_min and v_max
        self.v_min.setEnabled(1)
        self.v_max.setEnabled(1)
        
        # Clear the elev_filters mappers and actors
        for scan_name in self.actor_dict:
            self.renderer.RemoveActor(self.actor_dict[scan_name])
        self.vtkWidget.GetRenderWindow().Render()
        self.elev_filt_dict.clear()
        self.class_filt_dict.clear()
        self.mapper_dict.clear()
        self.actor_dict.clear()
        self.selected_poly_dict.clear()
        
        for scan_name in self.project.scan_dict:
            # Create an elevation filter linked to transformFilter
            self.elev_filt_dict[scan_name] = vtk.vtkSimpleElevationFilter()
            self.elev_filt_dict[scan_name].SetInputConnection(
                self.project.scan_dict[scan_name].transformFilter.
                GetOutputPort())
            # We vtkSimpleElevationFilter will overwrite the active scalars
            # so let's set active scalars to a dummy array
            (self.elev_filt_dict[scan_name].GetInput().GetPointData().
             SetActiveScalars("elev"))
            self.elev_filt_dict[scan_name].GetInput().Modified()
            self.elev_filt_dict[scan_name].Update()
            
            # Now create a threshold filters for each of our classes and 
            # set the outputs of these to go to the mapper and actor. The
            # keys for these dictionaries will be (scan_name, category)
            
            #for category in self.class_dict:
                
            
            # Create mapper and set scalar range
            self.mapper_dict[scan_name] = vtk.vtkPolyDataMapper()
            self.mapper_dict[scan_name].SetInputConnection(
                self.elev_filt_dict[scan_name].GetOutputPort())
            self.mapper_dict[scan_name].ScalarVisibilityOn()
            self.mapper_dict[scan_name].SetLookupTable(
                pydar.mplcmap_to_vtkLUT(float(self.v_min.text()),
                                        float(self.v_max.text())))
            self.mapper_dict[scan_name].SetScalarRange(
                float(self.v_min.text()), float(self.v_max.text()))
            
            # Create actor and link to mapper
            self.actor_dict[scan_name] = vtk.vtkActor()
            self.actor_dict[scan_name].SetMapper(self.mapper_dict[scan_name])
            
            # Create appendPolyData, mappers and actors in selection dicts
            self.selected_poly_dict[scan_name] = vtk.vtkAppendPolyData()
            pdata = vtk.vtkPolyData()
            self.selected_poly_dict[scan_name].AddInputData(pdata)
            self.selected_append.AddInputConnection(
                self.selected_poly_dict[scan_name].GetOutputPort())

        
        # Update the fields available in field_selector
        self.field_selector.clear()
        self.field_selector.addItem('Elevation')
        for name in self.project.scan_dict['ScanPos001'].dsa_raw.PointData.keys():
            self.field_selector.addItem(name)
        self.field_selector.setEnabled(1)
        self.field_selector.setCurrentText('Elevation')
        
        # Update scan checkboxes this is the step that will actually lead to
        # scans being rendered.
        self.update_scan_checks(self.project.scan_dict.keys())
        
        # Update man_class_dict
        self.update_man_class_dict()
        self.show_class_checkbox.setEnabled(1)
    
    def on_scan_checkbox_changed(self, button, checked):
        """
        When a scan checkbox is changed, add or remove actor from renderer
        accordingly.

        Parameters
        ----------
        i : int
            Describes state of checkbox, unused.

        Returns
        -------
        None.

        """
        
        if checked:
            self.renderer.AddActor(self.actor_dict[button.text()])
            self.vtkWidget.GetRenderWindow().Render()
        else:
            self.renderer.RemoveActor(self.actor_dict[button.text()])
            self.vtkWidget.GetRenderWindow().Render()
    
    def on_clear_button_click(self, s):
        """
        Clear selected points when user clicks the clear button.

        Parameters
        ----------
        s : int
            Button status, not used.

        Returns
        -------
        None.

        """
        
        self.clear_selection()
        self.class_button.setEnabled(0)
    
    def on_class_button_click(self, s):
        """
        Classify selected points when the user clicks the classify button.

        Parameters
        ----------
        s : int
            Button status, not used.

        Returns
        -------
        None.

        """
        
        # Send Picked Points to SingleScan to be added to classified points
        for scan_name in self.project.scan_dict:
            if (self.selected_poly_dict[scan_name].GetOutput()
                .GetNumberOfPoints()>0):
                self.project.scan_dict[scan_name].update_man_class(
                    self.selected_poly_dict[scan_name].GetOutput(),
                    self.class_dict[self.class_combobox.currentText()])
        
        self.clear_selection()
        self.class_button.setEnabled(0)
        # Update man_class_dict
        self.update_man_class_dict()
    
    def on_train_button_click(self, s):
        """
        Train a classifier using all of the manually classified points.

        Parameters
        ----------
        s : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # First we need to pull all of the manually classified points
        if self.train_combobox.currentText()=='This Scan':
            df_list = []
            for scan_name in self.project.scan_dict:
                df_list.append(self.project.scan_dict[scan_name].man_class)
            
            df = pd.concat(df_list, ignore_index=True)
        elif self.train_combobox.currentText()=='Scan Area':
            project_tuples = []
            with os.scandir(self.scan_area.project_path) as it:
                for entry in it:
                    if entry.is_dir():
                        if re.search('.RiSCAN$', entry.name):
                            project_tuples.append((self.scan_area.project_path
                                                   , entry.name))
            df = pydar.get_man_class(project_tuples)
        elif self.train_combobox.currentText()=='All':
            project_tuples = []
            # Get the path to the scan area
            scan_area_path = os.path.split(self.scan_area.project_path)[0]
            #scan_area_path = (self.scan_area.project_path.rsplit(sep='\\',
            #                                                    maxsplit=2)[0]
            #                  + '\\')
            scan_areas = ['Snow1', 'Snow2', 'ROV']
            for scan_area in scan_areas:
                project_path = os.path.join(scan_area_path, scan_area)
                with os.scandir(project_path) as it:
                    for entry in it:
                        if entry.is_dir():
                            if re.search('.RiSCAN$', entry.name):
                                project_tuples.append((project_path
                                                       , entry.name))
            df = pydar.get_man_class(project_tuples)
        
        # Get feature list to train on
        feature_list = []
        for feature in self.feature_check_dict:
            if self.feature_check_dict[feature].isChecked():
                feature_list.append(feature)
        
        # Init and train classifier, just use default settings for now
        # and hardcode feature list
        self.classifier = pydar.Classifier(df_labeled=df)
        #self.classifier.init_randomforest()
        self.classifier.init_histgradboost()
        self.classifier.train_classifier(feature_list, reduce_ground=False,
                                         surface_only=True)
    
    def on_apply_button_click(self, s):
        """
        Apply the classifier to all scans in the project.

        Parameters
        ----------
        s : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # We just care about displaying here, so we want to apply to the
        # elevation filter output
        for scan_name in self.elev_filt_dict:
            # create dist field
            self.project.scan_dict[scan_name].add_dist()
            pdata = self.project.scan_dict[scan_name].polydata_raw
            self.classifier.classify_pdata(pdata)
            self.project.scan_dict[scan_name].transformFilter.Update()
            self.project.scan_dict[scan_name].currentFilter.Update()
            (self.elev_filt_dict[scan_name].GetInput().GetPointData().
             SetActiveScalars("elev"))
            self.elev_filt_dict[scan_name].Update()
        # !!! Need to add elevation if we want it as feature
        
        # Update renderwindow
        if self.field_selector.currentText()=='Classification':
            self.vtkWidget.GetRenderWindow().Render()
        else:
            self.field_selector.setCurrentText('Classification')
    
    def on_write_button_click(self, s):
        """
        Write scan to vtp files.

        Parameters
        ----------
        s : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.project.write_scans()
    
    def on_field_selector_changed(self, text):
        """
        When the field selector is changed, change the rendering to display
        the new field.

        Parameters
        ----------
        text : string
            The new value of the field selector.

        Returns
        -------
        None.

        """
        
        if text=='Classification':
            #raise NotImplementedError('Cannot display classification yet')
            # Create LookupTable
            colors={0 : (153/255, 153/255, 153/255, 1),
                    1 : (153/255, 153/255, 153/255, 1),
                    2 : (55/255, 126/255, 184/255, 1),
                    6 : (166/255, 86/255, 40/255, 1),
                    7 : (255/255, 255/255, 51/255, 1),
                    64: (255/255, 255/255, 51/255, 1),
                    65: (255/255, 255/255, 51/255, 1),
                    66: (255/255, 255/255, 51/255, 1),
                    67: (228/255, 26/255, 28/255, 1),
                    68: (77/255, 175/255, 74/255, 1),
                    69: (247/255, 129/255, 191/255, 1),
                    70: (152/255, 78/255, 163/255, 1),
                    71: (255/255, 127/255, 0/255, 1),
                    72: (253/255, 191/255, 111/255, 1)}
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(max(colors) + 1)
            lut.SetTableRange(0, max(colors))
            for key in colors:
                lut.SetTableValue(key, colors[key])
            lut.Build()
            
            # Apply to each scan
            for scan_name in self.elev_filt_dict:
                (self.elev_filt_dict[scan_name].GetOutput().GetPointData().
                  SetActiveScalars(text))
                self.elev_filt_dict[scan_name].GetOutput().Modified()
                self.mapper_dict[scan_name].SetLookupTable(lut)
                self.mapper_dict[scan_name].SetScalarRange(min(colors), 
                                                            max(colors))
                self.mapper_dict[scan_name].SetScalarVisibility(1)
                self.mapper_dict[scan_name].SetColorModeToMapScalars()
            
            # Disable v_min and v_max
            self.v_min.setEnabled(0)
            self.v_max.setEnabled(0)
        else:
            for scan_name in self.elev_filt_dict:
                (self.elev_filt_dict[scan_name].GetOutput().GetPointData().
                 SetActiveScalars(text))
                self.elev_filt_dict[scan_name].GetOutput().Modified()
                self.mapper_dict[scan_name].ScalarVisibilityOn()
                self.mapper_dict[scan_name].SetLookupTable(
                    pydar.mplcmap_to_vtkLUT(float(self.v_min.text()),
                                            float(self.v_max.text())))
                self.mapper_dict[scan_name].SetScalarRange(
                    float(self.v_min.text()), float(self.v_max.text()))
            
            # Enable v_min and v_max
            self.v_min.setEnabled(1)
            self.v_max.setEnabled(1)
        
        self.vtkWidget.GetRenderWindow().Render()
    
    def on_v_edit(self):
        """
        When one of the value boxes is edited update the color limits.

        Returns
        -------
        None.

        """
        if float(self.v_min.text())<float(self.v_max.text()):
            for scan_name in self.mapper_dict:
                self.mapper_dict[scan_name].SetLookupTable(
                    pydar.mplcmap_to_vtkLUT(float(self.v_min.text()),
                                            float(self.v_max.text())))
                self.mapper_dict[scan_name].SetScalarRange(
                    float(self.v_min.text()), float(self.v_max.text()))
            self.vtkWidget.GetRenderWindow().Render()
    
    def on_clip_changed(self):
        """
        

        Returns
        -------
        None.

        """
        
        self.renderer.GetActiveCamera().SetClippingRange(
            float(self.near_label.text()), float(self.far_label.text()))
        self.vtkWidget.GetRenderWindow().Render()
    
    def look_down(self):
        """
        Set camera view to be looking straight down.

        Returns
        -------
        None.

        """
        
        camera_pos = self.renderer.GetActiveCamera().GetPosition()
        self.renderer.GetActiveCamera().SetFocalPoint(camera_pos[0],
                                                      camera_pos[1],
                                                      -5)
        self.vtkWidget.GetRenderWindow().Render()
    
    def on_show_class_toggled(self, checked):
        """
        If we checked the checkbutton add the actor to the renderer

        Parameters
        ----------
        checked : bool
            Whether the show class checkbox was toggled.

        Returns
        -------
        None.

        """
        
        if checked:
            self.renderer.AddActor(self.man_class_actor)
            self.vtkWidget.GetRenderWindow().Render()
        else:
            self.renderer.RemoveActor(self.man_class_actor)
            self.vtkWidget.GetRenderWindow().Render()
    
    def update_scan_checks(self, scan_name_list):
        """
        Remove all scan checkbox widgets and create new ones from
        scan_name_list

        Parameters
        ----------
        scan_name_list : list
            List of scan names.

        Returns
        -------
        None.

        """
        
        # Remove all prior checkboxes
        for key in self.scan_check_dict:
            self.scan_button_group.removeButton(self.scan_check_dict[key])
            self.scan_layout.removeWidget(self.scan_check_dict[key])
            self.scan_check_dict[key].deleteLater()
            self.scan_check_dict[key] = None
        self.scan_check_dict.clear()
        
        # Create a new 
        # Add new check boxes
        for name in scan_name_list:
            self.scan_check_dict[name] = Qt.QCheckBox(name)
            self.scan_check_dict[name].setChecked(0)
            self.scan_layout.addWidget(self.scan_check_dict[name])
            self.scan_button_group.addButton(self.scan_check_dict[name])
            self.scan_check_dict[name].setChecked(1)
    
    def clear_selection(self):
        """
        Clears out the selected points from the viewport and dicts.

        Returns
        -------
        None.

        """
        
        # Clear the inputs from each appendPolyData in the dict
        for scan_name in self.selected_poly_dict:
            self.selected_poly_dict[scan_name].RemoveAllInputs()
            pdata = vtk.vtkPolyData()
            self.selected_poly_dict[scan_name].AddInputData(pdata)
            self.selected_poly_dict[scan_name].Update()
        
        self.vtkWidget.GetRenderWindow().Render()
    
    def update_man_class_dict(self):
        """
        Updates the polydatas in man_class_dict

        Returns
        -------
        None.

        """
        
        for scan_name in self.project.scan_dict:
            # If this scan_name is already in man_class_dict remove it from
            # append PolyData
            if scan_name in self.man_class_dict:
                self.man_class_append.RemoveInputData(
                    self.man_class_dict[scan_name])
            # Check if the man_class table is empty, if so create an empty
            # polydata
            if self.project.scan_dict[scan_name].man_class.shape[0]==0:
                pdata = vtk.vtkPolyData()
            else:
                # Otherwise, use pedigreeID selection to get the manually
                # classified points
                # Get PointId's from index of man_class
                pedigreeIds = vtk.vtkTypeUInt32Array()
                pedigreeIds.SetNumberOfComponents(1)
                pedigreeIds.SetNumberOfTuples(
                    self.project.scan_dict[scan_name].man_class.shape[0])
                np_pedigreeIds = vtk_to_numpy(pedigreeIds)
                if np.max(self.project.scan_dict[scan_name].man_class
                          .index.values)>np.iinfo(np.uint32).max:
                    raise RuntimeError('PointId exceeds size of uint32')
                np_pedigreeIds[:] = (self.project.scan_dict[scan_name]
                                     .man_class.index.values.astype(np.uint32))
                pedigreeIds.Modified()
                # Use PedigreeId selection to get points
                selectionNode = vtk.vtkSelectionNode()
                selectionNode.SetFieldType(1) # we want to select points
                selectionNode.SetContentType(2) # 2 corresponds to pedigreeIds
                selectionNode.SetSelectionList(pedigreeIds)
                selection = vtk.vtkSelection()
                selection.AddNode(selectionNode)
                extractSelection = vtk.vtkExtractSelection()
                extractSelection.SetInputData(0, self.mapper_dict[scan_name].
                                              GetInput())
                extractSelection.SetInputData(1, selection)
                extractSelection.Update()
                vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
                vertexGlyphFilter.SetInputConnection(
                    extractSelection.GetOutputPort())
                vertexGlyphFilter.Update()
                
                pdata = vertexGlyphFilter.GetOutput()
                
            # Put the corresponding pdata in man_class_dict
            self.man_class_dict[scan_name] = pdata
            # And add to appendPolyData
            self.man_class_append.AddInputData(self.man_class_dict[scan_name])
        
        # Updates
        self.man_class_append.Update()
        self.man_class_mapper.Update()
        if self.show_class_checkbox.isChecked():
            self.vtkWidget.GetRenderWindow().Render()
                
    ### VTK methods ###
    def on_end_pick(self, obj, event):
        """
        After a pick action concludes we want to select those points.

        Parameters
        ----------
        obj : TYPE
            DESCRIPTION.
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Variable for checking if we successfully selected any points
        pts_selected = False
        
        for scan_name in self.mapper_dict:
            
            # Check if this scan is visible
            if self.scan_check_dict[scan_name].isChecked():
                
                # Select points via frustum selection
                arr = vtk.vtkDoubleArray()
                arr.SetNumberOfComponents(4)
                arr.SetNumberOfTuples(8)
                for i in range(8):
                    pt = obj.GetClipPoints().GetPoint(i)
                    tup = (pt[0], pt[1], pt[2], 1)
                    arr.SetTuple(i, tup)
                selectionNode = vtk.vtkSelectionNode()
                selectionNode.SetFieldType(1) # we want to select points
                selectionNode.SetContentType(5) # Frustum selection is 5 
                selectionNode.SetSelectionList(arr)
                selection = vtk.vtkSelection()
                selection.AddNode(selectionNode)
                extractSelection = vtk.vtkExtractSelection()
                extractSelection.SetInputData(0, self.mapper_dict[scan_name].
                                              GetInput())
                extractSelection.SetInputData(1, selection)
                extractSelection.Update()
                vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
                vertexGlyphFilter.SetInputConnection(
                    extractSelection.GetOutputPort())
                vertexGlyphFilter.Update()
                
                if vertexGlyphFilter.GetOutput().GetNumberOfPoints()>0:
                    pts_selected = True
                
                # Add selected points to appropriate appendPolyData
                self.selected_poly_dict[scan_name].AddInputData(
                    vertexGlyphFilter.GetOutput())
        
        # If we selected any points enable class_button
        if pts_selected:
            self.class_button.setEnabled(1)
    
    def on_modified_renderwindow(self, obj, event):
        """
        When the renderwindow is modified, update near and far clipping
        plane labels.

        Parameters
        ----------
        obj : TYPE
            DESCRIPTION.
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Get current clipping distances
        clipping_dists = self.renderer.GetActiveCamera().GetClippingRange()
        
        # update labels
        self.near_label.setText(str(clipping_dists[0]))
        self.far_label.setText(str(clipping_dists[1]))


if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())