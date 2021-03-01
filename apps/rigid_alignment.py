# -*- coding: utf-8 -*-
"""

rigid_alignment.py

An application for examining the offset between successive scans and
exploring rigid transformations.

Created on Thu Jan  7 10:48:45 2021

@author: d34763s
"""

import sys
import vtk
import math
import copy
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
        
        self.resize(2000, 1500)
        
        # Create the main layout
        self.frame = Qt.QFrame()
        main_layout = Qt.QHBoxLayout()
        
        # Create the visualization layout, will contain the renderwindow
        # and a toolbar with options
        vis_layout = Qt.QVBoxLayout()
        
        # Create the vis_tools_layout to sit beneath the renderwindow
        vis_tools_layout = Qt.QHBoxLayout()
        
        # Populate the vis_tools_layout
        look_down_button = Qt.QPushButton('Look Down')
        vis_tools_layout.addWidget(look_down_button)
        
        # Populate the vis_layout
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vtkWidget.setSizePolicy(Qt.QSizePolicy.Expanding, 
                                     Qt.QSizePolicy.Expanding)
        
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
        
        # ComboBoxes containing available projects
        proj_label_0 = Qt.QLabel("Project 0: ")
        self.proj_combobox_0 = Qt.QComboBox()
        self.proj_combobox_0.setEnabled(0)
        self.proj_combobox_0.setSizeAdjustPolicy(0)
        opt_layout.addWidget(proj_label_0)
        opt_layout.addWidget(self.proj_combobox_0)
        proj_label_1 = Qt.QLabel("Project 1: ")
        self.proj_combobox_1 = Qt.QComboBox()
        self.proj_combobox_1.setEnabled(0)
        self.proj_combobox_1.setSizeAdjustPolicy(0)
        opt_layout.addWidget(proj_label_1)
        opt_layout.addWidget(self.proj_combobox_1)
        
        # Button to prompt us to select a project
        self.sel_proj_button = Qt.QPushButton("Select Project")
        self.sel_proj_button.setEnabled(0)
        opt_layout.addWidget(self.sel_proj_button)
        
        # Create combobox to select which scan to display
        self.scan_combobox = Qt.QComboBox()
        self.scan_combobox.setEnabled(0)
        self.scan_combobox.setSizeAdjustPolicy(0)
        opt_layout.addWidget(self.scan_combobox)

        # Create interface for modifying rigid transformation parameters
        self.param_dict = {}
        param_list = ['dx', 'dy', 'dz', 'roll', 'pitch', 'yaw']
        for param in param_list:
            temp_layout = Qt.QHBoxLayout()
            temp_label = Qt.QLabel(param + ": ")
            self.param_dict[param] = Qt.QLineEdit('0.0')
            self.param_dict[param].setValidator(Qt.QDoubleValidator())
            temp_layout.addWidget(temp_label)
            temp_layout.addWidget(self.param_dict[param])
            opt_layout.addLayout(temp_layout)

        # Create buttons for reseting or update transformation
        update_param_button = Qt.QPushButton("Update Transform")
        opt_layout.addWidget(update_param_button)
        reset_param_button = Qt.QPushButton("Reset Transform")
        opt_layout.addWidget(reset_param_button)

        # Populate the main layout
        main_layout.addLayout(vis_layout, stretch=5)
        main_layout.addLayout(opt_layout)
        
        # Set layout for the frame and set central widget
        self.frame.setLayout(main_layout)
        self.setCentralWidget(self.frame)
        
        # Signals and slots
        self.sel_scan_area_button.clicked.connect(
            self.on_sel_scan_area_button_click)
        self.sel_proj_button.clicked.connect(self.on_sel_proj_button_click)
        self.proj_dialog.fileSelected.connect(self.on_scan_area_selected)
        self.scan_combobox.currentTextChanged.connect(self.on_scan_changed)
        reset_param_button.clicked.connect(self.on_reset_param_button_click)
        look_down_button.clicked.connect(self.look_down)
        
        self.show()
        
        # VTK setup
        
        # Renderer and interactor
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        #self.vtkWidget.GetRenderWindow().AddObserver("ModifiedEvent", 
        #                                             self.
        #                                             on_modified_renderwindow)
        #style = vtk.vtkInteractorStyleRubberBandPick()
        #areaPicker = vtk.vtkAreaPicker()
        #areaPicker.AddObserver("EndPickEvent", self.on_end_pick)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        #self.iren.SetPicker(areaPicker)
        self.iren.Initialize()
        #self.iren.SetInteractorStyle(style)
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
            
            registration_list = [Registration('mosaic_rov_190120.RiSCAN', 
                                  'mosaic_rov_190120.RiSCAN'),
                     Registration('mosaic_rov_190120.RiSCAN',
                                  'mosaic_rov_110120.RiSCAN',
                                  ['r05', 'r28', 'r29', 'r31', 'r32', 'r33',
                                   'r34'],
                                  'LS'),
                     Registration('mosaic_rov_190120.RiSCAN',
                                  'mosaic_rov_040120.RiSCAN',
                                  ['r28', 'r29', 'r30', 'r31', 'r32', 'r33'],
                                  'LS'),
                     Registration('mosaic_rov_190120.RiSCAN',
                                  'mosaic_rov_250120.RiSCAN',
                                  ['r28', 'r29', 'r30', 'r32', 'r34', 'r35',
                                   'r22'],
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
        
        # Update proj_comboboxes with available scans
        self.proj_combobox_0.addItems(project_names)
        self.proj_combobox_0.setEnabled(1)
        self.proj_combobox_1.addItems(project_names)
        self.proj_combobox_1.setEnabled(1)
        
        # Enable sel_proj_button
        self.sel_proj_button.setEnabled(1)
        
        # Disable further scan area changes
        self.sel_scan_area_button.setEnabled(0)
    
    def on_sel_proj_button_click(self, s):
        """
        Load projects.

        Parameters
        ----------
        s : int
            Button status. Not used.

        Returns
        -------
        None.

        """

        # Parse project path and names and load projects
        project_name_0 = self.proj_combobox_0.currentText()
        self.scan_area.add_project(project_name_0, import_las=True, 
                                   create_id=False,
                                   las_fieldnames=['Points'])
        project_name_1 = self.proj_combobox_1.currentText()
        self.scan_area.add_project(project_name_1, import_las=True, 
                                   create_id=False,
                                   las_fieldnames=['Points'])
        self.scan_area.register_all()
        self.project_0 = self.scan_area.project_dict[project_name_0]
        self.project_1 = self.scan_area.project_dict[project_name_1]

        # Render project 0 (we will never need to change this one)
        for scan_name in self.project_0.scan_dict:
            # Create solid color pipeline
            self.project_0.scan_dict[scan_name].create_solid_pipeline('Cyan')
            self.renderer.AddActor(self.project_0.scan_dict[scan_name].actor)
        self.vtkWidget.GetRenderWindow().Render()
        
        # Update scan combobox for project_1
        self.scan_combobox.addItems(self.project_1.scan_dict.keys())
        self.scan_combobox.setEnabled(1)
        # by adding items we will trigger the currentTextChanged signal
        # and load the first scan position

    def on_scan_changed(self, s):
        """
        Render the scan from project_1 we requested and reset transform

        Parameters
        ----------
        s : str
            scan_name

        Returns
        -------
        None.

        """

        # If we were previously rendering a singlescan, delete it
        if hasattr(self, 'ss'):
            # Remove it's actor from the renderer
            self.renderer.RemoveActor(self.ss.actor)
            self.vtkWidget.GetRenderWindow().Render()

        # Store reference to requested scan
        self.ss = self.project_1.scan_dict[s]

        # Create Actor and add to renderer
        self.ss.create_solid_pipeline('Lime')
        self.renderer.AddActor(self.ss.actor)

        # Call on_reset_param_button_click to update entry fields and render
        self.on_reset_param_button_click(1)
    
    def on_reset_param_button_click(self, s):
        """
        Reset transformation parameters to those derived from reflectors.

        Parameters
        ----------
        s : int
            Button status. Not used.

        Returns
        -------
        None.

        """

        # Set currentTransform in ss to match the transform stored in the
        # project
        self.ss.apply_transforms(self.project_1.current_transform_list)

        # Update values in param_dict
        pos = np.float32(self.ss.transform.GetPosition())
        self.param_dict['dx'].setText(str(pos[0]))
        self.param_dict['dy'].setText(str(pos[1]))
        self.param_dict['dz'].setText(str(pos[2]))
        ori = np.float32(self.ss.transform.GetOrientation())
        ori = ori * math.pi / 180
        self.param_dict['roll'].setText(str(ori[0]))
        self.param_dict['pitch'].setText(str(ori[1]))
        self.param_dict['yaw'].setText(str(ori[2]))

        # Render
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
                
    ### VTK methods ###


if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())