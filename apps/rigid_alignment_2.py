# -*- coding: utf-8 -*-
"""

rigid_alignment_2.py

An updated rigid_alignment app specifically designed to assist with
manually updating registration

matplotlib from https://www.learnpyqt.com/tutorials/plotting-matplotlib/

Created on Fri Mar  26  2021

@author: d34763s
"""

import sys
import vtk
import math
import numpy as np
from scipy.stats import mode
from vtk.util.numpy_support import vtk_to_numpy
from vtk.numpy_interface import dataset_adapter as dsa
from PyQt5 import Qt
from collections import namedtuple
import platform
if platform.system()=='Windows':
    sys.path.append('C:/Users/d34763s/Desktop/DavidCS/PhD/code/pydar/')
else:
    sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar
import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=8, height=8, dpi=600):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainWindow(Qt.QMainWindow):
    
    def __init__(self, parent=None):
        Qt.QMainWindow.__init__(self, parent)
        
        self.resize(2000, 1500)
        
        # Create the main layout
        self.frame = Qt.QFrame()
        main_layout = Qt.QHBoxLayout()
        
        # Populate the vis_layout
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vtkWidget.setSizePolicy(Qt.QSizePolicy.Expanding, 
                                     Qt.QSizePolicy.Expanding)

        # Create the vis_tools_layout to sit beneath the renderwindow
        vis_tools_layout = Qt.QHBoxLayout()
        
        # Populate the vis_tools_layout
        # Look down button
        look_down_button = Qt.QPushButton('Look Down')
        vis_tools_layout.addWidget(look_down_button)
        # Radio dial for which point we want to pick
        self.point_button_group = Qt.QButtonGroup()
        p0_radio = Qt.QRadioButton('p0')
        p0_radio.setChecked(1)
        self.point_button_group.addButton(p0_radio, 0)
        vis_tools_layout.addWidget(p0_radio)
        p1_radio = Qt.QRadioButton('p1')
        self.point_button_group.addButton(p1_radio, 1)
        vis_tools_layout.addWidget(p1_radio)
        p2_radio = Qt.QRadioButton('p2')
        self.point_button_group.addButton(p2_radio, 2)
        vis_tools_layout.addWidget(p2_radio)
        p3_radio = Qt.QRadioButton('p3')
        self.point_button_group.addButton(p3_radio, 3)
        vis_tools_layout.addWidget(p3_radio)

        # Transect endpoints
        t_list = ['x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2'
                  , 'x3', 'y3', 'z3', 'd']
        self.transect_dict = {}
        for t in t_list:
            t_label = Qt.QLabel(t + ': ')
            vis_tools_layout.addWidget(t_label)
            self.transect_dict[t] = Qt.QLineEdit('0.0')
            self.transect_dict[t].setValidator(Qt.QDoubleValidator())
            vis_tools_layout.addWidget(self.transect_dict[t])
        plot_transect_button = Qt.QPushButton('Plot Transect')
        vis_tools_layout.addWidget(plot_transect_button)
        
        # Create a matplotlib plot widget to plot transects in
        self.mpl_widget = MplCanvas(self)
        # Create toolbar for interacting with plot
        toolbar = NavigationToolbar(self.mpl_widget, self)

        # Create the visualization layout, will contain the renderwindow
        # and a toolbar with options
        vis_layout = Qt.QVBoxLayout()

        vis_layout.addWidget(self.vtkWidget)
        vis_layout.addLayout(vis_tools_layout)
        vis_layout.addWidget(toolbar)

        # Need to add vis_layout to a widget to use QSplitter
        vis_widget = Qt.QWidget()
        vis_widget.setLayout(vis_layout)
        
        # Create a second matplotlib canvas for the second transect
        self.mpl_widget2 = MplCanvas(self)
        toolbar2 = NavigationToolbar(self.mpl_widget2, self)

        # now create Qsplitter and add vis_widget and plot
        vis_splitter = Qt.QSplitter(0) # 0 here creates vertical splitter
        vis_splitter.addWidget(vis_widget)
        vis_splitter.addWidget(self.mpl_widget)
        vis_splitter.addWidget(toolbar2)
        vis_splitter.addWidget(self.mpl_widget2)

        # Create the Options layout, which will contain tools to select files
        # classify points, etc
        opt_layout = Qt.QVBoxLayout()
        
        # Populate the opt_layout
        
        # Scan Area button
        self.sel_scan_area_button = Qt.QPushButton("Select Scan Area")
        # Create the file dialog that we'll use
        self.proj_dialog = Qt.QFileDialog(self)
        self.proj_dialog.setDirectory('/media/thayer/Data/mosaic_lidar/')
        self.proj_dialog.setFileMode(4) # set file mode to pick directories
        opt_layout.addWidget(self.sel_scan_area_button)
        
        # ComboBoxes containing available projects
        proj_label_0 = Qt.QLabel("Project 0: ")
        self.proj_combobox_0 = Qt.QComboBox()
        self.proj_combobox_0.setEnabled(0)
        self.proj_combobox_0.setSizeAdjustPolicy(0)
        opt_layout.addWidget(proj_label_0)
        opt_layout.addWidget(self.proj_combobox_0)
        self.proj_suffix_0 = Qt.QLineEdit('slfsnow')
        opt_layout.addWidget(self.proj_suffix_0)
        proj_label_1 = Qt.QLabel("Project 1: ")
        self.proj_combobox_1 = Qt.QComboBox()
        self.proj_combobox_1.setEnabled(0)
        self.proj_combobox_1.setSizeAdjustPolicy(0)
        opt_layout.addWidget(proj_label_1)
        opt_layout.addWidget(self.proj_combobox_1)
        self.proj_suffix_1 = Qt.QLineEdit('slfsnow')
        opt_layout.addWidget(self.proj_suffix_1)
        
        
        # Button to prompt us to select a project
        self.sel_proj_button = Qt.QPushButton("Select Project")
        self.sel_proj_button.setEnabled(0)
        opt_layout.addWidget(self.sel_proj_button)
        
        # Create combobox to select which scan to display
        self.scan_combobox = Qt.QComboBox()
        self.scan_combobox.setEnabled(0)
        self.scan_combobox.setSizeAdjustPolicy(0)
        opt_layout.addWidget(self.scan_combobox)
        
        # Create interface for Gridded minima alignment
        self.z_align_dict = {}
        z_align_list = ['cell_w', 'min_dens', 'max_diff']
        for param in z_align_list:
            temp_layout = Qt.QHBoxLayout()
            temp_label = Qt.QLabel(param + ": ")
            self.z_align_dict[param] = Qt.QLineEdit('0.0')
            self.z_align_dict[param].setValidator(Qt.QDoubleValidator())
            temp_layout.addWidget(temp_label)
            temp_layout.addWidget(self.z_align_dict[param])
            opt_layout.addLayout(temp_layout)
        temp_layout = Qt.QHBoxLayout()
        temp_label = Qt.QLabel("Bin Reduc Mode: ")
        self.z_align_mode = Qt.QComboBox()
        self.z_align_mode.addItems(['min', 'mean', 'mode'])
        temp_layout.addWidget(temp_label)
        temp_layout.addWidget(self.z_align_mode)
        opt_layout.addLayout(temp_layout)
        z_align_button = Qt.QPushButton("Compute Z Align")
        opt_layout.addWidget(z_align_button)
        self.frac_exceed_label = Qt.QLabel('frac>max_diff: nan')
        opt_layout.addWidget(self.frac_exceed_label)
        diff_mode_list = ['mean', 'median', 'mode']
        self.diff_mode_dict = {}
        self.diff_mode_buttongroup = Qt.QButtonGroup()
        for param in diff_mode_list:
            temp_layout = Qt.QHBoxLayout()
            temp_label = Qt.QLabel(param + ": ")
            self.diff_mode_dict[param] = Qt.QPushButton('')
            self.diff_mode_buttongroup.addButton(self.diff_mode_dict[param])
            temp_layout.addWidget(temp_label)
            temp_layout.addWidget(self.diff_mode_dict[param])
            opt_layout.addLayout(temp_layout)
        self.z_change = Qt.QLineEdit('0.0')
        self.z_change.setValidator(Qt.QDoubleValidator())
        opt_layout.addWidget(self.z_change)
        z_update_button = Qt.QPushButton("Apply Z Change")
        opt_layout.addWidget(z_update_button)

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
        main_layout.addWidget(vis_splitter, stretch=5)
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
        z_align_button.clicked.connect(self.on_z_align_button_clicked)
        self.diff_mode_buttongroup.buttonPressed.connect(self.diff_mode_changed)
        z_update_button.clicked.connect(self.on_z_update_button_clicked)
        update_param_button.clicked.connect(self.on_update_param_button_click)
        reset_param_button.clicked.connect(self.on_reset_param_button_click)
        look_down_button.clicked.connect(self.look_down)
        for t in t_list:
            self.transect_dict[t].editingFinished.connect(
                self.update_trans_endpoints)
        plot_transect_button.clicked.connect(self.plot_transect)
        
        self.show()
        
        # VTK setup
        
        # Transect endpoints
        pts0 = vtk.vtkPoints()
        pts0.SetNumberOfPoints(1)
        pts0.SetPoint(0, 0.0, 0.0, 0.0)
        pt_0 = vtk.vtkPolyData()
        pt_0.SetPoints(pts0)
        self.vgf_pt_0 = vtk.vtkVertexGlyphFilter()
        self.vgf_pt_0.SetInputData(pt_0)
        self.vgf_pt_0.Update()
        mapper_pt_0 = vtk.vtkPolyDataMapper()
        mapper_pt_0.SetInputConnection(self.vgf_pt_0.GetOutputPort())
        actor_pt_0 = vtk.vtkActor()
        actor_pt_0.SetMapper(mapper_pt_0)
        actor_pt_0.GetProperty().RenderPointsAsSpheresOn()
        actor_pt_0.GetProperty().SetPointSize(20)
        actor_pt_0.GetProperty().SetColor(1, 1, 0)

        pts1 = vtk.vtkPoints()
        pts1.SetNumberOfPoints(1)
        pts1.SetPoint(0, 0.0, 0.0, 0.0)
        pt_1 = vtk.vtkPolyData()
        pt_1.SetPoints(pts1)
        self.vgf_pt_1 = vtk.vtkVertexGlyphFilter()
        self.vgf_pt_1.SetInputData(pt_1)
        self.vgf_pt_1.Update()
        mapper_pt_1 = vtk.vtkPolyDataMapper()
        mapper_pt_1.SetInputConnection(self.vgf_pt_1.GetOutputPort())
        actor_pt_1 = vtk.vtkActor()
        actor_pt_1.SetMapper(mapper_pt_1)
        actor_pt_1.GetProperty().RenderPointsAsSpheresOn()
        actor_pt_1.GetProperty().SetPointSize(20)
        actor_pt_1.GetProperty().SetColor(1, 0, 0)
        
        pts2 = vtk.vtkPoints()
        pts2.SetNumberOfPoints(1)
        pts2.SetPoint(0, 0.0, 0.0, 0.0)
        pt_2 = vtk.vtkPolyData()
        pt_2.SetPoints(pts2)
        self.vgf_pt_2 = vtk.vtkVertexGlyphFilter()
        self.vgf_pt_2.SetInputData(pt_2)
        self.vgf_pt_2.Update()
        mapper_pt_2 = vtk.vtkPolyDataMapper()
        mapper_pt_2.SetInputConnection(self.vgf_pt_2.GetOutputPort())
        actor_pt_2 = vtk.vtkActor()
        actor_pt_2.SetMapper(mapper_pt_2)
        actor_pt_2.GetProperty().RenderPointsAsSpheresOn()
        actor_pt_2.GetProperty().SetPointSize(20)
        actor_pt_2.GetProperty().SetColor(0, 0, 1)
        
        pts3 = vtk.vtkPoints()
        pts3.SetNumberOfPoints(1)
        pts3.SetPoint(0, 0.0, 0.0, 0.0)
        pt_3 = vtk.vtkPolyData()
        pt_3.SetPoints(pts3)
        self.vgf_pt_3 = vtk.vtkVertexGlyphFilter()
        self.vgf_pt_3.SetInputData(pt_3)
        self.vgf_pt_3.Update()
        mapper_pt_3 = vtk.vtkPolyDataMapper()
        mapper_pt_3.SetInputConnection(self.vgf_pt_3.GetOutputPort())
        actor_pt_3 = vtk.vtkActor()
        actor_pt_3.SetMapper(mapper_pt_3)
        actor_pt_3.GetProperty().RenderPointsAsSpheresOn()
        actor_pt_3.GetProperty().SetPointSize(20)
        actor_pt_3.GetProperty().SetColor(1, 0, 1)

        # Renderer and interactor
        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(actor_pt_0)
        self.renderer.AddActor(actor_pt_1)
        self.renderer.AddActor(actor_pt_2)
        self.renderer.AddActor(actor_pt_3)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        style = vtk.vtkInteractorStyleTrackballCamera()
        pointPicker = vtk.vtkPointPicker()
        pointPicker.SetTolerance(0.03)
        pointPicker.AddObserver("EndPickEvent", self.on_end_pick)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.SetPicker(pointPicker)
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
        self.project_path = dir_str    
        
        
        # Depending on the scan_area_name load and register project
        if scan_area_name=='Snow2':
            project_names = ['mosaic_02_110619.RiSCAN',
                             'mosaic_02_111319.RiSCAN']
            
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
            
        else:
            raise ValueError("You have selected a nonexistant scan area."
                             " please start again")
        
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
        project_name_1 = self.proj_combobox_1.currentText()
        self.scan_area = pydar.ScanArea(self.project_path)
        
        self.scan_area.add_project(project_name_0, import_mode='read_scan',
                                   las_fieldnames=['Points'], create_id=False,
                                   class_list='all', 
                                   suffix=self.proj_suffix_0.text())
        self.scan_area.project_dict[project_name_0].read_transforms(
            suffix=self.proj_suffix_0.text())
        self.scan_area.project_dict[project_name_0].apply_transforms(
            ['current_transform'])
        
        if not project_name_0==project_name_1:
            self.scan_area.add_project(project_name_1, import_mode=
                                       'read_scan', las_fieldnames=['Points'],
                                       create_id=False, class_list='all',
                                       suffix=self.proj_suffix_1.text())
            
            self.scan_area.project_dict[project_name_1].read_transforms(
                suffix=self.proj_suffix_1.text())
            self.scan_area.project_dict[project_name_1].apply_transforms(
                ['current_transform'])
        
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
        
        # If we are looking at just one project the behavior is abit different
        if self.project_0==self.project_1:
            # If we were previously rendering a singlescan, remove it from
            # renderwindow put it back into
            # the project's scan_dict, create cyan pipeline, render.
            # Pop requested scan position
            if hasattr(self, 'ss'):
                # Remove it's actor from the renderer
                self.renderer.RemoveActor(self.ss.actor)
                # Create pipeline add back to scan_dict
                self.ss.create_solid_pipeline('Cyan')
                self.project_0.scan_dict[self.ss.scan_name] = self.ss
                self.renderer.AddActor(self.ss.actor)
            self.renderer.RemoveActor(self.project_0.scan_dict[s].actor)
            self.ss = self.project_0.scan_dict.pop(s)
            
        else:
            # If we were previously rendering a singlescan, delete it
            if hasattr(self, 'ss'):
                # Remove it's actor from the renderer
                self.renderer.RemoveActor(self.ss.actor)
                self.vtkWidget.GetRenderWindow().Render()
    
            # Store reference to requested scan
            self.ss = self.project_1.scan_dict[s]
            
            # If we are looking at just one project, render all of the scans
            # from project 0 except the requested.
        

        # Create Actor and add to renderer
        self.ss.create_solid_pipeline('Lime')
        self.renderer.AddActor(self.ss.actor)

        # Call on_reset_param_button_click to update entry fields and render
        self.on_reset_param_button_click(1)
        
        # Set the endpoints of the transect to be +- 100 m along the scanners
        # x axis
        pt_0 = self.ss.transform.TransformPoint(-100, 0, 0)
        pt_1 = self.ss.transform.TransformPoint(100, 0, 0)
        self.transect_dict['x0'].setText(str(pt_0[0]))
        self.transect_dict['y0'].setText(str(pt_0[1]))
        self.transect_dict['z0'].setText(str(pt_0[2]))
        self.transect_dict['x1'].setText(str(pt_1[0]))
        self.transect_dict['y1'].setText(str(pt_1[1]))
        self.transect_dict['z1'].setText(str(pt_1[2]))
        
        pt_2 = self.ss.transform.TransformPoint(0, -100, 0)
        pt_3 = self.ss.transform.TransformPoint(0, 100, 0)
        self.transect_dict['x2'].setText(str(pt_2[0]))
        self.transect_dict['y2'].setText(str(pt_2[1]))
        self.transect_dict['z2'].setText(str(pt_2[2]))
        self.transect_dict['x3'].setText(str(pt_3[0]))
        self.transect_dict['y3'].setText(str(pt_3[1]))
        self.transect_dict['z3'].setText(str(pt_3[2]))
        self.update_trans_endpoints()

    def on_z_align_button_clicked(self, s):
        """
        Run align z between the ss and project by gridded minima

        Parameters
        ----------
        s : int
            Button status. Not used.

        Returns
        -------
        None.

        """
        
        # Get gridded difference
        frac_exceed_max_diff, diff = self.scan_area.z_alignment_ss(
            self.project_0.project_name, self.project_1.project_name, 
            self.ss.scan_name, float(self.z_align_dict['cell_w'].text()),
            float(self.z_align_dict['cell_w'].text()), 
            float(self.z_align_dict['min_dens'].text()), 
            float(self.z_align_dict['max_diff'].text()),
            bin_reduc_op=self.z_align_mode.currentText())
        # Update text output
        self.frac_exceed_label.setText('frac>max_diff: ' 
                                       + str(frac_exceed_max_diff))
        # Set the text for each of the diff_mode buttons
        diff_notnan = np.ravel(diff)[np.logical_not(np.isnan(
                        np.ravel(diff)))]
        self.diff_mode_dict['mean'].setText(str(round(
            -1*diff_notnan.mean(), 3)))
        self.diff_mode_dict['median'].setText(str(round(
            -1*np.median(diff_notnan), 3)))
        m, _ = mode(np.around(diff_notnan, 3))
        self.diff_mode_dict['mode'].setText(str(round(-1*m[0], 3)))
        
        # Update plot
        # Clear plot canvas
        self.mpl_widget.axes.cla()
        self.mpl_widget.axes.imshow(diff)
        self.mpl_widget.axes.axis('equal')
        self.mpl_widget.draw()
    
    def diff_mode_changed(self, button):
        """
        When we click one of the diff mode buttons update the z_change
        line edit and apply the change in z.

        Parameters
        ----------
        button : Qt.QAbstractButton
            The button that was clicked

        Returns
        -------
        None.

        """
        
        print('Diff mode changed')
        self.z_change.setText(button.text())
        self.on_z_update_button_clicked(1)
    
    def on_z_update_button_clicked(self, s):
        """
        Change dz by the amount in z_change

        Parameters
        ----------
        s : int
            Button status. Not used.

        Returns
        -------
        None.

        """
        
        # Update dz
        self.param_dict['dz'].setText(str(
            float(self.param_dict['dz'].text()) + float(self.z_change.text())))
        # Update transformation applied to scan
        self.on_update_param_button_click(1)
        
    def on_update_param_button_click(self, s):
        """
        Set transformation for ss to match user defined parameters.

        Parameters
        ----------
        s : int
            Button status. Not used.

        Returns
        -------
        None.

        """

        # Step 1
        # Create a 4x4 homologous transform from the user defined parameters
        u = np.float32(self.param_dict['roll'].text())
        v = np.float32(self.param_dict['pitch'].text())
        w = np.float32(self.param_dict['yaw'].text())
        dx = np.float32(self.param_dict['dx'].text())
        dy = np.float32(self.param_dict['dy'].text())
        dz = np.float32(self.param_dict['dz'].text())
        c = np.cos
        s = np.sin
        
        Rx = np.array([[1, 0, 0, 0],
                      [0, c(u), -s(u), 0],
                      [0, s(u), c(u), 0],
                      [0, 0, 0, 1]])
        Ry = np.array([[c(v), 0, s(v), 0],
                       [0, 1, 0, 0],
                       [-s(v), 0, c(v), 0],
                       [0, 0, 0, 1]])
        Rz = np.array([[c(w), -s(w), 0, 0],
                      [s(w), c(w), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        # Order of rotations in vtk is Pitch, then Roll, then Yaw
        M = Rz @ Rx @ Ry
        # Now add translation components
        M[0, 3] = dx
        M[1, 3] = dy
        M[2, 3] = dz

        # Step 2:
        # Add transform matrix to ss's transform dict, apply it and update
        # renderwindow
        self.ss.add_transform('app', M)
        self.ss.apply_transforms(['app'])
        self.vtkWidget.GetRenderWindow().Render()


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
    
    def plot_transect(self):
        """
        Plot the transect with the current endpoints

        Returns
        -------
        None.

        """

        # Step 1, let's plot the transect line on the viewport
        # If there's already a t_actor present remove it and clear plot
        if hasattr(self, 't_actor'):
            self.renderer.RemoveActor(self.t_actor)
            del self.t_actor
        
        # Clear plot canvas
        self.mpl_widget.axes.cla()
        self.mpl_widget.axes.axis('auto')
        line = vtk.vtkLineSource()
        line.SetPoint1(float(self.transect_dict['x0'].text()),
                       float(self.transect_dict['y0'].text()),
                       float(self.transect_dict['z0'].text()))
        line.SetPoint2(float(self.transect_dict['x1'].text()),
                       float(self.transect_dict['y1'].text()),
                       float(self.transect_dict['z1'].text()))
        t_mapper = vtk.vtkPolyDataMapper()
        t_mapper.SetInputConnection(line.GetOutputPort())
        self.t_actor = vtk.vtkActor()
        self.t_actor.SetMapper(t_mapper)
        self.t_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        self.t_actor.GetProperty().SetOpacity(0.8)
        self.t_actor.GetProperty().RenderLinesAsTubesOn()
        self.t_actor.GetProperty().SetLineWidth(5)
        self.renderer.AddActor(self.t_actor)
        self.vtkWidget.GetRenderWindow().Render()
        print('Line Created')

        # Step 2 Extract Points
        length = np.sqrt((float(self.transect_dict['x1'].text())
                          -float(self.transect_dict['x0'].text()))**2 + 
                         (float(self.transect_dict['y1'].text())
                          -float(self.transect_dict['y0'].text()))**2)
        # we need a transformation first brings point 0 to the origin, then yaws
        # transect axis onto x-axis
        t_trans = vtk.vtkTransform()
        # set mode to post multiply, so we will first translate and then rotate
        t_trans.PostMultiply()
        t_trans.Translate(-1*float(self.transect_dict['x0'].text()), 
                          -1*float(self.transect_dict['y0'].text()), 0)
        # Get yaw angle of line
        yaw = np.arctan2(float(self.transect_dict['y1'].text())
                         -float(self.transect_dict['y0'].text()), 
                         float(self.transect_dict['x1'].text())
                         -float(self.transect_dict['x0'].text())) * 180/np.pi
        t_trans.RotateZ(-1*yaw)
        # Transform points, for project_0 we have to aggregate
        t_trans_filter_0 = vtk.vtkTransformFilter()
        t_trans_filter_0.SetTransform(t_trans)
        t_trans_filter_0.SetInputConnection(self.project_0.
                                            get_merged_points(port=True))
        t_trans_filter_0.Update()
        t_trans_filter_1 = vtk.vtkTransformFilter()
        t_trans_filter_1.SetTransform(t_trans)
        t_trans_filter_1.SetInputConnection(self.ss.currentFilter.
                                            GetOutputPort())
        t_trans_filter_1.Update()
        # Extract transformed points
        box = vtk.vtkBox()
        box.SetBounds(0, length, -1*float(self.transect_dict['d'].text())
                      , float(self.transect_dict['d'].text()), -1000, 1000)
        extractPoints_0 = vtk.vtkExtractPoints()
        extractPoints_0.SetImplicitFunction(box)
        extractPoints_0.SetInputConnection(t_trans_filter_0.GetOutputPort())
        extractPoints_0.Update()
        extractPoints_1 = vtk.vtkExtractPoints()
        extractPoints_1.SetImplicitFunction(box)
        extractPoints_1.SetInputConnection(t_trans_filter_1.GetOutputPort())
        extractPoints_1.Update()
        # Store output so we can use it for other things (and it's not garbage
        # collected?)
        self.t_pdata_0 = extractPoints_0.GetOutput()
        self.t_pdata_1 = extractPoints_1.GetOutput()
        # dataset adapters for points arrays
        t_dsa_0 = dsa.WrapDataObject(self.t_pdata_0)
        t_dsa_1 = dsa.WrapDataObject(self.t_pdata_1)
        print('Extracted Points')

        # Step 3 plot points
        self.mpl_widget.axes.scatter(t_dsa_0.Points[:,0],
                                     t_dsa_0.Points[:,2],
                                     c='Cyan', s=0.1)
        self.mpl_widget.axes.scatter(t_dsa_1.Points[:,0],
                                     t_dsa_1.Points[:,2],
                                     c='Lime', s=0.1)
        self.mpl_widget.axes.set_facecolor('k')
        self.mpl_widget.draw()
        
        # Step 1, let's plot the transect line on the viewport
        # If there's already a t_actor present remove it and clear plot
        if hasattr(self, 't2_actor'):
            self.renderer.RemoveActor(self.t2_actor)
            del self.t2_actor
        
        # Clear plot canvas
        self.mpl_widget2.axes.cla()
        self.mpl_widget2.axes.axis('auto')
        line = vtk.vtkLineSource()
        line.SetPoint1(float(self.transect_dict['x2'].text()),
                       float(self.transect_dict['y2'].text()),
                       float(self.transect_dict['z2'].text()))
        line.SetPoint2(float(self.transect_dict['x3'].text()),
                       float(self.transect_dict['y3'].text()),
                       float(self.transect_dict['z3'].text()))
        t_mapper = vtk.vtkPolyDataMapper()
        t_mapper.SetInputConnection(line.GetOutputPort())
        self.t2_actor = vtk.vtkActor()
        self.t2_actor.SetMapper(t_mapper)
        self.t2_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        self.t2_actor.GetProperty().SetOpacity(0.8)
        self.t2_actor.GetProperty().RenderLinesAsTubesOn()
        self.t2_actor.GetProperty().SetLineWidth(5)
        self.renderer.AddActor(self.t2_actor)
        self.vtkWidget.GetRenderWindow().Render()
        print('Line Created')

        # Step 2 Extract Points
        length = np.sqrt((float(self.transect_dict['x3'].text())
                          -float(self.transect_dict['x2'].text()))**2 + 
                         (float(self.transect_dict['y3'].text())
                          -float(self.transect_dict['y2'].text()))**2)
        # we need a transformation first brings point 0 to the origin, then yaws
        # transect axis onto x-axis
        t_trans = vtk.vtkTransform()
        # set mode to post multiply, so we will first translate and then rotate
        t_trans.PostMultiply()
        t_trans.Translate(-1*float(self.transect_dict['x2'].text()), 
                          -1*float(self.transect_dict['y2'].text()), 0)
        # Get yaw angle of line
        yaw = np.arctan2(float(self.transect_dict['y3'].text())
                         -float(self.transect_dict['y2'].text()), 
                         float(self.transect_dict['x3'].text())
                         -float(self.transect_dict['x2'].text())) * 180/np.pi
        t_trans.RotateZ(-1*yaw)
        # Transform points, for project_0 we have to aggregate
        t_trans_filter_0 = vtk.vtkTransformFilter()
        t_trans_filter_0.SetTransform(t_trans)
        t_trans_filter_0.SetInputConnection(self.project_0.
                                            get_merged_points(port=True))
        t_trans_filter_0.Update()
        t_trans_filter_1 = vtk.vtkTransformFilter()
        t_trans_filter_1.SetTransform(t_trans)
        t_trans_filter_1.SetInputConnection(self.ss.currentFilter.
                                            GetOutputPort())
        t_trans_filter_1.Update()
        # Extract transformed points
        box = vtk.vtkBox()
        box.SetBounds(0, length, -1*float(self.transect_dict['d'].text())
                      , float(self.transect_dict['d'].text()), -1000, 1000)
        extractPoints_0 = vtk.vtkExtractPoints()
        extractPoints_0.SetImplicitFunction(box)
        extractPoints_0.SetInputConnection(t_trans_filter_0.GetOutputPort())
        extractPoints_0.Update()
        extractPoints_1 = vtk.vtkExtractPoints()
        extractPoints_1.SetImplicitFunction(box)
        extractPoints_1.SetInputConnection(t_trans_filter_1.GetOutputPort())
        extractPoints_1.Update()
        # Store output so we can use it for other things (and it's not garbage
        # collected?)
        self.t2_pdata_0 = extractPoints_0.GetOutput()
        self.t2_pdata_1 = extractPoints_1.GetOutput()
        # dataset adapters for points arrays
        t2_dsa_0 = dsa.WrapDataObject(self.t2_pdata_0)
        t2_dsa_1 = dsa.WrapDataObject(self.t2_pdata_1)
        print('Extracted Points')

        # Step 3 plot points
        self.mpl_widget2.axes.scatter(t2_dsa_0.Points[:,0],
                                     t2_dsa_0.Points[:,2],
                                     c='Cyan', s=0.1)
        self.mpl_widget2.axes.scatter(t2_dsa_1.Points[:,0],
                                     t2_dsa_1.Points[:,2],
                                     c='Lime', s=0.1)
        self.mpl_widget2.axes.set_facecolor('k')
        self.mpl_widget2.draw()


    ### VTK methods ###
    def on_end_pick(self, obj, event):
        """
        When a pick is made set it to be an endpoint of the transect

        Parameters
        ----------
        obj: vtkPointPicker
            vtkPointPicker object containing picked location
        event: str
            event name, not used.

        Returns
        -------
        None.

        """
        print('pick made')

        # Get the picked point
        pt = obj.GetPickPosition()
        
        # Depending on which point we want to update, update that point
        if self.point_button_group.checkedId()==0:
            self.transect_dict['x0'].setText(str(pt[0]))
            self.transect_dict['y0'].setText(str(pt[1]))
            self.transect_dict['z0'].setText(str(pt[2]))
        elif self.point_button_group.checkedId()==1:
            self.transect_dict['x1'].setText(str(pt[0]))
            self.transect_dict['y1'].setText(str(pt[1]))
            self.transect_dict['z1'].setText(str(pt[2]))
        elif self.point_button_group.checkedId()==2:
            self.transect_dict['x2'].setText(str(pt[0]))
            self.transect_dict['y2'].setText(str(pt[1]))
            self.transect_dict['z2'].setText(str(pt[2]))
        elif self.point_button_group.checkedId()==3:
            self.transect_dict['x3'].setText(str(pt[0]))
            self.transect_dict['y3'].setText(str(pt[1]))
            self.transect_dict['z3'].setText(str(pt[2]))
        self.update_trans_endpoints()

    def update_trans_endpoints(self):
        """
        Update the spheres marking the transect endpoints

        Returns
        -------
        None.

        """

        # Update the transect endpoints
        pts0 = vtk.vtkPoints()
        pts0.SetNumberOfPoints(1)
        pts0.SetPoint(0, float(self.transect_dict['x0'].text()),
                         float(self.transect_dict['y0'].text()),
                         float(self.transect_dict['z0'].text()))
        pt_0 = vtk.vtkPolyData()
        pt_0.SetPoints(pts0)
        self.vgf_pt_0.SetInputData(pt_0)
        self.vgf_pt_0.Update()
        pts1 = vtk.vtkPoints()
        pts1.SetNumberOfPoints(1)
        pts1.SetPoint(0, float(self.transect_dict['x1'].text()),
                         float(self.transect_dict['y1'].text()),
                         float(self.transect_dict['z1'].text()))
        pt_1 = vtk.vtkPolyData()
        pt_1.SetPoints(pts1)
        self.vgf_pt_1.SetInputData(pt_1)
        self.vgf_pt_1.Update()
        pts2 = vtk.vtkPoints()
        pts2.SetNumberOfPoints(1)
        pts2.SetPoint(0, float(self.transect_dict['x2'].text()),
                         float(self.transect_dict['y2'].text()),
                         float(self.transect_dict['z2'].text()))
        pt_2 = vtk.vtkPolyData()
        pt_2.SetPoints(pts2)
        self.vgf_pt_2.SetInputData(pt_2)
        self.vgf_pt_2.Update()
        pts3 = vtk.vtkPoints()
        pts3.SetNumberOfPoints(1)
        pts3.SetPoint(0, float(self.transect_dict['x3'].text()),
                         float(self.transect_dict['y3'].text()),
                         float(self.transect_dict['z3'].text()))
        pt_3 = vtk.vtkPolyData()
        pt_3.SetPoints(pts3)
        self.vgf_pt_3.SetInputData(pt_3)
        self.vgf_pt_3.Update()
        
        self.vtkWidget.GetRenderWindow().Render()

if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())