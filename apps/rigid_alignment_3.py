# -*- coding: utf-8 -*-
"""

rigid_alignment_3.py

An updated rigid_alignment app specifically designed to assist with
manually updating registration

matplotlib from https://www.learnpyqt.com/tutorials/plotting-matplotlib/

Created on Fri Mar  26  2021

@author: d34763s
"""

import sys
import vtk
import math
import json
import os
import warnings
import numpy as np
import pyperclip
from scipy.stats import mode
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
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

class AreaPointList():
    
    def __init__(self, vtkWidget):
        """
        
        Returns
        -------
        None.

        """
        
        self.vtkWidget = vtkWidget
        # List for containing AreaPoint objects
        self.list = []
        # Qt objects
        self.scroll = Qt.QScrollArea()
        self.scroll.setMinimumWidth(200)
        self.scroll.setWidgetResizable(True)
        self.inner = Qt.QFrame(self.scroll)
        self.layout = Qt.QVBoxLayout()
        self.inner.setLayout(self.layout)
        self.scroll.setWidget(self.inner)
        self.button_group = Qt.QButtonGroup()
        
        # Add first areapoint
        self.add_areapoint()
    
    def get_scroll(self):
        return self.scroll
    
    def add_areapoint(self):
        self.list.append(AreaPoint(len(self.list), self))
    
    def update(self):
        # Handle displaying polyline
        renderer = (self.vtkWidget.GetRenderWindow().GetRenderers()
                    .GetFirstRenderer())
        if hasattr(self, 'actor'):
            # If a polyline already exists, delete it
            renderer.RemoveActor(self.actor)
            del self.actor
        if len(self.list)>=3:
            # If we have at least two picked points create line and render
            # Get the point coordinates for each area point
            pts_np = np.empty((len(self.list)-1, 3), dtype=np.float32)
            for i in np.arange(len(self.list)-1):
                pts_np[i,:] = (self.list[i].actor.GetMapper().GetInput().
                               GetPoint(0))
            pts = vtk.vtkPoints()
            pts.SetData(numpy_to_vtk(pts_np, deep=True, 
                                     array_type=vtk.VTK_FLOAT))
            pdata = vtk.vtkPolyData()
            pdata.SetPoints(pts)
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(len(self.list))
            for i in np.arange(len(self.list)-1):
                lines.InsertCellPoint(i)
            lines.InsertCellPoint(0) # closing the loop
            pdata.SetLines(lines)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(pdata)
            self.actor = vtk.vtkActor()
            self.actor.SetMapper(mapper)
            self.actor.GetProperty().SetLineWidth(5)
            self.actor.GetProperty().SetColor(1.0, 1.0, 0.0)
            self.actor.GetProperty().RenderLinesAsTubesOn()
            renderer.AddActor(self.actor)
        
        
        
        self.vtkWidget.GetRenderWindow().Render()
    
    def move(self, areapoint, direction):
        # Get the current position of the areapoint object
        ind = self.list.index(areapoint)
        # Handle cases where we're at the edge of the list
        if (ind==0) and (direction=='up'):
            return
        if (ind==(len(self.list)-2)) and (direction=='down'):
            return
        # Pop the element out of the list
        areapoint = self.list.pop(ind)
        # Remove the layout from the scroll
        self.layout.removeItem(areapoint.layout)
        if direction=='up':
            # If we want to move up subtract from ind
            self.list.insert(ind-1, areapoint)
            self.layout.insertLayout(ind-1, areapoint.layout)
        elif direction=='down':
            # If we want to move up subtract from ind
            self.list.insert(ind+1, areapoint)
            self.layout.insertLayout(ind+1, areapoint.layout)
        
        self.update()
    
    def insert_below(self, areapoint):
        # Take the areapoint at the bottom of the list and move it below
        # this areapoint
        # Get the current position of the areapoint object
        ind = self.list.index(areapoint)
        # Handle case where we're at the edge of the list
        if ind==(len(self.list)-2):
            return
        # Pop the bottom filled areapoint out of the list
        b_areapoint = self.list.pop(-2)
        # Remove the layout from the scroll
        self.layout.removeItem(b_areapoint.layout)
        # Insert areapoint and layout below this one
        self.list.insert(ind+1, b_areapoint)
        self.layout.insertLayout(ind+1, b_areapoint.layout)
        
        self.update()
    
    def delete(self, areapoint):
        # Get the current position of the areapoint object
        ind = self.list.index(areapoint)
        # Pop the element out of the list
        areapoint = self.list.pop(ind)
        # Remove button from button group
        self.button_group.removeButton(areapoint.radio)
        # Delete each item in the layout
        while areapoint.layout.count()>0:
            widget = areapoint.layout.takeAt(0).widget()
            widget.deleteLater()
        # Remove the layout from the scroll
        self.layout.removeItem(areapoint.layout)
        del areapoint.layout
        # Remove point from renderwindow
        renderer = (self.vtkWidget.GetRenderWindow().GetRenderers()
                    .GetFirstRenderer())
        renderer.RemoveActor(areapoint.actor)
        del areapoint.actor
        # Delete areapoint object, there should be no more references
        del areapoint
        
        self.update()
    
    def copy_areapoints(self, project_name):
        # Create the list of lists of the areapointlist as a json string
        
        output = []
        
        for ap in self.list:
            if ap.empty:
                continue
            output.append((ap.radio.text(), int(ap.PointId)))
        
        #print(output)
        pyperclip.copy(json.dumps({project_name: output}, indent=4))
    
    def save_areapoints(self, project_name, path):
        
        # Write json formatted areapoints to file given by path
        output = []
        
        for ap in self.list:
            if ap.empty:
                continue
            output.append((ap.radio.text(), int(ap.PointId)))
        
        #print(output)
        f = open(path, 'w')
        json.dump({project_name: output}, f, indent=4)
        f.close()

class AreaPoint():
    
    def __init__(self, position, areapointlist):
        """
        Create new AreaPoint

        Parameters
        ----------
        position : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Our button starts empty
        self.empty = True
        self.position = position
        self.areapointlist = areapointlist
        self.PointId = None
        
        # Qt stuff
        self.layout = Qt.QHBoxLayout()
        self.radio = Qt.QRadioButton('empty')
        self.layout.addWidget(self.radio)
        self.areapointlist.button_group.addButton(self.radio)
        self.radio.setChecked(True)
        self.areapointlist.layout.addLayout(self.layout)
    
    def set_point(self, PointId, ss, renderer):
        """
        Set the point in the singlescan that we've picked

        Parameters
        ----------
        PointId : int
            PointId, see singlescan's pedigree ids.
        ss : pydar.SingleScan
            SingleScan object that the point belongs too.
        renderer : vtk.vtkRenderer
            Renderer object for the render window
            
        Returns
        -------
        None.

        """
        
        self.PointId = PointId
        
        if self.empty:
            # If we were empty to begin with create buttons
            up = Qt.QPushButton("up")
            up.setMinimumWidth(40)
            self.layout.addWidget(up)
            up.clicked.connect(self.move_up)
            down = Qt.QPushButton("down")
            down.setMinimumWidth(40)
            self.layout.addWidget(down)
            down.clicked.connect(self.move_down)
            delete = Qt.QPushButton("del")
            delete.setMinimumWidth(40)
            self.layout.addWidget(delete)
            delete.clicked.connect(self.delete)
            ins = Qt.QPushButton("ins")
            ins.setMinimumWidth(40)
            self.layout.addWidget(ins)
            ins.clicked.connect(self.insert_below)
            # Create a new empty area point
            self.areapointlist.add_areapoint()
            self.empty = False
            # Connect with toggled slot
            self.radio.toggled.connect(self.toggled)
        else:
            # remove the existing point from the renderer
            renderer.RemoveActor(self.actor)
            del self.actor
        
        # Set the button text to the single scan name
        self.radio.setText(ss.scan_name)
        # Create VTK Selection pipeline
        selectionList = numpy_to_vtk(np.array([PointId], dtype=np.uint32),
                                     deep=True, 
                                     array_type=vtk.VTK_UNSIGNED_INT)
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
        selectionNode.SetContentType(vtk.vtkSelectionNode.PEDIGREEIDS)
        selectionNode.SetSelectionList(selectionList)
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)
        extractSelection = vtk.vtkExtractSelection()
        extractSelection.SetInputData(1, selection)
        extractSelection.SetInputConnection(0, ss.currentFilter
                                            .GetOutputPort())
        extractSelection.Update()
        geoFilter = vtk.vtkGeometryFilter()
        geoFilter.SetInputConnection(extractSelection.GetOutputPort())
        geoFilter.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(geoFilter.GetOutputPort())
        mapper.SetScalarVisibility(0)
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().RenderPointsAsSpheresOn()
        self.actor.GetProperty().SetPointSize(20)
        if self.radio.isChecked():
            self.actor.GetProperty().SetColor(0.5, 0.5, 0.5)
        else:
            self.actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        renderer.AddActor(self.actor)
        
        # Update polyline
        self.areapointlist.update()
    
    def move_up(self, s):
        """
        Tell our areapointlist to move this item up

        Parameters
        ----------
        s : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.areapointlist.move(self, 'up')
    
    def move_down(self, s):
        """
        Tell our areapointlist to move this item down

        Parameters
        ----------
        s : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.areapointlist.move(self, 'down')
    
    def delete(self, s):
        """
        

        Parameters
        ----------
        s : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.areapointlist.delete(self)
    
    def insert_below(self, s):
        """
        

        Parameters
        ----------
        s : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.areapointlist.insert_below(self)
    
    def toggled(self, checked):
        """
        Change the color when we toggle the radio button

        Parameters
        ----------
        checked : bool
            Whether radio button is now checked or unchecked.

        Returns
        -------
        None.

        """
        
        if checked:
            #print('toggled on')
            self.actor.GetProperty().SetColor(0.5, 0.5, 0.5)
        else:
            #print('toggled off')
            self.actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        self.areapointlist.vtkWidget.GetRenderWindow().Render()

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
        # Some additions here to make this scrollable
        opt_scroll = Qt.QScrollArea()
        opt_scroll.setWidgetResizable(True)
        opt_inner = Qt.QFrame(opt_scroll)
        opt_layout = Qt.QVBoxLayout()
        opt_inner.setLayout(opt_layout)
        opt_scroll.setWidget(opt_inner)
        
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
        temp_layout = Qt.QHBoxLayout()
        temp_layout.addWidget(Qt.QLabel("Points Suffix:"))
        self.proj_suffix_0 = Qt.QLineEdit('')
        temp_layout.addWidget(self.proj_suffix_0)
        opt_layout.addLayout(temp_layout)
        temp_layout = Qt.QHBoxLayout()
        temp_layout.addWidget(Qt.QLabel("Trans Suffix:"))
        self.proj_t_suffix_0 = Qt.QLineEdit('')
        temp_layout.addWidget(self.proj_t_suffix_0)
        opt_layout.addLayout(temp_layout)
        proj_label_1 = Qt.QLabel("Project 1: ")
        self.proj_combobox_1 = Qt.QComboBox()
        self.proj_combobox_1.setEnabled(0)
        self.proj_combobox_1.setSizeAdjustPolicy(0)
        opt_layout.addWidget(proj_label_1)
        opt_layout.addWidget(self.proj_combobox_1)
        temp_layout = Qt.QHBoxLayout()
        temp_layout.addWidget(Qt.QLabel("Points Suffix:"))
        self.proj_suffix_1 = Qt.QLineEdit('')
        temp_layout.addWidget(self.proj_suffix_1)
        opt_layout.addLayout(temp_layout)
        temp_layout = Qt.QHBoxLayout()
        temp_layout.addWidget(Qt.QLabel("Trans Suffix:"))
        self.proj_t_suffix_1 = Qt.QLineEdit('')
        temp_layout.addWidget(self.proj_t_suffix_1)
        opt_layout.addLayout(temp_layout)
        
        # Button to prompt us to select a project
        self.sel_proj_button = Qt.QPushButton("Select Project(s)")
        self.sel_proj_button.setEnabled(0)
        opt_layout.addWidget(self.sel_proj_button)
        
        # Create combobox to select which scan to display
        self.scan_combobox = Qt.QComboBox()
        self.scan_combobox.setEnabled(0)
        self.scan_combobox.setSizeAdjustPolicy(0)
        opt_layout.addWidget(self.scan_combobox)
        
        # Create interface for maxima alignment
        opt_layout.addWidget(Qt.QLabel('Max Alignment'))
        temp_layout = Qt.QHBoxLayout()
        temp_layout.addWidget(Qt.QLabel('cell_w: '))
        self.max_align_w = Qt.QLineEdit('5.0')
        self.max_align_w.setValidator(Qt.QDoubleValidator())
        temp_layout.addWidget(self.max_align_w)
        opt_layout.addLayout(temp_layout)
        temp_layout = Qt.QHBoxLayout()
        temp_layout.addWidget(Qt.QLabel('p_thresh: '))
        self.max_align_p = Qt.QLineEdit('0.1')
        self.max_align_p.setValidator(Qt.QDoubleValidator())
        temp_layout.addWidget(self.max_align_p)
        opt_layout.addLayout(temp_layout)
        temp_layout = Qt.QHBoxLayout()
        temp_layout.addWidget(Qt.QLabel('az_thresh: '))
        self.max_align_a = Qt.QLineEdit('0.0008')
        self.max_align_a.setValidator(Qt.QDoubleValidator())
        temp_layout.addWidget(self.max_align_a)
        opt_layout.addLayout(temp_layout)
        temp_layout = Qt.QHBoxLayout()
        temp_layout.addWidget(Qt.QLabel('z_intcpt: '))
        self.max_align_zi = Qt.QLineEdit('0.02')
        self.max_align_zi.setValidator(Qt.QDoubleValidator())
        temp_layout.addWidget(self.max_align_zi)
        opt_layout.addLayout(temp_layout)
        temp_layout = Qt.QHBoxLayout()
        temp_layout.addWidget(Qt.QLabel('z_slope: '))
        self.max_align_zs = Qt.QLineEdit('0.001')
        self.max_align_zs.setValidator(Qt.QDoubleValidator())
        temp_layout.addWidget(self.max_align_zs)
        opt_layout.addLayout(temp_layout)
        max_align_button = Qt.QPushButton("Compute Max Align")
        opt_layout.addWidget(max_align_button)
        self.count_label = Qt.QLabel('Keypoint Pair Count: nan')
        opt_layout.addWidget(self.count_label)
        # Let's show the output as changes from current
        max_align_list = ['ddx', 'ddy', 'ddz', 'droll', 'dpitch', 'dyaw']
        self.max_align_dict = {}
        for param in max_align_list:
            temp_layout = Qt.QHBoxLayout()
            temp_label = Qt.QLabel(param + ": ")
            self.max_align_dict[param] = Qt.QLineEdit('0.0')
            self.max_align_dict[param].setValidator(Qt.QDoubleValidator())
            temp_layout.addWidget(temp_label)
            temp_layout.addWidget(self.max_align_dict[param])
            opt_layout.addLayout(temp_layout)
        align_update_button = Qt.QPushButton("Apply Changes")
        opt_layout.addWidget(align_update_button)
        
        # Create interface for Gridded alignment
        opt_layout.addWidget(Qt.QLabel('Gridded z Alignment'))
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
        temp_layout = Qt.QHBoxLayout()
        temp_layout.addWidget(Qt.QLabel('droll: '))
        self.diff_roll = Qt.QPushButton('')
        temp_layout.addWidget(self.diff_roll)
        opt_layout.addLayout(temp_layout)
        temp_layout = Qt.QHBoxLayout()
        temp_layout.addWidget(Qt.QLabel('dpitch: '))
        self.diff_pitch = Qt.QPushButton('')
        temp_layout.addWidget(self.diff_pitch)
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
        copy_transform_button = Qt.QPushButton("Copy Transform")
        opt_layout.addWidget(copy_transform_button)
        update_param_button = Qt.QPushButton("Update Transform")
        opt_layout.addWidget(update_param_button)
        reset_param_button = Qt.QPushButton("Reset Transform")
        opt_layout.addWidget(reset_param_button)
        
        # Create a second options layout with options for visualization
        opt_layout_2 = Qt.QVBoxLayout()
        opt_layout_2.addWidget(Qt.QLabel('Classes to Display:'))
        self.class_layout = Qt.QVBoxLayout()
        class_group_box = Qt.QGroupBox()
        # Dictionary object containing checkboxes keyed on classes (or 'all')
        self.class_check_dict = {}
        # Add 'all' class 
        self.class_check_dict['all'] = Qt.QCheckBox('all')
        self.class_check_dict['all'].setChecked(1)
        self.class_layout.addWidget(self.class_check_dict['all'])
        class_group_box.setLayout(self.class_layout)
        opt_layout_2.addWidget(class_group_box)
        update_class_button = Qt.QPushButton('Update Class Display')
        opt_layout_2.addWidget(update_class_button)
        # Add interface for displaying elevation instead of solid color
        elev_layout = Qt.QHBoxLayout()
        self.elev_checkbox = Qt.QCheckBox("Elev. Range: ")
        self.z_min = Qt.QLineEdit("-5.0")
        self.z_min.setValidator(Qt.QDoubleValidator())
        self.z_max = Qt.QLineEdit("-1.0")
        self.z_max.setValidator(Qt.QDoubleValidator())
        elev_layout.addWidget(self.elev_checkbox)
        elev_layout.addWidget(self.z_min)
        elev_layout.addWidget(self.z_max)
        opt_layout_2.addLayout(elev_layout)
        # Add interface for creating enclosed areas
        opt_layout_2.addWidget(Qt.QLabel('Pointwise Area Selection'))
        self.edit_area_check = Qt.QCheckBox('Edit Area Points')
        opt_layout_2.addWidget(self.edit_area_check)
        self.area_point_list = AreaPointList(self.vtkWidget)
        opt_layout_2.addWidget(self.area_point_list.get_scroll())
        # Add buttons for copying selected points
        copy_areapoints_button = Qt.QPushButton('Copy areapoints')
        opt_layout_2.addWidget(copy_areapoints_button)
        # copy_cornercoords_button = Qt.QPushButton('Copy cornercoords')
        # opt_layout_2.addWidget(copy_cornercoords_button)
        sel_area_dir_button = Qt.QPushButton("Select areapoint dir")
        # Create the file dialog that we'll use
        self.area_dialog = Qt.QFileDialog(self)
        self.area_dialog.setDirectory('/media/thayer/Data/mosaic_lidar/')
        self.area_dialog.setFileMode(4) # set file mode to pick directories
        opt_layout_2.addWidget(sel_area_dir_button)
        self.area_filename_lineedit = Qt.QLineEdit('areapoint filename')
        opt_layout_2.addWidget(self.area_filename_lineedit)
        save_areapoint_button = Qt.QPushButton('Save areapoints')
        opt_layout_2.addWidget(save_areapoint_button)
        delete_areapoints_button = Qt.QPushButton('Delete areapoints')
        opt_layout_2.addWidget(delete_areapoints_button)
        load_areapoint_button = Qt.QPushButton('Load areapoints')
        opt_layout_2.addWidget(load_areapoint_button)
        # Create interface for rubberband selection
        rb_layout = Qt.QHBoxLayout()
        self.rb_checkbox = Qt.QCheckBox("Rubberband Pick")
        self.rb_checkbox.setEnabled(0)
        self.near_label = Qt.QLineEdit('0.1')
        self.near_label.setValidator(Qt.QDoubleValidator())
        self.far_label = Qt.QLineEdit('1000.0')
        self.far_label.setValidator(Qt.QDoubleValidator())
        rb_layout.addWidget(self.rb_checkbox)
        rb_layout.addWidget(self.near_label)
        rb_layout.addWidget(self.far_label)
        opt_layout_2.addLayout(rb_layout)
        rb_layout_2 = Qt.QHBoxLayout()
        rb_layout_2.addWidget(Qt.QLabel('Classification: '))
        self.class_lineedit = Qt.QLineEdit('0')
        self.class_lineedit.setValidator(Qt.QIntValidator(0, 255))
        rb_layout_2.addWidget(self.class_lineedit)
        clear_button = Qt.QPushButton("Clear")
        rb_layout_2.addWidget(clear_button)
        self.class_button = Qt.QPushButton("Classify")
        self.class_button.setEnabled(0)
        rb_layout_2.addWidget(self.class_button)

        opt_layout_2.addLayout(rb_layout_2)


        # Populate the main layout
        main_layout.addWidget(vis_splitter, stretch=5)
        #main_layout.addLayout(opt_layout)
        main_layout.addWidget(opt_scroll)
        main_layout.addLayout(opt_layout_2)
        
        # Set layout for the frame and set central widget
        self.frame.setLayout(main_layout)
        self.setCentralWidget(self.frame)
        
        # Signals and slots
        # vis_tools
        look_down_button.clicked.connect(self.look_down)
        for t in t_list:
            self.transect_dict[t].editingFinished.connect(
                self.update_trans_endpoints)
        plot_transect_button.clicked.connect(self.plot_transect)
        # opt_layout
        self.sel_scan_area_button.clicked.connect(
            self.on_sel_scan_area_button_click)
        self.sel_proj_button.clicked.connect(self.on_sel_proj_button_click)
        self.proj_dialog.fileSelected.connect(self.on_scan_area_selected)
        self.scan_combobox.currentTextChanged.connect(self.on_scan_changed)
        max_align_button.clicked.connect(self.on_max_align_button_clicked)
        align_update_button.clicked.connect(self.on_align_update_button_clicked)
        z_align_button.clicked.connect(self.on_z_align_button_clicked)
        self.diff_mode_buttongroup.buttonPressed.connect(self.diff_mode_changed)
        z_update_button.clicked.connect(self.on_z_update_button_clicked)
        copy_transform_button.clicked.connect(
            self.on_copy_transform_button_click)
        update_param_button.clicked.connect(self.on_update_param_button_click)
        reset_param_button.clicked.connect(self.on_reset_param_button_click)
        # opt_layout_2
        update_class_button.clicked.connect(self.on_update_class_button_click)
        self.elev_checkbox.toggled.connect(self.on_elev_checkbox_toggled)
        self.z_min.editingFinished.connect(self.on_z_edit)
        self.z_max.editingFinished.connect(self.on_z_edit)
        self.edit_area_check.toggled.connect(self.on_edit_area_check_toggled)
        copy_areapoints_button.clicked.connect(
            self.on_copy_areapoints_button_click)
        sel_area_dir_button.clicked.connect(self.on_sel_area_dir_button_click)
        self.area_dialog.fileSelected.connect(self.on_area_dir_selected)
        save_areapoint_button.clicked.connect(
            self.on_save_areapoint_button_click)
        delete_areapoints_button.clicked.connect(
            self.on_delete_areapoints_button)
        load_areapoint_button.clicked.connect(self.on_load_areapoint_button)
        self.rb_checkbox.toggled.connect(self.on_rb_checkbox_toggled)
        self.near_label.editingFinished.connect(self.on_clip_changed)
        self.far_label.editingFinished.connect(self.on_clip_changed)
        self.class_button.clicked.connect(self.on_classify_button_click)
        clear_button.clicked.connect(self.on_clear_button_click)
        
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
        self.selected_actor.GetProperty().SetColor(1, 1, 1)
        self.selected_actor.GetProperty().SetPointSize(2.0)

        # Renderer and interactor
        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(actor_pt_0)
        self.renderer.AddActor(actor_pt_1)
        self.renderer.AddActor(actor_pt_2)
        self.renderer.AddActor(actor_pt_3)
        self.renderer.AddActor(self.selected_actor)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtkWidget.GetRenderWindow().AddObserver("ModifiedEvent", 
                                                     self.
                                                     on_modified_renderwindow)
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.pointPicker = vtk.vtkPointPicker()
        self.pointPicker.SetTolerance(0.01)
        self.pointPicker.AddObserver("EndPickEvent", self.on_end_pick)
        self.areaPicker = vtk.vtkAreaPicker()
        self.areaPicker.AddObserver("EndPickEvent", self.on_area_pick)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.SetPicker(self.pointPicker)
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
                                   las_fieldnames=['Points', 'Classification',
                                                   'PointId'], 
                                   class_list='all', create_id=False,
                                   suffix=self.proj_suffix_0.text())
        self.scan_area.project_dict[project_name_0].load_man_class()
        self.scan_area.project_dict[project_name_0].apply_man_class()
        self.scan_area.project_dict[project_name_0].read_transforms(
            suffix=self.proj_t_suffix_0.text())
        self.scan_area.project_dict[project_name_0].apply_transforms(
            ['current_transform'])
        
        if not project_name_0==project_name_1:
            self.scan_area.add_project(project_name_1, import_mode=
                                       'read_scan', las_fieldnames=['Points',
                                        'Classification', 'PointId'],
                                       class_list='all', create_id=False,
                                       suffix=self.proj_suffix_1.text())
            self.scan_area.project_dict[project_name_1].load_man_class()
            self.scan_area.project_dict[project_name_1].apply_man_class()
            self.scan_area.project_dict[project_name_1].read_transforms(
                suffix=self.proj_t_suffix_1.text())
            self.scan_area.project_dict[project_name_1].apply_transforms(
                ['current_transform'])
        else:
            # Only enable rubberband selection if we are just looking at 1 proj
            self.rb_checkbox.setEnabled(1)
        
        self.project_0 = self.scan_area.project_dict[project_name_0]
        self.project_1 = self.scan_area.project_dict[project_name_1]

        # Create appendPolyData, Mappers and actors in selection dicts
        for scan_name in self.project_0.scan_dict:
            # Create appendPolyData, mappers and actors in selection dicts
            self.selected_poly_dict[(project_name_0, scan_name)
                ] = vtk.vtkAppendPolyData()
            pdata = vtk.vtkPolyData()
            self.selected_poly_dict[(project_name_0, scan_name)].AddInputData(
                pdata)
            self.selected_append.AddInputConnection(
                self.selected_poly_dict[(project_name_0, scan_name)
                ].GetOutputPort())

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
        
        # Add class checkboxes for every class present in the projects
        class_arr = np.empty(0, dtype='int')
        for scan_name in self.project_0.scan_dict:
            class_arr = np.concatenate((class_arr, np.unique(vtk_to_numpy(
                self.project_0.scan_dict[scan_name].polydata_raw
                .GetPointData().GetArray('Classification')))))
        if not project_name_0==project_name_1:
            for scan_name in self.project_1.scan_dict:
                class_arr = np.concatenate((class_arr, np.unique(vtk_to_numpy(
                    self.project_1.scan_dict[scan_name].polydata_raw
                    .GetPointData().GetArray('Classification')))))
        class_arr = np.unique(class_arr)
        for c in class_arr:
            self.class_check_dict[c] = Qt.QCheckBox(str(c))
            self.class_check_dict[c].setChecked(0)
            self.class_layout.addWidget(self.class_check_dict[c])

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
                if self.elev_checkbox.isChecked():
                    self.ss.create_elevation_pipeline(float(self.z_min.text()),
                                                      float(self.z_max.text()))
                else:
                    self.ss.create_solid_pipeline('cyan')
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
        if self.elev_checkbox.isChecked():
            self.ss.create_elevation_pipeline(float(self.z_min.text()),
                                              float(self.z_max.text()))
        else:
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
    
    def on_max_align_button_clicked(self, s):
        """
        Compute the adjustment to the current transform that would align
        the singlescan with the other project on the basis of shared maxima.

        Parameters
        ----------
        s : int
            Button status. Not used.

        Returns
        -------
        None.

        """
        
        # Get hyperparameters
        w0 = w1 = float(self.max_align_w.text())
        #max_diff = float(self.max_align_d.text())
        p_thresh = float(self.max_align_p.text())
        az_thresh = float(self.max_align_a.text())
        z_intcpt = float(self.max_align_zi.text())
        z_slope = float(self.max_align_zs.text())
        
        # Compute alignment
        A, _, count = self.scan_area.max_alignment_ss(
            self.project_0.project_name, self.project_1.project_name, 
            self.ss.scan_name, w0=w0, w1=w1, max_diff=None, 
            return_count=True, use_closest=True, p_thresh=p_thresh,
            az_thresh=az_thresh, z_intcpt=z_intcpt, z_slope=z_slope)
        
        # Update count label
        self.count_label.setText('Keypoint Pair Count: ' + str(count))
        
        # Compute the difference between new transform and current
        # Go from numpy 4x4 matrix to vtkTransform
        # Create vtk transform object
        vtk4x4 = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk4x4.SetElement(i, j, A[i, j])
        new_transform = vtk.vtkTransform()
        new_transform.SetMatrix(vtk4x4)
        # Full transform
        full_transform = vtk.vtkTransform()
        full_transform.PostMultiply()
        full_transform.Concatenate(self.ss.transform)
        full_transform.Concatenate(new_transform)
        pos = full_transform.GetPosition()
        ori = np.array(full_transform.GetOrientation()) * np.pi / 180
        
        # Update max_align_dict's deltas
        param_list = ['dx', 'dy', 'dz', 'roll', 'pitch', 'yaw']
        param_arr = np.concatenate((pos, ori))
        for i in range(6):
            self.max_align_dict['d'+param_list[i]].setText(str(round(
                param_arr[i] - float(self.param_dict[param_list[i]].text()),
                6)))
        
    def on_align_update_button_clicked(self, s):
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
        
        # Update each parameter
        param_list = ['dx', 'dy', 'dz', 'roll', 'pitch', 'yaw']
        for param in param_list:
            self.param_dict[param].setText(str(
                float(self.param_dict[param].text()) +
                float(self.max_align_dict['d'+param].text())))
        
        # Update transformation applied to scan
        self.on_update_param_button_click(1)
        
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
        frac_exceed_max_diff, diff, grid = pydar.z_alignment_ss(
            self.project_0, self.ss, float(self.z_align_dict['cell_w'].text()),
            float(self.z_align_dict['cell_w'].text()), 
            float(self.z_align_dict['min_dens'].text()), 
            float(self.z_align_dict['max_diff'].text()),
            bin_reduc_op=self.z_align_mode.currentText(), return_grid=True)
        # Update text output
        self.frac_exceed_label.setText('frac>max_diff: ' 
                                       + str(frac_exceed_max_diff))
        # Compute the least squares fit
        ind = np.logical_not(np.isnan(diff.ravel()))
        A = np.hstack((np.ones((ind.sum(),1)), grid[ind,:2]))
        b = diff.ravel()[ind, np.newaxis]
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Set the text for each of the diff_mode buttons
        diff_notnan = np.ravel(diff)[np.logical_not(np.isnan(
                        np.ravel(diff)))]
        self.diff_mode_dict['mean'].setText(str(round(
            -1*diff_notnan.mean(), 3)))
        self.diff_mode_dict['median'].setText(str(round(
            -1*np.median(diff_notnan), 3)))
        m, _ = mode(np.around(diff_notnan, 3))
        self.diff_mode_dict['mode'].setText(str(round(-1*m[0], 3)))
        
        # Note that because of the righthand rule, the change we want is the
        # negative slope with respect to y for droll but the positive slope
        # with respect to x for dpitch
        self.diff_roll.setText(str(round(-1*x[2,0], 6)))
        self.diff_pitch.setText(str(round(x[1,0], 6)))
        
        # Update plot
        # Clear plot canvas
        self.mpl_widget.axes.cla()
        self.mpl_widget.axes.contourf(grid[:,0].reshape(diff.shape), 
             grid[:,1].reshape(diff.shape), diff, vmin=-0.02, vmax=0.02, 
             cmap='RdBu_r', antialiased=False)
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
    
    def on_copy_transform_button_click(self, s):
        """
        Copy function call to the clipboard with the current transform

        Parameters
        ----------
        s : int
            Button status. Not used

        Returns
        -------
        None.

        """
        
        output = ('replace_current_transform(\n    "' + self.ss.project_path 
                  + '",\n    "' + self.ss.project_name + '",\n    "' +
                  self.ss.scan_name + '",\n    "' + self.proj_t_suffix_1.text()
                  + '",\n    ' + self.param_dict['dx'].text() + ",\n    " +
                  self.param_dict['dy'].text() + ",\n    " +
                  self.param_dict['dz'].text() + ",\n    " +
                  self.param_dict['roll'].text() + ",\n    " +
                  self.param_dict['pitch'].text() + ",\n    " +
                  self.param_dict['yaw'].text() + ")\n" )
        pyperclip.copy(output)
        
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
        #self.vtkWidget.GetRenderWindow().Render()
        # Update area point list
        self.area_point_list.update()


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
        #self.vtkWidget.GetRenderWindow().Render()
        # Update area point list
        self.area_point_list.update()
        
    def on_update_class_button_click(self, s):
        """
        Update the classes we are filtering in current filter

        Parameters
        ----------
        s : int
            Button status. Not used.

        Returns
        -------
        None.

        """
        
        # First check if the 'all' button is checked
        if self.class_check_dict['all'].isChecked():
            class_list = 'all'
        # Otherwise get all checked classes
        else:
            class_list = []
            for c in self.class_check_dict:
                if self.class_check_dict[c].isChecked():
                    class_list.append(c)
        
        # Update current filter
        for scan_name in self.project_0.scan_dict:
            self.project_0.scan_dict[scan_name].update_current_filter(
                class_list)
            self.renderer.RemoveActor(self.project_0.scan_dict[scan_name]
                                      .actor)
            self.project_0.scan_dict[scan_name].create_solid_pipeline('Cyan')
            self.renderer.AddActor(self.project_0.scan_dict[scan_name].actor)
        if self.project_0==self.project_1:
            self.ss.update_current_filter(class_list)
        else:
            for scan_name in self.project_1.scan_dict:
                self.project_1.scan_dict[scan_name].update_current_filter(
                    class_list)
        self.renderer.RemoveActor(self.ss.actor)
        self.ss.create_solid_pipeline('Lime')
        self.renderer.AddActor(self.ss.actor)
        self.vtkWidget.GetRenderWindow().Render()
    
    def on_elev_checkbox_toggled(self, checked):
        """
        Switch visualization to color by elevation

        Parameters
        ----------
        checked : bool
            Check button state.

        Returns
        -------
        None.

        """
        
        if checked:
            for scan_name in self.project_0.scan_dict:
                self.renderer.RemoveActor(self.project_0.scan_dict[scan_name]
                                          .actor)
                self.project_0.scan_dict[scan_name].create_elevation_pipeline(
                    float(self.z_min.text()), float(self.z_max.text()))
                self.renderer.AddActor(self.project_0.scan_dict[scan_name]
                                       .actor)
            self.renderer.RemoveActor(self.ss.actor)
            if self.project_0==self.project_1:
                self.ss.create_elevation_pipeline(float(self.z_min.text()), 
                                                  float(self.z_max.text()))
            else:
                for scan_name in self.project_1.scan_dict:
                    (self.project_1.scan_dict[scan_name]
                     .create_elevation_pipeline(float(self.z_min.text()), 
                                                float(self.z_max.text())))
            self.renderer.AddActor(self.ss.actor)
        else:
            for scan_name in self.project_0.scan_dict:
                self.renderer.RemoveActor(self.project_0.scan_dict[scan_name]
                                          .actor)
                self.project_0.scan_dict[scan_name].create_solid_pipeline(
                    "cyan")
                self.renderer.AddActor(self.project_0.scan_dict[scan_name]
                                       .actor)
            self.renderer.RemoveActor(self.ss.actor)
            if self.project_0==self.project_1:
                self.ss.create_solid_pipeline("lime")
            else:
                for scan_name in self.project_1.scan_dict:
                    self.project_1.scan_dict[scan_name].create_solid_pipeline(
                        "lime")
            self.renderer.AddActor(self.ss.actor)
        self.vtkWidget.GetRenderWindow().Render()
    
    def on_z_edit(self):
        """
        Update elevation colorscale

        Returns
        -------
        None.

        """
        
        if self.elev_checkbox.isChecked():
            if float(self.z_min.text())<float(self.z_max.text()):
                for scan_name in self.project_0.scan_dict:
                    self.project_0.scan_dict[scan_name].mapper.SetLookupTable(
                        pydar.mplcmap_to_vtkLUT(float(self.z_min.text()),
                                                float(self.z_max.text())))
                    self.project_0.scan_dict[scan_name].mapper.SetScalarRange(
                        float(self.z_min.text()), float(self.z_max.text()))
                if self.project_0==self.project_1:
                    self.ss.mapper.SetLookupTable(
                        pydar.mplcmap_to_vtkLUT(float(self.z_min.text()),
                                                float(self.z_max.text())))
                    self.ss.mapper.SetScalarRange(
                        float(self.z_min.text()), float(self.z_max.text()))
                else:
                    for scan_name in self.project_1.scan_dict:
                        self.project_1.scan_dict[scan_name].mapper.SetLookupTable(
                            pydar.mplcmap_to_vtkLUT(float(self.z_min.text()),
                                                    float(self.z_max.text())))
                        self.project_1.scan_dict[scan_name].mapper.SetScalarRange(
                            float(self.z_min.text()), float(self.z_max.text()))
                self.vtkWidget.GetRenderWindow().Render()
    
    def on_edit_area_check_toggled(self, checked):
        """
        Switch what we want picking a point to do from transects to area and
        vice versa.

        Parameters
        ----------
        checked : bool
            Check button state.

        Returns
        -------
        None.

        """
        
        if checked:
            # Disable transect endpoint selection
            for i in range(4):
                self.point_button_group.button(i).setEnabled(0)
            # Enable area selection
        else:
            # Enable transect endpoint selection
            for i in range(4):
                self.point_button_group.button(i).setEnabled(1)
            # Disable area selection
        
    def on_copy_areapoints_button_click(self, s):
        """
        copy areapointlist to clipboard in convenient format

        Parameters
        ----------
        s : int
            Button status. Not used.

        Returns
        -------
        None.

        """
        
        self.area_point_list.copy_areapoints(self.ss.project_name)
    
    def on_sel_area_dir_button_click(self, s):
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
        
        self.area_dialog.exec_()
        
    def on_area_dir_selected(self, dir_str):
        """
        Save the area directory path for when we want to save areapoints

        Parameters
        ----------
        dir_str : str
            filepath we just selected.

        Returns
        -------
        None.

        """
        
        self.areapoint_dir_str = dir_str
    
    def on_save_areapoint_button_click(self, s):
        """
        Write areapoints as a json formatted string to a file

        Parameters
        ----------
        s : int
            Button status not used.

        Returns
        -------
        None.

        """
        
        try:
            path = os.path.join(self.areapoint_dir_str, 
                                self.area_filename_lineedit.text())
            self.area_point_list.save_areapoints(self.ss.project_name, path)
        except:
            warnings.warn('Save areapoints failed. Have you selected a '
                          'directory?')
    
    def on_delete_areapoints_button(self, s):
        """
        Delete all areapoints

        Parameters
        ----------
        s : int
            Button status. Not used

        Returns
        -------
        None.

        """
        
        while len(self.area_point_list.list)>1:
            self.area_point_list.delete(self.area_point_list.list[0])

    def on_load_areapoint_button(self, s):
        """
        Load areapoints from file. Most useful if we have emptied the list
        first

        Parameters
        ----------
        s : int
            Button status not used.

        Returns
        -------
        None.

        """
        
        # Try loading the file
        try:
            path = os.path.join(self.areapoint_dir_str, 
                                self.area_filename_lineedit.text())
            f = open(path, 'r')
            areapoint_dict = json.load(f)
            f.close()
        except:
            warnings.warn('File not found. Aborting')
            return
        
        # Check that the project matches our project_1
        if not (self.project_1.project_name in areapoint_dict):
            warnings.warn('areapoints project does not match project 1.'
                          + ' Aborting.')
            return
        
        # For each areapoint in the list add it to self.area_point_list
        # the only tricky part here is that we need to get the singlescan
        # associated with it.
        for scan_name, PointId in areapoint_dict[self.project_1.project_name]:
            if scan_name==self.ss.scan_name:
                ss = self.ss
            else:
                ss = self.project_1.scan_dict[scan_name]
            self.area_point_list.list[-1].set_point(PointId, ss, self.renderer)
        self.vtkWidget.GetRenderWindow().Render()

    def on_rb_checkbox_toggled(self, checked):
        """
        Turn rubberband area selection on or off

        Parameters:
        -----------
        checked : bool
            Whether the check is on or off

        Returns:
        --------
        None.

        """

        if checked:
            self.iren.SetPicker(self.areaPicker)
            self.iren.SetInteractorStyle(vtk.vtkInteractorStyleRubberBandPick())
        else:
            self.iren.SetPicker(self.pointPicker)
            self.iren.SetInteractorStyle(
                vtk.vtkInteractorStyleTrackballCamera())


    def on_clip_changed(self):
        """
        

        Returns
        -------
        None.

        """
        
        self.renderer.GetActiveCamera().SetClippingRange(
            float(self.near_label.text()), float(self.far_label.text()))
        self.vtkWidget.GetRenderWindow().Render()

    def on_clear_button_click(self, s):
        """
        Clears out the selected points from the viewport and dicts.

        Returns
        -------
        None.

        """
        
        # Clear the inputs from each appendPolyData in the dict
        for key in self.selected_poly_dict:
            self.selected_poly_dict[key].RemoveAllInputs()
            pdata = vtk.vtkPolyData()
            self.selected_poly_dict[key].AddInputData(pdata)
            self.selected_poly_dict[key].Update()
        
        self.vtkWidget.GetRenderWindow().Render()
        self.class_button.setEnabled(0)

    def on_classify_button_click(self, s):
        """
        Adds the classified points to man_class and updates classifications

        Returns:
        --------
        None.

        """

        for key in self.selected_poly_dict:
            if (self.selected_poly_dict[key].GetOutput().GetNumberOfPoints()>0):
                # Get the singlescan that this poly came from
                if key[1]==self.ss.scan_name:
                    ss = self.ss
                else:
                    ss = self.project_0.scan_dict[key[1]]
                # Add points to man_class table
                c = np.uint8(self.class_lineedit.text())
                ss.update_man_class(self.selected_poly_dict[key].GetOutput(), c)
                # And update
                ss.apply_man_class()
        # Clear points
        self.on_clear_button_click(1)

        if not c in self.class_check_dict.keys():
            self.class_check_dict[c] = Qt.QCheckBox(str(c))
            self.class_check_dict[c].setChecked(0)
            self.class_layout.addWidget(self.class_check_dict[c])



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
        
        if self.edit_area_check.isChecked():
            # Find the pedigree id of the point in the current singlescan
            # That's closest to the picked point (usually this will be picked)
            # point
            ind = self.ss.currentFilter.GetOutput().FindPoint(pt)
            PointId = vtk_to_numpy(self.ss.currentFilter.GetOutput()
                                   .GetPointData().GetPedigreeIds())[ind]
            # Find the checked area point and set it
            for ap in self.area_point_list.list:
                if ap.radio.isChecked():
                    ap.set_point(PointId, self.ss, self.renderer)
                    self.vtkWidget.GetRenderWindow().Render()
                    break
            
        else:
            # We are updating the transect endpoints
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

    def on_area_pick(self, obj, event):
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
        
        for scan_name in self.project_0.scan_dict:
                
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
            extractSelection.SetInputData(0, self.project_0.scan_dict
                                          [scan_name].mapper.GetInput())
            extractSelection.SetInputData(1, selection)
            extractSelection.Update()
            vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
            vertexGlyphFilter.SetInputConnection(
                extractSelection.GetOutputPort())
            vertexGlyphFilter.Update()
            
            if vertexGlyphFilter.GetOutput().GetNumberOfPoints()>0:
                pts_selected = True
            
            # Add selected points to appropriate appendPolyData
            self.selected_poly_dict[(self.project_0.project_name, scan_name)
                ].AddInputData(vertexGlyphFilter.GetOutput())

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
        extractSelection.SetInputData(0, self.ss.mapper.GetInput())
        extractSelection.SetInputData(1, selection)
        extractSelection.Update()
        vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
        vertexGlyphFilter.SetInputConnection(
            extractSelection.GetOutputPort())
        vertexGlyphFilter.Update()
        
        if vertexGlyphFilter.GetOutput().GetNumberOfPoints()>0:
            pts_selected = True
        
        # Add selected points to appropriate appendPolyData
        self.selected_poly_dict[(self.ss.project_name, self.ss.scan_name)
            ].AddInputData(vertexGlyphFilter.GetOutput())
        
        # If we selected any points enable class_button
        if pts_selected:
            self.class_button.setEnabled(1)

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