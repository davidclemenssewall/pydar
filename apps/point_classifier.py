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
import pyperclip
import warnings
import json
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from PyQt5 import QtCore, QtGui
from PyQt5 import Qt
from collections import namedtuple
import re
import os
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

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
        
        print(output)
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
        self.vtkWidget.setSizePolicy(Qt.QSizePolicy.Expanding, 
                                     Qt.QSizePolicy.Expanding)
        vis_layout.addWidget(self.vtkWidget)
        vis_layout.addLayout(vis_tools_layout)
        
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
        if os.path.isdir('/media/thayer/Data/mosaic_lidar/'):
            self.proj_dialog.setDirectory('/media/thayer/Data/mosaic_lidar/')
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
        
        # Options for which data version and transform to use
        temp_layout = Qt.QHBoxLayout()
        temp_layout.addWidget(Qt.QLabel("Points Suffix:"))
        self.proj_suffix_0 = Qt.QLineEdit('')
        temp_layout.addWidget(self.proj_suffix_0)
        opt_layout.addLayout(temp_layout)
        # Lineedit to enter the Classification suffix
        temp_layout = Qt.QHBoxLayout()
        temp_layout.addWidget(Qt.QLabel('Classification suffix:'))
        self.class_lineedit = Qt.QLineEdit('')
        temp_layout.addWidget(self.class_lineedit)
        opt_layout.addLayout(temp_layout)
        trans_checkbox = Qt.QCheckBox('Use PRCS')
        opt_layout.addWidget(trans_checkbox)
        temp_layout = Qt.QHBoxLayout()
        temp_layout.addWidget(Qt.QLabel("Trans Suffix:"))
        self.proj_t_suffix_0 = Qt.QLineEdit('')
        temp_layout.addWidget(self.proj_t_suffix_0)
        opt_layout.addLayout(temp_layout)
        
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
        opt_layout.addWidget(Qt.QLabel('Classes to Display:'))
        self.class_layout = Qt.QVBoxLayout()
        class_group_box = Qt.QGroupBox()
        # Dictionary object containing checkboxes keyed on classes (or 'all')
        self.class_check_dict = {}
        # Add 'all' class 
        self.class_check_dict['all'] = Qt.QCheckBox('all')
        self.class_check_dict['all'].setChecked(1)
        self.class_layout.addWidget(self.class_check_dict['all'])
        class_group_box.setLayout(self.class_layout)
        opt_layout.addWidget(class_group_box)
        update_class_button = Qt.QPushButton('Update Class Display')
        opt_layout.addWidget(update_class_button)
        #opt_layout.addWidget(self.class_group_box)
        
        # Add write scans button
        write_button = Qt.QPushButton("Write Scans")
        opt_layout.addWidget(write_button)

        # Create a second options layout with options for visualization
        opt_layout_2 = Qt.QVBoxLayout()
        # Radio button group for picking what kind of selection we want
        self.selection_buttongroup = Qt.QButtonGroup()
        selection_groupbox = Qt.QGroupBox('Selection Type')
        vbox = Qt.QVBoxLayout()
        rect_radio = Qt.QRadioButton('Rectangular')
        rect_radio.setChecked(True)
        vbox.addWidget(rect_radio)
        self.selection_buttongroup.addButton(rect_radio, id=0)
        ap_radio = Qt.QRadioButton('Area Points')
        vbox.addWidget(ap_radio)
        self.selection_buttongroup.addButton(ap_radio, id=1)
        labels_radio = Qt.QRadioButton('Labels')
        vbox.addWidget(labels_radio)
        self.selection_buttongroup.addButton(labels_radio, id=2)
        selection_groupbox.setLayout(vbox)
        opt_layout_2.addWidget(selection_groupbox)

        # Add interface for creating enclosed areas
        opt_layout_2.addWidget(Qt.QLabel('Pointwise Area Selection'))
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
        if os.path.isdir('/media/thayer/Data/mosaic_lidar/'):
            self.area_dialog.setDirectory('/media/thayer/Data/mosaic_lidar/')
        else:
            self.area_dialog.setDirectory(os.getcwd())
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

        # Add interface for working with Labels
        opt_layout_2.addWidget(Qt.QLabel('Labels'))
        load_labels = Qt.QPushButton('Load Labels')
        opt_layout_2.addWidget(load_labels)
        self.labels_checkbox = Qt.QCheckBox('Show Labels')
        self.labels_checkbox.setEnabled(0)
        opt_layout_2.addWidget(self.labels_checkbox)
        self.label_category_combo = Qt.QComboBox()
        self.label_category_combo.setEnabled(0)
        self.label_category_combo.setEditable(1)
        self.label_category_combo.lineEdit().setPlaceholderText(
            'label category')
        opt_layout_2.addWidget(self.label_category_combo)
        self.label_subcategory_combo = Qt.QComboBox()
        self.label_subcategory_combo.setEnabled(0)
        self.label_subcategory_combo.setEditable(1)
        self.label_subcategory_combo.lineEdit().setPlaceholderText(
            'label subcategory')
        opt_layout_2.addWidget(self.label_subcategory_combo)
        self.label_id_combo = Qt.QComboBox()
        self.label_id_combo.setEnabled(0)
        self.label_id_combo.setEditable(1)
        self.label_id_combo.lineEdit().setPlaceholderText('label id')
        opt_layout_2.addWidget(self.label_id_combo)
        self.save_label_button = Qt.QPushButton('Save Label')
        self.save_label_button.setEnabled(0)
        opt_layout_2.addWidget(self.save_label_button)
        
        # Populate the main layout
        main_layout.addLayout(vis_layout, stretch=5)
        main_layout.addWidget(opt_scroll)
        main_layout.addLayout(opt_layout_2)
        
        # Set layout for the frame and set central widget
        self.frame.setLayout(main_layout)
        self.setCentralWidget(self.frame)
        
        # Signals and slots
        # vis tools
        self.field_selector.currentTextChanged.connect(
            self.on_field_selector_changed)
        self.v_min.editingFinished.connect(self.on_v_edit)
        self.v_max.editingFinished.connect(self.on_v_edit)
        self.near_label.editingFinished.connect(self.on_clip_changed)
        self.far_label.editingFinished.connect(self.on_clip_changed)
        look_down_button.clicked.connect(self.look_down)
        self.show_class_checkbox.toggled.connect(self.on_show_class_toggled)
        # options layout
        self.sel_scan_area_button.clicked.connect(
            self.on_sel_scan_area_button_click)
        self.sel_proj_button.clicked.connect(self.on_sel_proj_button_click)
        self.proj_dialog.fileSelected.connect(self.on_scan_area_selected)
        self.read_scan_box.currentTextChanged.connect(
            self.on_read_scan_box_changed)
        trans_checkbox.toggled.connect(self.on_trans_checkbox_toggled)
        self.scan_button_group.buttonToggled.connect(
            self.on_scan_checkbox_changed)
        clear_button.clicked.connect(self.on_clear_button_click)
        self.class_button.clicked.connect(self.on_class_button_click)
        train_button.clicked.connect(self.on_train_button_click)
        apply_button.clicked.connect(self.on_apply_button_click)
        update_class_button.clicked.connect(self.on_update_class_button_click)
        write_button.clicked.connect(self.on_write_button_click)
        # opt_layout_2
        #self.edit_area_check.toggled.connect(self.on_edit_area_check_toggled)
        self.selection_buttongroup.buttonClicked.connect(
            self.on_selection_changed)
        copy_areapoints_button.clicked.connect(
            self.on_copy_areapoints_button_click)
        sel_area_dir_button.clicked.connect(self.on_sel_area_dir_button_click)
        self.area_dialog.fileSelected.connect(self.on_area_dir_selected)
        save_areapoint_button.clicked.connect(
            self.on_save_areapoint_button_click)
        delete_areapoints_button.clicked.connect(
            self.on_delete_areapoints_button)
        load_areapoint_button.clicked.connect(self.on_load_areapoint_button)
        load_labels.clicked.connect(self.on_load_labels_clicked)
        self.labels_checkbox.toggled.connect(self.on_show_labels_toggled)
        self.save_label_button.clicked.connect(self.on_save_label_button)

        self.show()
        
        # VTK setup

        # Label point
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
        self.renderer.AddActor(actor_pt_0)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtkWidget.GetRenderWindow().AddObserver("ModifiedEvent", 
                                                     self.
                                                     on_modified_renderwindow)
        style = vtk.vtkInteractorStyleRubberBandPick()
        self.pointPicker = vtk.vtkPointPicker()
        self.pointPicker.SetTolerance(0.001)
        self.pointPicker.AddObserver("EndPickEvent", self.on_end_pick)
        self.areaPicker = vtk.vtkAreaPicker()
        self.areaPicker.AddObserver("EndPickEvent", self.on_area_pick)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.SetPicker(self.areaPicker)
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
            project_names = os.listdir(dir_str)
        
        # Update proj_combobox with available scans
        self.proj_combobox.addItems(project_names)
        self.proj_combobox.setEnabled(1)
        
        # Enable sel_proj_button
        self.sel_proj_button.setEnabled(1)
        
        # Disable further scan area changes
        #self.sel_scan_area_button.setEnabled(0)
        
    def on_read_scan_box_changed(self, s):
        """
        If user selects option other than read saved, make suffix not enabled.

        Parameters
        ----------
        s : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        if s=='Read Saved':
            self.proj_suffix_0.setEnabled(1)
        else:
            self.proj_suffix_0.setEnabled(0)
    
    def on_trans_checkbox_toggled(self, s):
        """
        If the trans_checkbox is toggled disable loading a transform

        Parameters
        ----------
        s : bool
            Whether or not the button is checked

        Returns
        -------
        None.

        """
        if s:
            self.proj_t_suffix_0.setEnabled(0)
        else:
            self.proj_t_suffix_0.setEnabled(1)
    
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
            import_mode = 'read_scan'
        elif self.read_scan_box.currentIndex()==1:
            import_mode = 'import_las'
        else:
            import_mode = 'poly'
            
        # Parse project path and name and load project
        project_name = self.proj_combobox.currentText()
        self.project = pydar.Project(self.project_path, project_name, 
                                     import_mode=import_mode,
                                     class_list='all', 
                                     suffix=self.proj_suffix_0.text(),
                                     class_suffix=self.class_lineedit.text())
        
        # Set the transform appropriately
        if self.proj_t_suffix_0.isEnabled():
            # If the transform suffix is enabled use that
            self.project.read_transforms(suffix=self.proj_t_suffix_0.text())
            self.project.apply_transforms(['current_transform'])
        else:
            self.project.apply_transforms(['sop'])
        
        # Load the man_class table
        for scan_name in self.project.scan_dict:
            self.project.scan_dict[scan_name].load_man_class()
        
        # Enable v_min and v_max
        self.v_min.setEnabled(1)
        self.v_max.setEnabled(1)
        
        # Clear the elev_filters mappers and actors
        # !!! update to get all of the actors in the renderer...
        #for scan_name in self.actor_dict:
        #    self.renderer.RemoveActor(self.actor_dict[scan_name])
        #self.vtkWidget.GetRenderWindow().Render()
        # self.elev_filt_dict.clear()
        # self.class_filt_dict.clear()
        # self.mapper_dict.clear()
        # self.actor_dict.clear()
        # self.selected_poly_dict.clear()
        
        for scan_name in self.project.scan_dict:
            # Create actor
            self.project.scan_dict[scan_name].create_elevation_pipeline(
                float(self.v_min.text()), float(self.v_max.text()))
            # Create Scanner Actors
            self.project.scan_dict[scan_name].create_scanner_actor(
                    color='Grey', length=150)

            self.renderer.AddActor(self.project.scan_dict[scan_name].actor)
            self.renderer.AddActor(self.project.scan_dict[scan_name]
                                   .scannerActor)
            self.renderer.AddActor(self.project.scan_dict[scan_name]
                                   .scannerText)
            self.project.scan_dict[scan_name].scannerText.SetCamera(
                self.renderer.GetActiveCamera())

            # Create appendPolyData, mappers and actors in selection dicts
            self.selected_poly_dict[scan_name] = vtk.vtkAppendPolyData()
            pdata = vtk.vtkPolyData()
            self.selected_poly_dict[scan_name].AddInputData(pdata)
            self.selected_append.AddInputConnection(
                self.selected_poly_dict[scan_name].GetOutputPort())

        
        # Update the fields available in field_selector
        self.field_selector.clear()
        self.field_selector.addItem('Elevation')
        for name in list(self.project.scan_dict.values()
                         )[0].dsa_raw.PointData.keys():
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
            self.renderer.AddActor(self.project.scan_dict[button.text()].actor)
            self.renderer.AddActor(self.project.scan_dict[button.text()]
                                   .scannerActor)
            self.renderer.AddActor(self.project.scan_dict[button.text()]
                                   .scannerText)
            self.vtkWidget.GetRenderWindow().Render()
        else:
            self.renderer.RemoveActor(self.project.scan_dict[button.text()]
                                      .actor)
            self.renderer.RemoveActor(self.project.scan_dict[button.text()]
                                   .scannerActor)
            self.renderer.RemoveActor(self.project.scan_dict[button.text()]
                                   .scannerText)
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
        
        c = self.class_dict[self.class_combobox.currentText()]
        # Send Picked Points to SingleScan to be added to classified points
        for scan_name in self.project.scan_dict:
            if (self.selected_poly_dict[scan_name].GetOutput()
                .GetNumberOfPoints()>0):
                self.project.scan_dict[scan_name].update_man_class(
                    self.selected_poly_dict[scan_name].GetOutput(), c)
        
        self.clear_selection()
        self.class_button.setEnabled(0)
        # Update man_class_dict
        self.update_man_class_dict()
        
        # add to class check dict
        if not c in self.class_check_dict.keys():
            self.class_check_dict[c] = Qt.QCheckBox(str(c))
            self.class_check_dict[c].setChecked(0)
            self.class_layout.addWidget(self.class_check_dict[c])
    
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
            with os.scandir(self.project.project_path) as it:
                for entry in it:
                    if entry.is_dir():
                        if re.search('.RiSCAN$', entry.name):
                            project_tuples.append((self.project.project_path
                                                   , entry.name))
            df = pydar.get_man_class(project_tuples)
        elif self.train_combobox.currentText()=='All':
            project_tuples = []
            # Get the path to the scan area
            scan_area_path = os.path.split(self.project.project_path)[0]
            #scan_area_path = (self.project.project_path.rsplit(sep='\\',
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
            # Apply to each scan
            for scan_name in self.project.scan_dict:
                ss = self.project.scan_dict[scan_name]
                self.renderer.RemoveActor(ss.actor)
                ss.create_filter_pipeline()
                self.renderer.AddActor(ss.actor)
            
            # Disable v_min and v_max
            self.v_min.setEnabled(0)
            self.v_max.setEnabled(0)
        else:
            for scan_name in self.project.scan_dict:
                ss = self.project.scan_dict[scan_name]
                
                if (ss.mapper.GetInput().GetPointData()
                    .HasArray('Classification')):
                    self.renderer.RemoveActor(ss.actor)
                    ss.create_elevation_pipeline(float(self.v_min.text()),
                                                 float(self.v_max.text()))
                    self.renderer.AddActor(ss.actor)
                else:
                    ss.mapper.GetInput().GetPointData().SetActiveScalars(text)
                    ss.mapper.GetInput().Modified()
                    ss.mapper.ScalarVisibilityOn()
                    ss.mapper.SetLookupTable(
                        pydar.mplcmap_to_vtkLUT(float(self.v_min.text()),
                                                float(self.v_max.text())))
                    ss.mapper.SetScalarRange(
                        float(self.v_min.text()), float(self.v_max.text()))
                    ss.mapper_sub.GetInput().GetPointData().SetActiveScalars(
                        text)
                    ss.mapper_sub.GetInput().Modified()
                    ss.mapper_sub.ScalarVisibilityOn()
                    ss.mapper_sub.SetLookupTable(
                        pydar.mplcmap_to_vtkLUT(float(self.v_min.text()),
                                                float(self.v_max.text())))
                    ss.mapper_sub.SetScalarRange(
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
            for scan_name in self.project.scan_dict:
                ss = self.project.scan_dict[scan_name]
                ss.mapper.SetLookupTable(
                    pydar.mplcmap_to_vtkLUT(float(self.v_min.text()),
                                            float(self.v_max.text())))
                ss.mapper.SetScalarRange(
                    float(self.v_min.text()), float(self.v_max.text()))
                ss.mapper_sub.SetLookupTable(
                    pydar.mplcmap_to_vtkLUT(float(self.v_min.text()),
                                            float(self.v_max.text())))
                ss.mapper_sub.SetScalarRange(
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
                          .index.get_level_values('PointId')
                          )>np.iinfo(np.uint32).max:
                    raise RuntimeError('PointId exceeds size of uint32')
                np_pedigreeIds[:] = (self.project.scan_dict[scan_name]
                                     .man_class.index
                                     .get_level_values('PointId')
                                     .astype(np.uint32))
                pedigreeIds.Modified()
                # Use PedigreeId selection to get points
                selectionNode = vtk.vtkSelectionNode()
                selectionNode.SetFieldType(1) # we want to select points
                selectionNode.SetContentType(2) # 2 corresponds to pedigreeIds
                selectionNode.SetSelectionList(pedigreeIds)
                selection = vtk.vtkSelection()
                selection.AddNode(selectionNode)
                extractSelection = vtk.vtkExtractSelection()
                extractSelection.SetInputData(0, self.project.scan_dict
                                              [scan_name].mapper.GetInput())
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
        for scan_name in self.project.scan_dict:
            self.project.scan_dict[scan_name].update_current_filter(
                class_list)
            self.renderer.RemoveActor(self.project.scan_dict[scan_name]
                                      .actor)
            # !!! Change to create correct pipeline
            self.project.scan_dict[scan_name].create_solid_pipeline('Cyan')
            self.renderer.AddActor(self.project.scan_dict[scan_name].actor)
        
        self.vtkWidget.GetRenderWindow().Render()

    def on_selection_changed(self, button):
        """
        Change the selection mode based on the user's request.

        Parameters
        ----------
        button : QAbstractButton
            Object of the button that was clicked.

        Returns
        -------
        None.

        """

        if button.text() in ['Area Points', 'Labels']:
            self.iren.SetPicker(self.pointPicker)
            self.iren.SetInteractorStyle(
                vtk.vtkInteractorStyleTrackballCamera())
        elif button.text()=='Rectangular':
            self.iren.SetPicker(self.areaPicker)
            self.iren.SetInteractorStyle(vtk.vtkInteractorStyleRubberBandPick())
        else:
            raise RuntimeError('Invalid id passed to on_selection_changed')
        
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
        
        self.area_point_list.copy_areapoints(self.project.project_name)
    
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
            self.area_point_list.save_areapoints(self.project.project_name, 
                                                 path)
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
        if not (self.project.project_name in areapoint_dict):
            warnings.warn('areapoints project does not match project 1.'
                          + ' Aborting.')
            return
        
        # For each areapoint in the list add it to self.area_point_list
        # the only tricky part here is that we need to get the singlescan
        # associated with it.
        for scan_name, PointId in areapoint_dict[self.project.project_name]:
            ss = self.project.scan_dict[scan_name]
            self.area_point_list.list[-1].set_point(PointId, ss, self.renderer)
        self.vtkWidget.GetRenderWindow().Render()

    def on_load_labels_clicked(self, s):
        """
        Load labels for the project

        Parameters
        ----------
        s : int
            Button status not used.

        Returns
        -------
        None.

        """

        self.project.load_labels()
        self.labels_checkbox.setEnabled(1)

        for scan_name in self.project.scan_dict:
                self.project.scan_dict[scan_name].create_labels_actors()
                for i in range(self.project.scan_dict[scan_name]
                               .labels_actors.shape[0]):
                    (self.project.scan_dict[scan_name].labels_actors
                     ['text_actor'].iat[i].SetCamera(self.renderer
                                                     .GetActiveCamera()))

        # Enable labels comboboxes and fill with values.
        labels = self.project.get_labels()
        self.label_category_combo.addItems(list(labels.index.levels[0]))
        self.label_category_combo.setEnabled(1)
        self.label_subcategory_combo.addItems(list(labels.index.levels[1]))
        self.label_subcategory_combo.setEnabled(1)
        self.label_id_combo.addItems(list(labels.index.levels[2]))
        self.label_id_combo.setEnabled(1)
        self.save_label_button.setEnabled(1)

    def on_save_label_button(self, s):
        """
        Save the label with the current values of the comboboxes chosen point.

        Parameters
        ----------
        s : int
            Button status not used.

        Returns
        -------
        None.

        """

        cat = self.label_category_combo.currentText()
        subcat = self.label_subcategory_combo.currentText()
        id_str = self.label_id_combo.currentText()

        # Get current picked point
        pt = self.vgf_pt_0.GetInput().GetPoint(0)

        # First, if show labels is checked go through SingleScans and remove
        # actors if we're overwriting a label
        if self.labels_checkbox.isChecked():
            for scan_name in self.project.scan_dict:
                if (self.project.scan_dict[scan_name].labels_actors.index
                    .isin([(cat, subcat, id_str)]).any()):
                    self.renderer.RemoveActor(self.project.scan_dict[scan_name]
                                           .labels_actors.loc[(cat, subcat, 
                                                               id_str),
                                           'text_actor'])
                    self.renderer.RemoveActor(self.project.scan_dict[scan_name]
                                           .labels_actors.loc[(cat, subcat, 
                                                               id_str),
                                           'point_actor'])

        # Get whichever SingleScan this point is in.
        best_point = ['', 1, np.inf]
        for scan_name in self.project.scan_dict:
            # Find the pedigree id of the point in any active singlescan
            # That's closest to the picked point (usually this will be picked)
            # point
            if self.scan_check_dict[scan_name].isChecked():
                ss = self.project.scan_dict[scan_name]
                ind = ss.currentFilter.GetOutput().FindPoint(pt)
                PointId = vtk_to_numpy(ss.currentFilter.GetOutput()
                                       .GetPointData().GetPedigreeIds())[ind]
                dist = np.square(pt - 
                                 np.array(ss.currentFilter.GetOutput()
                                          .GetPoint(ind))
                                 ).sum()
                if dist < best_point[2]:
                    best_point = [ss, PointId, dist]

        # Add this point to that SingleScan's labels dataframe
        ss.add_label(cat, subcat, id_str, pt[0], pt[1], pt[2])
        
        # Create the actors for displaying
        ss.create_labels_actors(row_index=(cat, subcat, id_str))
        ss.labels_actors.loc[(cat, subcat, id_str),'text_actor'].SetCamera(
            self.renderer.GetActiveCamera())

        # If show labels is checked add the actors
        if self.labels_checkbox.isChecked():
            self.renderer.AddActor(ss.labels_actors.loc[(cat, subcat, id_str),
                                           'text_actor'])
            self.renderer.AddActor(ss.labels_actors.loc[(cat, subcat, id_str),
                                           'point_actor'])

        self.vtkWidget.GetRenderWindow().Render()


    def on_show_labels_toggled(self, checked):
        """
        Add or remove labels glyphs from the renderwindow.

        Parameters
        ----------
        checked : bool
            Whether show labels is checked or not

        Returns
        -------
        None.

        """

        if checked:
            for scan_name in self.project.scan_dict:
                for i in range(self.project.scan_dict[scan_name]
                               .labels_actors.shape[0]):
                    self.renderer.AddActor(self.project.scan_dict[scan_name]
                                           .labels_actors['text_actor'].iat[i])
                    self.renderer.AddActor(self.project.scan_dict[scan_name]
                                           .labels_actors['point_actor'].iat[i])
        else:
            for scan_name in self.project.scan_dict:
                for i in range(self.project.scan_dict[scan_name]
                               .labels_actors.shape[0]):
                    self.renderer.RemoveActor(self.project.scan_dict[scan_name]
                                           .labels_actors['text_actor'].iat[i])
                    self.renderer.RemoveActor(self.project.scan_dict[scan_name]
                                           .labels_actors['point_actor'].iat[i])
        self.vtkWidget.GetRenderWindow().Render()


    ### VTK methods ###
    def on_end_pick(self, obj, event):
        """
        When a pick is made set it as an areapoint or a label

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

        # Get the picked point
        pt = np.array(obj.GetPickPosition())
        
        # If we're selecting an areapoint
        # Find the checked area point and set it
        if self.selection_buttongroup.checkedButton().text()=='Area Points':
            best_point = ['', 1, np.inf]
            for scan_name in self.project.scan_dict:
                # Find the pedigree id of the point in any active singlescan
                # That's closest to the picked point (usually this will be picked)
                # point
                if self.scan_check_dict[scan_name].isChecked():
                    ss = self.project.scan_dict[scan_name]
                    ind = ss.currentFilter.GetOutput().FindPoint(pt)
                    PointId = vtk_to_numpy(ss.currentFilter.GetOutput()
                                           .GetPointData().GetPedigreeIds())[ind]
                    dist = np.square(pt - 
                                     np.array(ss.currentFilter.GetOutput()
                                              .GetPoint(ind))
                                     ).sum()
                    if dist < best_point[2]:
                        best_point = [ss, PointId, dist]

            for ap in self.area_point_list.list:
                if ap.radio.isChecked():
                    ap.set_point(best_point[1], best_point[0], self.renderer)
                    break

        elif self.selection_buttongroup.checkedButton().text()=='Labels':
            pts0 = vtk.vtkPoints()
            pts0.SetNumberOfPoints(1)
            pts0.SetPoint(0, pt[0], pt[1], pt[2])
            pt_0 = vtk.vtkPolyData()
            pt_0.SetPoints(pts0)
            self.vgf_pt_0.SetInputData(pt_0)
            self.vgf_pt_0.Update()
        else:
            raise RuntimeError('One of above conditions should be met')

        self.vtkWidget.GetRenderWindow().Render()

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
        
        for scan_name in self.project.scan_dict:
            
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
                extractSelection.SetInputData(0, self.project.scan_dict
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