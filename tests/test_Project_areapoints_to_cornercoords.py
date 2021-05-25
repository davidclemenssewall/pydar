#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:33:20 2021

@author: thayer
"""

import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% load scan

project_path = "/media/thayer/Data/mosaic_lidar/ROV"
project_name = "mosaic_rov_250120.RiSCAN"


project = pydar.Project(project_path, project_name, import_mode=
                      'read_scan', las_fieldnames=['Points', 'PointId',
                                                   'Classification'])

project.apply_transforms(['sop'])


# %% Test function 

# Sediment Trap Transect
areapoints = [('ScanPos001', 7359197),
('ScanPos001', 7341129),
('ScanPos001', 7338616),
('ScanPos001', 7215402),
('ScanPos001', 7209340),
('ScanPos001', 7208056),
('ScanPos001', 7194987),
('ScanPos001', 7184960),
('ScanPos001', 7184505),
('ScanPos001', 6975953),
('ScanPos007', 17385176),
('ScanPos007', 17382874),
('ScanPos001', 1652285),
('ScanPos001', 1650480),
('ScanPos001', 1650242),
('ScanPos001', 1668267),
('ScanPos001', 1666369),
('ScanPos001', 1664318),
('ScanPos001', 1664338),
('ScanPos001', 1626216),
('ScanPos001', 1663775),
('ScanPos001', 1608876),
('ScanPos001', 1610069),
('ScanPos001', 1610203),
('ScanPos001', 1648622),
('ScanPos001', 1650141),
('ScanPos001', 6919488),
('ScanPos001', 6920582),
('ScanPos001', 7177732),
('ScanPos001', 7179342),
('ScanPos001', 7182797),
('ScanPos001', 7183279),
('ScanPos001', 7212034),
('ScanPos001', 7213736),
('ScanPos001', 7337292),
('ScanPos001', 7340392),
('ScanPos001', 7343436),
('ScanPos001', 7359857),
('ScanPos001', 7360083),
]

cornercoords = project.areapoints_to_cornercoords(areapoints)

# %% Let's check if we manually filtered cornercoords correctly

project.apply_manual_filter(cornercoords)
project.display_project(-4, -1, field='Classification')