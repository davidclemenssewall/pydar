#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:33:20 2021

@author: thayer
"""

import os
import json
import time
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

areapoints = [('ScanPos001', 7360029),
('ScanPos001', 7340964),
('ScanPos001', 7216185),
('ScanPos001', 7195406),
('ScanPos001', 6975956),
('ScanPos001', 1652256),
('ScanPos001', 1626568),
('ScanPos001', 1626099),
('ScanPos001', 1608876),
('ScanPos001', 6920481),
('ScanPos001', 7208620),
('ScanPos001', 7337325),
('ScanPos001', 7343198),
]

areapoints = [('ScanPos001', 7360029),
('ScanPos001', 7340964),
('ScanPos001', 7216185),
('ScanPos001', 7195406),
('ScanPos001', 6975956),
('ScanPos001', 1652256),
('ScanPos001', 1626568),
('ScanPos001', 1626099),
('ScanPos001', 1608876),
('ScanPos001', 6920481),
('ScanPos001', 7208620),
('ScanPos001', 7337325),
('ScanPos001', 7343198),
('ScanPos001', 7342017),
('ScanPos001', 7387297),
('ScanPos001', 7394479),
('ScanPos001', 7394647),
('ScanPos001', 7396223),
('ScanPos001', 7396502),
('ScanPos001', 7408189),
('ScanPos001', 10186909),
('ScanPos001', 10198985),
('ScanPos001', 10305196),
('ScanPos001', 10307636),
('ScanPos001', 10315765),
('ScanPos001', 10314985),
('ScanPos001', 15563255),
('ScanPos001', 15638186),
('ScanPos001', 15921474),
('ScanPos001', 15983468),
('ScanPos001', 15996092),
('ScanPos001', 17082077),
('ScanPos001', 17379535),
('ScanPos001', 17381214),
('ScanPos001', 17088320),
('ScanPos001', 17102605),
('ScanPos001', 17105058),
('ScanPos001', 17132994),
('ScanPos001', 17012953),
('ScanPos001', 17012506),
('ScanPos001', 17012269),
('ScanPos001', 17073223),
('ScanPos001', 17072297),
('ScanPos001', 17069279),
('ScanPos001', 17049041),
('ScanPos001', 15883172),
('ScanPos001', 13771806),
('ScanPos001', 13623810),
('ScanPos001', 13614198),
('ScanPos001', 13563223),
('ScanPos001', 13562053),
('ScanPos001', 13469050),
('ScanPos001', 13468528),
('ScanPos001', 13464351),
('ScanPos001', 13452348),
('ScanPos001', 13437621),
('ScanPos001', 13437720),
('ScanPos001', 13437219),
('ScanPos001', 13436667),
('ScanPos001', 13436657),
('ScanPos001', 10608509),
('ScanPos001', 10607505),
('ScanPos001', 10607853),
('ScanPos001', 10675985),
('ScanPos001', 13455001),
('ScanPos001', 10708213),
('ScanPos001', 10682376),
('ScanPos001', 10689032),
('ScanPos001', 13546874),
('ScanPos001', 13549944),
('ScanPos001', 13586782),
('ScanPos001', 13687952),
('ScanPos001', 15859127),
('ScanPos001', 17042838),
('ScanPos001', 17077758),
('ScanPos001', 17076628),
('ScanPos001', 15981192),
('ScanPos001', 15655967),
('ScanPos001', 15634916),
('ScanPos001', 15546060),
('ScanPos001', 15102562),
('ScanPos001', 15091706),
('ScanPos001', 15096092),
('ScanPos001', 15094714),
('ScanPos001', 15064477),
('ScanPos001', 15098852),
('ScanPos001', 15076197),
('ScanPos001', 15009821),
('ScanPos001', 14809684),
('ScanPos001', 14805355),
('ScanPos001', 14791526),
('ScanPos001', 14788878),
('ScanPos001', 14523403),
('ScanPos001', 14517217),
('ScanPos001', 14432771),
('ScanPos001', 14430730),
('ScanPos001', 14333699),
('ScanPos001', 14258116),
('ScanPos001', 14267723),
('ScanPos001', 14234945),
('ScanPos001', 14198157),
('ScanPos001', 14032277),
('ScanPos001', 13803488),
('ScanPos001', 4972416),
('ScanPos001', 4617411),
('ScanPos001', 5093392),
('ScanPos001', 5099939),
('ScanPos001', 4821201),
('ScanPos001', 4458086),
('ScanPos001', 4206828),
('ScanPos001', 4224329),
('ScanPos001', 2465495),
('ScanPos001', 2491362),
('ScanPos001', 2498335),
('ScanPos001', 2473854),
('ScanPos001', 2440890),
('ScanPos001', 2394758),
('ScanPos001', 2299080),
('ScanPos001', 2170812),
('ScanPos001', 2111834),
('ScanPos001', 2283237),
('ScanPos001', 2219295),
('ScanPos001', 2216229),
('ScanPos001', 2013277),
('ScanPos001', 2127845),
('ScanPos001', 2011356),
('ScanPos001', 2200848),
('ScanPos001', 2200495),
('ScanPos001', 2003162),
('ScanPos001', 1989830),
('ScanPos001', 1116067),
('ScanPos001', 1115966),
('ScanPos001', 1113144),
('ScanPos001', 1987686),
('ScanPos001', 1092817),
('ScanPos001', 1104198),
('ScanPos001', 1090543),
('ScanPos001', 1087604),
('ScanPos001', 1086144),
('ScanPos001', 1021848),
('ScanPos001', 1022602),
('ScanPos001', 1027721),
('ScanPos001', 1030437),
('ScanPos001', 1988577),
('ScanPos001', 2093597),
('ScanPos001', 2101965),
('ScanPos001', 2140487),
('ScanPos001', 2144341),
('ScanPos001', 2182903),
('ScanPos001', 2721816),
('ScanPos001', 4306192),
('ScanPos001', 5944088),
('ScanPos002', 17597685),
('ScanPos009', 23887),
('ScanPos001', 9244458),
('ScanPos001', 14367779),
('ScanPos001', 9549774),
('ScanPos001', 9922830),
('ScanPos001', 14377521),
('ScanPos001', 14873054),
('ScanPos001', 15087319),
('ScanPos001', 15493615),
('ScanPos001', 14974852),
('ScanPos001', 10093468),
('ScanPos001', 15493037),
('ScanPos001', 10278029),
('ScanPos001', 10253209),
('ScanPos001', 10168626),
('ScanPos001', 10139676),
('ScanPos001', 10111571),
('ScanPos001', 7368910),
('ScanPos001', 7360987),
]
if False:
    areapoint_dict = {project_name: areapoints}
    f = open(os.path.join(project_path, project_name, 'manualclassification',
                          'ftridge.txt'), 'w')
    json.dump(areapoint_dict, f, indent=4)
    f.close()


# areapoint_dict = {
#     "mosaic_rov_250120.RiSCAN": [
#         [
#             "ScanPos001",
#             17133064
#         ],
#         [
#             "ScanPos001",
#             17012541
#         ],
#         [
#             "ScanPos001",
#             17011924
#         ],
#         [
#             "ScanPos001",
#             17012794
#         ]
#     ]
# }
# areapoints = areapoint_dict["mosaic_rov_250120.RiSCAN"]

cornercoords = project.areapoints_to_cornercoords(areapoints)
print(cornercoords)
#cornercoords[:,2] = 0
#print(cornercoords)

# %% Let's check if we manually filtered cornercoords correctly
t0 = time.perf_counter()
project.apply_manual_filter(cornercoords, mode='currentFilter')
t1 = time.perf_counter()
print(t1 - t0)


project.display_project(-4, -1, field='Classification')