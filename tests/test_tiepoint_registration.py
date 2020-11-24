# -*- coding: utf-8 -*-
"""
Test the registration functionality in TiePointList class

Created on Mon Sep 14 16:47:14 2020

@author: d34763s
"""

import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\lidar_processing\\')
import pydar
import numpy as np
import matplotlib.pyplot as plt

# %% load tiepointlists

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01_110819.RiSCAN'

tp1 = pydar.TiePointList(project_path, project_name)

project_name = 'mosaic_01_111519.RiSCAN'

tp2 = pydar.TiePointList(project_path, project_name)

# %% Compare Tiepoints

tp1.calc_pairwise_dist()
tp2.calc_pairwise_dist()

tp2.compare_pairwise_dist(tp1)

# %% Plot comparison

tp2.plot_map(tp1.project_name, delaunay=True)

# %% Test out calc_transformation

reflector_list = ['r01', 'r02', 'r05', 'r06', 'r07', 'r10']

tp2.calc_transformation(tp1, reflector_list)
tp2.calc_transformation(tp1, reflector_list, mode='Yaw')


# %% Now Apply transformation

index = (tp1.project_name+'_Yaw', ', '.join(reflector_list))
tp2.apply_tranform(index)

tp2.plot_map(tp1.project_name, delaunay=True, use_tiepoints_transformed=True)

# %% And Compare results
print(tp1.tiepoints_transformed)
print(tp2.tiepoints_transformed)
