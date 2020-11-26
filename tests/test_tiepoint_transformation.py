# -*- coding: utf-8 -*-
"""
Test functionality of tiepoints class, especially transformation.

Created on Mon Sep 14 13:50:06 2020

@author: d34763s
"""

import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\pydar\\')
import pydar
import numpy as np
import matplotlib.pyplot as plt

# %% load tiepointlist

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01_220320.RiSCAN'

tp = pydar.TiePointList(project_path, project_name)

# %% add a transform

rot90 = np.array([[0, -1, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
tp.add_transform('rot90', rot90)
tp.transforms

# %% Apply that transform

f, ax = plt.subplots(1, 1)

tp.apply_tranform(('identity', ''))

ax.plot(tp.tiepoints_transformed['X[m]'],
        tp.tiepoints_transformed['Y[m]'],
        'b')

tp.apply_tranform(('rot90',''))

ax.plot(tp.tiepoints_transformed['X[m]'],
        tp.tiepoints_transformed['Y[m]'],
        'r')