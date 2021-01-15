# -*- coding: utf-8 -*-
"""
test_Classifier.py

test the new classifier class.

Created on Thu Jan 14 09:49:36 2021

@author: d34763s
"""

import pandas as pd
import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\pydar\\')
import pydar

# %% init

project_path = 'D:\\mosaic_lidar\\Snow2\\'
project_name = 'mosaic_02_110619.RiSCAN'

project = pydar.Project(project_path, project_name, load_scans=True,
                        read_scans=True)

project.apply_transforms(['sop'])

# %% load all manually classified points and combine into single
# dataframe

df_list = []

for scan_name in project.scan_dict:
    project.scan_dict[scan_name].load_man_class()
    df_list.append(project.scan_dict[scan_name].man_class)
    
df = pd.concat(df_list, ignore_index=True)

# %% Create classifier object

classifier = pydar.Classifier(df_labeled=df)
classifier.init_randomforest()

# %% Train classifier

feature_list = ['Linearity', 'Planarity', 'Scattering', 'Verticality']
classifier.train_classifier(feature_list)

# %% Label points

for scan_name in project.scan_dict:
    classifier.classify_pdata(project.scan_dict[scan_name].polydata_raw)

# %% Display

project.display_project(-1, 0, field='Classification')

# %% Save classifications to test viewer

if False:
    project.write_scans()