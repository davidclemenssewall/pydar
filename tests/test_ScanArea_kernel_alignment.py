#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ScanArea_kernel_alignment_ss.py

test our applying our kernel alignment after subsampling to areas of high
mutual point density

Created on Thu Mar 25 12:12:05 2021

@author: thayer
"""

import numpy as np
from collections import namedtuple
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

Registration = namedtuple('Registration', ['project_name_0', 'project_name_1',
                                           'reflector_list', 'mode', 
                                           'yaw_angle'], 
                          defaults=[None, None, None])

project_names = ['mosaic_rov_250120.RiSCAN',
                 'mosaic_rov_190120.RiSCAN']#,
                 #,
                 #'mosaic_rov_040220.RiSCAN']

registration_list = [#Registration('mosaic_rov_190120.RiSCAN', 
                     #             'mosaic_rov_190120.RiSCAN'),
                      # Registration('mosaic_rov_190120.RiSCAN',
                      #              'mosaic_rov_250120.RiSCAN',
                      #              ['r28', 'r29', 'r30', 'r32', 'r34', 'r35',
                      #               'r22'],
                      #              'LS')
                      Registration('mosaic_rov_250120.RiSCAN',
                                   'mosaic_rov_250120.RiSCAN'),
                      # Registration('mosaic_rov_250120.RiSCAN',
                      #               'mosaic_rov_040220.RiSCAN',
                      #               ['r28', 'r29', 'r30', 'r31', 'r32', 'r34', 
                      #                'r35', 'r36'],
                      #               'LS')
                       Registration('mosaic_rov_250120.RiSCAN',
                                    'mosaic_rov_190120.RiSCAN',
                                    ['r28', 'r29', 'r30', 'r32', 'r34', 'r35',
                                     'r22'],
                                    'LS')
                      ]

# %% Init
project_path = '/media/thayer/Data/mosaic_lidar/ROV/'

# scan_area = pydar.ScanArea(project_path, project_names,
#                            registration_list, import_mode='read_scan',
#                            las_fieldnames=['Points', 'Classification', 'PointID'],
#                            class_list=[0])


# scan_area.register_all()

# # Apply z_alignment
# frac_exceed_diff_cutoff=0.25
# w0=1
# w1=1
# min_pt_dens=25
# max_diff=0.2
# bin_reduc_op='mean'
# diff_mode='mode'
# scan_area.z_align_all(w0=w0, w1=w1,
#                             min_pt_dens=min_pt_dens, max_diff=max_diff, 
#                             frac_exceed_diff_cutoff=frac_exceed_diff_cutoff, 
#                             bin_reduc_op=bin_reduc_op,
#                             diff_mode=diff_mode)

# for project_name in project_names:
#     scan_area.project_dict[project_name].write_current_transforms()

# %% Let's look at what some of the z offsets look like

def ss_print_z_offset(ss):
    print(ss.scan_name + ': ' +
          str(ss.transform_dict["z_offset"].GetPosition()[2]))

# for project_name in project_names:
#     print(project_name)
#     for scan_name in scan_area.project_dict[project_name].scan_dict:
#         ss_print_z_offset(
#             scan_area.project_dict[project_name].scan_dict[scan_name])
        
        
# %% compare

scan_area2 = pydar.ScanArea(project_path, project_names,
                           registration_list, import_mode='read_scan',
                           las_fieldnames=['Points', 'Classification', 'PointID'],
                           class_list='all', suffix='slfsnow')


scan_area2.register_all()

# Apply z_alignment
frac_exceed_diff_cutoff=0.25
w0=1
w1=1
min_pt_dens=25
max_diff=0.2
bin_reduc_op='mean'
diff_mode='mode'
scan_area2.z_align_all(w0=w0, w1=w1,
                            min_pt_dens=min_pt_dens, max_diff=max_diff, 
                            frac_exceed_diff_cutoff=frac_exceed_diff_cutoff, 
                            bin_reduc_op=bin_reduc_op,
                            diff_mode=diff_mode)

for project_name in project_names:
    print(project_name)
    for scan_name in scan_area2.project_dict[project_name].scan_dict:
        ss_print_z_offset(
            scan_area2.project_dict[project_name].scan_dict[scan_name])

# %% What if we try writing and reading transforms instead
suffix = 'slfsnow'

scan_area2 = pydar.ScanArea(project_path, project_names,
                           registration_list, import_mode='empty',#'read_scan',
                           las_fieldnames=['Points', 'Classification', 'PointID'],
                           class_list='all', suffix='slfsnow')


scan_area2.register_all()

# Let's save the current transform for ScanPos001
old_trans = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        old_trans[i,j] = (scan_area2.project_dict[project_names[1]]
                          .scan_dict['ScanPos001'].transform.GetMatrix().GetElement(i,j))

for project_name in project_names:
    # scan_area2.project_dict[project_name].write_current_transforms(suffix=
    #                                                                 suffix)
    scan_area2.project_dict[project_name].write_current_transforms()

del scan_area2

# Now load and try to z_align
scan_area2 = pydar.ScanArea(project_path, project_names, registration_list,
                                import_mode='read_scan', create_id=True,
                                las_fieldnames=['Points', 'Classification', 'PointID'],
                                class_list='all', suffix='slfsnow')
# Load the current transforms and apply
for project_name in scan_area2.project_dict:
    #scan_area2.project_dict[project_name].read_transforms(suffix=suffix)
    scan_area2.project_dict[project_name].read_transforms()
    scan_area2.project_dict[project_name].apply_transforms(
        ['current_transform'])

new_trans = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        new_trans[i,j] = (scan_area2.project_dict[project_names[1]]
                          .scan_dict['ScanPos001'].transform.GetMatrix().GetElement(i,j))
        
# scan_area2.z_align_all(w0=w0, w1=w1,    
#                             min_pt_dens=min_pt_dens, max_diff=max_diff, 
#                             frac_exceed_diff_cutoff=frac_exceed_diff_cutoff, 
#                             bin_reduc_op=bin_reduc_op,
#                             diff_mode=diff_mode)

frac_exceed_diff_cutoff=0.25
w0=1
w1=1
min_pt_dens=25
max_diff=0.2
bin_reduc_op='mean'
diff_mode='mode'

scan_area2.z_alignment(project_names[0], project_names[1], w0=w0, w1=w1,    
                            min_pt_dens=min_pt_dens, max_diff=max_diff, 
                            frac_exceed_diff_cutoff=frac_exceed_diff_cutoff, 
                            bin_reduc_op=bin_reduc_op,
                            diff_mode=diff_mode)



for project_name in project_names[1:]:
    print(project_name)
    for scan_name in scan_area2.project_dict[project_name].scan_dict:
        ss_print_z_offset(
            scan_area2.project_dict[project_name].scan_dict[scan_name])

# %% test kernel_alignment_ss
# Get pointclouds to register
project_name_0 = project_names[0]
project_name_1 = project_names[1]
scan_names = ['ScanPos004']
blur = 0.005

max_points = 800000

for scan_name in scan_names:
    print(scan_name)
    scan_area.kernel_alignment_ss(project_name_0, project_name_1, scan_name,
                                  max_points=max_points, plot_optimization=True,
                                  blur=blur)

# %% 
