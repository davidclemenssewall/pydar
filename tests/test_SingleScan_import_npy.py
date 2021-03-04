#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_SingleScan_import_npy.py

Test importing all point from numpy and see if it's  any faster than 
import_las

Created on Thu Mar  4 10:47:49 2021

@author: thayer
"""

import time
import tracemalloc
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% location

project_path = "/media/thayer/Data/mosaic_lidar/ROV"
project_name = "mosaic_rov_190120.RiSCAN"
scan_name = "ScanPos004"

# %% Import las

# t0 = time.perf_counter()
# ss = pydar.SingleScan(project_path, project_name, scan_name, 
#                       import_mode='import_las')
# t1 = time.perf_counter()

# print(t1 - t0)

# RAM before: 5.3 GB
# RAM after: 8.3 GB
# Time 22.07 sec

# %% Import npy

# t0 = time.perf_counter()
# ss = pydar.SingleScan(project_path, project_name, scan_name, 
#                       import_mode='import_npy')
# t1 = time.perf_counter()

# print(t1 - t0)

# Cleared cache this time
# RAM before: 5.8 GB
# RAM after: 7.7 GB
# Time 2.45 sec

# %% See if we drop importing ReturnIndex if that helps

# t0 = time.perf_counter()
# ss = pydar.SingleScan(project_path, project_name, scan_name, 
#                       import_mode='import_npy', las_fieldnames=['Points'])
# t1 = time.perf_counter()

# print(t1 - t0)

# Cleared cache this time
# RAM before: 6.0 GB
# RAM after: 7.8 GB
# Time 2.27 sec

# %% And also try not creating Id's
tracemalloc.start()

t0 = time.perf_counter()
ss = pydar.SingleScan(project_path, project_name, scan_name, 
                      import_mode='import_npy', las_fieldnames=['Points'],
                      create_id=False)
t1 = time.perf_counter()

print(t1 - t0)

current, peak = tracemalloc.get_traced_memory()
print('RAM Used in init')
print(current)
print(peak)
tracemalloc.stop()


# Cleared cache this time 1.79 seconds
# RAM before: 5
# RAM After: 6.3

# %% And repeat with no class filtering

tracemalloc.start()

t0 = time.perf_counter()
ss = pydar.SingleScan(project_path, project_name, scan_name, 
                      import_mode='import_npy', las_fieldnames=['Points'],
                      create_id=False, class_list='all')
t1 = time.perf_counter()

print(t1 - t0)

current, peak = tracemalloc.get_traced_memory()
print('RAM Used in init')
print(current)
print(peak)
tracemalloc.stop()

# Cleared cache time 0.38 seconds