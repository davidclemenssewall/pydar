#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_kernel_alignment.py

Use a kernel distance metric (from the geomloss or keops libraries) to evaluate
the distance between a SingleScan and another collection of SingleScans 
(typically a Project) and apply rigid transformations to optimize the distance
between the two. The output is a rigid transformation applied to the 

Created on Sat Mar 13 13:13:04 2021

@author: thayer
"""

import argparse
import os
import sys

import numpy as np
import json

# Create the parser
parser = argparse.ArgumentParser(description="Compute kernel distance" +
                                 "-based rigid transformation",
                                 allow_abbrev=False)


# Typical Path Arguments
parser.add_argument('--scan_area_name',
                    metavar='Scan Area Name',
                    type=str,
                    help='The Scan Area that these scans belong to. Should be' +
                        'one of ["Snow1", "Snow2", "ROV"]')

parser.add_argument('--project_name_0',
                    metavar='Project Name 0',
                    type=str,
                    help='The name of the project that we are aligning ' +
                        'relative to.')


parser.add_argument('--project_name_1',
                    metavar='Project Name 1',
                    type=str,
                    help='The name of the project that the scan which we ' +
                        'are aligning comes from.')

parser.add_argument('--scan_name_1',
                    metavar='Scan Name 1',
                    type=str,
                    help='The name of the scan we are aligning')

# Custom Path Arguments
parser.add_argument('--root_data_dir',
                    type=str,
                    action='store',
                    help='Path to the directory where the scan areas are stored'
                    + '. The default is: "/mosaic_lidar"')

parser.add_argument('--set_paths_directly',
                    action='store_true')

parser.add_argument('--scan_1_path',
                    type=str,
                    action='store',
                    help='Path where scan we are alignings Points file is')

parser.add_argument('--trans_1_path',
                    type=str,
                    action='store',
                    help='Path where current_transform files for scan 1 is')

parser.add_argument('--scan_0_paths',
                    type=list,
                    action='store',
                    help='List of paths where reference scan Points files are')

parser.add_argument('--trans_0_paths',
                    type=list,
                    action='store',
                    help='List of paths where reference scans transforms are')

# Execute the parse_args() method
args = parser.parse_args()

# Set defaults for arguments
if args.root_data_dir is None:
    args.root_data_dir = "/mosaic_lidar"

# Set the path arguments for the scans (if they were not set directly)
if not args.set_paths_directly:
    # Set the paths for the scan we are aligning
    args.scan_1_path = os.path.join(args.root_data_dir, args.scan_area_name,
                                    args.project_name_1, 'npyfiles', 
                                    args.scan_name_1)
    args.trans_1_path = os.path.join(args.root_data_dir, args.scan_area_name,
                                    args.project_name_1, 'transforms', 
                                    args.scan_name_1)
    # Set the paths for the reference scan
    scan_0_names = os.listdir(os.path.join(args.root_data_dir, 
                                           args.scan_area_name,
                                           args.project_name_0, 'npyfiles'))
    args.scan_0_paths = []
    args.trans_0_paths = []
    for name in scan_0_names:
        args.scan_0_paths.append(os.path.join(args.root_data_dir, 
                                              args.scan_area_name, 
                                              args.project_name_0, 
                                              'npyfiles', name))
        args.trans_0_paths.append(os.path.join(args.root_data_dir, 
                                                args.scan_area_name, 
                                                args.project_name_0, 
                                                'transforms', name))

# Check that the number of transforms matches the number of scans
if not len(args.scan_0_paths)==len(args.trans_0_paths):
    raise ValueError("Number of scans and transforms do not match")

print(args.scan_0_paths)
print(args.trans_0_paths)
print(args.scan_1_path)
print(args.trans_1_path)


