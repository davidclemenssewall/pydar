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
import copy

# Create the parser
parser = argparse.ArgumentParser(description="Compute kernel distance" +
                                 "-based rigid transformation",
                                 allow_abbrev=False)

# The one mandatory argument is the git hash
parser.add_argument('--git_hash',
                    type=str,
                    help='The git hash for the current version of pydar',
                    required=True)

# Typical Path Arguments
parser.add_argument('--scan_area_name',
                    metavar='Scan Area Name',
                    type=str,
                    help='The Scan Area that these scans belong to. Should be' +
                        'one of ["Snow1", "Snow2", "ROV"]',
                    choices=["Snow1", "Snow2", "ROV"])

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
                    + '. The default is: "/mosaic_lidar"',
                    default="/mosaic_lidar")

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

# Optional parameters for the optimization
parser.add_argument('--max_dist',
                    type=float,
                    action='store',
                    help='The maximum distance (in m) from the scanner in scan'
                        + ' 1 to look at points',
                    default=250)

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

# Function for applying rigid transformations in numpy
def rigid_transform_np(yaw, pitch, roll, del_x, del_y, del_z, pts):
    # Create a rigid transformation matrix
    u = np.float32(roll)
    v = np.float32(pitch)
    w = np.float32(yaw)
    c = np.cos
    s = np.sin
    
    Rx = np.array([[1, 0, 0],
                  [0, c(u), -s(u)],
                  [0, s(u), c(u)]])
    Ry = np.array([[c(v), 0, s(v)],
                   [0, 1, 0],
                   [-s(v), 0, c(v)]])
    Rz = np.array([[c(w), -s(w), 0],
                  [s(w), c(w), 0],
                  [0, 0, 1]])
    # Order of rotations in vtk is Pitch, then Roll, then Yaw
    M = Rz @ Rx @ Ry
    
    translate = np.array([del_x, del_y, del_z], dtype=np.float32)[np.newaxis,:]
    return pts @ M.T + translate

# Load Scan 1
scan_1_trans = np.load(os.path.join(args.trans_1_path, 'current_transform.npy'))
scan_1_pts = np.load(os.path.join(args.scan_1_path, 'Points.npy'))
scan_1_class = np.load(os.path.join(args.scan_1_path, 'Classification.npy'))
scan_1_pts = scan_1_pts[scan_1_class==0, :]
# Create the Scan 1 history dict
f_trans = open(os.path.join(args.trans_1_path, 'current_transform.txt'))
f_pts = open(os.path.join(args.scan_1_path, 'raw_history_dict.txt'))
scan_1_history_dict = {
    "type": "Transformer",
    "git_hash": args.git_hash,
    "method": "compute_kernel_alignment.py",
    "input_0": {
        "type": "Filter",
        "git_hash": args.git_hash,
        "method": "compute_kernel_alignment.py",
        "input_0": json.load(f_pts),
        "params": {"class_list": [0]}
        },
    "input_1": json.load(f_trans)
    }
f_trans.close()
f_pts.close()

print(scan_1_pts.dtype)
print(scan_1_pts.shape)
print(json.dumps(scan_1_history_dict, indent=4))

# Load Scan 0
for i in range(len(args.scan_0_paths)):
    trans = np.load(os.path.join(args.trans_0_paths[i], 
                                 'current_transform.npy'))
    pts = np.load(os.path.join(args.scan_0_paths[i], 'Points.npy'))
    scan_0_class = np.load(os.path.join(args.scan_0_paths[i], 
                                        'Classification.npy'))
    pts = pts[scan_0_class==0, :]
    # Apply transform to points
    pts = rigid_transform_np(trans['w0'][0], trans['v0'][0],
                             trans['u0'][0], trans['x0'][0], 
                             trans['y0'][0], trans['z0'][0],
                             pts)
    # Limit to just points within max_dist of scan 1
    pts = pts[((pts[:,0]-scan_1_trans['x0'][0])**2 
               + (pts[:,1]-scan_1_trans['y0'][0])**2)
                 <= args.max_dist**2]

    # Append points
    if i==0:
        scan_0_pts = copy.deepcopy(pts)
    else:
        scan_0_pts = np.vstack((scan_0_pts, pts))

    # Update history dict
    f_trans = open(os.path.join(args.trans_0_paths[i], 
                                'current_transform.txt'))
    f_pts = open(os.path.join(args.scan_0_paths[i], 'raw_history_dict.txt'))
    if i==0:
        scan_0_history_dict = {
            "type": "Pointset Aggregator",
            "git_hash": args.git_hash,
            "method": "compute_kernel_alignment.py",
            "input_0": {
                "type": "Transformer",
                "git_hash": args.git_hash,
                "method": "compute_kernel_alignment.py",
                "input_0": {
                    "type": "Filter",
                    "git_hash": args.git_hash,
                    "method": "compute_kernel_alignment.py",
                    "input_0": json.load(f_pts),
                    "params": {"class_list": [0]}
                    },
                "input_1": json.load(f_trans)
                }
            }
    elif i==1:
        scan_0_history_dict["input_1"] = {
            "type": "Transformer",
            "git_hash": args.git_hash,
            "method": "compute_kernel_alignment.py",
            "input_0": {
                "type": "Filter",
                "git_hash": args.git_hash,
                "method": "compute_kernel_alignment.py",
                "input_0": json.load(f_pts),
                "params": {"class_list": [0]}
                },
            "input_1": json.load(f_trans)
            }
    else:
        scan_0_history_dict = {
            "type": "Pointset Aggregator",
            "git_hash": args.git_hash,
            "method": "compute_kernel_alignment.py",
            "input_0": {
                "type": "Transformer",
                "git_hash": args.git_hash,
                "method": "compute_kernel_alignment.py",
                "input_0": {
                    "type": "Filter",
                    "git_hash": args.git_hash,
                    "method": "compute_kernel_alignment.py",
                    "input_0": json.load(f_pts),
                    "params": {"class_list": [0]}
                    },
                "input_1": json.load(f_trans)
                },
            "input_1": json.loads(json.dumps(scan_0_history_dict))
            }
    f_trans.close()
    f_pts.close()

# Include distance filter
scan_0_history_dict = {
    "type": "Filter",
    "git_hash": args.git_hash,
    "method": "compute_kernel_alignment.py",
    "input_0": json.loads(json.dumps(scan_0_history_dict)),
    "params": {
        "max_dist": args.max_dist,
        "x0": scan_1_trans['x0'][0],
        "y0": scan_1_trans['y0'][0]
        }
    }


print(scan_0_pts.dtype)
print(scan_0_pts.shape)
print(json.dumps(scan_0_history_dict, indent=4))


