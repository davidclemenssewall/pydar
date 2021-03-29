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

import json
import copy

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import geomloss

use_cuda = torch.cuda.is_available()
dtype    = torch.float32 if use_cuda else torch.float32
device = torch.device('cuda:0')

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
                    type=str,
                    nargs='+',
                    action='store',
                    help='List of paths where reference scan Points files are')

parser.add_argument('--trans_0_paths',
                    type=str,
                    nargs='+',
                    action='store',
                    help='List of paths where reference scans transforms are')

# Optional parameters for the optimization
parser.add_argument('--class_list',
                    nargs='+',
                    action='store',
                    help='If we want to restrict to just certain categories' +
                    '. If this is empty then pass all classes.',
                    default=None)

parser.add_argument('--max_dist',
                    type=float,
                    action='store',
                    help='The maximum distance (in m) from the scanner in scan'
                        + ' 1 to look at points',
                    default=250)

parser.add_argument('--max_pts',
                    type=int,
                    action='store',
                    help='The maximum number of points to keep in the ' +
                    'pointclouds (will strided downsample to below this)',
                    default=750000)

parser.add_argument('--blur',
                    type=float,
                    action='store',
                    help='Sigma for the gaussian blur applied to points in m',
                    default=0.005)

parser.add_argument('--max_steps',
                    type=int,
                    action='store',
                    help='Maximum number of steps for the optimization to take',
                    default=100)

parser.add_argument('--cutoff_t',
                    type=float,
                    action='store',
                    help='Cutoff to stop optimization if all translation steps'
                    + ' are below this cutoff in m',
                    default=0.0005)

parser.add_argument('--cutoff_r',
                    type=float,
                    action='store',
                    help='Cutoff to stop optimization if all rotation steps'
                    + ' are below this cutoff in radians',
                    default=0.000005)

# Optional parameters for the output
parser.add_argument('--trans_output_path',
                    type=str,
                    action='store',
                    help='Path to directory where rigid transformation output'
                    + ' should be stored. If not set will default to ' +
                    'overwriting trans_1_path')

parser.add_argument('--trans_output_name',
                    type=str,
                    action='store',
                    help='Name that the output transform should be stored. ' +
                    'The default is "current_transform"',
                    default='current_transform')

parser.add_argument('--plot_optimization',
                    action='store_true',
                    help='Whether to write out a plot displaying the progress'
                    + ' of the optimization')

parser.add_argument('--plot_output_path',
                    type=str,
                    action='store',
                    help='Path to location to store plots in. If not given then'
                    + ' defaults to root_data_dir/scan_area_name/snapshots')

# Execute the parse_args() method
args = parser.parse_args()

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

# Set output defaults
if args.trans_output_path is None:
    args.trans_output_path = copy.deepcopy(args.trans_1_path)
if args.plot_optimization and (args.plot_output_path is None):
    args.plot_output_path = os.path.join(args.root_data_dir,
                                         args.scan_area_name,
                                         'snapshots')

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
                  [0, s(u), c(u)]],
                  dtype=np.float32)
    Ry = np.array([[c(v), 0, s(v)],
                   [0, 1, 0],
                   [-s(v), 0, c(v)]],
                  dtype=np.float32)
    Rz = np.array([[c(w), -s(w), 0],
                  [s(w), c(w), 0],
                  [0, 0, 1]],
                  dtype=np.float32)
    # Order of rotations in vtk is Pitch, then Roll, then Yaw
    M = Rz @ Rx @ Ry
    
    translate = np.array([del_x, del_y, del_z], dtype=np.float32)[np.newaxis,:]
    return pts @ M.T + translate

# Load Scan 1
scan_1_trans = np.load(os.path.join(args.trans_1_path, 'current_transform.npy'))
scan_1_pts = np.load(os.path.join(args.scan_1_path, 'Points.npy'))
scan_1_class = np.load(os.path.join(args.scan_1_path, 'Classification.npy'))
if args.class_list:
    scan_1_pts = scan_1_pts[np.isin(scan_1_class, args.class_list), :]
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
        "params": {"class_list": args.class_list}
        },
    "input_1": json.load(f_trans)
    }
f_trans.close()
f_pts.close()

# Load Scan 0
for i in range(len(args.scan_0_paths)):
    trans = np.load(os.path.join(args.trans_0_paths[i], 
                                 'current_transform.npy'))
    pts = np.load(os.path.join(args.scan_0_paths[i], 'Points.npy'))
    scan_0_class = np.load(os.path.join(args.scan_0_paths[i], 
                                        'Classification.npy'))
    if args.class_list:
        pts = pts[np.isin(scan_0_class, args.class_list), :]
    # Apply transform to points
    pts = rigid_transform_np(trans['w0'][0], trans['v0'][0],
                             trans['u0'][0], trans['x0'][0], 
                             trans['y0'][0], trans['z0'][0],
                             pts)
    # Limit to just points within max_dist of scan 1
    pts = pts[((pts[:,0]-np.float32(scan_1_trans['x0'][0]))**2 
               + (pts[:,1]-np.float32(scan_1_trans['y0'][0]))**2)
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
                    "params": {"class_list": args.class_list}
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
                "params": {"class_list": args.class_list}
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
                    "params": {"class_list": args.class_list}
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

# Reduce size of points arrays to be less than max_pts
scan_0_pts = scan_0_pts[::scan_0_pts.shape[0]//args.max_pts + 1, :]
scan_1_pts = scan_1_pts[::scan_1_pts.shape[0]//args.max_pts + 1, :]

# Define a pytorch class for modeling rigid transformations
class RigidTransformation(nn.Module):
    def __init__(self, x0=0, y0=0, z0=0, u0=0, v0=0, w0=0):
        super().__init__()
        # Initialize translation and rotation parameters
        self.t = nn.ParameterDict({
            'dx': nn.Parameter(torch.tensor([x0], device=device, dtype=dtype)
                               .view(1,1).requires_grad_()),
            'dy': nn.Parameter(torch.tensor([y0], device=device, dtype=dtype)
                               .view(1,1).requires_grad_()),
            'dz': nn.Parameter(torch.tensor([z0], device=device, dtype=dtype)
                               .view(1,1).requires_grad_())
        })
        self.r = nn.ParameterDict({
            'u': nn.Parameter(torch.tensor([u0], device=device, dtype=dtype)
                               .view(1,1).requires_grad_()),
            'v': nn.Parameter(torch.tensor([v0], device=device, dtype=dtype)
                               .view(1,1).requires_grad_()),
            'w': nn.Parameter(torch.tensor([w0], device=device, dtype=dtype)
                               .view(1,1).requires_grad_())
        })
   
    def forward(self, pts):
        # Computes the output of the rigid transformation applied to Nx3 array of points
        # Initialize Rotation matrix and translation vector
        T = torch.cat((self.t.dx, self.t.dy, self.t.dz), dim=1)
        c = torch.cos
        s = torch.sin
        Rx = torch.cat([torch.tensor([1], device=device, dtype=dtype).view(1,1), 
                        torch.tensor([0], device=device, dtype=dtype).view(1,1),
                        torch.tensor([0], device=device, dtype=dtype).view(1,1),
                        
                        torch.tensor([0], device=device, dtype=dtype).view(1,1), 
                        c(self.r.u), 
                        -1*s(self.r.u),
                        
                        torch.tensor([0], device=device, dtype=dtype).view(1,1), 
                        s(self.r.u), 
                        c(self.r.u)]
                      ).view(3, 3)
        
        Ry = torch.cat([c(self.r.v), 
                        torch.tensor([0], device=device, dtype=dtype).view(1,1), 
                        s(self.r.v),
                        
                        torch.tensor([0, 1, 0], device=device, dtype=dtype).view(3,1),
                        
                        -1*s(self.r.v), 
                        torch.tensor([0], device=device, dtype=dtype).view(1,1),
                        c(self.r.v)]
                      ).view(3, 3)
        
        Rz = torch.cat([c(self.r.w), 
                        -1*s(self.r.w), 
                        torch.tensor([0], device=device, dtype=dtype).view(1,1),
                        
                        s(self.r.w), 
                        c(self.r.w), 
                        torch.tensor([0], device=device, dtype=dtype).view(1,1),
                        
                        torch.tensor([0, 0, 1], device=device, dtype=dtype).view(3,1)]
                      ).view(3, 3)
        # Order of rotations in vtk is Pitch, then Roll, then Yaw
        M = Rz @ Rx @ Ry
        
        return pts @ M.t() + T

# Pass pointclouds to the gpu
alpha_t = torch.tensor(scan_0_pts, device=device, dtype=dtype)
beta_t = torch.tensor(scan_1_pts, device=device, dtype=dtype)

# Create model and send to gpu
model = RigidTransformation(x0=np.float32(scan_1_trans['x0'][0]), 
                            y0=np.float32(scan_1_trans['y0'][0]),
                            z0=np.float32(scan_1_trans['z0'][0]), 
                            u0=np.float32(scan_1_trans['u0'][0]),
                            v0=np.float32(scan_1_trans['v0'][0]), 
                            w0=np.float32(scan_1_trans['w0'][0])).to(device)

# Create loss function
loss = geomloss.SamplesLoss(loss="laplacian", blur=args.blur, 
                            backend="multiscale")

# Create Optimizer:
optimizer = torch.optim.Rprop([{'params': model.t.parameters(), 
                              'lr': 1e-2, 
                              'step_sizes': (1e-5, 1e-2)},
                               {'params': model.r.parameters(), 
                               'lr': 1e-4, 
                               'step_sizes': (1e-7, 1e-3)}])

t_temp = np.zeros(3, dtype=np.float32)
r_temp = np.zeros(3, dtype=np.float32)
values_t = np.empty((args.max_steps,3), dtype=np.float32)
values_t[:] = np.nan
values_r = np.empty((args.max_steps,3), dtype=np.float32)
values_r[:] = np.nan
for t in range(args.max_steps):
    # Copy current dict values
    # Save current dict status in values
    for i, v in enumerate(model.t.values()):
        values_t[t,i] = v.item()
        t_temp[i] = v.item()
    for i, v in enumerate(model.r.values()):
        values_r[t,i] = v.item()
        r_temp[i] = v.item()
    
    # Set model in 'training' mode
    model.train()
    
    # Run model to generate prediction
    beta_t_pred = model(beta_t)
    
    # Compute (and print) loss
    L_ab = loss(alpha_t, beta_t_pred)
    L_ab.backward()
    #print(t)
    #print(L_ab)
    
    # Use Rprop to update weights
    optimizer.step()
    optimizer.zero_grad()
    
    # See if all of our changes was less than our cutoffs
    # stop if so
    stop = True
    for i, v in enumerate(model.t.values()):
        if np.abs(v.item()-t_temp[i])>args.cutoff_t:
            stop = False
            break
    if stop==False:
        pass
    else:
        for i, v in enumerate(model.r.values()):
            if np.abs(v.item()-r_temp[i])>args.cutoff_r:
                stop = False
                break
        if stop:
            break

# Now save the output as a rigid transformation in the requested location
# First, remove any prior transformation in that directory (if it exists)
filenames = os.listdir(args.trans_output_path)
for filename in filenames:
    if filename in [args.trans_output_name + '.txt', 
                    args.trans_output_name + '.npy']:
        os.remove(os.path.join(args.trans_output_path, filename))

transform_np = np.array([(values_t[t,0], values_t[t,1], values_t[t,2], 
                       values_r[t,0], values_r[t,1], values_r[t,2])],
                      dtype=[('x0', '<f4'), ('y0', '<f4'), 
                             ('z0', '<f4'), ('u0', '<f4'),
                             ('v0', '<f4'), ('w0', '<f4')])
np.save(os.path.join(args.trans_output_path, args.trans_output_name + '.npy'), 
        transform_np)
# Now save our new transform's history dict
transform_history_dict = {
    "type": "Transform Computer",
    "git_hash": args.git_hash,
    "method": "compute_kernel_alignment",
    "input_0": scan_0_history_dict,
    "input_1": scan_1_history_dict,
    "params": {
        "max_pts": str(args.max_pts),
        "cutoff_t": str(args.cutoff_t),
        "cutoff_r": str(args.cutoff_r),
        "n_steps": str(t),
        "blur": str(args.blur)
        }
    }
f = open(os.path.join(args.trans_output_path, args.trans_output_name + '.txt'),
         'w')
json.dump(transform_history_dict, f, indent=4)
f.close()

# Finally, if we requested a plot, write the plot to a file.
if args.plot_optimization:
    f, axs = plt.subplots(3,2, figsize=(10, 20))

    names = ['dx', 'dy', 'dz', 'u', 'v', 'w']

    for i in np.arange(3):
        axs[i,0].plot(values_t[:,i], '.')
        axs[i,0].set_title(names[i])
        
        axs[i,1].plot(values_r[:,i], '.')
        axs[i,1].set_title(names[i+3])
    
#    f.savefig(os.path.join(args.plot_output_path, 'keops_' 
#                           + args.project_name_0 + '_' + args.project_name_1
#                           + '_' + args.scan_name_1 + '.png'), dpi=600,
#                           transparent=True, bbox_inches='tight')
    f.savefig(os.path.join(args.plot_output_path, 'keops_temp.png'), dpi=600,
                           transparent=True, bbox_inches='tight')