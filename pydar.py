# -*- coding: utf-8 -*-
"""
Classes and functions for registering and displaying mosaic lidar scans.

In this module we take an object oriented approach to managing single scans, 
projects (collections of scans all from the same day or 2 days sometimes), 
tiepoint lists, and scan areas (collections of projects covering the same
physical regions). Each of these categories is represented by a class.

Created on Tue Sep  8 10:46:27 2020

@author: d34763s
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree, KDTree
from scipy.special import erf, erfinv
from scipy.signal import fftconvolve
from scipy.stats import mode
from numpy.linalg import svd
import scipy.sparse as sp
from scipy.optimize import minimize, minimize_scalar
import pandas as pd
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import os
import sys
import re
import copy
import json
import math
import warnings
import time
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

try:
    import pyximport
    pyximport.install(inplace=True, language_level=3)
except ImportError:
    print('Failed to build cython_util, is the C/C++ compiler installed?')
try:
    import cython_util
except ModuleNotFoundError:
    print('cython_util was not imported, functions relying on it will fail')
try:
    import pdal
except ModuleNotFoundError:
    print('pdal not imported, functions relying on it will fail')
try:
    import open3d as o3d
except ModuleNotFoundError:
    print('open3d not imported, functions relying on it will fail')
try:
    import cv2 as cv
except ModuleNotFoundError:
    print('opencv was not imported, functions relying on it will fail')
try:
    import torch
    import gpytorch
except ModuleNotFoundError:
    print('torch and gpytorch were not imported')
try:
    import utils_find_1st as utf1st
except ModuleNotFoundError:
    print('utf1st not loaded, local max finding will not work')

class TiePointList:
    """Class contains tiepointlist object and methods.
    
    ...
    
    Attributes
    ----------
    project_name : str
        Filename of the RiSCAN project the tiepointlist comes from
    project_path : str
        Directory location of the project.
    tiepoints : Pandas dataframe
        List of tiepoint names and coordinates in project coordinate system.
    tiepoints_transformed : Pandas dataframe
        List of tiepoints transformed into a new coordinate system.
    current_transform : tuple
        Tuple with index of current transform (the one used to create 
        tiepoints_transformed).
    pwdist : Pandas dataframe
        List of unique pairwise distances between reflectors.
    dict_compare : dict
        Stores comparisons of self pwdist with other tiepointlists' pwdists.
    transforms : dict
        Stores transformations that aligns self tiepoints with others. Keyed
        on tuples (name, str_reflector_list). Each entry is a tuple 
        (reflector_list, transform, std).
    raw_history_dict : dict
        A dict containing the history of modification dependencies as a tree
        see SingleScan docstring for more details
    transformed_history_dict : dict
        Same as raw history dict for tiepoints_transformed
    trans_history_dict : dict
        for each transformation in transforms gives the node of the history
        tree keyed the same as in transforms
        
    Methods
    -------
    add_transform(name, transform, reflector_list=[], std=np.NaN)
        Adds a transform to the transforms dataframe.
    get_transform(index)
        Returns the requested numpy array.
    apply_transform(index)
        Applies a transform in transforms to update tiepoints_transformed.
    calc_pairwise_dist()
        Calculates the distances between each unique pair of reflectors.
    compare_pairwise_dist(other_tiepointlist)
        Compares pwdist with other_tiepointlist, stores result in compare_dict
    calc_transformation(other_tiepointlist, reflector_list, mode='LS')
        Calculates best fitting rigid transformation to align with other.
    plot_map(other_project_name, delaunay=False, mode='dist')
        Plots a map of the change in reflector distances.
    """
    
    def __init__(self, project_path, project_name):
        """Stores the project_path and project_name variables and loads the 
        tiepoints into a pandas dataframe"""
        self.project_path = project_path
        self.project_name = project_name
        try:
            self.tiepoints = pd.read_csv(os.path.join(project_path, project_name,
                                         'tiepoints.csv'),
                                         index_col=0, usecols=[0,1,2,3])
            self.tiepoints.sort_index(inplace=True)
        except FileNotFoundError:
            self.tiepoints = pd.DataFrame()
            warnings.warn('No tiepoints found for ' 
                          + os.path.join(project_path, project_name))
        # Ignore any tiepoints that start with t
        for label in self.tiepoints.index:
            if label[0]=='t':
                self.tiepoints.drop(label, inplace=True)
        
        # Add the identity matrix to the transform list
        self.transforms = {('identity', '') : ([], np.eye(4), np.NaN)}
        
        # Create the tiepoints_transformed dataframe
        self.tiepoints_transformed = self.tiepoints.copy(deep=True)
        self.current_transform = ('identity', '')
        
        # Create and update history dicts
        git_hash = get_git_hash()
        self.raw_history_dict = {
            "type": "Pointset Source",
            "git_hash": git_hash,
            "method": "TiePointList.__init__",
            "filename": os.path.join(project_path, project_name, 
                                     'tiepoints.csv')
            }
        self.trans_history_dict = {}
        self.trans_history_dict[('identity', '')] = {
            "type": "Transform Source",
            "git_hash": git_hash,
            "method": "TiePointList.__init__",
            "filename": ''
            }
        self.transformed_history_dict = {
            "type": "Transformer",
            "git_hash": git_hash,
            "method": "TiePointList.__init__",
            "input_0": self.raw_history_dict,
            "input_1": self.trans_history_dict[('identity', '')]
            }
        
    def add_transform(self, name, transform, reflector_list=[], std=np.NaN,
                      history_dict=None):
        """
        Add a transform to the transforms dict.

        Parameters
        ----------
        name : str
            The name of the tranformation to put in self.transforms
        transform : 4x4 ndarray
            The affine transformation in homologous coordinates.
        reflector_list : list, optional
            List of reflectors used to find this transformation (if any). 
            The default is None.
        std : float, optional
            Standard deviation of residuals between aligned reflectors in m
            if transformation is from reflectors. The default is None.
        history_dict : dict
            dict tree containing history of transform. If None then we create
            a Transform Source node with the matrix as a param. The default
            is None.

        Returns
        -------
        key : tuple
            The tuple that the tranform is keyed (indexed) in on transforms.

        """
        
        # Index into the transforms dict 
        str_reflector_list = ', '.join(reflector_list)
        self.transforms[(name, str_reflector_list)] = (reflector_list, 
                                                       transform,
                                                       std)
        
        if history_dict is None:
            warnings.warn('You are adding a transform with no history' + 
                          ' make sure this is intended')
            self.trans_history_dict[(name, str_reflector_list)] = {
                "type": "Transform Source",
                "git_hash": get_git_hash(),
                "method": "SingleScan.add_transform",
                "filename": '',
                "params": {"transform": transform.tolist()}
                }
        else:
            self.trans_history_dict[(name, str_reflector_list)] = history_dict
        
        # Return key (index) of transform
        return (name, str_reflector_list)
    
    def get_transform(self, index, history_dict=False):
        """
        Return the requested transform's array.

        Parameters
        ----------
        index : tuple
            Key for desired transform in self.transforms.

        Returns
        -------
        ndarray or (ndarray, dict)
            ndarray is 4x4 matrix of the requested transform. dict is the
            transform's history_dict

        """
        
        if history_dict:
            return (self.transforms[index][1], json.loads(json.dumps(
                self.trans_history_dict[index])))
        else:
            warnings.warn("You are getting a transform without its history")
            return self.transforms[index][1]
    
    def apply_transform(self, index):
        """
        Apply the transform in transforms to update tiepoints_transformed

        Parameters
        ----------
        index : tuple
            Index of the transform in self.transforms to apply

        Returns
        -------
        None.

        """
        
        # extract positions in homogenous coordinates
        x_vec = np.ones((4, self.tiepoints.shape[0]))
        x_vec[:-1, :] = self.tiepoints.to_numpy().T
        # apply transformation
        y_vec = np.matmul(self.transforms[index][1], x_vec)
        # replace values in self.tiepoints_transformed
        self.tiepoints_transformed[:] = y_vec[:3, :].T
        
        # Update current_transform
        self.current_transform = index
        
        # Update history dict
        self.transformed_history_dict["input_1"] = self.trans_history_dict[
            index]

    def calc_pairwise_dist(self):
        """Calculate the pairwise distances between each unique pair of 
        reflectors"""
        # Use list of dictionaries approach: https://stackoverflow.com/questions/10715965/add-one-row-to-pandas-dataframe
        rows_list = []
        for indexA, rowA in self.tiepoints.iterrows():
            for indexB, rowB in self.tiepoints.iterrows():
                if (indexA >= indexB):
                    continue
                else:
                    dict1 = {}
                    dict1.update({'rA': indexA, 'rB': indexB,
                                  'dist': np.sqrt(
                        (rowA[0] - rowB[0])**2 +
                        (rowA[1] - rowB[1])**2 +
                        (rowA[2] - rowB[2])**2)})
                    rows_list.append(dict1)
        self.pwdist = pd.DataFrame(rows_list)
        self.pwdist.set_index(['rA', 'rB'], inplace=True)
        
    def compare_pairwise_dist(self, other_tiepointlist):
        """
        Compares pairwise distances of TiePointList with the other one and 
        stores results in a dictionary whose key is the name of the other.
        
        Note, our typical usage will be to compare the current tiepointlist
        with one from a week prior. Thus the difference we calculate is this
        list's distance minus the other distance.

        Parameters
        ----------
        other_tiepointlist : TiePointList to compare to.
            These tiepoints will be represented by rB and dist_y.

        Returns
        -------
        None. But stores results in a dictionary. Keyed on name of other.

        """
        # Create Dictionary if it doesn't exist
        if not hasattr(self, 'dict_compare'):
            self.dict_compare = {}
        
        # Now Calculate pairwise distances and store
        if not hasattr(self, 'pwdist'):
            self.calc_pairwise_dist()
        if not hasattr(other_tiepointlist, 'pwdist'):
            other_tiepointlist.calc_pairwise_dist()
        df = pd.merge(self.pwdist, other_tiepointlist.pwdist, how='inner', 
                      left_index=True, right_index=True)
        df['diff'] = df['dist_x'] - df['dist_y']
        df['diff_abs'] = abs(df['diff'])
        df['strain'] = df['diff']/df['dist_y']
        df.sort_values(by='diff_abs', inplace=True)
        self.dict_compare.update({other_tiepointlist.project_name: df})
    
    def calc_transformation(self, other_tiepointlist, reflector_list, 
                            mode='LS', use_tiepoints_transformed=True,
                            yaw_angle=0):
        """
        Calculate the rigid transformation to align with other tiepointlist.
        
        See info under mode. In either mode, we start by computing the 
        centroids of the selected reflectors in both lists and create arrays
        of position relative to centroid. Then we use the singular value
        decomposition to find the rotation matrix (either in 3 dimensions or
        1 depending on mode) that best aligns the reflectors. Finally, we 
        solve for the appropriate translation based on the centroids. We store
        the result as a 4x4 matrix in self.transforms.
        
        The method used here is based off of Arun et al 1987:
            https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4767965
        
        Parameters
        ----------
        other_tiepointlist : TiePointList
            TiePointList to compare with.
        reflector_list : list
            List of tiepoints to align.
        mode : str, optional
            Mode of the transformation, in 'LS' transformation can rotate on 
            all 3 axes to find best fit (must have at least 3 reflectors). In
            'Yaw' the z axis is fixed and we are only rotating around it (it 
            still translates in all 3 dimensions). If 'trans', we only have 1
            reflector and just translate. The default is 'LS'.
        use_tiepoints_transformed : bool, optional
            Whether to use the tiepoints_transformed from the other tiepoint
            list or the raw ones. The default is True.
        yaw_angle : float, optional
            If the mode is 'Trans' this is the angle (in radians) by which to
            change the heading of the scan. The default is 0.

        Returns
        -------
        key : tuple
            The tuple that the tranform is keyed (indexed) in on transforms.

        """
        
        # Create history_dict for this operation
        history_dict = {
            "type": "Transform Computer",
            "git_hash": get_git_hash(),
            "method": 'TiePointList.calc_transformation',
            "input_1": json.loads(json.dumps(self.raw_history_dict)),
            "params": {"reflector_list": reflector_list,
                       "mode": mode,
                       "yaw_angle": yaw_angle}
            }
        # extract point lists and name as in Arun et al.
        if use_tiepoints_transformed:
            psubi_prime = other_tiepointlist.tiepoints_transformed.loc[
                reflector_list].to_numpy().T
            history_dict["input_0"] = json.loads(json.dumps(
                other_tiepointlist.transformed_history_dict))
        else:
            psubi_prime = other_tiepointlist.tiepoints.loc[
                reflector_list].to_numpy().T
            history_dict["input_0"] = json.loads(json.dumps(
                other_tiepointlist.raw_history_dict))
        psubi = self.tiepoints.loc[reflector_list].to_numpy().T
        
        # Compute centroids
        p_prime = psubi_prime.mean(axis=1).reshape((3,1))
        p = psubi.mean(axis=1).reshape((3,1))
        
        # Translate such that centroids are at zero
        qsubi_prime = psubi_prime - p_prime
        qsubi = psubi - p
        
        # Compute best fitting rotation matrix R
        if (mode=='LS'):
            # Calculate the 3x3 matrix H (Using all 3 axes)
            H = np.matmul(qsubi, qsubi_prime.T)
            # Find it's singular value decomposition
            U, S, Vh = svd(H)
            # Calculate X, the candidate rotation matrix
            X = np.matmul(Vh.T, U.T)
            # Check if the determinant of X is near 1, this should basically 
            # alsways be the case for our data
            if np.isclose(1, np.linalg.det(X)):
                R = X
            elif np.isclose(-1, np.linalg.det(X)):
                V_prime = np.array([Vh[0,:], Vh[1,:], -1*Vh[2,:]]).T
                R = np.matmul(V_prime, U.T)
                print(R)
        elif (mode=='Yaw'):
            # If we are locking the x-y plane we can modify the above process
            # to just find a rotation in 2 dimensions
            # Calculate the 2x2 matrix H
            H = np.matmul(qsubi[:2,:], qsubi_prime[:2,:].T)
            # Find it's singular value decomposition
            U, S, Vh = svd(H)
            # Calculate X, the candidate rotation matrix
            X = np.matmul(Vh.T, U.T)
            # Check if the determinant of X is near 1, this should basically 
            # alsways be the case for our data
            R = np.eye(3)
            if np.isclose(1, np.linalg.det(X)):
                R[:2, :2] = X
            elif np.isclose(-1, np.linalg.det(X)):
                V_prime = np.array([Vh[0,:], -1*Vh[1,:]]).T
                R[:2, :2] = np.matmul(V_prime, U.T)
        elif (mode=='Trans'):
            # If we are just in translational mode then rotation matrix is 
            # Determined by yaw angle.
            R = np.eye(3)
            R[0,0] = np.cos(yaw_angle)
            R[0,1] = -1*np.sin(yaw_angle)
            R[1,0] = np.sin(yaw_angle)
            R[1,1] = np.cos(yaw_angle)
        
        # Now find translation vector to align centroids
        T = p_prime - np.matmul(R, p)
        
        # Combine R and T into 4x4 matrix
        A = np.eye(4)
        A[:3, :3] = R
        A[:3, 3] = T.squeeze()
        
        # Compute standard deviation of euclidean distances.
        if not (mode=='Trans'):
            p_trans = np.matmul(R, psubi) + T
            dist = np.sqrt((np.square(psubi_prime - p_trans)).sum(axis=0))
            std = std=dist.std()
        else:
            std = np.NaN
        
        # Create history_dict for this operation
        # Add matrix to transforms, including it's history dict
        key = self.add_transform(other_tiepointlist.project_name + '_' + mode,
                                 A, reflector_list, std=std, 
                                 history_dict=history_dict)
        # Return key (index) of tranform in self.transforms.
        return key

    def plot_map(self, other_project_name, delaunay=False, mode='dist',
                 use_tiepoints_transformed=False):
        """
        Plot lines showing change in distances or strains between two scans.

        Parameters
        ----------
        other_project_name : str
            Other project must already be in dict_compare.
        delaunay : bool, optional
            Whether to plot just the lines that are part of the Delaunay 
            triangulation. The default is False.
        mode : str, optional
            If 'dist' display differences as distances, if 'strain' display 
            differences as strains. The default is 'dist'.
        use_tiepoints_transformed : bool, optional
            If true plot tiepoints at locations given by tiepoints_transformed.
            The default is False.

        Returns
        -------
        Matplotlib figure and axes objects.

        """
        # First let's just plot the reflectors and their names.
        f, ax = plt.subplots(1, 1, figsize=(10, 10))
        # Looping is inefficient but we have very few points.
        # Let's limit to just plotting the reflectors present in both
        tup = self.dict_compare[other_project_name].index.tolist()
        as_arr = np.array(list(map(lambda x: x[0], tup)) + 
                          list(map(lambda x: x[1], tup)))
        common_reflectors = np.unique(as_arr)
        if use_tiepoints_transformed:
            for index, row in self.tiepoints_transformed.iterrows():
                if (index in common_reflectors):
                    ax.scatter(row[0], row[1], s=10, c='k')
                    ax.text(row[0]+5, row[1]+5, s=index, fontsize=12)
        else:
            for index, row in self.tiepoints.iterrows():
                if (index in common_reflectors):
                    ax.scatter(row[0], row[1], s=10, c='k')
                    ax.text(row[0]+5, row[1]+5, s=index, fontsize=12)
        
        # If we are just plotting delaunay lines calculate delaunay triang.
        if delaunay:
            tri = Delaunay(self.tiepoints.loc[common_reflectors].
                           to_numpy()[:,:2])
            # Now we want to create a list of tuples matching the multiindex
            # in dict_compare
            indptr, indices = tri.vertex_neighbor_vertices
            delaunay_links = []
            for i in range(common_reflectors.size): #range(self.tiepoints.shape[0]):
                #r_start = self.tiepoints.index[i]
                r_start = common_reflectors[i]
                for neighbor in indices[indptr[i]:indptr[i+1]]:
                    #r_end = self.tiepoints.index[neighbor]
                    r_end = common_reflectors[neighbor]
                    # Preserve ordering style in multiindex
                    if (r_end > r_start):
                        delaunay_links.append((r_start, r_end))
        
        # Create appropriate diverging color map for changes in distance
        # We'll use RdBu for now and flip it so that Blue is contraction
        if mode=='dist':
            max_abs_diff = (self.dict_compare[other_project_name]
                            ['diff_abs'].to_numpy().max())
        elif mode=='strain':
            max_abs_diff = (abs(self.dict_compare[other_project_name]
                            ['strain'].to_numpy()).max())
        
        norm_color = Normalize(vmin=-1*max_abs_diff, vmax=max_abs_diff)
        
            
        # Now plot lines between pairs of reflectors
        if use_tiepoints_transformed:
            for index, row in self.dict_compare[other_project_name].iterrows():
                if (delaunay and not (index in delaunay_links)):
                    continue
                # Color indicates change, blue is shortening, red is lengthening
                if mode=='dist':
                    c = cm.RdBu_r(row['diff']/max_abs_diff + .5)
                elif mode=='strain':
                    c = cm.RdBu_r(row['strain']/max_abs_diff + .5)
                ax.plot([self.tiepoints_transformed.loc[index[0],'X[m]'],
                         self.tiepoints_transformed.loc[index[1],'X[m]']],
                        [self.tiepoints_transformed.loc[index[0],'Y[m]'],
                         self.tiepoints_transformed.loc[index[1],'Y[m]']],
                        c=c)
                if delaunay:
                    if mode=='dist':
                        ax.text((self.tiepoints_transformed.loc[index[0],'X[m]'] +
                             self.tiepoints_transformed.loc[index[1],'X[m]'])/2,
                            (self.tiepoints_transformed.loc[index[0],'Y[m]'] +
                             self.tiepoints_transformed.loc[index[1],'Y[m]'])/2,
                            s = format(row['diff'], '.2f'))
                    elif mode=='strain':
                        ax.text((self.tiepoints_transformed.loc[index[0],'X[m]'] +
                             self.tiepoints_transformed.loc[index[1],'X[m]'])/2,
                            (self.tiepoints_transformed.loc[index[0],'Y[m]'] +
                             self.tiepoints_transformed.loc[index[1],'Y[m]'])/2,
                            s = format(row['strain'], '.4f'))
        else:
            for index, row in self.dict_compare[other_project_name].iterrows():
                if (delaunay and not (index in delaunay_links)):
                    continue
                # Color indicates change, blue is shortening, red is lengthening
                if mode=='dist':
                    c = cm.RdBu_r(row['diff']/max_abs_diff + .5)
                elif mode=='strain':
                    c = cm.RdBu_r(row['strain']/max_abs_diff + .5)
                ax.plot([self.tiepoints.loc[index[0],'X[m]'],
                         self.tiepoints.loc[index[1],'X[m]']],
                        [self.tiepoints.loc[index[0],'Y[m]'],
                         self.tiepoints.loc[index[1],'Y[m]']],
                        c=c)
                if delaunay:
                    if mode=='dist':
                        ax.text((self.tiepoints.loc[index[0],'X[m]'] +
                             self.tiepoints.loc[index[1],'X[m]'])/2,
                            (self.tiepoints.loc[index[0],'Y[m]'] +
                             self.tiepoints.loc[index[1],'Y[m]'])/2,
                            s = format(row['diff'], '.3f'))
                    elif mode=='strain':
                        ax.text((self.tiepoints.loc[index[0],'X[m]'] +
                             self.tiepoints.loc[index[1],'X[m]'])/2,
                            (self.tiepoints.loc[index[0],'Y[m]'] +
                             self.tiepoints.loc[index[1],'Y[m]'])/2,
                            s = format(row['strain'], '.4f'))
            
        # Add colorbar
        f.colorbar(cm.ScalarMappable(norm=norm_color, cmap='RdBu_r'), ax=ax)
        ax.axis('equal')
        ax.set_title('Change from ' + other_project_name + ' to ' 
                     + self.project_name)
        
class SingleScan:
    """
    Container for single lidar scan and methods for displaying it.
    
    ...
    
    Attributes
    ----------
    project_path : str
        Path to folder containing all Riscan projects.
    project_name : str
        Name of Riscan project.
    scan_name : str
        Typically ScanPos0XX where XX is the scan number.
    poly : int
        Which polydata from the list to take.
    transform_dict : dict
        dict of vtkTransforms linked with this single scan.
    transform : vtkTransform
        pipelined, concatenated vtkTransform to apply to this scan.
    transformFilter : vtkTransformPolyDataFilter
        filter that applies the transform above
    filterName : str
        name of the current filter whose output is in currentFilter
    currentFilter : vtkThresholdPoints
        vtkThresholdPoints with all transformed points that haven't been 
        filtered out.
    filteredPoints : vtkThresholdPoints
        vtkThresholdPoints containing all points that have been filtered out.
    filterDict : dict
        dict of vtkFilter objects
    mapper : vtkPolyDataMapper
        vtk mapper object
    actor : vtkActor
        vtk actor object
    polydata_raw : vtkPolyData
        Raw data read in from Riscan, we will add arrays to PointData. This
        polydata's PointData includes an array Classification. This is a uint8
        array with the classification of points defined as in the LAS 
        specification from ASPRS: 
        https://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf 
        Plus additional catagories defined here
        0 : Created, Never Classified
        1 : Unclassified
        2 : Ground
        6 : Building
        7 : Low Point (Noise)
        64: High Elevation (Classified by elevation_filter)
        65: Snowflake (Classified by returnindex filter)
        66: Reflectance (high reflectance points and neighborhoods if desired)
        67: Pole
        68: Human
        69: Snowmobile
        70: Road
        71: Flag
        72: Wire
        73: Manually Removed (mostly the ship and logistics area)
    dsa_raw : vtk.numpy_interface.dataset_adapter.Polydata
        dataset adaptor object for interacting with polydata_raw
    man_class : pandas dataframe
        Dataframe containing information on manually classified points. The
        dataframe is keyed on PointId and contains: 
            X, Y, Z: position of the point in the scanner's own coordinate 
                system
            trans_XX: transformation matrix where the point was selected. To
                simplify I/O we break into the 12 components 00-11. We don't
                need 16 because the last row of a rigid transformation is
                0 0 0 1.
            Linearity, Planarity, Scattering, Verticality: the four
                Demantke2011 dimensionalities in the reference frame 
                corresponding to that transformation matrix. 
            elev: The vertical position of the point in the given reference 
                frame
            dist: The distance from the scanner
            Amplitude: measures of the pulse quality
            Classification: Manual classification (number)
        The expected usage here is that the scan was orginally loaded from a
        LAS file and that the PointId field created on that original loading
        is the same as the PointId's of the points we add to this dataframe.
        Doing otherwise may cause duplicate points which could cause errors
        later on.
    raw_history_dict : dict
        A dict containing the history of modification dependencies to the
        SingleScan as a tree. Every node in the tree contains the following
        attributes as strings: "type", "git_hash", "method". It then contains
        0 (if it's a source), 1 (filters), or 2 geometric 
        inputs, as history_dicts of themselves. Then it contains an arbitrary
        number of parameters. If the node has two input geometries, then the
        output is considered to be the first geometry acting on the zeroth
        (if some sense of ordering is required). There are two kinds of 
        geometric objects, pointsets (e.g. lidar point clouds, reflector lists)
        and transforms. 
        Node examples:
        "type": "Transformer"
        "git_hash": 
        "method":
        "input_0": pointset
        "input_1": transform
        the output of this is a pointset (input 0 transformed by input 1)
        
        "type": "Transform Computer"
        "git_hash": 
        "method":
        "input_0": pointset
        "input_1": pointset
        "params": {...}
        the output of this is a transform (aligning input 1 with input 0)
        
        "type": "Transform Concatenator"
        "git_hash": 
        "method":
        "input_0": transform
        "input_1": transform
        "params": {...}
        the output of this  a transform, the result of transforming by 
        input_0 followed by input_1
        
        "type": "Pointset Aggregator"
        "git_hash": 
        "method":
        "input_0": pointset
        "input_1": pointset
        the output of this is a pointset (input_0 and input_1 concatenated)
        
        "type": "Filter"
        "git_hash":
        "method":
        "input_0": pointset
        "params": {...}
        the output of this is a pointset that's a subset of input_0
        
        "type": "Scalar Modifier"
        "git_hash":
        "method":
        "name":
        "input_0": pointset
        "params": {...}
        the output of this is a pointset with the same geometry as input_0
        
        "type": "Pointset Source"
        "git_hash":
        "method":
        "filename": str
        "params": {...}
        the output of this is pointset. Note, this is only appropriate for
        the first time we import a pointset from RiSCAN, if we are loading
        saved intermediate data we should also load its history_dict.
        
        "type": "Transform Source"
        "git_hash":
        "method":
        "filename": str
        "params": {...}
        the output of this is a transform. If filename is empty and params
        is empty then identity transform is assumed.
    transformed_history_dict : dict
        Same structure as raw_history_dict. self.raw_history_dict should be
        the "input_0" value.
        NOTE: here input_0 is passed by reference so that if for example,
        we add arrays to polydata_raw (via "Scalar_Modifier") that carries
        through.
        "type": "Transformer"
        "git_hash": 
        "method":
        "input_0": self.raw_history_dict
        "input_1": (dict corresponding to the current transform,)
    filt_history_dict : dict
        Same structure as raw_history_dict. self.transformed_history_dict
        should be the input_0 value:
        "type": "Filter"
        "git_hash":
        "method":
        "input_0": self.transformed_history_dict
        "params": {...}
        NOTE: here input_0 is passed by reference so that if for example,
        we add arrays to polydata_raw (via "Scalar_Modifier") that carries
        through.
    trans_history_dict : dict
        for each transformation in transform_dict gives the node to the 
        history tree, keyed off the same key.
    labels : pandas dataframe
        dataframe containing category, subcategory, id, x, y, z for manually 
        labelled points (e.g. stakes). x, y, z coordinates are in the Scanners 
        Own Coordinate System.
    
    Methods
    -------
    write_scan()
        Writes the scan to a file to save the filters
    write_current_transform()
        Writes the current transform and its history_dict to files.
    read_transform()
        Reads a transform from file and places it in current transforms.
    load_man_class()
        Load the manual classification dataframe
    load_labels()
        Load the labels dataframe
    add_label(category, subcategory, id, x, y, z, transform="current")
        Add a label to the labels dataframe
    get_label_point(category, subcategory, id, transform="current")
        Get the x, y, z for a given label.
    get_labels()
        Get the labels dataframe, includes columns for transformed coordinates
    add_transform(key, matrix, history_dict=None)
        add a transform to transform_dict
    update_current_filter(class_list)
        update the current filter object with a new class_list
    add_sop()
        load the appropriate SOP matrix into transform_dict
    add_z_offset(z_offset, history_dict=None)
        add a z_offset transformation to transform_dict
    get_polydata(port=False, history_dict=False)
        Returns the polydata object for the current settings of transforms
        and filters.
    apply_transforms(transform_list)
        updates transform to be concatenation of transforms in transform list.
    random_voxel_downsample_filter(wx, wy, wz=None, seed=1234)
        Downsample point cloud with one random point per voxel.
    clear_classification(category=None)
        Reset all Classification values to 0.
    apply_man_class()
        For points that we have manually classified, set their classification.
    update_man_class(pdata, classification)
        Update the points in man_class with the points in pdata.
    update_man_class_fields(update_fields='all', update_trans=True)
        Update the man_class table with values from the fields currently in
        polydata_raw. Useful, for example if we've improved the HAG filter and
        don't want to have to repick all points.
    create_normals(radius=2, max_nn=30)
        Estimate point normals (using Open3D).
    create_z_sigma()
        For the current value of the transformation, project the pointwise
        uncertainty spheroids onto the z-axis and save in PointData.
    apply_elevation_filter(z_max)
        Filter out all points above a certain height. Sets the flag in 
        Classification to 64.
    apply_rmin_filter(buffer=0.05, count=150000)
        Assign all points very close to the scanner as snowflakes.
    apply_snowflake_filter_3(z_std_mult, leafsize):
        Filter points as snowflakes based on whether their z value in the
        transformed reference frame exceeds z_std_mult multiples of the mean
        z values for points nearby (within a bucket of size, leafsize)
    apply_snowflake_filter_returnindex(cylinder_rad, radial_precision)
        Filter snowflakes based on their return index and whether they are on
        the border of the visible region.
    apply_manual_filter(corner_coords, normal=(0, 0, 1), category=73)
        Classify points by a selection loop.
    apply_cylinder_filter(x, y, r, category=73)
        Classify all points within a horizontal distance r of a given point.
    create_scanner_actor(color="Gray", length=150)
        Create an actor for visualizing the scanner and its orientation.
    create_labels_actors(color='White', row_index=None)
        Create dataframe containing actors for each label
    create_filter_pipeline(colors)
        Create mapper and actor displaying points colored by Classification
    create_solid_pipeline(color='Green')
        Create vtk visualization pipeline with solid colors
    create_elevation_pipeline(z_min, z_max, lower_threshold=-1000,
                              upper_threshold=1000, LOD=True, 
                              cmap_name='rainbow')
        create mapper and actor for displaying points with colors by elevation
    create_reflectance()
        Create reflectance field in polydata_raw according to RiSCAN instructs.
    create_reflectance_pipeline(v_min, v_max, field='Reflectance')
        create mapper and actor for displaying points with colors by reflectance
    correct_reflectance_radial(mode)
        Adjust reflectance for radial artifact.
    reflectance_filter(threshold, radius=0, field='reflectance_radial')
        Set Classification values for high reflectance objects (and neighborhood
        if desired) to 66.
    write_npy_pdal(output_dir, filename, mode)
        Write SingleScan to numpy structured array that can be read by pdal.
    write_pdal_transformation_json(mode, input_dir, output_dir)
        Write a JSON string for PDAL such that it transforms raw scan data
        by self.transform.
    add_dist()
        Add distance from scanner to polydata_raw
    get_local_max(z_threshold, rmax, return_dist=False, return_zs=False)
        Returns the set of points that are locally maximal in the current
        transformation and, optionally, their distance from the scanner
        and z sigma.
    """
    
    def __init__(self, project_path, project_name, scan_name, 
                 import_mode=None, poly='.1_.1_.01',
                 read_scan=False, import_las=False, create_id=True,
                 las_fieldnames=None, 
                 class_list=[0, 1, 2, 70], read_dir=None, suffix='',
                 class_suffix=''):
        """
        Creates SingleScan object and transformation pipeline.
        
        Note, if a polydata folder with the desired suffix does not exist then
        we will produce many vtk warnings (so I don't recommend this)

        Parameters
        ----------
        project_path : str
            Path to folder containing all Riscan projects.
        project_name : str
            Name of Riscan project.
        scan_name : str
            Typically ScanPos0XX where XX is the scan number.
        import_mode : str, optional
            How to create polydata_raw, the base data for this SingleScan. 
            Options are: 'poly' (read from Riscan generated poly), 'read_scan'
            (read saved vtp file), 'import_las' (use pdal to import from las
            file generate by Riscan), 'empty' (create an empty polydata, 
            useful if we just want to work with transformations). 'import_npy'
            (import from npyfiles directories) If value is None, then code 
            will interpret values of read_scan and import_las
            (deprecated method of specifying which to import) to maintain
            backwards compatibility. The default is None.
        poly : str, optional
            The suffix describing which polydata to load. The default is
            '.1_.1_.01'.
        read_scan : bool, optional
            Whether to read a saved scan from file. Typically useful for
            handling filtered scans. The default is False. Deprecated,
            use import_mode.
        import_las: bool, optional
            If true (and read_scan is False) read in the las file instead of
            the polydata. The default is False.
        create_id: bool, optional
            If true and PointId's do not exist create PointIds. The default
            is True.
        las_fieldnames: list, optional
            List of fieldnames to load if we are importing from a las file
            Must include 'Points'. If None, and we are loading scans, read
            all arrays. If None and we are importing las then set to
            ['Points', 'NumberOfReturns', 'ReturnIndex', 'Reflectance',
             'Amplitude']. The default is None.
        class_list : list, optional
            List of categories this filter will return, if special value: 
            'all' Then we do not have a selection filter and we pass through 
            all points. The default is [0, 1, 2, 70].
        read_dir : str, optional
            Directory to read scan from. Defaults to npyfiles if None. The
            default is None.
        suffix : str, optional
            Suffix for npyfiles directory if we are reading scans. The default
            is '' which corresponds to the regular npyfiles directory.
        class_suffix : str, optional
            Suffix for which Classification[class_suffix].npy file to load as
            'Classification' array. The default is '' (load Classification.npy)

        Returns
        -------
        None.

        """
        # Store instance attributes
        self.project_path = project_path
        self.project_name = project_name
        self.scan_name = scan_name
        self.poly = poly
        self.class_suffix = class_suffix
        # Get git_hash
        git_hash = get_git_hash()
        
        if import_mode is None:
            # Issue a deprecated warning
            warnings.warn("Use import_mode to indicate how SingleScan object" +
                          " should load polydata.", FutureWarning)
            if read_scan:
                import_mode = 'read_scan'
            elif not import_las:
                import_mode = 'poly'
            elif import_las:
                import_mode = 'import_las'
            else:
                raise RuntimeError("You have specified an invalid combination"
                                   + " of import flags")

        # Read scan
        if import_mode=='read_scan':
            # Import directly from numpy files that we've already saved
            if read_dir is None:
                npy_path = os.path.join(self.project_path, self.project_name,
                                        'npyfiles' + suffix, self.scan_name)
            else:
                npy_path = read_dir
            
            if not os.path.isdir(npy_path):
                raise ValueError('npyfiles directory does not exist')
            # If las_fieldnames is None load all numpy files
            if las_fieldnames is None:
                filenames = os.listdir(npy_path)
                las_fieldnames = []
                for filename in filenames:
                    if re.search('.*npy$', filename):
                        las_fieldnames.append(filename)
            else:
                las_fieldnames = copy.deepcopy(las_fieldnames)
                for i in range(len(las_fieldnames)):
                    # Adjust for different Classification arrays
                    if las_fieldnames[i]=='Classification':
                        las_fieldnames[i] = 'Classification' + class_suffix
                    las_fieldnames[i] = las_fieldnames[i] + '.npy'
            
            pdata = vtk.vtkPolyData()
            self.np_dict = {}
            for k in las_fieldnames:
                try:
                    name = k.split('.')[0]
                    # Adjust for class_suffix
                    if k==('Classification' + class_suffix + '.npy'):
                        name = 'Classification'
                    self.np_dict[name] = np.load(os.path.join(npy_path, k))
                    if name=='Points':
                        pts = vtk.vtkPoints()
                        if self.np_dict[name].dtype=='float64':
                            arr_type = vtk.VTK_DOUBLE
                        elif self.np_dict[name].dtype=='float32':
                            arr_type = vtk.VTK_FLOAT
                        else:
                            raise RuntimeError('Unrecognized dtype in ' + k)
                        pts.SetData(numpy_to_vtk(self.np_dict[name], 
                                                 deep=False, 
                                                 array_type=arr_type))
                        pdata.SetPoints(pts)
                    elif name=='Normals':
                        vtk_arr = numpy_to_vtk(self.np_dict[name], 
                                               deep=False, 
                                               array_type=vtk.VTK_FLOAT)
                        vtk_arr.SetName('Normals')
                        pdata.GetPointData().SetNormals(vtk_arr)
                    elif name=='PointId':
                        vtkarr = numpy_to_vtk(self.np_dict[name], deep=False,
                                              array_type=vtk.VTK_UNSIGNED_INT)
                        vtkarr.SetName(name)
                        pdata.GetPointData().SetPedigreeIds(vtkarr)
                        pdata.GetPointData().SetActivePedigreeIds('PointId')
                    else:
                        if self.np_dict[name].dtype=='float64':
                            arr_type = vtk.VTK_DOUBLE
                        elif self.np_dict[name].dtype=='float32':
                            arr_type = vtk.VTK_FLOAT
                        elif self.np_dict[name].dtype=='int8':
                            arr_type = vtk.VTK_SIGNED_CHAR
                        elif self.np_dict[name].dtype=='uint8':
                            arr_type = vtk.VTK_UNSIGNED_CHAR
                        elif self.np_dict[name].dtype=='uint32':
                            arr_type = vtk.VTK_UNSIGNED_INT
                        else:
                            raise RuntimeError('Unrecognized dtype in ' + k)
                        vtkarr = numpy_to_vtk(self.np_dict[name], deep=False,
                                              array_type=arr_type)
                        vtkarr.SetName(name)
                        pdata.GetPointData().AddArray(vtkarr)                
                except IOError:
                    print(k + ' does not exist in ' + npy_path)
                
            # Create VertexGlyphFilter so that we have vertices for
            # displaying
            pdata.Modified()
            #vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
            #vertexGlyphFilter.SetInputData(pdata)
            #vertexGlyphFilter.Update()
            self.polydata_raw = pdata#vertexGlyphFilter.GetOutput()
            
            # Load in history dict
            # If the directory is frozen, just set the raw history 
            # dict to be a string with the location of the directory relative
            # to the root of the mosaic_lidar directory. e.g.:
            # "/ROV/[PROJECT_NAME]/npyfiles/[SCAN_NAME]/"
            try:
                f = open(os.path.join(npy_path, 'raw_history_dict.txt'))
                temp_hist_dict = json.load(f)
                f.close()
                if type(temp_hist_dict) is list:
                    if temp_hist_dict[0]=='frozen':
                        if read_dir is None:
                            # Get the scan area from the project path.
                            temp = os.path.split(self.project_path)
                            if temp[1]=='':
                                scan_area_name = os.path.split(temp[0])[1]
                            else:
                                scan_area_name = temp[1]
                            # Create name
                            self.raw_history_dict = os.path.join(
                                scan_area_name, self.project_name,
                                'npyfiles' + suffix, self.scan_name)
                        else:
                            self.raw_history_dict = read_dir
                    else:
                        raise RuntimeError('history dict in ' + npy_path +
                                           ' is a list, but first element is' +
                                           ' not "frozen"')
                else:
                    self.raw_history_dict = temp_hist_dict
            except FileNotFoundError:
                self.raw_history_dict = {
                        "type": "Pointset Source",
                        "git_hash": git_hash,
                        "method": "SingleScan.__init__",
                        "filename": os.path.join(self.project_path, 
                                                 self.project_name, 
                                                 "npyfiles"+suffix),
                        "params": {"import_mode": import_mode,
                                   "las_fieldnames": las_fieldnames}
                        }
                warnings.warn("No history dict found for " + scan_name)
        elif import_mode=='poly':
            # Match poly with polys
            reader = vtk.vtkXMLPolyDataReader()
            polys = os.listdir(os.path.join(self.project_path, 
                                            self.project_name, 'SCANS', 
                                            self.scan_name, 'POLYDATA'))
            for name in polys:
                if re.search(poly + '$', name):
                    reader.SetFileName(os.path.join(self.project_path, self.project_name,
                                            'SCANS', self.scan_name, 
                                            'POLYDATA', name, 
                                            '1.vtp'))
                    reader.Update()
                    self.polydata_raw = reader.GetOutput()
                    # We are reading from a RiSCAN output so initialize
                    # raw_history_dict as a Pointset Source"
                    self.raw_history_dict = {
                        "type": "Pointset Source",
                        "git_hash": git_hash,
                        "method": "SingleScan.__init__",
                        "filename": os.path.join(self.project_path, 
                                                 self.project_name,
                                                 'SCANS', self.scan_name, 
                                                 'POLYDATA', name, '1.vtp'),
                        "params": {"import_mode": import_mode}
                        }
                    break
        elif import_mode=='import_las':
            # If las_fieldnames is None set it
            if las_fieldnames is None:
                las_fieldnames = ['Points', 'NumberOfReturns', 'ReturnIndex', 
                                  'Reflectance', 'Amplitude']
            # import las file from lasfiles directory in project_path
            filenames = os.listdir(os.path.join(self.project_path, 
                                                self.project_name, 
                                                "lasfiles"))
            pattern = re.compile(self.scan_name + '.*las')
            matches = [pattern.fullmatch(filename) for filename in filenames]
            if any(matches):
                # Create filename input
                #filename = next(f for f, m in zip(filenames, matches) if m)
                filenames = [f for f, m in zip(filenames, matches) if m]
                json_list = [os.path.join(self.project_path, self.project_name, 
                             "lasfiles", filename) for filename in filenames]
                if len(filenames) > 1:
                    json_list.append({"type": "filters.merge"})
                json_data = json.dumps(json_list, indent=4)
                #print(json_data)
                # Load scan into numpy array
                pipeline = pdal.Pipeline(json_data)
                _ = pipeline.execute()
                
                # Create pdata and populate with points from las file
                pdata = vtk.vtkPolyData()
                
                # np_dict stores references to underlying np arrays so that
                # they do not get garbage-collected
                self.np_dict = {}
                
                for k in las_fieldnames:
                    if k=='Points':
                        self.np_dict[k] = np.hstack((
                            np.float32(pipeline.arrays[0]['X'])[:, np.newaxis],
                            np.float32(pipeline.arrays[0]['Y'])[:, np.newaxis],
                            np.float32(pipeline.arrays[0]['Z'])[:, np.newaxis]
                            ))
                        pts = vtk.vtkPoints()
                        pts.SetData(numpy_to_vtk(self.np_dict[k], 
                                                 deep=False, 
                                                 array_type=vtk.VTK_FLOAT))
                        pdata.SetPoints(pts)
                    elif k in ['NumberOfReturns', 'ReturnIndex']:
                        if k=='ReturnIndex':
                            self.np_dict[k] = pipeline.arrays[0][
                                'ReturnNumber']
                            # Fix that return number 7 should be 0
                            self.np_dict[k][self.np_dict[k]==7] = 0
                            # Now convert to return index, so -1 is last return
                            # -2 is second to last return, etc
                            self.np_dict[k] = (self.np_dict[k] - 
                                               pipeline.arrays[0]
                                               ['NumberOfReturns'])
                            self.np_dict[k] = np.int8(self.np_dict[k])
                        else:
                            self.np_dict[k] = pipeline.arrays[0][k]
        
                        vtkarr = numpy_to_vtk(self.np_dict[k],
                                              deep=False,
                                              array_type=vtk.VTK_SIGNED_CHAR)
                        vtkarr.SetName(k)
                        pdata.GetPointData().AddArray(vtkarr)
                    elif k in ['Reflectance', 'Amplitude']:
                        self.np_dict[k] = pipeline.arrays[0][k]
                        vtkarr = numpy_to_vtk(self.np_dict[k],
                                              deep=False,
                                              array_type=vtk.VTK_DOUBLE)
                        vtkarr.SetName(k)
                        pdata.GetPointData().AddArray(vtkarr)
                
                # Create VertexGlyphFilter so that we have vertices for
                # displaying
                pdata.Modified()
                #vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
                #vertexGlyphFilter.SetInputData(pdata)
                #vertexGlyphFilter.Update()
                self.polydata_raw = pdata #vertexGlyphFilter.GetOutput()
                # We're importing from LAS (RiSCAN output) so initialize
                # raw_history_dict as a Pointset Source
                self.raw_history_dict = {
                        "type": "Pointset Source",
                        "git_hash": git_hash,
                        "method": "SingleScan.__init__",
                        "filenames": json_data,
                        "params": {"import_mode": import_mode,
                                   "las_fieldnames": las_fieldnames}
                        }
            else:
                raise RuntimeError('Requested LAS file not found')
        elif import_mode=='empty':
            # Just to smooth things out, let's create a polydata with just
            # one point (and no cells)
            pts = vtk.vtkPoints()
            pts.SetNumberOfPoints(1)
            pts.SetPoint(0, 0, 0, 0)
            self.polydata_raw = vtk.vtkPolyData()
            self.polydata_raw.SetPoints(pts)
            # The pointset source in this case is empty
            self.raw_history_dict = {
                    "type": "Pointset Source",
                    "git_hash": git_hash,
                    "method": "SingleScan.__init__",
                    "filename": '',
                    "params": {"import_mode": import_mode}
                    }
        else:
            raise ValueError('Invalid import_mode provided')
        
        # Create dataset adaptor for interacting with polydata_raw
        self.dsa_raw = dsa.WrapDataObject(self.polydata_raw)
        
        # Add Classification array to polydata_raw if it's not present
        if not self.polydata_raw.GetPointData().HasArray('Classification'):
            arr = vtk.vtkUnsignedCharArray()
            arr.SetName('Classification')
            arr.SetNumberOfComponents(1)
            arr.SetNumberOfTuples(self.polydata_raw.GetNumberOfPoints())
            arr.FillComponent(0, 0)
            self.polydata_raw.GetPointData().AddArray(arr)
            self.polydata_raw.GetPointData().SetActiveScalars('Classification')
            # Update raw_history_dict to indicate that we've added 
            # Classification field (note, use json method to deepcopy because
            # it is thread safe)
            self.raw_history_dict = {"type": "Scalar Modifier",
                                     "git_hash": git_hash,
                                     "method": "SingleScan.__init__",
                                     "name": "Add Classification",
                                     "input_0": json.loads(json.dumps(
                                         self.raw_history_dict))}
        # Set Classification array as active scalars
        self.polydata_raw.GetPointData().SetActiveScalars('Classification')
        
        
        # Add PedigreeIds if they are not already present
        if create_id and not ('PointId' in 
                              list(self.dsa_raw.PointData.keys())):
            pedigreeIds = vtk.vtkTypeUInt32Array()
            pedigreeIds.SetName('PointId')
            pedigreeIds.SetNumberOfComponents(1)
            pedigreeIds.SetNumberOfTuples(self.polydata_raw.
                                          GetNumberOfPoints())
            np_pedigreeIds = vtk_to_numpy(pedigreeIds)
            np_pedigreeIds[:] = np.arange(self.polydata_raw.
                                          GetNumberOfPoints(), dtype='uint32')
            self.polydata_raw.GetPointData().SetPedigreeIds(pedigreeIds)
            self.polydata_raw.GetPointData().SetActivePedigreeIds('PointId')
            # Update raw_history_dict to indicate that we've added 
            # PointId field (note, use json method to deepcopy because
            # it is thread safe)
            self.raw_history_dict = {"type": "Scalar Modifier",
                                     "git_hash": git_hash,
                                     "method": "SingleScan.__init__",
                                     "name": "Add PointId",
                                     "input_0": json.loads(json.dumps(
                                         self.raw_history_dict))}
        
        self.polydata_raw.Modified()
        
        self.transform = vtk.vtkTransform()
        # Set mode to post-multiply, so concatenation is successive transforms
        self.transform.PostMultiply()
        self.transformFilter = vtk.vtkTransformPolyDataFilter()
        self.transformFilter.SetTransform(self.transform)
        self.transformFilter.SetInputData(self.polydata_raw)
        self.transformFilter.Update()
        # Create the transformed_history_dict, in this case 
        self.transformed_history_dict = {
            "type": "Transformer",
            "git_hash": git_hash,
            "method": "SingleScan.__init__",
            "input_0": self.raw_history_dict,
            "input_1": {"type": "Transform Source",
                        "git_hash": git_hash,
                        "method": "SingleScan.__init__",
                        "filename": ''}
            }
   
        # Create other attributes
        self.transform_dict = {}
        self.trans_history_dict = {}
        self.filterName = 'None'
        self.filterDict = {}
        
        if class_list=='all':
            self.currentFilter = self.transformFilter
        else:
            selectionList = vtk.vtkUnsignedCharArray()
            for v in class_list:
                selectionList.InsertNextValue(v)
            selectionNode = vtk.vtkSelectionNode()
            selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
            selectionNode.SetContentType(vtk.vtkSelectionNode.VALUES)
            selectionNode.SetSelectionList(selectionList)
            
            selection = vtk.vtkSelection()
            selection.AddNode(selectionNode)
            
            self.extractSelection = vtk.vtkExtractSelection()
            self.extractSelection.SetInputData(1, selection)
            self.extractSelection.SetInputConnection(0, 
                                        self.transformFilter.GetOutputPort())
            self.extractSelection.Update()
            
            # Unfortunately, extractSelection produces a vtkUnstructuredGrid
            # so we need to use vtkGeometryFilter to convert to polydata
            self.currentFilter = vtk.vtkGeometryFilter()
            self.currentFilter.SetInputConnection(self.extractSelection
                                                  .GetOutputPort())
            self.currentFilter.Update()
        # Create filt_history_dict
        self.filt_history_dict = {
            "type": "Filter",
            "git_hash": git_hash,
            "method": "SingleScan.__init__",
            "input_0": self.transformed_history_dict,
            "params": {"class_list": class_list}}
        
    def write_scan(self, write_dir=None, class_list=None, suffix='',
                   freeze=False, overwrite_frozen=False):
        """
        Write the scan to a collection of numpy files.
        
        This enables us to save the Classification field so we don't need to 
        run all of the filters each time we load data. Additionally, npy files
        are much faster to load than vtk files. Finally, we need to write
        the history_dict to this directory as well.
        
        Parameters
        ----------
        write_dir: str, optional
            Directory to write scan files to. If None write default npyfiles
            location. The default is None.
        class_list: list, optional
            Whether to first filter the data so that we only write points whose
            Classification values are in class_list. If None do not filter.
            The default is None.
        suffix: str, optional
            Suffix for writing to the correct npyfiles directory. The default
            is ''.
        freeze: bool, optional
            Indicate whether the written files should be 'frozen'. Frozen files
            have the first element of their history dict set as 'frozen', and 
            we will store the path to the file in subsequent history dicts
            rather than the history dict itself to save space. The default 
            is False.
        overwrite_frozen : bool, optional
            If the pre-existing files are frozen, overwrite (by default
            attempting to delete a frozen file will raise an error)
            The default is False.

        Returns
        -------
        None.

        """
        
        npy_dir = "npyfiles" + suffix
        
        
        if write_dir is None:
            # If the write directory doesn't exist, create it
            if not os.path.isdir(os.path.join(self.project_path, 
                                              self.project_name, npy_dir)):
                os.mkdir(os.path.join(self.project_path, self.project_name, 
                                      npy_dir))
            # Within npyfiles we need a directory for each scan
            if not os.path.isdir(os.path.join(self.project_path, 
                                              self.project_name, npy_dir, 
                                              self.scan_name)):
                os.mkdir(os.path.join(self.project_path, self.project_name, 
                                      npy_dir, self.scan_name))
            write_dir = os.path.join(self.project_path, self.project_name, 
                                     npy_dir, self.scan_name)
        
        # Delete old saved SingleScan files in the directory
        # If there is a history_dict file, check if it's frozen first
        try:
            f = open(os.path.join(write_dir, 'raw_history_dict.txt'))
            old_history_dict = json.load(f)
            f.close()
        except FileNotFoundError:
            old_history_dict = [None, None]
        if (type(old_history_dict) is list) and (
            old_history_dict[0]=='frozen') and (overwrite_frozen==False):
            raise RuntimeError('You are trying to overwrite files in ' +
                               write_dir + ' which is frozen. If you want ' +
                               'to do this set overwrite_frozen=True')
        else:
            for f in os.listdir(write_dir):
                os.remove(os.path.join(write_dir, f))

        # If class_list is None just write raw data
        if class_list is None:
            # Save Points
            np.save(os.path.join(write_dir, 'Points.npy'), self.dsa_raw.Points)
            # Save Normals
            if not self.polydata_raw.GetPointData().GetNormals() is None:
                np.save(os.path.join(write_dir, 'Normals.npy'), vtk_to_numpy(
                    self.polydata_raw.GetPointData().GetNormals()))
            # Save arrays
            for name in self.dsa_raw.PointData.keys():
                np.save(os.path.join(write_dir, name), 
                        self.dsa_raw.PointData[name])
            # Save history_dict
            new_hist_dict = self.raw_history_dict
        else:
            ind = np.isin(self.dsa_raw.PointData['Classification'], class_list)

            # Save Points
            np.save(os.path.join(write_dir, 'Points.npy'), 
                    self.dsa_raw.Points[ind, :])
            # Save Normals if we have them
            if not self.polydata_raw.GetPointData().GetNormals() is None:
                np.save(os.path.join(write_dir, 'Normals.npy'), vtk_to_numpy(
                    self.polydata_raw.GetPointData().GetNormals())[ind, :])
            # Save arrays
            for name in self.dsa_raw.PointData.keys():
                np.save(os.path.join(write_dir, name), 
                        self.dsa_raw.PointData[name][ind])
            # Save history_dict
            new_hist_dict = {
                "type": "Filter",
                "git_hash": get_git_hash(),
                "method": "SingleScan.write_scan",
                "input_0": self.raw_history_dict,
                "params": {"class_list": class_list}
                }

        # If freezing files, set to be 'read-only'
        if freeze:
            f = open(os.path.join(write_dir, 'raw_history_dict.txt'), 'w')
            json.dump(['frozen', new_hist_dict], f, indent=4)
            f.close()
        else:
            f = open(os.path.join(write_dir, 'raw_history_dict.txt'), 'w')
            json.dump(new_hist_dict, f, indent=4)
            f.close()
    
    def write_current_transform(self, write_dir=None, name='current_transform'
                                , mode='rigid', suffix='', freeze=False,
                                overwrite_frozen=False):
        """
        Write the current tranform and its history_dict to files.

        Parameters
        ----------
        write_dir : str, optional
            The path where to write the transform. If None we'll write to 
            project_path/project_name/transforms/scan_name. The default is 
            None.
        name : str, optional
            The name to give the output files. The default is 
            'current_transform'.
        mode : str, optional
            What type of transformation it is. Currently the only option is
            'rigid' (6 components). The default is 'rigid'.
        suffix : str, optional
            Suffix for transforms directory if we are reading scans. 
            The default is '' which corresponds to the regular transforms
            directory.
        freeze: bool, optional
            Indicate whether the written files should be 'frozen'. Frozen files
            have the first element of their history dict set as 'frozen', and 
            we will store the path to the file in subsequent history dicts
            rather than the history dict itself to save space. The default 
            is False.
        overwrite_frozen : bool, optional
            If the pre-existing files are frozen, overwrite (by default
            attempting to delete a frozen file will raise an error)
            The default is False.

        Returns
        -------
        None.

        """
        
        if write_dir is None:
            write_dir = os.path.join(self.project_path, self.project_name,
                                     'transforms' + suffix, self.scan_name)
        # If the write_dir doesn't exist, create it
        if not os.path.isdir(os.path.join(self.project_path, self.project_name
                                          , 'transforms' + suffix)):
            os.mkdir(os.path.join(self.project_path, self.project_name
                                          , 'transforms' + suffix))
        if not os.path.isdir(write_dir):
            os.mkdir(write_dir)
        
        # If the files already exist remove them, check if preexisting frozen
        try:
            f = open(os.path.join(write_dir, name + '.txt'))
            old_history_dict = json.load(f)
            f.close()
        except FileNotFoundError:
            old_history_dict = [None, None]
        if (type(old_history_dict) is list) and (
            old_history_dict[0]=='frozen') and (overwrite_frozen==False):
            raise RuntimeError('You are trying to overwrite file ' +
                               name + '.txt which is frozen. If you want ' +
                               'to do this set overwrite_frozen=True')
        else:
            filenames = os.listdir(write_dir)
            for filename in filenames:
                if filename in [name + '.txt', name + '.npy']:
                    os.remove(os.path.join(write_dir, filename))
        
        if mode=='rigid':
            # Convert the current transform into position and orientation
            # NOTE VTK ANGLES IN DEGREES, WE CONVERT TO radians
            pos = self.transform.GetPosition()
            ori = np.array(self.transform.GetOrientation()) * np.pi / 180
            transform_np = np.array([(pos[0], pos[1], pos[2], 
                                   ori[0], ori[1], ori[2])],
                                  dtype=[('x0', '<f8'), ('y0', '<f8'), 
                                         ('z0', '<f8'), ('u0', '<f8'),
                                         ('v0', '<f8'), ('w0', '<f8')])
            np.save(os.path.join(write_dir, name + '.npy'), transform_np)
        else:
            raise ValueError('mode must be rigid')
        
        # Now write the history dict
        f = open(os.path.join(write_dir, name + '.txt'), 'w')
        if freeze:
            json.dump(['frozen', self.transformed_history_dict["input_1"]], f, 
                      indent=4)
        else:
            json.dump(self.transformed_history_dict["input_1"], f, indent=4)
        f.close()
    
    def read_transform(self, read_dir=None, name='current_transform', 
                       suffix=''):
        """
        Read the requested transform (if it exists)

        Parameters
        ----------
        read_dir : str, optional
            Path to read transform from. If None read from /project_path/
            project_name/transforms/scan_name/. The default is None.
        name : str, optional
            Name of the transform to read. The default is 'current_transform'.
        suffix : str, optional
            Suffix for transforms directory if we are reading scans. 
            The default is '' which corresponds to the regular transforms 
            directory.

        Returns
        -------
        None.

        """
        
        # If read_dir is None set to default
        if read_dir is None:
            read_dir = os.path.join(self.project_path, self.project_name,
                                    'transforms' + suffix, self.scan_name)
            default_dir = True
        # Load the transform
        transform_np = np.load(os.path.join(read_dir, name + '.npy'))

        # Load the history dict
        try:
            f = open(os.path.join(read_dir, name + '.txt'))
            temp_hist_dict = json.load(f)
            f.close()
            if type(temp_hist_dict) is list:
                if temp_hist_dict[0]=='frozen':
                    if default_dir:
                        # Get the scan area from the project path.
                        temp = os.path.split(self.project_path)
                        if temp[1]=='':
                            scan_area_name = os.path.split(temp[0])[1]
                        else:
                            scan_area_name = temp[1]
                        # Create name
                        new_hist_dict = os.path.join(
                            scan_area_name, self.project_name,
                            'transforms' + suffix, self.scan_name,
                            name + '.txt')
                    else:
                        new_hist_dict = read_dir
                else:
                    raise RuntimeError('history dict in ' + read_dir +
                                       ' is a list, but first element is' +
                                       ' not "frozen"')
            else:
                new_hist_dict = temp_hist_dict
        except FileNotFoundError:
            new_hist_dict = {
                    "type": "Transform Source",
                    "git_hash": git_hash,
                    "method": "SingleScan.read_transform",
                    "filename": os.path.join(read_dir, name + '.npy'),
                    }
            warnings.warn("No history dict found for " + name)
        
        # Check which kind of transformation it is
        if transform_np.dtype==[('x0', '<f8'), ('y0', '<f8'), ('z0', '<f8'),
                                ('u0', '<f8'), ('v0', '<f8'), ('w0', '<f8')]:
            # Then we're dealing with a rigid transformation
            # Create 4x4 matrix
            u = transform_np[0]['u0']
            v = transform_np[0]['v0']
            w = transform_np[0]['w0']
            c = np.cos
            s = np.sin
            
            Rx = np.array([[1, 0, 0, 0],
                          [0, c(u), -s(u), 0],
                          [0, s(u), c(u), 0],
                          [0, 0, 0, 1]])
            Ry = np.array([[c(v), 0, s(v), 0],
                           [0, 1, 0, 0],
                           [-s(v), 0, c(v), 0],
                           [0, 0, 0, 1]])
            Rz = np.array([[c(w), -s(w), 0, 0],
                          [s(w), c(w), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
            # Order of rotations in vtk is Pitch, then Roll, then Yaw
            M = Rz @ Rx @ Ry
            # Now add translation components
            M[0, 3] = transform_np[0]['x0']
            M[1, 3] = transform_np[0]['y0']
            M[2, 3] = transform_np[0]['z0']
        elif transform_np.dtype==[('x0', '<f4'), ('y0', '<f4'), ('z0', '<f4'),
                                ('u0', '<f4'), ('v0', '<f4'), ('w0', '<f4')]:
            # Then we're dealing with a rigid transformation
            # Create 4x4 matrix
            u = transform_np[0]['u0']
            v = transform_np[0]['v0']
            w = transform_np[0]['w0']
            c = np.cos
            s = np.sin
            
            Rx = np.array([[1, 0, 0, 0],
                          [0, c(u), -s(u), 0],
                          [0, s(u), c(u), 0],
                          [0, 0, 0, 1]])
            Ry = np.array([[c(v), 0, s(v), 0],
                           [0, 1, 0, 0],
                           [-s(v), 0, c(v), 0],
                           [0, 0, 0, 1]])
            Rz = np.array([[c(w), -s(w), 0, 0],
                          [s(w), c(w), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
            # Order of rotations in vtk is Pitch, then Roll, then Yaw
            M = Rz @ Rx @ Ry
            # Now add translation components
            M[0, 3] = transform_np[0]['x0']
            M[1, 3] = transform_np[0]['y0']
            M[2, 3] = transform_np[0]['z0']
        else:
            raise RuntimeError('transform does not match known format')

        # Add to transform dict, include history dict
        self.add_transform(name, M, new_hist_dict)
            
    
    def load_man_class(self):
        """
        Load the man_class dataframe. Create if it does not exist.

        Returns
        -------
        None.

        """
        
        # Check if directory for manual classifications exists and create
        # if it doesn't.
        create_df = False
        if os.path.isdir(os.path.join(self.project_path, self.project_name, 
                         'manualclassification')):
            # Check if file exists
            if os.path.isfile(os.path.join(self.project_path, self.project_name, 
                              'manualclassification', self.scan_name +
                              '.parquet')):
                self.man_class = pd.read_parquet(os.path.join(self.project_path, 
                                                 self.project_name,
                                                 'manualclassification', 
                                                 self.scan_name + '.parquet'),
                                                 engine="pyarrow")
            # otherwise create dataframe
            else:
                create_df = True
        else:
            # Create directory and dataframe
            create_df = True
            os.mkdir(os.path.join(self.project_path, self.project_name, 
                     'manualclassification'))
        
        if create_df:
            self.man_class = pd.DataFrame({'X':
                                           pd.Series([], 
                                                     dtype=np.float32),
                                           'Y':
                                           pd.Series([], 
                                                     dtype=np.float32),
                                           'Z':
                                           pd.Series([], 
                                                     dtype=np.float32),
                                           'trans_00':
                                           pd.Series([], 
                                                     dtype=np.double),
                                           'trans_01':
                                           pd.Series([], 
                                                     dtype=np.double),
                                           'trans_02':
                                           pd.Series([], 
                                                     dtype=np.double),
                                           'trans_03':
                                           pd.Series([], 
                                                     dtype=np.double),
                                           'trans_04':
                                           pd.Series([], 
                                                     dtype=np.double),
                                           'trans_05':
                                           pd.Series([], 
                                                     dtype=np.double),
                                           'trans_06':
                                           pd.Series([], 
                                                     dtype=np.double),
                                           'trans_07':
                                           pd.Series([], 
                                                     dtype=np.double),
                                           'trans_08':
                                           pd.Series([], 
                                                     dtype=np.double),
                                           'trans_09':
                                           pd.Series([], 
                                                     dtype=np.double),
                                           'trans_10':
                                           pd.Series([], 
                                                     dtype=np.double),
                                           'trans_11':
                                           pd.Series([], 
                                                     dtype=np.double),
                                           'Linearity':
                                           pd.Series([], 
                                                     dtype=np.float32),
                                           'Planarity':
                                           pd.Series([], 
                                                     dtype=np.float32),
                                           'Scattering':
                                           pd.Series([], 
                                                     dtype=np.float32),
                                           'Verticality':
                                           pd.Series([], 
                                                     dtype=np.float32),
                                           'Density':
                                           pd.Series([], 
                                                     dtype=np.float32),
                                           'Anisotropy':
                                           pd.Series([], 
                                                     dtype=np.float32),
                                           'HeightAboveGround':
                                           pd.Series([], 
                                                     dtype=np.float32),
                                           'dist':
                                           pd.Series([], 
                                                     dtype=np.float32),
                                           'Amplitude':
                                           pd.Series([], 
                                                     dtype=np.float32),
                                           'Classification':
                                           pd.Series([], 
                                                     dtype=np.uint8)})
            self.man_class.index.name = 'PointId'
        
    def load_labels(self):
        """
        Load the labels dataframe. Create if it does not exist.

        Returns
        -------
        None.

        """
        
        # Check if directory for manual classifications exists and create
        # if it doesn't.
        create_df = False
        if os.path.isdir(os.path.join(self.project_path, self.project_name, 
                         'manualclassification')):
            # Check if file exists
            if os.path.isfile(os.path.join(self.project_path, self.project_name, 
                              'manualclassification', self.scan_name +
                              '_labels.parquet')):
                self.labels = pd.read_parquet(os.path.join(self.project_path, 
                                                 self.project_name,
                                                 'manualclassification', 
                                                 self.scan_name 
                                                 + '_labels.parquet'),
                                                 engine="pyarrow")
            # otherwise create dataframe
            else:
                create_df = True
        else:
            # Create directory and dataframe
            create_df = True
            os.mkdir(os.path.join(self.project_path, self.project_name, 
                     'manualclassification'))
        
        if create_df:
            self.labels = pd.DataFrame({
                                        'category':
                                        pd.Series([], dtype='string'),
                                        'subcategory':
                                        pd.Series([], dtype='string'),
                                        'id':
                                        pd.Series([], dtype='string'),
                                        'x':
                                        pd.Series([], 
                                                 dtype=np.float32),
                                        'y':
                                        pd.Series([], 
                                                  dtype=np.float32),
                                        'z':
                                        pd.Series([], 
                                                  dtype=np.float32),
                                        })
            self.labels.set_index(['category', 'subcategory', 'id'], 
                                  inplace=True)

    def add_label(self, category, subcategory, id, x, y, z, 
                  transform='current'):
        """
        Add a label to the labels dataframe.

        Parameters
        ----------
        category : str
            The category that the label belongs to e.g. 'stake'
        subcategory : str
            The subcategory that the label belongs to e.g. 'ridge_ranch'
        id : str
            id for the label e.g. '52'. Note, will be cast to string
        x : float
            x coordinate for label
        y : float
            y coordinate for label
        z : float
            z coordinate for label
        transform : 'current' or 'raw'
            The reference frame for the (x, y, z) coordinates. The labels
            dataframe is in the scanner's own coordinate system. So if
            transform=='current', the point will be inverse transformed into
            the SOCS. Otherwise, if transform=='raw' add point as is. The
            default is 'current'

        Returns
        -------
        None.

        """

        if not hasattr(self, 'labels'):
            raise RuntimeError('labels dataframe does not exist, load it first')
        if transform=='current':
            invTransform = self.transform.GetInverse()
            x_socs, y_socs, z_socs = invTransform.TransformPoint(x, y, z)
        elif transform=='raw':
            x_socs = x
            y_socs = y
            z_socs = z
        else:
            raise ValueError('transform must be "current" or "raw"')

        # Add to labels df
        temp_df = pd.DataFrame(data={'category': category,
                               'subcategory': subcategory,
                               'id': str(id),
                               'x': x_socs,
                               'y': y_socs,
                               'z': z_socs,
                               }, index=[0])
        temp_df.set_index(['category', 'subcategory', 'id'], inplace=True)
        self.labels = temp_df.combine_first(self.labels)
        #self.labels.drop_duplicates(inplace=True)

        # Write to file to save
        self.labels.to_parquet(os.path.join(self.project_path, 
                                                 self.project_name, 
                                                 'manualclassification', 
                                                 self.scan_name 
                                                 + '_labels.parquet'),
                                                 engine="pyarrow", 
                                                 compression=None)

    def get_label_point(self, category, subcategory, id, transform='current'):
        """
        Get the x, y, z for a given label.

        Parameters
        ----------
        category : str
            The category that the label belongs to e.g. 'stake'
        subcategory : str
            The subcategory that the label belongs to e.g. 'ridge_ranch'
        id : str
            id for the label e.g. '52'. Note, will be cast to string
        transform : 'current' or 'raw'
            The reference frame for to return the point in. If 'current' apply
            current transform to point. If 'raw' return point as is. The default
            is 'current'

        Returns
        -------
        (3,) ndarray

        """

        # Get point
        point = self.labels.loc[category, subcategory, id].values

        if transform=='current':
            point = self.transform.TransformPoint(point)
        elif transform=='raw':
            # do nothing
            pass
        else:
            raise RuntimeError("Unsupported transform requested.")

        return point

    def get_labels(self):
        """
        Get the labels dataframe, includes columns for transformed coordinates

        Returns
        -------
        pandas Dataframe

        """

        df = self.labels.copy(deep=True)

        # Add columns for scan_name and project_name and add to index
        df['scan_name'] = self.scan_name
        df['project_name'] = self.project_name
        df.set_index(['project_name', 'scan_name'], append=True, inplace=True)

        # Transform points, we can do this in a for loop because we'll never
        # have that many labels
        df['x_trans'] = 0.0
        df['y_trans'] = 0.0
        df['z_trans'] = 0.0
        for i in range(df.shape[0]):
            df['x_trans'].iat[i], df['y_trans'].iat[i], df['z_trans'].iat[i] = (
                self.transform.TransformPoint(df['x'].iat[i],
                                              df['y'].iat[i],
                                              df['z'].iat[i]))

        return df
    
    def add_transform(self, key, matrix, history_dict=None):
        """
        Adds a new transform to the transform_dict

        Parameters
        ----------
        key : str
            Name of the tranform (e.g. 'sop')
        matrix : 4x4 array-like
            4x4 matrix of transformation in homologous coordinates.
        history_dict : dict
            dict tree containing history of transform. If None then we create
            a Transform Source node with the matrix as a param. The default
            is None.

        Returns
        -------
        None.

        """
        
        # Create vtk transform object
        vtk4x4 = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk4x4.SetElement(i, j, matrix[i, j])
        transform = vtk.vtkTransform()
        transform.SetMatrix(vtk4x4)
        # Add transform to transform_dict
        self.transform_dict.update({key : transform})
        if history_dict is None:
            warnings.warn('You are adding a transform with no history' + 
                          ' make sure this is intended')
            self.trans_history_dict[key] = {
                "type": "Transform Source",
                "git_hash": get_git_hash(),
                "method": "SingleScan.add_transform",
                "filename": '',
                "params": {"matrix": matrix.tolist()}
                }
        else:
            self.trans_history_dict[key] = history_dict
    
    def update_current_filter(self, class_list):
        """
        Set the current filter to the new class list

        Parameters
        ----------
        class_list : list
            List of categories this filter will return, if special value: 
            'all' Then we do not have a selection filter and we pass through 
            all points.

        Returns
        -------
        None.

        """
        
        if class_list=='all':
            self.currentFilter = self.transformFilter
        else:
            selectionList = vtk.vtkUnsignedCharArray()
            for v in class_list:
                selectionList.InsertNextValue(v)
            selectionNode = vtk.vtkSelectionNode()
            selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
            selectionNode.SetContentType(vtk.vtkSelectionNode.VALUES)
            selectionNode.SetSelectionList(selectionList)
            
            selection = vtk.vtkSelection()
            selection.AddNode(selectionNode)
            
            self.extractSelection = vtk.vtkExtractSelection()
            self.extractSelection.SetInputData(1, selection)
            self.extractSelection.SetInputConnection(0, 
                                        self.transformFilter.GetOutputPort())
            self.extractSelection.Update()
            
            # Unfortunately, extractSelection produces a vtkUnstructuredGrid
            # so we need to use vtkGeometryFilter to convert to polydata
            self.currentFilter = vtk.vtkGeometryFilter()
            self.currentFilter.SetInputConnection(self.extractSelection
                                                  .GetOutputPort())
            self.currentFilter.Update()
        
        # Create filt_history_dict
        self.filt_history_dict = {
            "type": "Filter",
            "git_hash": get_git_hash(),
            "method": "SingleScan.update_current_filter",
            "input_0": self.transformed_history_dict,
            "params": {"class_list": class_list}}
    
    def add_sop(self):
        """
        Add the sop matrix to transform_dict. Must have exported from RiSCAN

        Returns
        -------
        None.

        """
        
        trans = np.genfromtxt(os.path.join(self.project_path, self.project_name, 
                              self.scan_name + '.DAT'), delimiter=' ')
        self.add_transform('sop', trans, history_dict={
            "type": "Transform Source",
            "git_hash": get_git_hash(),
            "method": "SingleScan.add_sop",
            "filename": os.path.join(self.project_path, self.project_name, 
                              self.scan_name + '.DAT')
            })
        
    def add_z_offset(self, z_offset, history_dict=None):
        """
        Adds a uniform z offset to the scan

        Parameters
        ----------
        z_offset : float
            z offset to add in meters.
        history_dict : dict
            dict tree containing history of transform. If None then we create
            a Transform Source node with the z_offset as a param. If the z
            offset was computed from an upstream source (like ScanArea.z_align
            ) then that information should be passed in in history_dict.
            The default is None.

        Returns
        -------
        None.

        """
        
        trans = np.eye(4)
        trans[2, 3] = z_offset
        if history_dict is None:
            warnings.warn('You are adding a transform with no history' + 
                          ' make sure this is intended')
            history_dict = {
                "type": "Transform Source",
                "git_hash": get_git_hash(),
                "method": "SingleScan.add_z_offset",
                "filename": '',
                "params": {"z_offset": z_offset}
                }
        self.add_transform('z_offset', trans, history_dict=history_dict)
        
    def get_polydata(self, port=False, history_dict=False):
        """
        Returns vtkPolyData of scan with current transforms and filters.
        
        Parameters
        ----------
        port : bool, optional
            Whether to return an output connection instead of a polydata.
            The default is False
        history_dict : bool, optional
            If history, also return the history dict for this polydata. The
            default is False.

        Returns
        -------
        vtkPolyData or vtkAlgorithmOutput.

        """
        if not history_dict:
            warnings.warn("You are passing a SingleScan's PolyData without" +
                          "it's history.")
        
        if port:
            if history_dict:
                return (self.currentFilter.GetOutputPort(), 
                        self.filt_history_dict)
            else:    
                return self.currentFilter.GetOutputPort()
        else:
            if history_dict:
                return (self.currentFilter.GetOutput(), 
                        json.loads(json.dumps(self.filt_history_dict)))
            else:
                return self.currentFilter.GetOutput()
    
    def apply_transforms(self, transform_list):
        """
        Update transform to be a concatenation of transform_list.
        
        Clears existing transform!

        Parameters
        ----------
        transform_list : list
            str in list must be keys in transform_dict. Transformations are 
            applied in the same order as in the list (postmultiply order)

        Returns
        -------
        None.

        """
        # Reset transform to the identity
        self.transform.Identity()
        git_hash = get_git_hash()
        
        
        for i, key in enumerate(transform_list):
            try:
                self.transform.Concatenate(self.transform_dict[key])
                
            except Exception as e:
                print("Requested transform " + key + " is not in " +
                      "transform_dict")
                print(e)
            if i==0:
                # If we're the first transform in the list, start the tree
                # of transformations
                temp_trans_dict = json.loads(json.dumps(
                    self.trans_history_dict[key]))
            else:
                # Otherwise create a Transform concatenator node
                temp_trans_dict = {
                    "type": "Transform Concatenator",
                    "git_hash": git_hash,
                    "method": 'SingleScan.apply_transforms',
                    "input_0": json.loads(json.dumps(temp_trans_dict)),
                    "input_1": json.loads(json.dumps(
                        self.trans_history_dict[key]))
                    }
        # Overwrite input_1 in the transformed_history_dict so that it reflect
        # the current transform
        self.transformed_history_dict["input_1"] = json.loads(json.dumps(
            temp_trans_dict))
            
        self.transformFilter.Update()
        self.currentFilter.Update()

        
    def random_voxel_downsample_filter(self, wx, wy, wz=None, seed=1234):
        """
        Downsample point cloud with one random point per voxel.
        
        This filter takes, as input, the current transformed, filtered
        polydata.
        
        Executing this will overwrite polydata_raw!!

        Parameters
        ----------
        wx : float
            Voxel x dimension in m.
        wy : float
            Voxel y dimension in m.
        wz : float, optional
            Voxel z dimension in m. If none then we just downsample
            horizontally. The default is None.
        seed : int, optional
            Random seed for the shuffler. The default is 1234.

        Returns
        -------
        None.

        """
        
        # Step 1, create shuffled points from current polydata
        pdata, history_dict = self.get_polydata(history_dict=True)
        filt_pts = vtk_to_numpy(pdata.GetPoints().GetData())
        point_ids = vtk_to_numpy(pdata.GetPointData()
                                 .GetArray('PointId'))
        rng = np.random.default_rng(seed=seed)
        shuff_ind = np.arange(filt_pts.shape[0])
        rng.shuffle(shuff_ind)
        shuff_pts = filt_pts[shuff_ind, :]
        shuff_point_ids = point_ids[shuff_ind]
        
        # Step 2, bin and downsample
        if wz is None:
            w = [wx, wy]
            edges = 2*[None]
            nbin = np.empty(2, np.int_)
            
            for i in range(2):
                edges[i] = (np.arange(int(np.ceil((shuff_pts.max(axis=0)[i] - 
                                                  shuff_pts.min(axis=0)[i])
                                                 /w[i]))
                                     + 1, dtype=np.float32) 
                            * w[i] + shuff_pts.min(axis=0)[i])
                # needed to avoid min point falling out of bounds
                edges[i][0] = edges[i][0] - 0.0001 
                nbin[i] = len(edges[i]) + 1
            
            Ncount = tuple(np.searchsorted(edges[i], shuff_pts[:,i], 
                                           side='right') for i in range(2))
        else:
            w = [wx, wy, wz]
            edges = 3*[None]
            nbin = np.empty(3, np.int_)
            
            for i in range(3):
                edges[i] = (np.arange(int(np.ceil((shuff_pts.max(axis=0)[i] - 
                                                  shuff_pts.min(axis=0)[i])
                                                 /w[i]))
                                     + 1, dtype=np.float32) 
                            * w[i] + shuff_pts.min(axis=0)[i])
                # needed to avoid min point falling out of bounds
                edges[i][0] = edges[i][0] - 0.0001 
                nbin[i] = len(edges[i]) + 1
            
            Ncount = tuple(np.searchsorted(edges[i], shuff_pts[:,i], 
                                           side='right') for i in range(3))
        
        xyz = np.ravel_multi_index(Ncount, nbin)
        
        # We want to take just one random point from each bin. Since we've 
        # shuffled the points, the first point we find in each bin will 
        # suffice. Thus we use the unique function
        _, inds = np.unique(xyz, return_index=True)
        
        # Now apply pedigree id selection to update polydata raw
        np_pedigreeIds = shuff_point_ids[inds]
        if not np_pedigreeIds.dtype==np.uint32:
            raise RuntimeError('np_pedigreeIds is not type np.uint32')
        pedigreeIds = numpy_to_vtk(np_pedigreeIds, deep=False, array_type=
                                   vtk.VTK_UNSIGNED_INT)
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(1) # we want to select points
        selectionNode.SetContentType(2) # 2 corresponds to pedigreeIds
        selectionNode.SetSelectionList(pedigreeIds)
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)
        extractSelection = vtk.vtkExtractSelection()
        extractSelection.SetInputData(0, self.polydata_raw)
        extractSelection.SetInputData(1, selection)
        extractSelection.Update()
        #vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
        #vertexGlyphFilter.SetInputConnection(extractSelection.GetOutputPort())
        #vertexGlyphFilter.Update()
        geoFilter = vtk.vtkGeometryFilter()
        geoFilter.SetInputConnection(extractSelection.GetOutputPort())
        geoFilter.Update()
        self.polydata_raw = geoFilter.GetOutput()
        self.dsa_raw = dsa.WrapDataObject(self.polydata_raw)
        
        # Update filters
        self.transformFilter.SetInputData(self.polydata_raw)
        self.transformFilter.Update()
        self.currentFilter.Update()
        
        # Update history dicts
        self.raw_history_dict = {
            "type": "Filter",
            "git_hash": get_git_hash(),
            "method": "SingleScan.random_voxel_downsample_filter",
            "input_0": history_dict,
            "params": {"wx": wx, "wy": wy, "wz": wz, "seed": seed}
            }
        self.transformed_history_dict["input_0"] = self.raw_history_dict
        
    
    def clear_classification(self, category=None):
        """
        Reset Classification for all points to 0

        Parameters
        ----------
        category : int, optional
            If given, will only reset classifications for the given category
            The default is None (reset all).

        Returns
        -------
        None.

        """
        
        if category is None:
            self.dsa_raw.PointData['Classification'][:] = 0
        else:
            self.dsa_raw.PointData['Classification'][
             self.dsa_raw.PointData['Classification']==category] = 0
        # Update currentTransform
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()
        # Update raw_history_dict
        self.raw_history_dict = {
            "type": "Scalar Modifier",
            "git_hash": get_git_hash(),
            "method": "SingleScan.clear_classification",
            "name": "Reset Classification to zero",
            "input_0": json.loads(json.dumps(self.raw_history_dict)),
            "params": {"category": category}
            }
        self.transformed_history_dict["input_0"] = self.raw_history_dict
    
    def apply_man_class(self):
        """
        Set the classification of manually classified points in polydata_raw

        Returns
        -------
        None.

        """
        
        # Get the unique classes in man_class table
        unique_classes = np.unique(self.man_class['Classification'])
        
        # For each class, get the pointids from 
        for uq in unique_classes:
            # Get pointIds for points with given classification
            PointIdsClass = np.uint32(self.man_class.index.values[
                np.where(self.man_class['Classification']==uq)[0]])
            # Set those points in polydata_raw to their classification
            Classification = vtk_to_numpy(self.polydata_raw.GetPointData()
                                          .GetArray('Classification'))
            PointIds = vtk_to_numpy(self.polydata_raw.GetPointData()
                                          .GetArray('PointId'))
            Classification[np.isin(PointIds, PointIdsClass, assume_unique=True)
                           ] = np.uint8(uq)
        
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()
    
    def update_man_class(self, pdata, classification):
        """
        Update the points in man_class with the points in pdata.
        
        See documentation under SingleScan for description of man_class

        Parameters
        ----------
        pdata : vtkPolyData
            PolyData containing the points to add to man_class.
        classification : uint8
            The classification code of the points. See SingleScan 
            documentation for mapping from code to text

        Returns
        -------
        None.

        """
        
        # Raise exception if man class table doesn't exist
        if not hasattr(self, 'man_class'):
            raise RuntimeError('man_class table does not exist. '
                               + 'load it first?')
        
        # Inverse Transform to get points in Scanners Own Coordinate System
        invTransform = vtk.vtkTransformFilter()
        invTransform.SetTransform(self.transform.GetInverse())
        invTransform.SetInputData(pdata)
        invTransform.Update()
        pdata_inv = invTransform.GetOutput()
        
        # Create a dataframe from selected points
        dsa_pdata = dsa.WrapDataObject(pdata_inv)
        n_pts = pdata_inv.GetNumberOfPoints()
        df_trans = pd.DataFrame({'X' : dsa_pdata.Points[:,0],
                                 'Y' : dsa_pdata.Points[:,1],
                                 'Z' : dsa_pdata.Points[:,2],
                                 'trans_00' : (self.transform.GetMatrix()
                                               .GetElement(0, 0) * np.ones(
                                                   n_pts, dtype=np.double)),
                                 'trans_01' : (self.transform.GetMatrix()
                                               .GetElement(0, 1) * np.ones(
                                                   n_pts, dtype=np.double)),
                                 'trans_02' : (self.transform.GetMatrix()
                                               .GetElement(0, 2) * np.ones(
                                                   n_pts, dtype=np.double)),
                                 'trans_03' : (self.transform.GetMatrix()
                                               .GetElement(0, 3) * np.ones(
                                                   n_pts, dtype=np.double)),
                                 'trans_04' : (self.transform.GetMatrix()
                                               .GetElement(1, 0) * np.ones(
                                                   n_pts, dtype=np.double)),
                                 'trans_05' : (self.transform.GetMatrix()
                                               .GetElement(1, 1) * np.ones(
                                                   n_pts, dtype=np.double)),
                                 'trans_06' : (self.transform.GetMatrix()
                                               .GetElement(1, 2) * np.ones(
                                                   n_pts, dtype=np.double)),
                                 'trans_07' : (self.transform.GetMatrix()
                                               .GetElement(1, 3) * np.ones(
                                                   n_pts, dtype=np.double)),
                                 'trans_08' : (self.transform.GetMatrix()
                                               .GetElement(2, 0) * np.ones(
                                                   n_pts, dtype=np.double)),
                                 'trans_09' : (self.transform.GetMatrix()
                                               .GetElement(2, 1) * np.ones(
                                                   n_pts, dtype=np.double)),
                                 'trans_10' : (self.transform.GetMatrix()
                                               .GetElement(2, 2) * np.ones(
                                                   n_pts, dtype=np.double)),
                                 'trans_11' : (self.transform.GetMatrix()
                                               .GetElement(2, 3) * np.ones(
                                                   n_pts, dtype=np.double)),
                                 'Linearity' : (dsa_pdata
                                                .PointData['Linearity']),
                                 'Planarity' : (dsa_pdata
                                                .PointData['Planarity']),
                                 'Scattering' : (dsa_pdata
                                                .PointData['Scattering']),
                                 'Verticality' : (dsa_pdata
                                                .PointData['Verticality']),
                                 'Density' : (dsa_pdata
                                                .PointData['Density']),
                                 'Anisotropy' : (dsa_pdata
                                                .PointData['Anisotropy']),
                                 'HeightAboveGround' : (dsa_pdata
                                                .PointData['HeightAboveGround'
                                                           ]),
                                 'dist' : np.sqrt(np.sum(
                                     np.square(dsa_pdata.Points), axis=1)),
                                 'Amplitude' : (dsa_pdata
                                                .PointData['Amplitude']),
                                 'HorizontalClosestPoint' : (dsa_pdata
                                    .PointData['HorizontalClosestPoint']),
                                 'VerticalClosestPoint' : (dsa_pdata
                                    .PointData['VerticalClosestPoint']),
                                 'Classification' : classification * np.ones(
                                     n_pts, dtype=np.uint8)
                                 },
                                index=dsa_pdata.PointData['PointId'], 
                                copy=True)
        df_trans.index.name = 'PointId'
        
        # Join the dataframe with the existing one, overwrite points if we
        # have repicked some points.
        self.man_class = df_trans.combine_first(self.man_class)
        
        # drop columns that we don't have. Because they show up as 
        # vtkNoneArray their datatype is object.
        self.man_class = self.man_class.select_dtypes(exclude=['object'])
        
        # Write to file to save
        self.man_class.to_parquet(os.path.join(self.project_path, 
                                                 self.project_name, 
                                                 'manualclassification', 
                                                 self.scan_name + '.parquet'),
                                                 engine="pyarrow", 
                                                 compression=None)
        
    def update_man_class_fields(self, update_fields='all', update_trans=True):
        """
        Update man_class table with the fields that are currently in
        polydata_raw.
        
        Requires that PointId's haven't changed!!!
        
        Parameters
        ----------
        update_fields : list or 'all', optional
            Which fields in man_class we want to update. If 'all' update all 
            but Classification. The default is 'all'.
        update_trans : bool, optional
            Whether to update the transformation matrix values with the
            current transformation. The default is True.

        Returns
        -------
        None.

        """
        
        # Raise exception if man class table doesn't exist
        if not hasattr(self, 'man_class'):
            raise RuntimeError('man_class table does not exist. '
                               + 'load it first?')
        if self.man_class.shape[0]==0:
            return
        
        # Get PointID's of picked points
        pedigreeIds = vtk.vtkTypeUInt32Array()
        pedigreeIds.SetNumberOfComponents(1)
        pedigreeIds.SetNumberOfTuples(self.man_class.shape[0])
        np_pedigreeIds = vtk_to_numpy(pedigreeIds)
        if np.max(self.man_class.index.values)>np.iinfo(np.uint32).max:
            raise RuntimeError('PointId exceeds size of uint32')
        np_pedigreeIds[:] = self.man_class.index.values.astype(np.uint32)
        pedigreeIds.Modified()
        
        # Selection points from polydata_raw by PedigreeId
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(1) # we want to select points
        selectionNode.SetContentType(2) # 2 corresponds to pedigreeIds
        selectionNode.SetSelectionList(pedigreeIds)
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)
        extractSelection = vtk.vtkExtractSelection()
        extractSelection.SetInputData(0, self.polydata_raw)
        extractSelection.SetInputData(1, selection)
        extractSelection.Update()
        pdata = extractSelection.GetOutput()
        dsa_pdata = dsa.WrapDataObject(pdata)
        
        # Handle if the update fields are all
        if update_fields=='all':
            update_fields = ['Linearity', 'Planarity', 'Scattering', 
                             'Verticality', 'Density', 'Anisotropy', 
                             'HeightAboveGround', 'dist', 'Amplitude',
                             'HorizontalClosestPoint', 'VerticalClosestPoint']
        
        # If update_trans we also want to update the tranform matrix
        n_pts = pdata.GetNumberOfPoints()
        if update_trans:
            df_dict = {
                        'trans_00' : (self.transform.GetMatrix()
                                      .GetElement(0, 0) * np.ones(
                                          n_pts, dtype=np.double)),
                        'trans_01' : (self.transform.GetMatrix()
                                      .GetElement(0, 1) * np.ones(
                                          n_pts, dtype=np.double)),
                        'trans_02' : (self.transform.GetMatrix()
                                      .GetElement(0, 2) * np.ones(
                                          n_pts, dtype=np.double)),
                        'trans_03' : (self.transform.GetMatrix()
                                      .GetElement(0, 3) * np.ones(
                                          n_pts, dtype=np.double)),
                        'trans_04' : (self.transform.GetMatrix()
                                      .GetElement(1, 0) * np.ones(
                                          n_pts, dtype=np.double)),
                        'trans_05' : (self.transform.GetMatrix()
                                      .GetElement(1, 1) * np.ones(
                                          n_pts, dtype=np.double)),
                        'trans_06' : (self.transform.GetMatrix()
                                      .GetElement(1, 2) * np.ones(
                                          n_pts, dtype=np.double)),
                        'trans_07' : (self.transform.GetMatrix()
                                      .GetElement(1, 3) * np.ones(
                                          n_pts, dtype=np.double)),
                        'trans_08' : (self.transform.GetMatrix()
                                      .GetElement(2, 0) * np.ones(
                                          n_pts, dtype=np.double)),
                        'trans_09' : (self.transform.GetMatrix()
                                      .GetElement(2, 1) * np.ones(
                                          n_pts, dtype=np.double)),
                        'trans_10' : (self.transform.GetMatrix()
                                      .GetElement(2, 2) * np.ones(
                                          n_pts, dtype=np.double)),
                        'trans_11' : (self.transform.GetMatrix()
                                      .GetElement(2, 3) * np.ones(
                                          n_pts, dtype=np.double))
                        }
        else:
            df_dict = {}
        
        # Now add column for each field in update_fields
        for column_name in update_fields:
            df_dict[column_name] = dsa_pdata.PointData[column_name]
        
        # Create new data frame
        df_new = pd.DataFrame(df_dict, index=dsa_pdata.PointData['PointId'],
                              copy=True)
        df_new.index.name = 'PointId'
        
        # drop columns that we don't have. Because they show up as 
        # vtkNoneArray their datatype is object.
        df_new = df_new.select_dtypes(exclude=['object'])
        
        # Join with self.man_class we want to replace any columns that are in 
        # both dataframes with the columns in df_new
        self.man_class = pd.merge(df_new, self.man_class, how='inner',
                                  left_index=True, right_index=True, 
                                  sort=False, suffixes=('', '_y'))
        self.man_class.drop(self.man_class.filter(regex='_y$').columns
                            .tolist(), axis=1, inplace=True)
        
        # Write to file to save
        self.man_class.to_parquet(os.path.join(self.project_path, 
                                                 self.project_name, 
                                                 'manualclassification', 
                                                 self.scan_name + '.parquet'),
                                                 engine="pyarrow", 
                                                 compression=None)
        
    def create_normals(self, radius=2, max_nn=30):
        """
        Use Open3d to compute pointwise normals and store.

        Parameters
        ----------
        radius : float, optional
            Max distance to look for nearby points. The default is 2.
        max_nn : int, optional
            max number of points to use in normal estimation. 
            The default is 30.

        Returns
        -------
        None.

        """
        
        # Create Open3d pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vtk_to_numpy(
            self.polydata_raw.GetPoints().GetData()))
        # Estimate Normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn))
        # Orient Normals to point towards scanner
        pcd.orient_normals_towards_camera_location(camera_location=
                                                   np.zeros(3))
        
        # Save normals in polydata_raw, We should be able to just save these
        # as floats (doubles are probably overkill)
        # First create npy_dict if it doesn't exist:
        if not hasattr(self, 'np_dict'):
            self.np_dict = {}
        self.np_dict['Normals'] = np.array(pcd.normals, dtype=np.float32)
        # Now add normals to polydata_raw
        vtk_arr = numpy_to_vtk(self.np_dict['Normals'], deep=False, 
                               array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName('Normals')
        self.polydata_raw.GetPointData().SetNormals(vtk_arr)
        self.polydata_raw.Modified()
        self.dsa_raw = dsa.WrapDataObject(self.polydata_raw)
        self.transformFilter.Update()
        self.currentFilter.Update()
        # Update raw_history_dict
        self.raw_history_dict = {
            "type": "Create Normals",
            "git_hash": get_git_hash(),
            "method": "SingleScan.create_normals",
            "name": "Create Normals by PCA Estimation",
            "input_0": json.loads(json.dumps(self.raw_history_dict)),
            "params": {"radius": radius,
                       'max_nn': max_nn}
            }
        self.transformed_history_dict["input_0"] = self.raw_history_dict
        
    def create_z_sigma(self, sigma_ro=0.008, sigma_bw=0.0003/4):
        """
        Estimate the pointwise uncertainty in the z-direction.
        
        For each point, project the pointwise uncertainty spheroid onto the z 
        unit vector in the reference frame defined by the current transform.
        
        Each point measured by the scanner has a positional uncertainty that
        can be described as a gaussian spheroid. The symmetry axis of this
        spheroid is aligned with the direction of the laser beam, which we
        will name p_hat. Along p_hat the uncertainty is determined by the
        ranging uncertainty of the laser (sigma_ro). Perpendicular to p_hat
        in all directions, our uncertainty is due to the bandwidth spreading
        of the laser beam--sigma_bw. sigma_bw is measured in radians and to 
        get sigma_b measured in m we need to multiply by the distance.
        
        In order to find the uncertainty in the z direction of the current
        transform, we first project the z unit vector in the current 
        transform's reference frame into the scanner's reference 
        frame--z_prime. Then we use the dot product of z_prime with p_hat for
        each point to get the cosine of the angle between them. Finally,
        the distance from the origin of the uncertainty spheroid to the 1
        sigma surface along the direction z_prime can be computed from the
        squared cosine of the angle between z_prime and p_hat.
        
        Finally, we save the result as an array in polydata_raw.

        Parameters
        ----------
        sigma_ro : float, optional
            The standard deviation of the laser's ranging uncertainty in m.
            The default is 0.008 (value for VZ-1000)
        sigma_bw : float, optional
            The standard deviation of the laser's bandwidth spreading in
            radians. The defaults is 0.0003/4 (value for VZ-1000)
        
        Returns
        -------
        None.

        """
        
        # Go from z axis in transformed coordinate system to scanners own 
        # coordinate system
        z_prime = np.zeros(3, dtype=np.float32)
        self.transform.GetInverse().TransformNormal((0.0, 0.0, 1.0), z_prime)
        
        # Get the distance from the scanner for each point
        pts_np = vtk_to_numpy(self.polydata_raw.GetPoints().GetData())
        d = np.sqrt(np.square(pts_np).sum(axis=1))
        
        # Get cos(theta)**2 the square of the dot product of each point's 
        # direction vector with z_prime
        cos_theta_sq = np.square(np.dot(pts_np, z_prime)/d)
        
        # Now the uncertainty in the direction of our z_vector is the distance from
        # each point to it's error spheroid along that direction
        # The symmetry axis of the error spheroid runs in the direction of the
        # points direction vector
        u = np.sqrt((sigma_ro**2)*cos_theta_sq 
                    + ((sigma_bw*d)**2) * (1-cos_theta_sq))
        
        # Add uncertainty as an array in polydata_raw
        vtk_arr = numpy_to_vtk(u, deep=False, array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName('z_sigma')
        self.polydata_raw.GetPointData().AddArray(vtk_arr)
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()
        
        # Update raw_history_dict
        self.raw_history_dict = {
            "type": "Scalar Modifier",
            "git_hash": get_git_hash(),
            "method": "SingleScan.create_z_sigma",
            "name": "Create z_sigma field",
            "input_0": json.loads(json.dumps(self.transformed_history_dict)),
            "params": {"sigma_ro": sigma_ro, "sigma_bw": sigma_bw}
            }
        self.transformed_history_dict["input_0"] = self.raw_history_dict
    
    def apply_elevation_filter(self, z_max):
        """
        Set Classification for all points above z_max to be 64. 

        Parameters
        ----------
        z_max : float
            Maximum z-value (in reference frame of currentTransform).

        Returns
        -------
        None.

        """
        
        # If the current filter output has no points, return
        if self.currentFilter.GetOutput().GetNumberOfPoints()==0:
            return
        # Get the points of the currentTransform as a numpy array
        Points = vtk_to_numpy(self.currentFilter.GetOutput().GetPoints()
                              .GetData())
        PointIds = vtk_to_numpy(self.currentFilter.GetOutput().GetPointData().
                        GetArray('PointId'))
        # Set the in Classification for points whose z-value is above z_max to 
        # 64
        self.dsa_raw.PointData['Classification'][np.isin(self.dsa_raw.PointData
            ['PointId'], PointIds[Points[:,2]>z_max], assume_unique=True)] =64
        # Update currentTransform
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()
        # Update raw_history_dict
        self.raw_history_dict = {
            "type": "Scalar Modifier",
            "git_hash": get_git_hash(),
            "method": "SingleScan.apply_elevation_filter",
            "name": "Set Classification for points above z_max to be 64",
            "input_0": json.loads(json.dumps(self.filt_history_dict)),
            "params": {"z_max": z_max}
            }
        self.transformed_history_dict["input_0"] = self.raw_history_dict
        
    def apply_rmin_filter(self, buffer=0.05, count=150000):
        """
        Assign all points very close to the scanner as snowflakes. This filter
        is applied to the output of current filter (and hence depends on)
        the current transform we have applied.

        Parameters
        ----------
        buffer : float, optional
            How far past the cylindrical shell defined by count to go in m. 
            The default is 0.05.
        count : int, optional
            How many points to count before we decide that we're at the edge.
            The default is 150000

        Returns
        -------
        None.

        """
        
        # Compute xy squared distance
        sq_dist = (np.square(vtk_to_numpy(self.currentFilter.GetOutput()
                                         .GetPoints().GetData())[:,:2]
                            - np.array(self.transform.GetPosition()[:2]
                                       , ndmin=2))
                   .sum(axis=1))
        PointIds = vtk_to_numpy(self.currentFilter.GetOutput().GetPointData().
                        GetArray('PointId'))
        # Get the sq_distance corresponding to the countth point
        k_sq_dist = np.partition(sq_dist, count)[count]
        
        # Now set the classification value for every point that is within
        # counteth's point's xy distance plus the buffer to 65
        self.dsa_raw.PointData['Classification'][np.isin(self.dsa_raw.PointData
            ['PointId'], PointIds[sq_dist<(k_sq_dist + buffer**2)], 
            assume_unique=True)] = 65
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()

        # update history_dicts appropriately
        # Update raw_history_dict
        self.raw_history_dict = {
            "type": "Scalar Modifier",
            "git_hash": get_git_hash(),
            "method": "SingleScan.apply_rmin_filter",
            "name": "Set snowflake Classification to 65",
            "input_0": json.loads(json.dumps(self.filt_history_dict)),
            "params": {"buffer": buffer,
                       "count": count}
            }
        self.transformed_history_dict["input_0"] = self.raw_history_dict
        
    def apply_snowflake_filter_3(self, z_std_mult, leafsize):
        """
        Filter points as snowflakes based on whether their z value in the
        transformed reference frame exceeds z_std_mult multiples of the mean
        z values for points nearby (within a bucket of size leafsize).

        We apply this only to the output of currentFilter!

        All points that this filter identifies as snowflakes are set to
        Classification=65

        Parameters
        ----------
        z_std_mult : float
            The number of positive z standard deviations greater than other
            nearby points for us to classify it as a snowflake.
        leafsize : int
            maximum number of points in each bucket (we use scipy's
            KDTree)

        """
        
        # If the current filter output has no points, return
        if self.currentFilter.GetOutput().GetNumberOfPoints()==0:
            return
        # Step 1, get pointer to points array and create tree
        Points = vtk_to_numpy(self.currentFilter.GetOutput().GetPoints()
                              .GetData())
        PointIds = vtk_to_numpy(self.currentFilter.GetOutput().GetPointData().
                        GetArray('PointId'))
        tree = cKDTree(Points[:,:2], leafsize=leafsize)
        # Get python accessible version
        ptree = tree.tree

        # Step 2, define the recursive function that we'll use
        def z_std_filter(node, z_std_mult, Points, bool_arr):
            # If we are not at a leaf, call this function on each child
            if not node.split_dim==-1:
                # Call this function on the lesser node
                z_std_filter(node.lesser, z_std_mult, Points, bool_arr)
                # Call this function on the greater node
                z_std_filter(node.greater, z_std_mult, Points, bool_arr)
            else:
                # We are at a leaf. Compute distance from mean
                ind = node.indices
                z_mean = Points[ind, 2].mean()
                z_std = Points[ind, 2].std()
                bool_arr[ind] = (Points[ind, 2] - z_mean) > (z_std_mult * z_std)

        # Step 3, Apply function
        bool_arr = np.empty(Points.shape[0], dtype=np.bool_)
        z_std_filter(ptree, z_std_mult, Points, bool_arr)

        # Step 4, modify Classification field in polydata_raw
        # Use bool_arr to index into PointIds, use np.isin to find indices
        # in dsa_raw
        self.dsa_raw.PointData['Classification'][np.isin(self.dsa_raw.PointData
            ['PointId'], PointIds[bool_arr], assume_unique=True)] = 65
        del ptree, tree, PointIds, Points
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()

        # Step 5, update history_dicts appropriately
        # Update raw_history_dict
        self.raw_history_dict = {
            "type": "Scalar Modifier",
            "git_hash": get_git_hash(),
            "method": "SingleScan.apply_snowflake_filter_3",
            "name": "Set snowflake Classification to 65",
            "input_0": json.loads(json.dumps(self.filt_history_dict)),
            "params": {"z_std_mult": z_std_mult,
                       "leafsize": leafsize}
            }
        self.transformed_history_dict["input_0"] = self.raw_history_dict


    def apply_snowflake_filter_returnindex(self, cylinder_rad=0.025*np.sqrt(2)
                                           *np.pi/180, radial_precision=0):
        """
        Filter snowflakes using return index visible space.
        
        Snowflakes are too small to fully occlude the laser pulse. Therefore
        all snowflakes will be one of multiple returns (returnindex<-1).
        However, the edges of shadows will also be one of multiple returns. To
        address this we look at each early return and check if it's on the 
        border of the visible area from the scanner's perspective. We do this
        by finding all points within cylinder_rad of the point in question
        in panorama space. Then, if the radial value of the point in question
        is greater than any of these radial values that means the point
        in question is on the border of the visible region and we should keep
        it.
        
        All points that this filter identifies as snowflakes are set to
        Classification=65

        Parameters
        ----------
        cylinder_rad : float, optional
            The radius of a cylinder, in radians around an early return
            to look for last returns. The default is 0.025*np.sqrt(2)*np.pi/
            180.
        radial_precision : float, optional
            If an early return's radius is within radial_precision of an
            adjacent last return accept it as surface. The default is 0.

        Returns
        -------
        None.

        """
        
        # Convert to polar coordinates
        sphere2cart = vtk.vtkSphericalTransform()
        cart2sphere = sphere2cart.GetInverse()
        transformFilter = vtk.vtkTransformFilter()
        transformFilter.SetTransform(cart2sphere)
        transformFilter.SetInputData(self.polydata_raw)
        transformFilter.Update()
        
        # Get only last returns
        (transformFilter.GetOutput().GetPointData().
         SetActiveScalars('ReturnIndex'))
        thresholdFilter = vtk.vtkThresholdPoints()
        thresholdFilter.ThresholdByUpper(-1.5)
        thresholdFilter.SetInputConnection(transformFilter.GetOutputPort())
        thresholdFilter.Update()
        
        # Transform such that points are  in x and y and radius is in Elevation field
        swap_r_phi = vtk.vtkTransform()
        swap_r_phi.SetMatrix((0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1))
        filter_r_phi = vtk.vtkTransformFilter()
        filter_r_phi.SetTransform(swap_r_phi)
        filter_r_phi.SetInputConnection(thresholdFilter.GetOutputPort())
        filter_r_phi.Update()
        radialElev = vtk.vtkSimpleElevationFilter()
        radialElev.SetVector(0, 0, 1.0)
        radialElev.SetInputConnection(filter_r_phi.GetOutputPort())
        radialElev.Update()
        flattener = vtk.vtkTransformFilter()
        transFlat = vtk.vtkTransform()
        transFlat.Scale(1, 1, 0)
        flattener.SetTransform(transFlat)
        flattener.SetInputConnection(radialElev.GetOutputPort())
        flattener.Update()
        
        # Create locator for last returns
        locator = vtk.vtkStaticPointLocator2D()
        flat_last_returns = flattener.GetOutput()
        flat_last_returns.SetPointLocator(locator)
        locator.SetDataSet(flat_last_returns)
        flat_last_returns.BuildPointLocator()
        
        # Get early returns as possible snowflakes
        thresholdFilterL = vtk.vtkThresholdPoints()
        thresholdFilterL.ThresholdByLower(-1.5)
        thresholdFilterL.SetInputConnection(transformFilter.GetOutputPort())
        thresholdFilterL.Update()
        early_returns = thresholdFilterL.GetOutput()
        
        # Allocate objects needed to find nearby points
        result = vtk.vtkIdList()
        pt = np.zeros(3)
        snowflake = True
        
        for i in np.arange(early_returns.GetNumberOfPoints()):
            # Get the point in question
            early_returns.GetPoint(i, pt)
            # Get the adjacent points from last_returns and place id's in result
            (flat_last_returns.GetPointLocator().FindPointsWithinRadius(
                cylinder_rad, pt[2], pt[1], 0, result))
            # If the radius of the point in question is larger than that of 
            # any of the adjacent point, then that means we are on the edge of
            # the lidar's vision and this point is probably not a snowflake
            snowflake = True
            for j in range(result.GetNumberOfIds()):
                if pt[0] >= (flat_last_returns.GetPointData().
                             GetAbstractArray('Elevation').GetTuple(result.
                                                                    GetId(j)
                                                                    )[0]
                             -radial_precision):
                    snowflake = False
                    break
            if snowflake:
                self.dsa_raw.PointData['Classification'][self.dsa_raw.PointData[
                    'PointId']==early_returns.GetPointData().
                    GetPedigreeIds().GetValue(i)] = 65
        
        # Update currentTransform
        self.polydata_raw.GetPointData().SetActiveScalars('Classification')
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()
        
        # Update raw_history_dict
        self.raw_history_dict = {
            "type": "Scalar Modifier",
            "git_hash": get_git_hash(),
            "method": "SingleScan.apply_snowflake_filter_returnindex",
            "name": "Set snowflake Classification to 65",
            "input_0": json.loads(json.dumps(self.raw_history_dict)),
            "params": {"cylinder_rad": cylinder_rad,
                       'radial_precision': radial_precision}
            }
        self.transformed_history_dict["input_0"] = self.raw_history_dict

    def apply_manual_filter(self, corner_coords, normal=(0.0, 0.0, 1.0), 
                            category=73, mode='currentFilter'):
        """
        Manually classify points using a selection loop.

        Parameters
        ----------
        corner_coords : ndarray
            Nx3 array of points defining corners of selection.
        normal : array-like, optional
            Three element vector describing normal (axis) of loop. 
            The default is (0, 0, 1).
        category : uint8, optional
            Which category to classify points as. The default is 73.
        mode : str, optional
            What to apply filter to. Options are 'currentFilter' and
            'transformFilter'. The default is 'currentFilter'.

        Returns
        -------
        None.

        """
        
        # Convert corner_coords to vtk points object
        pts = vtk.vtkPoints()
        arr_type = (vtk.VTK_FLOAT if corner_coords.dtype=='float32' else 
                    vtk.VTK_DOUBLE)
        pts.SetData(numpy_to_vtk(corner_coords, array_type=arr_type))
        
        # Create implicit selection loop
        selectionLoop = vtk.vtkImplicitSelectionLoop()
        selectionLoop.AutomaticNormalGenerationOff()
        selectionLoop.SetNormal(normal[0], normal[1], normal[2])
        selectionLoop.SetLoop(pts)
        
        # The extract points function is just too slow with the entire 
        # dataset. Let's see if we can quickly subset the points first.
        if mode=='currentFilter':
            pdata = self.currentFilter.GetOutput()
        elif mode=='transformFilter':
            pdata = self.transformFilter.GetOutput()
        else:
            raise ValueError('mode must be currentFilter or transformFilter')
        # If this pdata has no points in it, we can just return
        if pdata.GetNumberOfPoints()==0:
            return
        minx, miny, _ = corner_coords.min(axis=0)
        maxx, maxy, _ = corner_coords.max(axis=0)
        pts_np = vtk_to_numpy(pdata.GetPoints().GetData())
        PointIds_np = vtk_to_numpy(pdata.GetPointData().GetArray('PointId'))
        mask = ((pts_np[:,0]>=minx) & (pts_np[:,0]<=maxx) &
                (pts_np[:,1]>=miny) & (pts_np[:,1]<=maxy))
        pts_np = pts_np[mask, :]
        # set the z-coordinate of points to -1, for some reason the selection
        # loop only selects negative points...
        pts_np[:, 2] = -1
        PointIds_np = PointIds_np[mask]
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(pts_np, array_type=vtk.VTK_FLOAT))
        pdata = vtk.vtkPolyData()
        pdata.SetPoints(pts)
        vtkarr = numpy_to_vtk(PointIds_np, array_type=vtk.VTK_UNSIGNED_INT)
        vtkarr.SetName('PointId')
        pdata.GetPointData().AddArray(vtkarr)
        
        # Create Extract Points
        extractPoints = vtk.vtkExtractPoints()
        extractPoints.GenerateVerticesOff()
        extractPoints.SetImplicitFunction(selectionLoop)
        extractPoints.SetInputData(pdata)
        extractPoints.Update()
        
        # If extractPoints has zero points in it then we can just return
        if extractPoints.GetOutput().GetNumberOfPoints()==0:
            return
        
        # Update classification
        PointIds = vtk_to_numpy(extractPoints.GetOutput().GetPointData().
                                GetArray('PointId'))
        self.dsa_raw.PointData['Classification'][np.isin(self.dsa_raw.PointData
            ['PointId'], PointIds, assume_unique=True)] = category
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()
        
        # Update History Dict
        self.raw_history_dict = {
            "type": "Scalar Modifier",
            "git_hash": get_git_hash(),
            "method": "SingleScan.apply_manual_filter",
            "name": "Manually classify points",
            "input_0": (json.loads(json.dumps(self.filt_history_dict)) 
                        if mode=='currentFilter' else 
                        json.loads(json.dumps(self.transformed_history_dict))),
            "params": {"corner_coords": corner_coords.tolist(),
                       "normal": list(normal),
                       "category": category}
            }
        self.transformed_history_dict["input_0"] = self.raw_history_dict
        

    def apply_cylinder_filter(self, x, y, r, category=73):
        """
        Classify all points within distance r of (x,y)

        Parameters:
        -----------
        x : float
            x coordinate of the center point in transformed reference frame
        y : float
            y coordinate of the center point in transformed reference frame
        r : float
            Distance (in m) from center point to classify points as
        category : int, optional
            Category to classify points as. The defaults is 73


        Returns:
        --------
        None

        """

        # Subset points to square around center point
        pdata = self.currentFilter.GetOutput()
        # If this pdata has no points in it, we can just return
        if pdata.GetNumberOfPoints()==0:
            return
        pts_np = vtk_to_numpy(pdata.GetPoints().GetData())
        PointIds_np = vtk_to_numpy(pdata.GetPointData().GetArray('PointId'))
        mask = ((pts_np[:,0]>=(x-r)) & (pts_np[:,0]<=(x+r)) &
                (pts_np[:,1]>=(y-r)) & (pts_np[:,1]<=(y+r)))
        pts_np = pts_np[mask, :]
        PointIds_np = PointIds_np[mask]
        # Get PointIds of points within r of center point
        mask = (r**2 >= (np.square(pts_np[:,0]-x) + np.square(pts_np[:,1]-y)))
        PointIds = PointIds_np[mask]

        # Update classification
        self.dsa_raw.PointData['Classification'][np.isin(self.dsa_raw.PointData
            ['PointId'], PointIds, assume_unique=True)] = category
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()
        
        # Update History Dict
        self.raw_history_dict = {
            "type": "Scalar Modifier",
            "git_hash": get_git_hash(),
            "method": "SingleScan.apply_cylinder_filter",
            "name": "Manually classify points around center point",
            "input_0": json.loads(json.dumps(self.filt_history_dict)) ,
            "params": {"x": x,
                       "y": y,
                       "r": r,
                       "category": category}
            }
        self.transformed_history_dict["input_0"] = self.raw_history_dict
    
    def create_scanner_actor(self, color='Gray', length=150):
        """
        Create an actor for visualizing the scanner and its orientation.
        
        Parameters:
        -----------
        color : str, optional
            Name of the color to display as. The default is 'Gray'
        length : float, optional
            Length of the ray indicating the scanner's start orientation in m.
            The default is 150

        Returns:
        --------
        None.

        """

        # Named colors object
        nc = vtk.vtkNamedColors()

        # Create a cylinder to represent the scanner
        cylinderSource = vtk.vtkCylinderSource()
        cylinderSource.SetCenter(0, 0, 0)
        cylinderSource.SetRadius(0.25)
        cylinderSource.SetHeight(0.6)
        cylinderSource.SetResolution(12)
        cylinderSource.CappingOn()
        cylinderSource.Update()
        pdata = cylinderSource.GetOutput()

        # Add line along Scanner's x-axis
        ctr_id = pdata.GetPoints().InsertNextPoint(0, 0, 0)
        ray_id = pdata.GetPoints().InsertNextPoint(150, 0, 0)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(2)
        lines.InsertCellPoint(ctr_id)
        lines.InsertCellPoint(ray_id)
        pdata.SetLines(lines)
        pdata.Modified()

        # Mapper and Actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pdata)
        mapper.SetScalarVisibility(0)
        if hasattr(self, 'scannerActor'):
            del self.scannerActor
            del self.scannerText
        self.scannerActor = vtk.vtkActor()
        self.scannerActor.SetMapper(mapper)
        self.scannerActor.GetProperty().SetLineWidth(3)
        self.scannerActor.GetProperty().RenderLinesAsTubesOn()
        self.scannerActor.GetProperty().SetColor(nc.GetColor3d(color))
        self.scannerActor.RotateX(90) # because cylinder is along y axis
        self.scannerActor.SetUserTransform(self.transform)

        # Create Text with the scan position name
        text = vtk.vtkVectorText()
        text.SetText(self.scan_name)
        text.Update()
        textMapper = vtk.vtkPolyDataMapper()
        textMapper.SetInputData(text.GetOutput())
        self.scannerText = vtk.vtkFollower()
        self.scannerText.SetMapper(textMapper)
        self.scannerText.SetScale(1, 1, 1)
        self.scannerText.SetPosition(self.transform.GetPosition())
        self.scannerText.AddPosition(0, 0, 0.5)
        self.scannerText.GetProperty().SetColor(nc.GetColor3d(color))

    def create_labels_actors(self, color='White', row_index=None):
        """
        Create dataframe containing actors for each label
        
        Parameters:
        -----------
        color : str, optional
            Name of the color to display as. The default is 'Gray'
        length : float, optional
            Length of the ray indicating the scanner's start orientation in m.
            The default is 150
        row_index : tuple, optional
            Sometimes we want to create new actors for a single row, and not
            redo everything else, if so, this is tne index for that row. The
            default is None.

        Returns:
        --------
        None.

        """

        # Named colors object
        nc = vtk.vtkNamedColors()

        # First, copy the labels DataFrame
        if row_index is None:
            self.labels_actors = self.labels.copy(deep=True)
            # Now create columns for the point actor and text actor
            self.labels_actors['point_actor'] = None
            self.labels_actors['text_actor'] = None
        else:
            self.labels_actors.loc[row_index] = self.labels.loc[row_index]

        for row in self.labels_actors.itertuples():
            if (not row_index is None) and (not row.Index==row_index):
                continue
            
            # Create point and transform into project coordinate system
            pts = vtk.vtkPoints()
            pts.InsertNextPoint(row.x, row.y, row.z)
            pdata = vtk.vtkPolyData()
            pdata.SetPoints(pts)
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetInputData(pdata)
            transformFilter.SetTransform(self.transform)
            transformFilter.Update()
            vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
            vertexGlyphFilter.SetInputConnection(
                transformFilter.GetOutputPort())
            vertexGlyphFilter.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(vertexGlyphFilter.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().RenderPointsAsSpheresOn()
            actor.GetProperty().SetPointSize(20)
            actor.GetProperty().SetColor(nc.GetColor3d(color))
            # Add actor to the dataframe
            self.labels_actors.at[row.Index, 'point_actor'] = actor

            # Create Text
            text = vtk.vtkVectorText()
            text.SetText(row.Index[0] + ', ' + row.Index[1] + ', ' 
                         + row.Index[2])
            text.Update()
            textMapper = vtk.vtkPolyDataMapper()
            textMapper.SetInputConnection(text.GetOutputPort())
            labelText = vtk.vtkFollower()
            labelText.SetMapper(textMapper)
            labelText.SetScale(0.1, 0.1, 1)
            labelText.SetPosition(transformFilter.GetOutput().GetPoints()
                                  .GetPoint(0))
            labelText.AddPosition(0, 0, 0.2)
            labelText.GetProperty().SetColor(nc.GetColor3d(color))
            # add to dataframe
            self.labels_actors.at[row.Index, 'text_actor'] = labelText

    def create_filter_pipeline(self,colors={0 : (153/255, 153/255, 153/255, 1),
                                            1 : (153/255, 153/255, 153/255, 1),
                                            2 : (55/255, 126/255, 184/255, 1),
                                            6 : (166/255, 86/255, 40/255, 1),
                                            7 : (255/255, 255/255, 51/255, 1),
                                            64: (255/255, 255/255, 51/255, 1),
                                            65: (255/255, 255/255, 51/255, 1),
                                            66: (255/255, 255/255, 51/255, 1),
                                            67: (228/255, 26/255, 28/255, 1),
                                            68: (77/255, 175/255, 74/255, 1),
                                            69: (247/255, 129/255, 191/255, 1),
                                            70: (152/255, 78/255, 163/255, 1),
                                            71: (255/255, 127/255, 0/255, 1),
                                            72: (253/255, 191/255, 111/255, 1),
                                            73: (255/255, 0/255, 0/255, 1)
                                            }):
        """
        Create mapper and actor displaying points colored by Classification

        Parameters
        ----------
        colors : dict, optional
            Mapping from value in Classification to color. 
            The default is {0 : (0, 255, 0), 1 : (255, 0, 0)}.

        Returns
        -------
        None.

        """
        
        # Set active scalars
        self.currentFilter.GetOutput().GetPointData().SetActiveScalars(
            'Classification')
        
        # Create Lookuptable
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(max(colors) + 1)
        lut.SetTableRange(0, max(colors))
        for key in colors:
            lut.SetTableValue(key, colors[key])
        lut.Build()
        
        # Create vertices
        vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
        vertexGlyphFilter.SetInputConnection(
            self.currentFilter.GetOutputPort())
        vertexGlyphFilter.Update()
        
        # Create mapper
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(vertexGlyphFilter.GetOutputPort())
        self.mapper.SetLookupTable(lut)
        self.mapper.SetScalarRange(min(colors), max(colors))
        self.mapper.SetScalarVisibility(1)
        self.mapper.SetColorModeToMapScalars()
        
        # Create subsampled for LOD rendering
        maskPoints = vtk.vtkPMaskPoints()
        maskPoints.ProportionalMaximumNumberOfPointsOn()
        maskPoints.SetOnRatio(10)
        maskPoints.GenerateVerticesOn()
        maskPoints.SetInputConnection(self.transformFilter.GetOutputPort())
        maskPoints.Update()
        self.mapper_sub = vtk.vtkPolyDataMapper()
        self.mapper_sub.SetInputConnection(maskPoints.GetOutputPort())
        self.mapper_sub.SetLookupTable(lut)
        self.mapper_sub.SetScalarRange(min(colors), max(colors))
        self.mapper_sub.SetScalarVisibility(1)
        self.mapper_sub.SetColorModeToMapScalars()
        
        # Create actor
        self.actor = vtk.vtkLODProp3D()
        self.actor.AddLOD(self.mapper, 0.0)
        self.actor.AddLOD(self.mapper_sub, 0.0)
    
    def create_solid_pipeline(self, color='Green'):
        """
        Create vtk visualization pipeline with solid colors

        Parameters
        ----------
        color : string, optional
            Name of color (in vtkNamedColors). The default is 'Green'.

        Returns
        -------
        None.

        """
        
        # Named colors object
        nc = vtk.vtkNamedColors()
        
        # Create vertices
        vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
        vertexGlyphFilter.SetInputConnection(
            self.currentFilter.GetOutputPort())
        vertexGlyphFilter.Update()
        
        # Create mapper
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(vertexGlyphFilter.GetOutputPort())
        self.mapper.SetScalarVisibility(0)
        prop = vtk.vtkProperty()
        prop.SetColor(nc.GetColor3d(color))
        
        # Create subsampled for LOD rendering
        maskPoints = vtk.vtkPMaskPoints()
        maskPoints.ProportionalMaximumNumberOfPointsOn()
        maskPoints.SetOnRatio(10)
        maskPoints.GenerateVerticesOn()
        maskPoints.SetInputConnection(self.currentFilter.GetOutputPort())
        maskPoints.Update()
        self.mapper_sub = vtk.vtkPolyDataMapper()
        self.mapper_sub.SetInputConnection(maskPoints.GetOutputPort())
        self.mapper_sub.SetScalarVisibility(0)
        prop_sub = vtk.vtkProperty()
        prop_sub.SetColor(nc.GetColor3d(color))
        
        # Create actor
        self.actor = vtk.vtkLODProp3D()
        self.actor.AddLOD(self.mapper, prop, 0.0)
        self.actor.AddLOD(self.mapper_sub, prop_sub, 0.0)
        
    
    def create_elevation_pipeline(self, z_min, z_max, lower_threshold=-1000,
                                  upper_threshold=1000, LOD=True,
                                  cmap_name='rainbow'):
        """
        create mapper and actor displaying points colored by elevation.

        Parameters
        ----------
        z_min : float
            Lower cutoff for plotting colors.
        z_max : float
            Upper cutoff for plotting colors.
        lower_threshold : float, optional
            Minimum elevation of point to display. The default is -1000.
        upper_threshold : float, optional
            Maximum elevation of point to display. The default is 1000.
        LOD : bool, optional
            Whether to generate a level of detail actor. The default is True
        cmap_name : str, optional
            Name of matplotlib colormap to use. The default is 'rainbow'.

        Returns
        -------
        None.

        """
        
        # Create vertices
        vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
        vertexGlyphFilter.SetInputConnection(
            self.currentFilter.GetOutputPort())
        vertexGlyphFilter.Update()
        
        # # Create elevation filter
        elevFilter = vtk.vtkSimpleElevationFilter()
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()
        elevFilter.SetInputConnection(vertexGlyphFilter.GetOutputPort())
        # needed to prevent simpleelevationfilter from overwriting 
        # Classification array
        elevFilter.Update()
        
        # Create Threshold filter
        thresholdFilter = vtk.vtkThresholdPoints()
        thresholdFilter.SetInputConnection(elevFilter.GetOutputPort())
        thresholdFilter.ThresholdBetween(lower_threshold, upper_threshold)
        thresholdFilter.Update()
        
        # Create mapper, hardcode LUT for now
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(thresholdFilter.GetOutputPort())
        self.mapper.SetLookupTable(mplcmap_to_vtkLUT(z_min, z_max,
                                                     name=cmap_name))
        self.mapper.SetScalarRange(z_min, z_max)
        self.mapper.SetScalarVisibility(1)
        
        if LOD:
            # Create subsampled for LOD rendering
            maskPoints = vtk.vtkPMaskPoints()
            maskPoints.ProportionalMaximumNumberOfPointsOn()
            maskPoints.SetOnRatio(10)
            maskPoints.GenerateVerticesOn()
            maskPoints.SetInputConnection(thresholdFilter.GetOutputPort())
            maskPoints.Update()
            self.mapper_sub = vtk.vtkPolyDataMapper()
            self.mapper_sub.SetInputConnection(maskPoints.GetOutputPort())
            self.mapper_sub.SetLookupTable(mplcmap_to_vtkLUT(z_min, z_max,
                                                             name=
                                                             cmap_name))
            self.mapper_sub.SetScalarRange(z_min, z_max)
            self.mapper_sub.SetScalarVisibility(1)
            
            # Create actor
            self.actor = vtk.vtkLODProp3D()
            self.actor.AddLOD(self.mapper, 0.0)
            self.actor.AddLOD(self.mapper_sub, 0.0)
        else:
            self.actor = vtk.vtkActor()
            self.actor.SetMapper(self.mapper)
    
    def create_reflectance(self):
        """
        Create Reflectance field in polydata_raw according to RiSCAN instructs.

        Returns
        -------
        None.

        """
        
        # If Reflectance array doesn't exist, create it.
        if not 'Reflectance' in self.dsa_raw.PointData.keys():
            arr = vtk.vtkFloatArray()
            arr.SetName('Reflectance')
            arr.SetNumberOfComponents(1)
            arr.SetNumberOfTuples(self.polydata_raw.GetNumberOfPoints())
            arr.FillComponent(0, 0)
            self.polydata_raw.GetPointData().AddArray(arr)
        
            self.dsa_raw.PointData['Reflectance'][:] = (np.array(self.dsa_raw.
                                                                 PointData
                                              ['intensity'], dtype=np.float32) 
                                              - 32768)/100.0
            # Update
            self.polydata_raw.Modified()
            self.transformFilter.Update()
            self.currentFilter.Update()
            # Update raw_history_dict
            self.raw_history_dict = {
                "type": "Scalar Modifier",
                "git_hash": get_git_hash(),
                "method": "SingleScan.create_reflectance",
                "name": "Create reflectance field",
                "input_0": json.loads(json.dumps(self.raw_history_dict))
                }
                
    def create_reflectance_pipeline(self, v_min, v_max, field='Reflectance'):
        """
        create mapper and actor displaying points colored by elevation.

        Parameters
        ----------
        v_min : float
            Lower cutoff for plotting colors.
        v_max : float
            Upper cutoff for plotting colors.
        field : str, optional
            Which array in pointdata to display. The default is 'Reflectance'

        Returns
        -------
        None.

        """
        
        # Create vertices
        vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
        vertexGlyphFilter.SetInputConnection(
            self.currentFilter.GetOutputPort())
        vertexGlyphFilter.Update()
        
        # Create mapper, hardcode LUT for now
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(vertexGlyphFilter.GetOutputPort())
        self.mapper.GetInput().GetPointData().SetActiveScalars(field)
        self.mapper.SetLookupTable(mplcmap_to_vtkLUT(v_min, v_max, 
                                                     name='plasma'))
        self.mapper.SetScalarRange(v_min, v_max)
        self.mapper.SetScalarVisibility(1)
        
        # Create subsampled for LOD rendering
        maskPoints = vtk.vtkPMaskPoints()
        maskPoints.ProportionalMaximumNumberOfPointsOn()
        maskPoints.SetOnRatio(10)
        maskPoints.GenerateVerticesOn()
        maskPoints.SetInputConnection(self.currentFilter.GetOutputPort())
        maskPoints.Update()
        self.mapper_sub = vtk.vtkPolyDataMapper()
        self.mapper_sub.SetInputConnection(maskPoints.GetOutputPort())
        self.mapper_sub.GetInput().GetPointData().SetActiveScalars(field)
        self.mapper_sub.SetLookupTable(mplcmap_to_vtkLUT(v_min, v_max,
                                                         name='plasma'))
        self.mapper_sub.SetScalarRange(v_min, v_max)
        self.mapper_sub.SetScalarVisibility(1)
        
        # Create actor
        self.actor = vtk.vtkLODProp3D()
        self.actor.AddLOD(self.mapper, 0.0)
        self.actor.AddLOD(self.mapper_sub, 0.0)
    
    def correct_reflectance_radial(self, mode, r_min=None, r_max=None, 
                                   num=None, base=None):
        """
        Corrects radial artifact in reflectance. result: 'reflectance_radial'
        
        Attempts to correct radial artifact in reflectance. Still developing
        the best way to do this.
        
        If mode is 'median': bin the raw reflectances by radial distance.

        Parameters
        ----------
        mode : str
            Method for correcting radial artifact in reflectances. Currently
            only coded for 'median'.
        r_min : float, optional
            Needed for method 'median' minimum radius to bin
        r_max : float, optional
            Needed for method 'median' maximum radius to bin
        num : int, optional
            Needed for method 'median', number of bins
        base : float, optional
            Needed for method 'median', base for logspaced bins

        Returns
        -------
        None.

        """
        warnings.warn("History tracking has not been implemented for this " +
                      "function yet")
        
        # If reflectance_radial array doesn't exist, create it.
        if not 'reflectance_radial' in self.dsa_raw.PointData.keys():
            arr = vtk.vtkFloatArray()
            arr.SetName('reflectance_radial')
            arr.SetNumberOfComponents(1)
            arr.SetNumberOfTuples(self.polydata_raw.GetNumberOfPoints())
            arr.FillComponent(0, 0)
            self.polydata_raw.GetPointData().AddArray(arr)
        
        if mode=='median':
            start = np.log(r_min)/np.log(base)
            end = np.log(r_max)/np.log(base)
            
            log_bin_edges = np.logspace(start, end, num=num, base=base)
            
            bin_centers = np.zeros(num-1)
            median_refl = np.zeros(num-1)
            
            dist = np.sqrt(self.dsa_raw.Points[:,0]**2 + 
                           self.dsa_raw.Points[:,1]**2 +
                           self.dsa_raw.Points[:,2]**2)
            
            median_refl_pts = np.zeros(dist.shape)
            
            for i in np.arange(num-1):
                in_bin = np.logical_and(dist>=log_bin_edges[i], 
                                        dist<log_bin_edges[i+1])
                bin_centers[i] = np.median(dist[in_bin])
                median_refl[i] = np.median(self.dsa_raw.PointData
                                           ['Reflectance'][in_bin])
                median_refl_pts[in_bin] = median_refl[i]
            
            # Set median values outside range to be edges
            median_refl_pts[dist<r_min] = median_refl[0]
            median_refl_pts[dist>=r_max] = median_refl[-1]
        
        
        self.dsa_raw.PointData['reflectance_radial'][:] = (
            self.dsa_raw.PointData['Reflectance'] - median_refl_pts)
        
        # Update
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()
    
    def reflectance_filter(self, threshold, radius=0,
                           field='reflectance_radial'):
        """
        Set Classification values for high reflectance objects (and neighborhood
        if desired) to 66.

        Parameters
        ----------
        threshold : float
            reflectance threshold above which to flag
        radius : float, optional
            Radius around flagged points to also flag. The default is 0.
        field : str, optional
            The field in PointData to threshold. The default is 
            'reflectance_radial'.

        Returns
        -------
        None.

        """
        warnings.warn("History tracking has not been implemented for this " +
                      "function yet")
        
        # Undo previously flagged points (just reflectance, not snowflake)
        self.dsa_raw.PointData['Classification'][self.dsa_raw.PointData
                                              ['Classification']==66] = 0
        
        # Flag all points that exceed threshold and are not already flagged
        self.dsa_raw.PointData['Classification'][
            np.logical_and(self.dsa_raw.PointData[field]>threshold,
                           self.dsa_raw.PointData['Classification'] in [0, 1]
                           )] = 66
        
        # Update currentTransform
        self.polydata_raw.GetPointData().SetActiveScalars('Classification')
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()
        
        if radius>0:
            # If radius is greater than 0 we want to also flag all points within
            # a vertical cylinder of radius of one of our flagged points.
            #cylinder = vtk.vtkCylinder()
            # Set the axis of our cylinder to align with vertical in the current
            # transform
            #cylinder.SetRadius(radius)
            #cylinder.SetAxis(0.0, 0.0, 1.0)
            implicitBoolean = vtk.vtkImplicitBoolean()
            implicitBoolean.SetOperationTypeToUnion()
            # Get all points that meet our criteria
            thresholdPoints = vtk.vtkThresholdPoints()
            thresholdPoints.ThresholdBetween(2.5, 3.5)
            thresholdPoints.SetInputData(self.transformFilter.GetOutput())
            thresholdPoints.Update()
            pdata = thresholdPoints.GetOutput()
            #print(pdata.GetNumberOfPoints())
            # Create an implicit function of cylinders around all of our
            # points
            for i in np.arange(pdata.GetNumberOfPoints()):
                cylinder = vtk.vtkCylinder()
                cylinder.SetRadius(radius)
                cylinder.SetAxis(0.0, 0.0, 1.0)
                cylinder.SetCenter(pdata.GetPoint(i))
                implicitBoolean.AddFunction(cylinder)
            # Get all points inside this implicit function
            extractPoints = vtk.vtkExtractPoints()
            extractPoints.SetInputConnection(self.transformFilter.GetOutputPort())
            extractPoints.SetImplicitFunction(implicitBoolean)
            extractPoints.Update()
            for i in np.arange(extractPoints.GetOutput().GetNumberOfPoints()):
                pt_id = self.transformFilter.GetOutput().FindPoint(
                    extractPoints.GetOutput().GetPoint(i))
                self.dsa_raw.PointData['Classification'][pt_id] = 66

            # Update currentTransform
            self.polydata_raw.Modified()
            self.transformFilter.Update()
            self.currentFilter.Update()
    
    def write_npy_pdal(self, output_dir=None, filename=None, 
                       mode='transformed', skip_fields=[]):
        """
        Write scan to structured numpy array that can be read by PDAL.

        Parameters
        ----------
        output_dir : str, optional
            Directory to write to. If none will write to the 'temp' folder
            under the project name.
        filename : str, optional
            Filename to write, if None will write PROJECT_NAME_SCAN_NAME. 
            The default is None.
        mode : str, optional
            Whether to write 'raw' points, 'transformed' points, or 'filtered'
            points. The default is 'transformed'.
        skip_fields : list, optional
            Fields to skip in writing. If this is 'all' then only write x,
            y, z. Otherwise should be a list of field names. The default is []

        Returns
        -------
        None.

        """
        
        if mode=='raw':
            pdata = self.polydata_raw
            dsa_pdata = self.dsa_raw
        elif mode=='transformed':
            pdata = self.transformFilter.GetOutput()
            dsa_pdata = dsa.WrapDataObject(pdata)
        elif mode=='filtered':
            pdata = self.get_polydata()
            dsa_pdata = dsa.WrapDataObject(pdata)
        else:
            raise ValueError('mode must be raw, transformed, or filtered')
        
        n_pts = pdata.GetNumberOfPoints()
        if n_pts == 0:
            return None
        
        # Create numpy output
        names = []
        for name in dsa_pdata.PointData.keys():
            if name=='PointId':
                names.append(name)
            else:
                if skip_fields=='all':
                    continue
                elif name in skip_fields:
                    continue
                else:
                    names.append(name)
        formats = []
        for name in names:
            formats.append(dsa_pdata.PointData[name].dtype)
        names = tuple(names + ['X', 'Y', 'Z'])
        formats.append(np.float32)
        formats.append(np.float32)
        formats.append(np.float32)
        formats = tuple(formats)
        output_npy = np.zeros(n_pts, dtype={'names':names, 'formats':formats})
        for name in names:
            if name=='X':
                output_npy['X'] = dsa_pdata.Points[:,0]
            elif name=='Y':
                output_npy['Y'] = dsa_pdata.Points[:,1]
            elif name=='Z':
                output_npy['Z'] = dsa_pdata.Points[:,2]                
            else:
                output_npy[name] = dsa_pdata.PointData[name]
                
        if output_dir is None:
            output_dir = os.path.join(self.project_path, 'temp')
        if filename is None:
            filename = self.project_name + '_' + self.scan_name + '.npy'
        
        npy_filepath = os.path.join(output_dir, filename)
        np.save(npy_filepath, output_npy)
        return npy_filepath
    
    def write_pdal_transformation_json(self, mode='las', input_dir='./', 
                                       output_dir='./pdal_output/'):
        """
        Write pdal formatted JSON for transforming raw data to current.

        Parameters
        ----------
        mode : str, optional
            Datatype of the input. The default is 'las'.
        input_dir : str, optional
            Location to find file. The default is './'.
        output_dir : TYPE, optional
            Location to put pdal output. The default is './pdal_output/'.

        Returns
        -------
        None.

        """
        
        json_list = [None, None, None]
        # Find the first filename in input_dir that matches this scan position
        # and filetype
        filenames = os.listdir(input_dir)
        pattern = re.compile(self.scan_name + '.*' + mode)
        matches = [pattern.fullmatch(filename) for filename in filenames]
        if any(matches):
            # Create filename input
            filename = next(f for f, m in zip(filenames, matches) if m)
            json_list[0] = input_dir + filename
            # Get current transform 4x4 matrix as an array
            arr = np.ones(16, dtype='double')
            self.transform.GetMatrix().DeepCopy(arr, 
                                                self.transform.GetMatrix())
            json_list[1] = {"type": "filters.transformation",
                            "matrix": np.array2string(arr, max_line_width=1000
                                                      , formatter={
                                                          'float':lambda x: 
                                                              "%0.16f" % x})
                                [1:-1]}
            # Create output
            json_list[2] = output_dir + filename
            # convert list to json and write to a file
            with open(input_dir + self.scan_name + '_pdal_transformation.txt',
                      'w') as outfile:
                json.dump(json_list, outfile, indent=4)
            
        else:
            raise UserWarning(self.scan_name + ' not found in ' + input_dir)
    
    def add_dist(self):
        """
        Add distance array to polydata_raw

        Returns
        -------
        None.

        """
        
        # Add dist field to scan
        if not 'dist' in self.dsa_raw.PointData.keys():
            arr = vtk.vtkFloatArray()
            arr.SetName('dist')
            arr.SetNumberOfComponents(1)
            arr.SetNumberOfTuples(self.polydata_raw.GetNumberOfPoints())
            self.polydata_raw.GetPointData().AddArray(arr)
        self.dsa_raw.PointData['dist'][:] = np.sqrt(np.sum(np.square(
            self.dsa_raw.Points), axis=1), dtype=np.float32)
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()

        # Update raw_history_dict
        self.raw_history_dict = {"type": "Scalar Modifier",
                                "git_hash": get_git_hash(),
                                "method": "SingleScan.add_dict",
                                "name": "Add distance from scanner",
                                "input_0": json.loads(json.dumps(
                                    self.raw_history_dict))}

    def get_local_max(self, z_threshold, rmax, return_dist=False, 
                      return_zs=False):
        """
        Get set of locally maxima points above z_threshold

        Parameters
        ----------
        z_threshold : float
            Minimum z value (in current transformation) for all points to 
            consider.
        rmax : float
            Horizontal distance (in m) on which points must be locally maximal
        return_dist : bool, optional
            Whether to return an array with the distance of each point from 
            scanner.
            The default is False.
        return_zs : bool, optional
            Whether to return the z-sigma of each point. The default is False.

        Returns
        -------
        [local_max, dist, zs]
            ndarrays, Nx3 of locally maximal points in current tranformation.
            optionally distance and z sigma as well.
            
        """

        # Add fields, if needed
        if return_dist and not 'dist' in self.dsa_raw.PointData.keys():
            self.add_dist()
        if return_zs and not 'z_sigma' in self.dsa_raw.PointData.keys():
            self.create_z_sigma()

        # Get points and limit to those that exceed z_threshold
        pts_np = vtk_to_numpy(self.get_polydata().GetPoints().GetData())
        if return_dist:
            dist_np = vtk_to_numpy(self.get_polydata().GetPointData(
                ).GetArray('dist'))
        if return_zs:
            z_sigma_np = vtk_to_numpy(self.get_polydata().GetPointData(
                ).GetArray('z_sigma'))    
        mask = pts_np[:,2] > z_threshold
        pts_np = pts_np[mask, :]
        if return_dist:
            dist_np = dist_np[mask]
        if return_zs:
            z_sigma_np = z_sigma_np[mask]
        # Sort in descending z order
        sort_ind = np.flip(np.argsort(pts_np[:,2]))
        pts_np = pts_np[sort_ind,:]
        if return_dist:
            dist_np = dist_np[sort_ind]
        if return_zs:
            z_sigma_np = z_sigma_np[sort_ind]
        # Create kdtree for quick nearest neighbor query
        tree = KDTree(pts_np[:,:2])
        # Mask keeps track of which points are still possibly local maxima
        max_mask = np.ones(pts_np.shape[0], dtype=np.bool_)
        # Counter for where we are in the array
        ctr = 0
        while True:
            # Find all points within r near our current point
            inds = np.array(tree.query_ball_point(pts_np[ctr, :2], rmax, 
                                                  workers=-1))
            # If any of the points are higher than our query point, the query
            # cannot be a local max
            if any(pts_np[inds,2]>pts_np[ctr,2]):
                max_mask[ctr] = False
            # Any of the points that are lower than the query also cannot be a
            # local max
            max_mask[inds[pts_np[inds,2]<pts_np[ctr,2]]] = False
            # Find the next potential point that could be a local max
            ind = utf1st.find_1st(max_mask[(ctr+1):], True, utf1st.cmp_equal)
            if ind==-1:
                break
            else:
                ctr = ctr + 1 + ind
        
        ret_list = [pts_np[max_mask,:]]
        if return_dist:
            ret_list.append(dist_np[max_mask])
        if return_zs:
            ret_list.append(z_sigma_np[max_mask])
        return ret_list

class Project:
    """Class linking relevant data for a project and methods for registration.
    
    ...
    
    Attributes
    ----------
    project_date : str
        Date that this project was started on (1st day for 2 day scans)
    project_name : str
        Filename of the RiSCAN project the tiepointlist comes from
    project_path : str
        Directory location of the project.
    tiepointlist : TiePointList
        Object containing tiepoints and methods for referencing them together.
    scan_dict : dict
        Dictionary of SingleScan objects keyed on scan names
    current_transform_list : list
        The list of transforms currently applied to the project.
    filterName : str
        Name of filter currently applied to all scans in project.
    mesh : vtkPolyData
        Polydata containing a mesh representation of the project.
    image_dict : dict
        Dictionary containing vtkImageData with gridded height information over
        specified regions. For backwards compatability images if a key isn't
        specified it will be ''
    image_transform_dict : dict
        Dictionary of vtkTransforms with same key as image_dict
        Transforms for going from mesh reference frame to image ref
        frame. Needed because vtkImageData can only be axis aligned.
    dsa_image_dict : dict
        Dictionary of datasetadapter wrappers to interact with image via numpy
        Same key as image_dict and image_transform_dict
    profile_dict : dict
        Dictionary holding polydata for each profile that we create in the
        project. Each entry is a Polydata whose points are a profile and the
        PointData optionally includes upper and lower confidence intervals
        Various functions can either access these profile data or display
        it. Each polydata has a single polyline as its cell.
    pdata_dict : dict
        Dictionary holding polydata objects that we create in the project.
        There is no requirement on a specific type of polydata, the purpose
        is to make accessing and displaying (e.g. on top of image) easy.
        
    Methods
    -------
    apply_transforms(transform_list)
        Update transform for each scan and update current_transform_list.
    write_scans(project_write_dir=None, suffix='',
                freeze=False, overwrite_frozen=False)
        Write all singlescans to files.
    write_current_transform(suffix='', name='current_transform',
                            freeze=False, overwrite_frozen=False)
        Have each SingleScan write its current transform to a file.
    read_transforms(name='current_transform', suffix=''):
        Have each SingleScan read a transform from file
    load_man_class()
        Direct each single scan to load it's man_class table.
    apply_man_class()
        Update all point Classifications with their values in man_class.
    create_normals(radius=2, max_nn=30)
        Use Open3d to compute pointwise normals and store.
    create_z_sigma()
        For the current value of the transformation, project the pointwise
        uncertainty spheroids onto the z-axis and save in PointData.
    load_labels()
        Load labels in each SingleScan
    get_labels()
        Get all labels as a DataFrame
    apply_manual_filter()
        Manually classify points within a selection loop.
    apply_snowflake_filter_returnindex(cylinder_rad, radial_precision)
        Filter snowflakes based on their return index and whether they are on
        the border of the visible region.
    apply_cylinder_filter(x, y, r, category=73)
        Classify all points within distance r of (x,y).
    update_man_class_fields(update_fields='all', update_trans=True)
        Update the man_class table with values from the fields currently in
        polydata_raw. Useful, for example if we've improved the HAG filter and
        don't want to have to repick all points.
    add_transform(key, matrix, history_dict=None)
        Add the provided transform to each SingleScan
    add_z_offset(z_offset)
        Add z offset to all scans in project.
    display_project(z_min, z_max, lower_threshold=-1000, 
                    upper_threshold=1000, colorbar=True, field='Elevation',
                    mapview=False, profile_list=[], show_scanners=False,
                    scanner_color='Gray', scanner_length=150, 
                    show_labels=False, pdata_list=[], addtl_actors=[])
        Display project in a vtk interactive window.
    project_to_image(z_min, z_max, focal_point, camera_position,
                     roll=0, image_scale=500, lower_threshold=-1000, 
                     upper_threshold=1000, mode=None, colorbar=True,
                     field='Elevation',
                     name='', window_size=(2000, 1000), path=None,
                     date=True, scale=True, addtl_actors=[])
        Write out an image of the project (in point cloud) to the snapshot
        folder.
    point_to_grid_average_image(nx, ny, dx, dy, x0, y0, yaw=0,
                                key='', overwrite=False)
        Convert a rectangular area of points to an image by gridded averaging
    merged_points_to_image(nx, ny, dx, dy, x0, y0, lengthscale, 
                           outputscale, nu, yaw=0, n_neighbors=50, max_pts=
                           64000, min_pts=100, mx=32, my=32, eps=0, 
                           corner_coords=None,
                           max_dist=None, optimize=False, learning_rate=0.1,
                           n_iter=5, max_time=0.5, optim_param=None,
                           multiply_outputscale=False, var_radius=None,
                           key='', overwrite=False)
        Convert a rectangular area of points to an image using gpytorch.
    mesh_to_image(z_min, z_max, nx, ny, dx, dy, x0, y0, key='')
        Interpolate mesh into image.
    get_image(field='Elevation', warp_scalars=False,
              v_min=-9999.0, nan_value=None, key='')
        Return image as vtkImageData or vtkPolyData depending on warp_scalars
    display_image(z_min, z_max, field='Elevation', warp_scalars=False, 
                  color_field=None, show_points=False, profile_list=[],
                  show_scanners=False, scanner_color='Gray', scanner_length=150,
                  pdata_list=[], key='')
        Display project image in a vtk interactive window.
    write_plot_image(z_min, z_max, focal_point, camera_position,
                     field='Elevation', warp_scalars=False,
                     roll=0, image_scale=500, lower_threshold=-1000, 
                     upper_threshold=1000, mode='map', colorbar=True,
                     name='', light=None, profile_list=[],
                     window_size=(2000,1000), key='')
        Write an image of the image to the snapshots folder.
    plot_image(z_min, z_max, cmap='inferno', key='')
        Plots the image using matplotlib
    get_np_nan_image(key='')
        Convenience function for copying the image to a numpy object.
    merged_points_to_mesh(depth=13, min_density=9, x0=None, y0=None,
                          wx=None, wy=None, yaw=0)
        Merge all transformed pointclouds and convert to mesh.
    transect_points(x0, y0, x1, y1, d)
        Get the points within a distance d of transect defined by points.
    transect_n_points(x0, y0, x1, y1, n_pts, tol=1000, d0=0.5, dmax=50)
        Get approximately n points around the transect.
    image_transect(x0, y0, x1, y1, N, key, image_key='')
        Sample a transect through the current image and save in profiles.
    merged_points_transect_gp(x0, y0, x1, y1, N, key, mx=256, n_neighbors=256, 
                              eps=0, use_z_sigma=True, lengthscale=None, 
                              outputscale=None, mean=None, nu=0.5, 
                              optimize=False, learning_rate=0.1, n_iter=None, 
                              max_time=60)
        Use gpytorch to infer a surface transect.
    get_profile(key)
        Returns the requested profile as a numpy array
    mesh_transect(x0, y0, x1, y1, N)
        Cut a transect through the mesh, return the transect
    get_merged_points(port=False, history_dict=False, x0=None, y0=None, 
                      wx=None, wy=None, yaw=0)
        Get the merged points as a polydata
    write_merged_points(output_name=None)
        Merge all transformed and filtered pointclouds and write to file.
    write_las_pdal(output_dir, filename)
        Merge all points and write to a las formatted output.
    write_mesh(output_path=None, suffix='', name='mesh')
        Write mesh to vtp file.
    read_mesh(output_path=None, suffix='', name='mesh')
        Read mesh from file.
    write_image(output_path=None, suffix='', name=None, key='')
        Write vtkImageData to file. Useful for saving im_nan_border.
    read_image(image_path=None, suffix='', name=None, overwrite=False)
        Read image from file.
    create_reflectance()
        Create reflectance for each scan.
    correct_reflectance_radial(mode, r_min=None, r_max=None, num=None, 
                               base=None)
        Attempt to correct for reflectance bias due to distance from scanner.
    areapoints_to_cornercoords(areapoints)
        Given a set of points, identified by their scan names and point ids,
        return the coordinates of those points in the current reference frame.
    get_local_max(z_threshold, rmax, return_dist=False, return_zs=False,
                  Closest_only=False)
        Get the set of locally maximal points.
    create_local_max(z_threshold, rmax, Closest_only=True, key='local_max')
        Add pdata containing locally maximal points to pdata_dict.
    save_local_max(suffix='', key='local_max')
        Saves the local max in the npyfiles directory NOT ROBUST
    load_local_max(suffix='', key='local_max')
        Loads the local max in the npyfiles directory NOT ROBUST

    """
    
    def __init__(self, project_path, project_name, poly='.1_.1_.01', 
                 import_mode=None, load_scans=True, read_scans=False, 
                 import_las=False, create_id=True, 
                 las_fieldnames=None, 
                 class_list=[0, 1, 2, 70], suffix='', class_suffix=''):
        """
        Generates project, also inits singlescan objects

        Parameters
        ----------
        project_path : str
            Directory location of the project.
        project_name : str
            Filename of the RiSCAN project.
        import_mode : str, optional
            How to create polydata_raw, the base data for this SingleScan. 
            Options are: 'poly' (read from Riscan generated poly), 'read_scan'
            (read saved vtp file), 'import_las' (use pdal to import from las
            file generate by Riscan), 'empty' (create an empty polydata, 
            useful if we just want to work with transformations). 'import_npy'
            (import from npyfiles directories) If value is None, then code 
            will interpret values of read_scan and import_las
            (deprecated method of specifying which to import) to maintain
            backwards compatibility. The default is None.
        poly : str, optional
            The suffix describing which polydata to load. The default is
            '.1_.1_.01'.
        load_scans : bool, optional
            Whether to actually load the scans. Often if we're just
            aligning successive scans loading all of them causes overhead.
        read_scans : bool, optional
            If False, each SingleScan object will be initialized to read the
            raw polydata from where RiSCAN saved it. If True, read the saved
            vtp file from in the scan area directory. Useful if we have saved
            already filtered scans. The default is False.
        import_las: bool, optional
            If true (and read_scan is False) read in the las file instead of
            the polydata. The default is False.
        create_id: bool, optional
            If true and PointID's do not exist create PointIDs. The default
            is True.
        las_fieldnames: list, optional
            List of fieldnames to load if we are importing from a las file
            Must include 'Points'. The default is ['Points', 'ReturnIndex'].
        class_list : list
            List of categories this filter will return, if special value: 
            'all' Then we do not have a selection filter and we pass through 
            all points. The default is [0, 1, 2, 70].
        suffix : str, optional
            Suffix for npyfiles directory if we are reading scans. The default
            is '' which corresponds to the regular npyfiles directory.
        class_suffix : str, optional
            Suffix for which Classification[class_suffix].npy file to load as
            'Classification' array. The default is '' (load Classification.npy)

        Returns
        -------
        None.

        """
        
        # Store instance attributes
        self.project_path = project_path
        self.project_name = project_name
        self.scan_dict = {}
        self.project_date = mosaic_date_parser(project_name)
        self.current_transform_list = []
        self.profile_dict = {}
        self.pdata_dict = {}
        self.image_dict = {}
        self.image_transform_dict = {}
        self.dsa_image_dict = {}
        self.image_history_dict_dict = {}
        
        if import_mode is None:
            # Issue a deprecated warning
            warnings.warn("Use import_mode to indicate how SingleScan object" +
                          " should load polydata.", FutureWarning)
            if not load_scans:
                import_mode = 'empty'
            elif read_scans:
                import_mode = 'read_scan'
            elif not import_las:
                import_mode = 'poly'
            elif import_las:
                import_mode = 'import_las'
            else:
                raise RuntimeError("You have specified an invalid combination"
                                   + " of import flags")
        
        # Add SingleScans, including their SOPs, 
        # we will only add a singlescan if it has an SOP, or, if we are 
        # reading scans we will only add a scan if it exists
        if import_mode=='read_scan':
            scan_names = os.listdir(os.path.join(self.project_path, 
                                                 project_name,
                                                'npyfiles'+suffix))
        else:
            ls = os.listdir(os.path.join(project_path, project_name))
            scan_names = [x.split(sep='.')[0] for x in ls 
                if re.fullmatch('ScanPos0[0-9][0-9]\.DAT', x)]
            
        for scan_name in scan_names:
            if import_mode=='read_scan':
                if os.path.isdir(os.path.join(self.project_path, 
                                              self.project_name, 'npyfiles' +
                                              suffix, scan_name)):
                    scan = SingleScan(self.project_path, self.project_name,
                                      scan_name, import_mode=import_mode,
                                      poly=poly, create_id=create_id,
                                      las_fieldnames=las_fieldnames,
                                      class_list=class_list, suffix=suffix,
                                      class_suffix=class_suffix)
                    scan.add_sop()
                    self.scan_dict[scan_name] = scan
            else:
                if os.path.isfile(os.path.join(self.project_path, 
                                               self.project_name, 
                                  scan_name + '.DAT')):
                    scan = SingleScan(self.project_path, self.project_name,
                                      scan_name, import_mode=import_mode,
                                      poly=poly, create_id=create_id,
                                      las_fieldnames=las_fieldnames,
                                      class_list=class_list, suffix=suffix)
                    scan.add_sop()
                    self.scan_dict[scan_name] = scan
        
        # Load TiePointList
        self.tiepointlist = TiePointList(self.project_path, self.project_name)
    
    def apply_transforms(self, transform_list):
        """
        Apply transforms in transform list to each SingleScan

        Parameters
        ----------
        transform_list : list
            str in list must be transforms in each SingleScan, see SingleScan 
            class for more details.

        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].apply_transforms(transform_list)
        self.current_transform_list = transform_list
    
    def write_scans(self, project_write_dir=None, suffix='',
                    freeze=False, overwrite_frozen=False):
        """
        Write all single scans to files.
        
        Parameters
        ----------
        project_write_dir : str, optional
            A directory to write all scans for this project to. If none write
            to default npyfiles location. The default is None.
        suffix : str, optional
            Suffix for npyfiles directory if we are reading scans. The default
            is '' which corresponds to the regular npyfiles directory.
        freeze: bool, optional
            Indicate whether the written files should be 'frozen'. Frozen files
            have the first element of their history dict set as 'frozen', and 
            we will store the path to the file in subsequent history dicts
            rather than the history dict itself to save space. The default 
            is False.
        overwrite_frozen : bool, optional
            If the pre-existing files are frozen, overwrite (by default
            attempting to delete a frozen file will raise an error)
            The default is False.

        Returns
        -------
        None.

        """
        
        if project_write_dir is None:
            for scan_name in self.scan_dict:
                self.scan_dict[scan_name].write_scan(suffix=suffix,
                                                     freeze=freeze,
                                                     overwrite_frozen=
                                                     overwrite_frozen)
        else:
            # For each scan name create a directory under project_write_dir
            # if it does not already exist.
            for scan_name in self.scan_dict:
                if not os.path.isdir(os.path.join(project_write_dir, 
                                                  scan_name)):
                    os.mkdir(os.path.join(project_write_dir, scan_name))
                self.scan_dict[scan_name].write_scan(os.path.join(
                    project_write_dir, scan_name), suffix=suffix,
                                                     freeze=freeze,
                                                     overwrite_frozen=
                                                     overwrite_frozen)
    
    def write_current_transforms(self, suffix='', name='current_transform',
                                 freeze=False, 
                                 overwrite_frozen=False):
        """
        Have each SingleScan write its current transform to a file.
        
        Parameters
        ----------
        suffix : str, optional
            Suffix for transforms directory if we are reading scans. 
            The default is '' which corresponds to the regular transforms
            directory.
        name : str, optional
            The name of the transform. The default is 'current_transform'.
        freeze: bool, optional
            Indicate whether the written files should be 'frozen'. Frozen files
            have the first element of their history dict set as 'frozen', and 
            we will store the path to the file in subsequent history dicts
            rather than the history dict itself to save space. The default 
            is False.
        overwrite_frozen : bool, optional
            If the pre-existing files are frozen, overwrite (by default
            attempting to delete a frozen file will raise an error)
            The default is False.
            
        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].write_current_transform(suffix=suffix,
                                                              name=name,
                                                              freeze=freeze,
                                                              overwrite_frozen=
                                                              overwrite_frozen)
    
    def read_transforms(self, name='current_transform', suffix=''):
        """
        Have each SingleScan read a transform from file

        Parameters
        ----------
        name : str, optional
            The name of the transform. The default is 'current_transform'.
        suffix : str, optional
            Suffix for transforms directory if we are reading scans. 
            The default is '' which corresponds to the regular transforms
            directory.
            
        Returns
        -------
        None.
name : str, optional
            The name of the transform. The default is 'current_transform'.
        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].read_transform(name=name, suffix=suffix)
    
    def load_man_class(self):
        """
        Direct each single scan to load it's man_class table

        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].load_man_class()
    
    def apply_man_class(self):
        """
        Update all point Classifications with their values in man_class

        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].apply_man_class()
    
    def create_normals(self, radius=2, max_nn=30):
        """
        Use Open3d to compute pointwise normals and store.

        Parameters
        ----------
        radius : float, optional
            Max distance to look for nearby points. The default is 2.
        max_nn : int, optional
            max number of points to use in normal estimation. 
            The default is 30.

        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].create_normals(radius=radius,
                                                     max_nn=max_nn)
    
    def create_z_sigma(self, sigma_ro=0.008, sigma_bw=0.0003/4):
        """
        Estimate the pointwise uncertainty in the z-direction.
        
        For each point, project the pointwise uncertainty spheroid onto the z 
        unit vector in the reference frame defined by the current transform.
        
        Each point measured by the scanner has a positional uncertainty that
        can be described as a gaussian spheroid. The symmetry axis of this
        spheroid is aligned with the direction of the laser beam, which we
        will name p_hat. Along p_hat the uncertainty is determined by the
        ranging uncertainty of the laser (sigma_ro). Perpendicular to p_hat
        in all directions, our uncertainty is due to the bandwidth spreading
        of the laser beam--sigma_bw. sigma_bw is measured in radians and to 
        get sigma_b measured in m we need to multiply by the distance.
        
        In order to find the uncertainty in the z direction of the current
        transform, we first project the z unit vector in the current 
        transform's reference frame into the scanner's reference 
        frame--z_prime. Then we use the dot product of z_prime with p_hat for
        each point to get the cosine of the angle between them. Finally,
        the distance from the origin of the uncertainty spheroid to the 1
        sigma surface along the direction z_prime can be computed from the
        squared cosine of the angle between z_prime and p_hat.
        
        Finally, we save the result as an array in polydata_raw.

        Parameters
        ----------
        sigma_ro : float, optional
            The standard deviation of the laser's ranging uncertainty in m.
            The default is 0.008 (value for VZ-1000)
        sigma_bw : float, optional
            The standard deviation of the laser's bandwidth spreading in
            radians. The defaults is 0.0003/4 (value for VZ-1000)
        
        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].create_z_sigma(sigma_ro=sigma_ro,
                                                     sigma_bw=sigma_bw)

    def load_labels(self):
        """
        Load the labels dataframes for each SingleScan

        Returns
        -------
        None.

        """

        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].load_labels()

    def get_labels(self):
        """
        Get the labels for all scans in this project

        Returns
        -------
        Pandas DataFrame containing labels

        """

        labels_list = []

        for scan_name in self.scan_dict:
            labels_list.append(self.scan_dict[scan_name].get_labels())

        return pd.concat(labels_list)

    def apply_manual_filter(self, corner_coords, normal=(0.0, 0.0, 1.0), 
                            category=73, mode='currentFilter'):
        """
        Manually classify points using a selection loop.

        Parameters
        ----------
        corner_coords : ndarray
            Nx3 array of points defining corners of selection.
        normal : array-like, optional
            Three element vector describing normal (axis) of loop. 
            The default is (0, 0, 1).
        category : uint8, optional
            Which category to classify points as. The default is 73.
        mode : str, optional
            What to apply filter to. Options are 'currentFilter' and
            'transformFilter'. The default is 'currentFilter'.

        Returns
        -------
        None.

        """

        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].apply_manual_filter(corner_coords,
                                                          normal=normal,
                                                          category=category,
                                                          mode=mode)

    def apply_snowflake_filter_returnindex(self, cylinder_rad=0.025*np.sqrt(2)
                                           *np.pi/180, radial_precision=0):
        """
        Filter snowflakes using return index visible space.
        
        Snowflakes are too small to fully occlude the laser pulse. Therefore
        all snowflakes will be one of multiple returns (returnindex<-1).
        However, the edges of shadows will also be one of multiple returns. To
        address this we look at each early return and check if it's on the 
        border of the visible area from the scanner's perspective. We do this
        by finding all points within cylinder_rad of the point in question
        in panorama space. Then, if the radial value of the point in question
        is greater than any of these radial values that means the point
        in question is on the border of the visible region and we should keep
        it.
        
        All points that this filter identifies as snowflakes are set to
        Classification=65

        Parameters
        ----------
        cylinder_rad : float, optional
            The radius of a cylinder, in radians around an early return
            to look for last returns. The default is 0.025*np.sqrt(2)*np.pi/
            180.
        radial_precision : float, optional
            If an early return's radius is within radial_precision of an
            adjacent last return accept it as surface. The default is 0.

        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].apply_snowflake_filter_returnindex(
                cylinder_rad, radial_precision)
        self.filterName = "snowflake_returnindex"

    def apply_cylinder_filter(self, x, y, r, category=73):
        """
        Classify all points within distance r of (x,y)

        Parameters:
        -----------
        x : float
            x coordinate of the center point in transformed reference frame
        y : float
            y coordinate of the center point in transformed reference frame
        r : float
            Distance (in m) from center point to classify points as
        category : int, optional
            Category to classify points as. The defaults is 73


        Returns:
        --------
        None

        """

        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].apply_cylinder_filter(x, y, r, category=
                                                            category)
        self.filterName = "cylinder_filter"
    
    def update_man_class_fields(self, update_fields='all', update_trans=True):
        """
        Update man_class table with the fields that are currently in
        polydata_raw.
        
        Requires that PointId's haven't changed!!!
        
        Parameters
        ----------
        update_fields : list or 'all', optional
            Which fields in man_class we want to update. If 'all' update all 
            but Classification. The default is 'all'.
        update_trans : bool, optional
            Whether to update the transformation matrix values with the
            current transformation. The default is True.

        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].update_man_class_fields(
                update_fields=update_fields, update_trans=update_trans)
    
    def add_transform(self, key, matrix, history_dict=None):
        """
        Add the provided transform to each single scan

        Parameters
        ----------
        key : const (could be string or tuple)
            Dictionary key for the transforms dictionary.
        matrix : TYPE
            DESCRIPTION.
        history_dict : dict
            dict tree containing history of transform. If None then we create
            a Transform Source node with the matrix as a param. The default
            is None.

        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].add_transform(key, matrix, history_dict=
                                                    history_dict)
            
    def add_transform_from_tiepointlist(self, key):
        """
        Add the transform in the tiepointlist to each single scan.

        Parameters
        ----------
        key : const (could be string or tuple)
            Dictionary key for the transforms dictionary.
        matrix : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            matrix, history_dict = self.tiepointlist.get_transform(key, 
                                                        history_dict=True)
            self.scan_dict[scan_name].add_transform(key, matrix, history_dict)

    def add_z_offset(self, z_offset, history_dict=None):
        """
        Add z_offset transform to each single scan in scan_dict

        Parameters
        ----------
        z_offset : float.
            z offset to add in meters.
        history_dict : dict
            dict tree containing history of transform. The default
            is None.

        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].add_z_offset(z_offset, history_dict=
                                                   history_dict)

    def create_reflector_actors(self, color='White', text_scale=1,
                                size=0.05):
        """
        Create actors for visualizing the reflectors.

        Note, this is part of Project because TiePointList is in Project.
        If the transform changes, this needs to be re-run.

        Todo: Dicts for each reflector actor and text actor
        
        Parameters:
        -----------
        color : str, optional
            Name of the color to display as. The default is 'White'
        text_scale : float, optional
            Scale for the text. Default is 1
        size : float, optional
            Radius of sphere used to represent reflector in m. Default is 0.05
        
        Returns:
        --------
        None.

        """

        # Named colors object
        nc = vtk.vtkNamedColors()

        # Dicts store reflector and text actors
        if hasattr(self, 'reflectorActorDict'):
            del self.reflectorActorDict
            del self.reflectorTextDict
        self.reflectorActorDict = {}
        self.reflectorTextDict = {}

        for index, row in self.tiepointlist.tiepoints_transformed.iterrows():
            # Create a cylinder to represent the scanner
            sphereSource = vtk.vtkSphereSource()
            sphereSource.SetCenter(row['X[m]'], row['Y[m]'], row['Z[m]'])
            sphereSource.SetRadius(size)
            sphereSource.Update()
            pdata = sphereSource.GetOutput()
            
            # Mapper and Actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(pdata)
            mapper.SetScalarVisibility(0)
            self.reflectorActorDict[index] = vtk.vtkActor()
            self.reflectorActorDict[index].SetMapper(mapper)
            self.reflectorActorDict[index].GetProperty().SetColor(
                                                        nc.GetColor3d(color))
            
            # Create Text with the scan position name
            text = vtk.vtkVectorText()
            text.SetText(index)
            text.Update()
            textMapper = vtk.vtkPolyDataMapper()
            textMapper.SetInputData(text.GetOutput())
            self.reflectorTextDict[index] = vtk.vtkFollower()
            self.reflectorTextDict[index].SetMapper(textMapper)
            self.reflectorTextDict[index].SetScale(text_scale, text_scale, 
                                                   text_scale)
            self.reflectorTextDict[index].SetPosition(np.array([row['X[m]'], 
                                                               row['Y[m]'], 
                                                               row['Z[m]']]))
            self.reflectorTextDict[index].AddPosition(0, 0.5, 0)
            self.reflectorTextDict[index].GetProperty().SetColor(nc.GetColor3d(color))
            
    def display_project(self, z_min, z_max, lower_threshold=-1000, 
                        upper_threshold=1000, colorbar=True, field='Elevation',
                        mapview=False, profile_list=[], show_scanners=False,
                        scanner_color='Gray', scanner_length=150, 
                        show_reflectors=False, reflector_color='White',
                        reflector_text_scale=1, reflector_size=0.05,
                        show_labels=False, pdata_list=[], addtl_actors=[]):
        """
        Display all scans in a vtk interactive window.
        
        Points will be colored by elevation, apply_transforms must be run 
        before this the transform data into desired reference frame.
        
        Currently, this renderwindowinteractor is set to write the camera
        position and focal point to std out when the user presses 'u'.

        Parameters
        ----------
        z_min : float
            Lower cutoff for plotting colors.
        z_max : float
            Upper cutoff for plotting colors.
        lower_threshold : float, optional
            Minimum elevation of point to display. The default is -1000.
        upper_threshold : float, optional
            Maximum elevation of point to display. The default is 1000.
        colorbar : bool, optional
            Display colorbar. The default is True.
        field : str, optional
            Which scalar field to display (elevation, reflectance, etc). The
            default is 'Elevation'
        mapview : bool, optional
            Whether or not to use a parallel projection. The default is False.
        profile_list : list, optional
            Which, if any, profiles to display along with the rendering. This
            list is composed of lists whose zeroth element is always the
            key of the profile in self.profile_dict. Element 1 is line width
            in pixels (optional) Elements 2, 3, 4, are color
            channels (optional) and element 5 is opacity (optional). The default
            is [].
        pdata_list : list, optional
            Which, if any, pdata to display along with the rendering. This
            list is composed of lists whose zeroth element is always the
            key of the profile in self.pdata_dict. Element 1 is point/line width
            in pixels (optional) Elements 2, 3, 4, are color
            channels (optional) and element 5 is opacity (optional). The default
            is [].
        show_scanners : bool, optional
            Whether or not to show the scanners. The default is False.
        scanner_color : str, optional
            Name of the color to display as. The default is 'Gray'
        scanner_length : float, optional
            Length of the ray indicating the scanner's start orientation in m.
            The default is 150
        show_reflectors : bool, optional
            Whether or not to show the reflectors. The default is False.
        reflector_color : str, optional
            Name of the color to display as. The default is 'White'
        reflector_text_scale : float, optional
            Text scale for reflector labels. The default is 1
        reflector_size : float, optional
            Radius of sphere representing reflector in m, the default is 0.05
        show_labels : bool, optional
            Whether to display the labels for each SingleScan. The default is
            False.
        addtl_actors : list, optional
            List of additonal actors to render, the default is []

        Returns
        -------
        None.

        """
        
        # Define function for writing the camera position and focal point to
        # std out when the user presses 'u'
        def cameraCallback(obj, event):
            print("Camera Pos: " + str(obj.GetRenderWindow().
                                           GetRenderers().GetFirstRenderer().
                                           GetActiveCamera().GetPosition()))
            print("Focal Point: " + str(obj.GetRenderWindow().
                                            GetRenderers().GetFirstRenderer().
                                            GetActiveCamera().GetFocalPoint()))
            print("Roll: " + str(obj.GetRenderWindow().
                                            GetRenderers().GetFirstRenderer().
                                            GetActiveCamera().GetRoll()))
        
        # Create renderer
        renderer = vtk.vtkRenderer()
        
        # Run create elevation pipeline for each scan and add each actor
        for scan_name in self.scan_dict:
            if field=='Elevation':
                self.scan_dict[scan_name].create_elevation_pipeline(z_min, 
                                                                    z_max, 
                                                                lower_threshold, 
                                                                upper_threshold)
            elif field=='Classification':
                self.scan_dict[scan_name].create_filter_pipeline()
            else:
                self.scan_dict[scan_name].create_reflectance_pipeline(z_min,
                                                                      z_max,
                                                                      field=
                                                                      field)
            renderer.AddActor(self.scan_dict[scan_name].actor)

        # Add scanners if requested
        if show_scanners:
            for scan_name in self.scan_dict:
                self.scan_dict[scan_name].create_scanner_actor(
                    color=scanner_color, length=scanner_length)
                renderer.AddActor(self.scan_dict[scan_name].scannerActor)
                renderer.AddActor(self.scan_dict[scan_name].scannerText)

        # Add reflectors if requested
        if show_reflectors:
            self.create_reflector_actors(color=reflector_color,
                                         text_scale=reflector_text_scale,
                                         size=reflector_size)
            for key in self.reflectorActorDict.keys():
                renderer.AddActor(self.reflectorActorDict[key])
                renderer.AddActor(self.reflectorTextDict[key])
                
        # Add labels if requested
        if show_labels:
            for scan_name in self.scan_dict:
                self.scan_dict[scan_name].create_labels_actors()
                for i in range(self.scan_dict[scan_name]
                               .labels_actors.shape[0]):
                    renderer.AddActor(self.scan_dict[scan_name].labels_actors
                                      ['point_actor'].iat[i])
                    renderer.AddActor(self.scan_dict[scan_name].labels_actors
                                      ['text_actor'].iat[i])
                
        
        # Add requested profiles
        for profile_tup in profile_list:
            #print(profile_tup)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.profile_dict[profile_tup[0]])
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            if len(profile_tup)>=2:
                actor.GetProperty().SetLineWidth(profile_tup[1])
            else:
                actor.GetProperty().SetLineWidth(5)
            if len(profile_tup)>=5:
                actor.GetProperty().SetColor(profile_tup[2:5])
            else:
                actor.GetProperty().SetColor(0.0, 1.0, 0.0)
            if len(profile_tup)>=6:
                actor.GetProperty().SetOpacity(profile_tup[5])
            else:
                actor.GetProperty().SetOpacity(0.8)
            actor.GetProperty().RenderLinesAsTubesOn()
            renderer.AddActor(actor)

        # Add requested pdata
        for profile_tup in pdata_list:
            #print(profile_tup)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.pdata_dict[profile_tup[0]])
            mapper.SetScalarVisibility(0)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            if len(profile_tup)>=2:
                actor.GetProperty().SetLineWidth(profile_tup[1])
                actor.GetProperty().SetPointSize(profile_tup[1])
            else:
                actor.GetProperty().SetLineWidth(5)
                actor.GetProperty().SetPointSize(10)
            if len(profile_tup)>=5:
                actor.GetProperty().SetColor(profile_tup[2:5])
            else:
                actor.GetProperty().SetColor(1.0, 1.0, 1.0)
            if len(profile_tup)>=6:
                actor.GetProperty().SetOpacity(profile_tup[5])
            else:
                actor.GetProperty().SetOpacity(0.8)
            actor.GetProperty().RenderLinesAsTubesOn()
            actor.GetProperty().RenderPointsAsSpheresOn()
            renderer.AddActor(actor)

        for actor in addtl_actors:
            renderer.AddActor(actor)

        if colorbar:
            scalarBar = vtk.vtkScalarBarActor()
            if field=='Elevation':
                scalarBar.SetLookupTable(mplcmap_to_vtkLUT(z_min, z_max))
                renderer.AddActor2D(scalarBar)
            elif field in ['Reflectance', 'reflectance_radial']:
                scalarBar.SetLookupTable(mplcmap_to_vtkLUT(z_min, z_max,
                                                           name='plasma'))
                renderer.AddActor2D(scalarBar)
            
        
        if mapview:
            camera = vtk.vtkCamera()
            camera.SetFocalPoint(0.0, 0.0, -5)
            camera.SetPosition(0.0, 0.0, 500)
            camera.ParallelProjectionOn()
            camera.SetParallelScale(500)
            renderer.SetActiveCamera(camera)
            legendScaleActor = vtk.vtkLegendScaleActor()
            renderer.AddActor(legendScaleActor)
            
        # Create RenderWindow and interactor, set style to trackball camera
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetSize(2000, 1000)
        renderWindow.AddRenderer(renderer)
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renderWindow)

        style = vtk.vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(style)
            
        iren.Initialize()
        renderWindow.Render()

        # Set camera for followers
        if show_scanners:
            for scan_name in self.scan_dict:
                self.scan_dict[scan_name].scannerText.SetCamera(
                    renderer.GetActiveCamera())
        if show_labels:
            for scan_name in self.scan_dict:
                for i in range(self.scan_dict[scan_name]
                               .labels_actors.shape[0]):
                    (self.scan_dict[scan_name].labels_actors['text_actor']
                     .iat[i].SetCamera(renderer.GetActiveCamera()))

        iren.AddObserver('UserEvent', cameraCallback)
        iren.Start()
        
        # Needed to get window to close on linux
        renderWindow.Finalize()
        iren.TerminateApp()
        
        del renderWindow, iren
    
    def project_to_image(self, z_min, z_max, focal_point, camera_position,
                         roll=0, image_scale=500, lower_threshold=-1000, 
                         upper_threshold=1000, mode=None, colorbar=True,
                         field='Elevation',
                         name='', window_size=(2000, 1000), path=None,
                         date=True, scale=True, 
                         show_reflectors=False, reflector_color='White',
                         reflector_text_scale=1, reflector_size=0.05,
                         addtl_actors=[]):
        """
        Write an image of the project to the snapshots folder.
        
        Assumes we want an orthorectified image (mode='map') and we want the
        default image name to just be the project name.

        Parameters
        ----------
        z_min : float
            Minimum z value to display colors.
        z_max : float
            Maximum z value to display colors.
        focal_point : 3 element array like
            Focal point of the camera in the project's reference frame.
        camera_position : 3 element array like
            Camera position in the project's reference frame.
        roll : float, optional
            Camera roll in degrees. The default is 0.
        image_scale : float, optional
            Image scale used in parallel projection. The default is 500.
        lower_threshold : float, optional
            Value of z to clip below. The default is -1000.
        upper_threshold : float, optional
            Value of z to clip above. The default is 1000.
        mode : str, optional
            What kind of projection system to use. 'map' indicates parallel
            or orthorectified projection. The default is None.
        colorbar : bool, optional
            Whether to display a colorbar.
        name : str, optional
            Name to append to this snapshot. The default is ''.
        window_size : tuple, optional
            Window size in pixels. The default is (2000, 1000)
        path : str, optional
            If provided, this is the path to write the image to. The default
            is None.
        show_reflectors : bool, optional
            Whether or not to show the reflectors. The default is False.
        reflector_color : str, optional
            Name of the color to display as. The default is 'White'
        reflector_text_scale : float, optional
            Text scale for reflector labels. The default is 1
        reflector_size : float, optional
            Radius of sphere representing reflector in m, the default is 0.05
        addtl_actors : list, optional
            List of additonal actors to render, the default is []

        Returns
        -------
        None.

        """
        
        # Create renderer
        renderer = vtk.vtkRenderer()
        
        # Run create elevation pipeline for each scan and add each actor
        for scan_name in self.scan_dict:
            if field=='Elevation':
                self.scan_dict[scan_name].create_elevation_pipeline(z_min, 
                                                                    z_max, 
                                                                lower_threshold, 
                                                                upper_threshold)
            elif field=='Classification':
                self.scan_dict[scan_name].create_filter_pipeline()
            else:
                self.scan_dict[scan_name].create_reflectance_pipeline(z_min,
                                                                      z_max,
                                                                      field=
                                                                      field)
            renderer.AddActor(self.scan_dict[scan_name].actor)

        if colorbar:
            scalarBar = vtk.vtkScalarBarActor()
            if field=='Elevation':
                scalarBar.SetLookupTable(mplcmap_to_vtkLUT(z_min, z_max))
                renderer.AddActor2D(scalarBar)
            elif field in ['Reflectance', 'reflectance_radial']:
                scalarBar.SetLookupTable(mplcmap_to_vtkLUT(z_min, z_max,
                                                           name='plasma'))
                renderer.AddActor2D(scalarBar)
            
        # Add project date as text
        if date:
            textActor = vtk.vtkTextActor()
            textActor.SetInput(self.project_date)
            textActor.SetPosition2(10, 40)
            textActor.SetTextScaleModeToNone()
            textActor.GetTextProperty().SetFontSize(24)
            textActor.GetTextProperty().SetColor(0.0, 1.0, 0.0)
            renderer.AddActor2D(textActor)
        
        for actor in addtl_actors:
            renderer.AddActor(actor)

        # Create RenderWindow
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetSize(window_size[0], window_size[1])
        renderWindow.AddRenderer(renderer)
        # Create Camera
        camera = vtk.vtkCamera()
        camera.SetFocalPoint(focal_point)
        camera.SetPosition(camera_position)
        if mode=='map':
            camera.ParallelProjectionOn()
            camera.SetParallelScale(image_scale)
            camera.SetViewUp(0, 1.0, 0)
            if scale:
                legendScaleActor = vtk.vtkLegendScaleActor()
                legendScaleActor.LegendVisibilityOff()
                legendScaleActor.TopAxisVisibilityOff()
                legendScaleActor.RightAxisVisibilityOff()
                legendScaleActor.LeftAxisVisibilityOff()

                # sort out this stuff in needed...
                #legendScaleActor.GetLeftAxis().SetFontFactor(3)
                legendScaleActor.GetBottomAxis().SetFontFactor(3)
                #legendScaleActor.SetLeftBorderOffset(130)
                legendScaleActor.SetBottomBorderOffset(80)
                
                renderer.AddActor(legendScaleActor)
            camera.SetRoll(roll)
        else:
            camera.SetRoll(roll)
        renderer.SetActiveCamera(camera)

        # Add reflectors if requested
        if show_reflectors:
            self.create_reflector_actors(color=reflector_color,
                                         text_scale=reflector_text_scale,
                                         size=reflector_size)
            for key in self.reflectorActorDict.keys():
                renderer.AddActor(self.reflectorActorDict[key])
                self.reflectorTextDict[key].SetCamera(camera)
                renderer.AddActor(self.reflectorTextDict[key])
        
        renderWindow.Render()

        # Sleep for a second to allow rendering to finish
        time.sleep(1)
        
        # Screenshot image to save
        w2if = vtk.vtkWindowToImageFilter()
        w2if.ShouldRerenderOff()
        w2if.SetInput(renderWindow)
        w2if.Update()
    
        writer = vtk.vtkPNGWriter()
        if path is None:
            writer.SetFileName(os.path.join(self.project_path, 'snapshots', 
                               self.project_name + '_' + name + '.png'))
        else:
            writer.SetFileName(path)
        writer.SetInputData(w2if.GetOutput())
        writer.Write()
    
        renderWindow.Finalize()
        del renderWindow
        
    def point_to_grid_average_image(self, nx, ny, dx, dy, x0, y0, yaw=0,
                                    key='', overwrite=False):
        """
        Convert a rectangular area of points to an image by gridded averaging

        Requires cython_util

        Parameters
        ----------
        nx : int
            Number of gridcells in x direction.
        ny : int
            Number of gridcells in y direction.
        dx : float
            Width of gridcells in x direction.
        dy : float
            Width of gridcells in y direction.
        x0 : float
            x coordinate of the origin in m.
        y0 : float
            y coordinate of the origin in m.
        yaw : float, optional
            yaw angle in degrees of image to create, for generating 
            non-axis aligned image. The default is 0.
        key : str, optional
            Key to index this image and transform in image_dict and
            image_transform_dict. The default is '' (for backward compatability)
        overwrite : bool, optional
            Whether to overwrite preexisting image. If False and key already
            exists in image_dict will raise a warning. The default is False.

        Returns:
        --------
        None

        """

        # Check if key already exists
        if key in self.image_dict.keys():
            if overwrite:
                warnings.warn('You are overwriting image: ' + str(key))
            else:
                raise RuntimeWarning(str(key) + 'already exists in image_dict ' +
                                     'set overwrite=True if you want to' + 
                                     'overwrite')
                return

        # Get the points in this rectangular region
        pdata, point_history_dict = self.get_merged_points(history_dict=True,
            x0=x0, y0=y0, wx=nx*dx, wy=ny*dy, yaw=yaw)

        # Transform points into image reference frame, needed because
        # vtkImageData can only be axis-aligned. We'll save the transform 
        # filter in case we need it for mapping flags, etc, into image
        self.image_transform_dict[key] = vtk.vtkTransform()
        self.image_transform_dict[key].PostMultiply()
        # Translate origin to be at (x0, y0)
        self.image_transform_dict[key].Translate(-x0, -y0, 0)
        # Rotate around this origin
        self.image_transform_dict[key].RotateZ(-yaw)
        
        # Create transform filter and apply
        imageTransformFilter = vtk.vtkTransformPolyDataFilter()
        imageTransformFilter.SetTransform(self.image_transform_dict[key])
        imageTransformFilter.SetInputData(pdata)
        imageTransformFilter.Update()

        # Get the points and the z_sigma as numpy arrays
        pts_np = vtk_to_numpy(imageTransformFilter.GetOutput().GetPoints()
                              .GetData())

        # Create grid
        edges = 2*[None]
        nbin = np.empty(2, np.int_)
        w = [dx, dy]
        n = [nx, ny]
        for i in range(2):
            edges[i] = np.arange(n[i]+ 1, dtype=np.float32) * w[i]
            # Adjust lower edge so we don't miss lower most point
            edges[i][0] = edges[i][0] - 0.0001
            # Adjust uppermost edge so the bin width is appropriate
            edges[i][-1] += 0.0001
            nbin[i] = len(edges[i]) + 1

        # Use cython to iterate through grid
        _, grid_mean = gridded_counts_means(pts_np, edges)

        # Create image
        self.image_dict[key] = vtk.vtkImageData()
        self.image_dict[key].SetDimensions(nx, ny, 1)
        self.image_dict[key].SetSpacing(dx, dy, 1)
        self.image_dict[key].SetOrigin(0, 0, 0)
        warnings.warn('Still not sure if origin should be offset half grid cell')
        # Store np arrays related to image
        if not hasattr(self, 'np_dict'):
            self.np_dict = {}
        self.np_dict[('image_z', key)] = np.ravel(grid_mean, order='F')
        vtk_arr = numpy_to_vtk(self.np_dict[('image_z', key)], deep=False, 
                               array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName('Elevation')
        self.image_dict[key].GetPointData().AddArray(vtk_arr)
        self.image_dict[key].GetPointData().SetActiveScalars('Elevation')
        self.image_dict[key].Modified()

        # Create history dict
        self.image_history_dict_dict[key] = {
            "type": "Image Generator",
            "git_hash": get_git_hash(),
            "method": "Project.point_to_grid_average_image",
            "input_0": point_history_dict,
            "params": {"x0": x0, "y0": y0, "yaw": yaw}
            }

        # Also wrap with a datasetadaptor for working with numpy
        self.dsa_image_dict[key] = dsa.WrapDataObject(self.image_dict[key])

    def merged_points_to_image(self, nx, ny, dx, dy, x0, y0, lengthscale, 
                               outputscale, nu, yaw=0, n_neighbors=50, max_pts=
                               64000, min_pts=100, mx=32, my=32, eps=0, 
                               corner_coords=None,
                               max_dist=None, optimize=False, learning_rate=0.1,
                               n_iter=5, max_time=0.5, optim_param=None,
                               multiply_outputscale=False, var_radius=None,
                               key='', overwrite=False):
        """
        Convert a rectangular area of points to an image using gpytorch.
        
        Take a rectangular area, create an array of gridded coordinates where
        we want to evaluate surface (grid_points). For each grid_point, find
        the n_neighbors nearest neighbors in the lidar point cloud. Group
        grid_points into squarish groups of mx x my grid_points. For each of
        these M groups pool the nearest neighbors. Now we have groups of
        at most mx x my x n_neighbors lidar points with which to estimate the
        surface height and mx x my grid_points. For each M group, run 
        gaussian process regression to estimate the surface height at the grid
        points. Combine all grid points together into one image and save the
        result in the self.image attribute.
        
        Parameters
        ----------
        nx : int
            Number of gridcells in x direction.
        ny : int
            Number of gridcells in y direction.
        dx : float
            Width of gridcells in x direction.
        dy : float
            Width of gridcells in y direction.
        x0 : float
            x coordinate of the origin in m.
        y0 : float
            y coordinate of the origin in m.
        yaw : float, optional
            yaw angle in degrees of image to create, for generating 
            non-axis aligned image. The default is 0.
        lengthscale : float
            Length scale for the matern kernel in m. 2 seems reasonable.
        outputscale : float
            Outputscale for the scale kernel. Around 0.01 seems reasonable.
        nu : float
            Nu for the Matern kernel, must be one of 0.5, 1.5, or 2.5
        n_neighbors : int, optional
            Number of neighbors around each grid point. The default is 50.
        max_pts : int, optional
            If a chunk has more points than max_pts, then reduce n_neighbors by
            10% until we're within max_pts.
        min_pts : int, optional
            If a chunk has fewer points than min_pts, increase n_neighbors by
            10% until we're greater than min_pts.

        mx : int, optional
            Number of grid points in the x direction to group together in M
            groups. The default is 32.
        my : int, optional
            Number of grid points in the y direction to group together in M
            groups. The default is 32.
        eps : float, optional
            Eps for nearest neighbors search (see scipy.spatial.cKDTree) for
            more details. The default is 0 (exact nearest neighbors)
        corner_coords : Nx3 array, optional
            Corner coordinates of selection if we want to limit output
        optimize : bool, optional
            Whether or not to search for optimal hyperparameters for the GP. The
            default is False.
        learning_rate : float, optional
            The learning rate if we are optimizing (currently just using ADAM). 
            The default is 0.1.
        n_iter : int, optional
            The number of iterations to run optimizer for. Note that we will 
            only ever run optimizer for max_time. The default is None.
        max_time : float, optional
            The maximum amount of time (in seconds) to run an optimization for. 
            The default is 60.
        optim_param : list, optional
            The parameters to optimize, as a list of dicts. If None, we will
            optimize across all parameters. The default is None.
        multiply_outputscale : bool, optional
            If True, set outputscale for each chunk to be outputscale times
            the variance of the z-values of points in that chunk. The default 
            is False.
        var_radius : float, optional
            If this is given and multiply_outputscale is True, instead of 
            using the variance of the points in the chunk, we use the variance
            of the z values of points within var_radius of the centroid of
            the chunk. The default is None.
        key : str, optional
            Key to index this image and transform in image_dict and
            image_transform_dict. The default is '' (for backward compatability)
        overwrite : bool, optional
            Whether to overwrite preexisting image. If False and key already
            exists in image_dict will raise a warning. The default is False.

        Returns
        -------
        None

        """

        # Check if key already exists
        if key in self.image_dict.keys():
            if overwrite:
                raise RuntimeWarning('You are overwriting image: ' + key)
            else:
                raise RuntimeWarning(key + 'already exists in image_dict ' +
                                     'set overwrite=True if you want to' + 
                                     'overwrite')
                return

        # Get the points in this rectangular region
        pdata, point_history_dict = self.get_merged_points(history_dict=True,
            x0=x0, y0=y0, wx=nx*dx, wy=ny*dy, yaw=yaw)

        # Transform points into image reference frame, needed because
        # vtkImageData can only be axis-aligned. We'll save the transform 
        # filter in case we need it for mapping flags, etc, into image
        self.image_transform_dict[key] = vtk.vtkTransform()
        self.image_transform_dict[key].PostMultiply()
        # Translate origin to be at (x0, y0)
        self.image_transform_dict[key].Translate(-x0, -y0, 0)
        # Rotate around this origin
        self.image_transform_dict[key].RotateZ(-yaw)
        
        # Create transform filter and apply
        imageTransformFilter = vtk.vtkTransformPolyDataFilter()
        imageTransformFilter.SetTransform(self.image_transform_dict[key])
        imageTransformFilter.SetInputData(pdata)
        imageTransformFilter.Update()

        # Get the points and the z_sigma as numpy arrays
        pts_np = vtk_to_numpy(imageTransformFilter.GetOutput().GetPoints()
                              .GetData())
        z_sigma_np = vtk_to_numpy(imageTransformFilter.GetOutput()
                                  .GetPointData().GetArray('z_sigma'))

        # Create grid for image
        x_vals = np.arange(nx, dtype=np.float32) * dx
        y_vals = np.arange(ny, dtype=np.float32) * dy
        X, Y = np.meshgrid(x_vals, y_vals)
        grid_points = np.hstack((X.ravel()[:,np.newaxis], 
                                 Y.ravel()[:,np.newaxis]))
        # If we have corner_coords, limit grid points to just be within them
        if corner_coords is None:
            grid_points_mask = np.ones(grid_points.shape[0], dtype=np.bool)
        else:
            # Transform each point in corner_coords into image space
            pts = vtk.vtkPoints()
            for i in range(corner_coords.shape[0]):
                pts.InsertNextPoint(self.imageTransform.TransformPoint(
                    corner_coords[i,0], corner_coords[i,1], corner_coords[i,2]))
            grid_points_all_vtk = vtk.vtkPoints()
            grid_points_all_vtk.SetData(numpy_to_vtk(np.hstack(
                (grid_points, np.zeros((grid_points.shape[0],1)))),
                                        array_type=vtk.VTK_DOUBLE))
            grid_points_pdata = vtk.vtkPolyData()
            grid_points_pdata.SetPoints(grid_points_all_vtk)
            vtkarr = numpy_to_vtk(np.arange(grid_points.shape[0], 
                                            dtype='uint64'), deep=True, 
                                  array_type=vtk.VTK_UNSIGNED_LONG)
            vtkarr.SetName('PointId')
            #grid_points_pdata.GetPointData().AddArray(vtkarr)
            grid_points_pdata.GetPointData().SetPedigreeIds(vtkarr)
            grid_points_pdata.Modified()
            # Create implicit selection loop
            selectionLoop = vtk.vtkImplicitSelectionLoop()
            selectionLoop.SetNormal(0.0, 0.0, 1.0)
            selectionLoop.SetLoop(pts)
            # Select points
            extractPoints = vtk.vtkExtractPoints()
            extractPoints.SetImplicitFunction(selectionLoop)
            extractPoints.SetInputData(grid_points_pdata)
            extractPoints.Update()
            # These are the indices of points inside our corner_coords
            PointIds = vtk_to_numpy(extractPoints.GetOutput().GetPointData().
                                    GetArray('PointId'))
            #grid_points = grid_points_all[PointIds,:]
            # Get as a mask of true or falses (for logical indexing)
            grid_points_mask = np.isin(vtk_to_numpy(vtkarr), PointIds,
                                       assume_unique=True)
        # If we have set max_dist, limit grid_points_mask to be within max_dist
        if not (max_dist is None):
            dist_mask = np.zeros(grid_points.shape[0], dtype=np.bool)
            for scan_name in self.scan_dict:
                # For each scan, get scanner position image coordinates
                scan_pos = np.array(self.imageTransform.TransformPoint(
                            self.scan_dict[scan_name].transform.GetPosition())
                            )[np.newaxis,:2]
                # set all points that are within max_dist of scanner to True
                dist_mask = dist_mask | (
                            np.square(grid_points - scan_pos).sum(axis=1)
                                <=max_dist**2)
            grid_points_mask = grid_points_mask & dist_mask

        
        # Build a cKDTree from the xy coordinates of pts_np
        kdtree = cKDTree(pts_np[:,:2])
        
        # Indices for stepping through grid_points in mx x my chunks
        i_mx = 0
        j_my = 0
        ctr = 0
        # Output
        grid_mean = np.empty(nx*ny, dtype=np.float32) * np.nan
        grid_lower = np.empty(nx*ny, dtype=np.float32) * np.nan
        grid_upper = np.empty(nx*ny, dtype=np.float32) * np.nan
        gp_mean = np.empty(nx*ny, dtype=np.float32) * np.nan
        gp_outputscale = np.empty(nx*ny, dtype=np.float32) * np.nan
        gp_lengthscale = np.empty(nx*ny, dtype=np.float32) * np.nan
        total_points = grid_points_mask.sum()
        t0 = time.perf_counter()
        true_ctr = 0
        while ctr<(nx*ny):
            if (ctr%10000)<=(mx*my):
                print(str(true_ctr) + ' / ' + str(total_points) + ' ~time: ' 
                      + str((total_points-true_ctr)*(time.perf_counter() - t0)
                            /true_ctr))

            # Get start and end indices for x and y values
            x_s = mx*i_mx
            x_e = nx if x_s+mx>nx else x_s+mx
            y_s = my*j_my
            y_e = ny if y_s+my>ny else y_s+my
            step = (y_e-y_s)*(x_e-x_s)
            
            # Get the grid points in this chunk and use kdtree to pull point 
            # indices
            igridX, igridY = np.meshgrid(np.arange(x_s, x_e), 
                                         np.arange(y_s, y_e))
            ind = igridX + nx*igridY
            ind = ind.ravel()
            # Subset to just those indices that are within corner coords
            # that is, indices where grid_points_mask is True
            ind = ind[grid_points_mask[ind]]
            true_ctr += ind.size
            # If ind is empty that means all grid points in this chunk have been
            # masked, skip this chunk and go to the next one
            if ind.size>0:
                m_grid = grid_points[ind, :]
                # This while loop enforces that we're below max_pts
                c_n_neighbors = n_neighbors
                while True:
                    _, pt_ind = kdtree.query(m_grid, c_n_neighbors, eps=eps, 
                                             workers=-1)
                    del _
                    pt_ind = np.unique(pt_ind.ravel())
                    if pt_ind.size<max_pts:
                        break

                    c_n_neighbors = (c_n_neighbors*9)//10
                # This while loop enforces that we're greater than min_pts
                while len(pt_ind)<min_pts:
                    c_n_neighbors = (c_n_neighbors*11)//10
                    _, pt_ind = kdtree.query(m_grid, c_n_neighbors, eps=eps, 
                                             workers=-1)
                    del _
                    pt_ind = np.unique(pt_ind.ravel())
                
                # If we are using a non-constant outputscale
                if multiply_outputscale:
                    # if we have no var_radius, set to multiple of variance
                    # of points in chunk
                    if var_radius is None:
                        c_outputscale = outputscale * pts_np[pt_ind,2].var()
                    else:
                        # query points around centroid of grid points and
                        # and multiply their variance by outputscale
                        # If there are few points within this radius increase it
                        var_ind = []
                        cntr = m_grid.mean(axis=0)
                        c_var_radius = var_radius
                        while len(var_ind)<1000:
                            var_ind = kdtree.query_ball_point(cntr,
                                                            r=c_var_radius,
                                                            eps=0.2, workers=-1)
                            c_var_radius = c_var_radius * 1.1

                        c_outputscale = outputscale * pts_np[np.array(var_ind), 
                                                            2].var()
                else:
                    # Otherwise just use the fixed outputscale from params.
                    c_outputscale = outputscale

                # Run our GP on this chunk and estimate values at grid points
                # use indices from above to place output in the right places
                grid_mean[ind], grid_lower[ind], grid_upper[ind], m, v, l = run_gp(
                    pts_np[pt_ind,:], pts_np[pt_ind,2], m_grid,
                    z_sigma=z_sigma_np[pt_ind], lengthscale=lengthscale, 
                    outputscale=c_outputscale, nu=nu, optimize=optimize,
                    learning_rate=learning_rate, iter=n_iter, max_time=max_time,
                    optim_param=optim_param, multiply_outputscale=False)
                gp_mean[ind] = np.float32(m)
                gp_outputscale[ind] = np.float32(v)
                gp_lengthscale[ind] = np.float32(l)
            
            # Move i_mx and j_my to the next grid_points group
            if x_e==nx:
                # If we are at the end of a row, return i_mx to 0 and 
                # increment j_my
                i_mx = 0
                j_my += 1
            else:
                # Otherwise, just increment i_mx and leave j_my
                i_mx += 1
            # Increment counter
            ctr += step



        # Create image
        self.image_dict[key] = vtk.vtkImageData()
        self.image_dict[key].SetDimensions(nx, ny, 1)
        self.image_dict[key].SetSpacing(dx, dy, 1)
        self.image_dict[key].SetOrigin(0, 0, 0)
        # Store np arrays related to image
        if not hasattr(self, 'np_dict'):
            self.np_dict = {}
        self.np_dict[('image_z', key)] = grid_mean
        vtk_arr = numpy_to_vtk(self.np_dict[('image_z', key)], deep=False, 
                               array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName('Elevation')
        self.image_dict[key].GetPointData().AddArray(vtk_arr)
        self.np_dict[('image_z_lower', key)] = grid_lower
        vtk_arr = numpy_to_vtk(self.np_dict[('image_z_lower', key)], deep=False, 
                               array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName('z_lower')
        self.image_dict[key].GetPointData().AddArray(vtk_arr)
        self.np_dict[('image_z_upper', key)] = grid_upper
        vtk_arr = numpy_to_vtk(self.np_dict[('image_z_upper', key)], deep=False, 
                               array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName('z_upper')
        self.image_dict[key].GetPointData().AddArray(vtk_arr)
        self.np_dict[('image_z_ci', key)] = grid_upper - grid_lower
        vtk_arr = numpy_to_vtk(self.np_dict[('image_z_ci', key)], deep=False, 
                               array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName('z_ci')
        self.image_dict[key].GetPointData().AddArray(vtk_arr)
        self.np_dict[('gp_mean', key)] = gp_mean
        vtk_arr = numpy_to_vtk(self.np_dict[('gp_mean', key)], deep=False, 
                               array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName('gp_mean')
        self.image_dict[key].GetPointData().AddArray(vtk_arr)
        self.np_dict[('gp_outputscale', key)] = gp_outputscale
        vtk_arr = numpy_to_vtk(self.np_dict[('gp_outputscale', key)], deep=False, 
                               array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName('gp_outputscale')
        self.image_dict[key].GetPointData().AddArray(vtk_arr)
        self.np_dict[('gp_lengthscale', key)] = gp_lengthscale
        vtk_arr = numpy_to_vtk(self.np_dict[('gp_lengthscale', key)], deep=False, 
                               array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName('gp_lengthscale')
        self.image_dict[key].GetPointData().AddArray(vtk_arr)
        self.image_dict[key].GetPointData().SetActiveScalars('Elevation')
        self.image_dict[key].Modified()

        # Create history dict
        self.image_history_dict_dict[key] = {
            "type": "Image Generator",
            "git_hash": get_git_hash(),
            "method": "Project.merged_points_to_image",
            "input_0": point_history_dict,
            "params": {"x0": x0, "y0": y0, "yaw": yaw, "nu": nu,
                       "n_neighbors": n_neighbors, "mx": mx, "my": my,
                       "eps": eps, "corner_coords": np.float64(corner_coords)}
            }

        # Also wrap with a datasetadaptor for working with numpy
        self.dsa_image_dict[key] = dsa.WrapDataObject(self.image_dict[key])

    def mesh_to_image(self, nx, ny, dx, dy, x0, y0, yaw=0, key='', 
                      overwrite=False):
        """
        Interpolate mesh at regularly spaced points.
        
        Currently this image can only be axis aligned, if you want a different
        orientation then you need to apply the appropriate transformation to
        the mesh.

        Parameters
        ----------
        nx : int
            Number of gridcells in x direction.
        ny : int
            Number of gridcells in y direction.
        dx : int
            Width of gridcells in x direction.
        dy : int
            Width of gridcells in y direction.
        x0 : float
            x coordinate of the origin in m.
        y0 : float
            y coordinate of the origin in m.
        yaw : float, optional
            yaw angle in degerees of image to create, for generating 
            non-axis aligned image. The default is 0
        key : str, optional
            Key to index this image and transform in image_dict and
            image_transform_dict. The default is '' (for backward compatability)
        overwrite : bool, optional
            Whether to overwrite preexisting image. If False and key already
            exists in image_dict will raise a warning. The default is False.

        Returns
        -------
        None.

        """
        
        # Check if key already exists
        if key in self.image_dict.keys():
            if overwrite:
                warnings.warn('You are overwriting image: ' + key)
            else:
                raise RuntimeWarning(key + 'already exists in image_dict ' +
                                     'set overwrite=True if you want to' + 
                                     'overwrite')
                return

        # Use elevation filter to write z data to scalars
        elevFilter = vtk.vtkSimpleElevationFilter()
        elevFilter.SetInputData(self.mesh)
        elevFilter.Update()
        
        # Flatten Mesh, z data is now redundant (its in scalars)
        transform = vtk.vtkTransform()
        transform.Scale(1, 1, 0)
        flattener = vtk.vtkTransformFilter()
        flattener.SetTransform(transform)
        flattener.SetInputData(elevFilter.GetOutput())
        flattener.Update()
        
        # Transform mesh into image reference frame, needed because
        # vtkImageData can only be axis-aligned. We'll save the transform 
        # filter in case we need it for mapping flags, etc, into image
        self.image_transform_dict[key] = vtk.vtkTransform()
        self.image_transform_dict[key].PostMultiply()
        # Translate origin to be at (x0, y0)
        self.image_transform_dict[key].Translate(-x0, -y0, 0)
        # Rotate around this origin
        self.image_transform_dict[key].RotateZ(-yaw)
        
        # Create transform filter and apply
        imageTransformFilter = vtk.vtkTransformPolyDataFilter()
        imageTransformFilter.SetTransform(self.image_transform_dict[key])
        imageTransformFilter.SetInputData(flattener.GetOutput())
        imageTransformFilter.Update()
        
        # Create image
        im = vtk.vtkImageData()
        im.SetDimensions(nx, ny, 1)
        im.SetSpacing(dx, dy, 1)
        im.SetOrigin(0, 0, 0)
        im.AllocateScalars(vtk.VTK_FLOAT, 1)
            
        
        # Use probe filter to interpolate
        probe = vtk.vtkProbeFilter()
        probe.SetSourceData(imageTransformFilter.GetOutput())
        probe.SetInputData(im)
        probe.Update()
        
        self.image_dict[key] = probe.GetOutput()
        
        # Create history dict
        self.image_history_dict_dict[key] = {
            "type": "Image Generator",
            "git_hash": get_git_hash(),
            "method": "Project.mesh_to_image",
            "input_0": self.mesh_history_dict,
            "params": {"x0": x0, "y0": y0, "yaw": yaw}
            }

        # Also wrap with a datasetadaptor for working with numpy
        self.dsa_image_dict[key] = dsa.WrapDataObject(self.image_dict[key])
        
        # and use dsa to set NaN values where we have no data
        bool_arr = self.dsa_image_dict[key].PointData['vtkValidPointMask']==0
        self.dsa_image_dict[key].PointData['Elevation'][bool_arr] = np.NaN
    
    def get_image(self, field='Elevation', warp_scalars=False,
                  v_min=-9999.0, nan_value=None, key=''):
        """
        Return image as vtkImageData or vtkPolyData depending on warp_scalars

        Parameters
        ----------
        field : str, optional
            Which field in PointData to set active. The default is 'Elevation'
        warp_scalars : bool, optional
            Whether to warp the scalars in the image to create 3D surface
        key : str, optional
            Key to index this image and transform in image_dict and
            image_transform_dict. The default is '' (for backward compatability)

        Returns
        -------
        image: vtkImageData or vtkPolyData

        """
        
        self.image_dict[key].GetPointData().SetActiveScalars(field)
        if warp_scalars:
            # copy image
            im = vtk.vtkImageData()
            im.DeepCopy(self.image_dict[key])
            field_np = vtk_to_numpy(im.GetPointData().GetArray(field))
            if nan_value:
                field_np[np.isnan(field_np)] = nan_value
            else:
                field_np[np.isnan(field_np)] = v_min - 0.1
            im.Modified()
            geometry = vtk.vtkImageDataGeometryFilter()
            geometry.SetInputData(im)
            geometry.SetThresholdValue(v_min - 0.05)
            geometry.ThresholdCellsOn()
            geometry.Update()
            tri = vtk.vtkTriangleFilter()
            tri.SetInputData(geometry.GetOutput())
            tri.Update()
            strip = vtk.vtkStripper()
            strip.SetInputData(tri.GetOutput())
            strip.Update()
            warp = vtk.vtkWarpScalar()
            warp.SetScaleFactor(1)
            warp.SetInputData(strip.GetOutput())
            warp.Update()
            #return warp.GetOutput()
            # Compute normals to make the shading look better
            normals = vtk.vtkPPolyDataNormals()
            normals.SetInputData(warp.GetOutput())
            normals.Update()
            
            return normals.GetOutput()
        
        else:
            return self.image_dict[key]
    
    def display_image(self, z_min, z_max, field='Elevation',
                      warp_scalars=False, color_field=None,
                      show_points=False, profile_list=[],
                      show_scanners=False,
                      scanner_color='Gray', scanner_length=150,
                      pdata_list=[], key=''):
        """
        Display image in vtk interactive window.

        Parameters
        ----------
        z_min : float
            Minimum z value in m for color.
        z_max : float
            Maximum z value in m for color.
        field : str, optional
            Which field in PointData to display. The default is 'Elevation'
        warp_scalars : bool, optional
            Whether to warp the scalars in the image to create 3D surface
        color_field : str, optional
            If we want to display the color for a different field overlain
            on the geometry from warp scalars use this option. The defaut
            is None.
        show_points : bool, optional
            Whether to also render the points. The default is False.
        profile_list : list, optional
            Which, if any, profiles to display along with the rendering. This
            list is composed of lists whose zeroth element is always the
            key of the profile in self.profile_dict. Element 1 is line width
            in pixels (optional) Elements 2, 3, 4, are color
            channels (optional) and element 5 is opacity (optional). The default
            is [].
        pdata_list : list, optional
            Which, if any, pdata to display along with the rendering. This
            list is composed of lists whose zeroth element is always the
            key of the profile in self.pdata_dict. Element 1 is point/line width
            in pixels (optional) Elements 2, 3, 4, are color
            channels (optional) and element 5 is opacity (optional). The default
            is [].
        show_scanners : bool, optional
            Whether or not to show the scanners. The default is False.
        scanner_color : str, optional
            Name of the color to display as. The default is 'Gray'
        scanner_length : float, optional
            Length of the ray indicating the scanner's start orientation in m.
            The default is 150
        key : str, optional
            Key to index this image and transform in image_dict and
            image_transform_dict. The default is '' (for backward compatability)

        Returns
        -------
        None.

        """
        
        # Define function for writing the camera position and focal point to
        # std out when the user presses 'u'
        def cameraCallback(obj, event):
            print("Camera Pos: " + str(obj.GetRenderWindow().
                                           GetRenderers().GetFirstRenderer().
                                           GetActiveCamera().GetPosition()))
            print("Focal Point: " + str(obj.GetRenderWindow().
                                            GetRenderers().GetFirstRenderer().
                                            GetActiveCamera().GetFocalPoint()))
            print("Roll: " + str(obj.GetRenderWindow().
                                            GetRenderers().GetFirstRenderer().
                                            GetActiveCamera().GetRoll()))
        
        if warp_scalars:
            mapper = vtk.vtkPolyDataMapper()
            min_scalar = np.nanmin(vtk_to_numpy(self.image_dict[key]
                                                .GetPointData().
                                                GetArray(field)))
            mapper.SetInputData(self.get_image(field, warp_scalars, 
                                               v_min=min_scalar, key=key))
            if not color_field is None:
                mapper.GetInput().GetPointData().SetActiveScalars(color_field)
            
        else:
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(self.get_image(field, warp_scalars, key=key))
            
        mapper.SetLookupTable(mplcmap_to_vtkLUT(z_min, z_max))
        mapper.SetScalarRange(z_min, z_max)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetSize(2000, 1000)
        renderer.AddActor(actor)
        
        if show_points:
            for scan_name in self.scan_dict:
                self.scan_dict[scan_name].create_reflectance_pipeline(z_min,
                                                                      z_max,
                                                                      field=
                                                                      field)
                self.scan_dict[scan_name].actor.SetUserTransform(
                    self.image_transform_dict[key])
                renderer.AddActor(self.scan_dict[scan_name].actor)

        # Add scanners if requested
        if show_scanners:
            for scan_name in self.scan_dict:
                self.scan_dict[scan_name].create_scanner_actor(
                    color=scanner_color, length=scanner_length)
                # copy the userTransform and concatenate the imageTransform
                transform = vtk.vtkTransform()
                transform.DeepCopy(self.scan_dict[scan_name].scannerActor
                                   .GetUserTransform())
                transform.Concatenate(self.image_transform_dict[key])
                self.scan_dict[scan_name].scannerActor.SetUserTransform(
                    transform)
                self.scan_dict[scan_name].scannerText.AddPosition(
                    self.image_transform_dict[key].GetPosition())
                renderer.AddActor(self.scan_dict[scan_name].scannerActor)
                renderer.AddActor(self.scan_dict[scan_name].scannerText)

        # Add requested profiles
        for profile_tup in profile_list:
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(self.image_transform_dict[key])
            transformFilter.SetInputData(self.profile_dict[profile_tup[0]])
            transformFilter.Update()
            mapper = vtk.vtkPolyDataMapper()
            if warp_scalars:
                mapper.SetInputData(transformFilter.GetOutput())
            else:
                flattener = vtk.vtkTransform()
                flattener.Scale(1, 1, 0)
                flatFilter = vtk.vtkTransformPolyDataFilter()
                flatFilter.SetTransform(flattener)
                flatFilter.SetInputData(transformFilter.GetOutput())
                flatFilter.Update()
                mapper.SetInputData(flatFilter.GetOutput())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            if len(profile_tup)>=2:
                actor.GetProperty().SetLineWidth(profile_tup[1])
            else:
                actor.GetProperty().SetLineWidth(5)
            if len(profile_tup)>=5:
                actor.GetProperty().SetColor(profile_tup[2:5])
            else:
                actor.GetProperty().SetColor(0.0, 1.0, 0.0)
            if len(profile_tup)>=6:
                actor.GetProperty().SetOpacity(profile_tup[5])
            else:
                actor.GetProperty().SetOpacity(0.8)
            actor.GetProperty().RenderLinesAsTubesOn()
            renderer.AddActor(actor)
        
        # Add requested pdata
        for profile_tup in pdata_list:
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(self.image_transform_dict[key])
            transformFilter.SetInputData(self.pdata_dict[profile_tup[0]])
            transformFilter.Update()
            mapper = vtk.vtkPolyDataMapper()
            if warp_scalars:
                mapper.SetInputData(transformFilter.GetOutput())
            else:
                flattener = vtk.vtkTransform()
                flattener.Scale(1, 1, 0)
                flatFilter = vtk.vtkTransformPolyDataFilter()
                flatFilter.SetTransform(flattener)
                flatFilter.SetInputData(transformFilter.GetOutput())
                flatFilter.Update()
                mapper.SetInputData(flatFilter.GetOutput())
            mapper.SetScalarVisibility(0)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            if len(profile_tup)>=2:
                actor.GetProperty().SetLineWidth(profile_tup[1])
                actor.GetProperty().SetPointSize(profile_tup[1])
            else:
                actor.GetProperty().SetLineWidth(5)
                actor.GetProperty().SetPointSize(10)
            if len(profile_tup)>=5:
                actor.GetProperty().SetColor(profile_tup[2:5])
            else:
                actor.GetProperty().SetColor(1.0, 1.0, 1.0)
            if len(profile_tup)>=6:
                actor.GetProperty().SetOpacity(profile_tup[5])
            else:
                actor.GetProperty().SetOpacity(0.8)
            actor.GetProperty().RenderLinesAsTubesOn()
            actor.GetProperty().RenderPointsAsSpheresOn()
            renderer.AddActor(actor)

        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetLookupTable(mplcmap_to_vtkLUT(z_min, z_max))
        renderer.AddActor2D(scalarBar)
        
        renderWindow.AddRenderer(renderer)
        
        # create a renderwindowinteractor
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renderWindow)
        
        iren.Initialize()
        renderWindow.Render()

        # Set camera for followers
        if show_scanners:
            for scan_name in self.scan_dict:
                self.scan_dict[scan_name].scannerText.SetCamera(
                    renderer.GetActiveCamera())

        iren.AddObserver('UserEvent', cameraCallback)
        iren.Start()
    
    def write_plot_image(self, z_min, z_max, focal_point, camera_position,
                         field='Elevation', warp_scalars=False,
                         roll=0, image_scale=500, lower_threshold=-1000, 
                         upper_threshold=1000, mode='map', colorbar=True,
                         name='', light=None, profile_list=[],
                         window_size=(2000,1000), key=''):
        """
        Write an image of the image to the snapshots folder.
        
        Assumes we want an orthorectified image (mode='map') and we want the
        default image name to just be the project name.

        Parameters
        ----------
        z_min : float
            Minimum z value to display colors.
        z_max : float
            Maximum z value to display colors.
        focal_point : 3 element array like
            Focal point of the camera in the project's reference frame.
        camera_position : 3 element array like
            Camera position in the project's reference frame.
        field : str, optional
            Which field in PointData to display. The default is 'Elevation'
        warp_scalars : bool, optional
            Whether to warp the scalars in the image to create 3D surface
        roll : float, optional
            Camera roll in degrees. The default is 0.
        image_scale : float, optional
            Image scale used in parallel projection. The default is 500.
        lower_threshold : float, optional
            Value of z to clip below. The default is -1000.
        upper_threshold : float, optional
            Value of z to clip above. The default is 1000.
        mode : str, optional
            What kind of projection system to use. 'Map' indicates parallel
            or orthorectified projection. The default is 'map'.
        colorbar : bool, optional
            Whether to display a colorbar.
        name : str, optional
            Name to append to this snapshot. The default is ''.
        key : str, optional
            Key to index this image and transform in image_dict and
            image_transform_dict. The default is '' (for backward compatability)

        Returns
        -------
        None.

        """
        
        # Get image
        if warp_scalars:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.get_image(field, warp_scalars, key=key))
        else:
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(self.get_image(field, warp_scalars, key=key))
        mapper.SetLookupTable(mplcmap_to_vtkLUT(z_min, z_max))
        mapper.SetScalarRange(z_min, z_max)
        
        # Create actor and renderer        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        
        # Add requested profiles
        for profile_tup in profile_list:
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(self.image_transform_dict[key])
            transformFilter.SetInputData(self.profile_dict[profile_tup[0]])
            transformFilter.Update()
            mapper = vtk.vtkPolyDataMapper()
            if warp_scalars:
                mapper.SetInputData(transformFilter.GetOutput())
            else:
                flattener = vtk.vtkTransform()
                flattener.Scale(1, 1, 0)
                flatFilter = vtk.vtkTransformPolyDataFilter()
                flatFilter.SetTransform(flattener)
                flatFilter.SetInputData(transformFilter.GetOutput())
                flatFilter.Update()
                mapper.SetInputData(flatFilter.GetOutput())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            if len(profile_tup)>=2:
                actor.GetProperty().SetLineWidth(profile_tup[1])
            else:
                actor.GetProperty().SetLineWidth(5)
            if len(profile_tup)>=5:
                actor.GetProperty().SetColor(profile_tup[2:5])
            else:
                actor.GetProperty().SetColor(0.0, 1.0, 0.0)
            if len(profile_tup)>=6:
                actor.GetProperty().SetOpacity(profile_tup[5])
            else:
                actor.GetProperty().SetOpacity(0.8)
            actor.GetProperty().RenderLinesAsTubesOn()
            renderer.AddActor(actor)

        if colorbar:
            scalarBar = vtk.vtkScalarBarActor()
            scalarBar.SetLookupTable(mplcmap_to_vtkLUT(z_min, z_max))
            renderer.AddActor2D(scalarBar)
            
        # Add project date as text
        textActor = vtk.vtkTextActor()
        textActor.SetInput(self.project_date)
        textActor.SetPosition2(10, 40)
        textActor.SetTextScaleModeToNone()
        textActor.GetTextProperty().SetFontSize(24)
        textActor.GetTextProperty().SetColor(0.0, 1.0, 0.0)
        renderer.AddActor2D(textActor)
        
        # Create RenderWindow
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetSize(window_size[0], window_size[1])
        renderWindow.AddRenderer(renderer)
        # Create Camera
        camera = vtk.vtkCamera()
        camera.SetFocalPoint(focal_point)
        camera.SetPosition(camera_position)
        if mode=='map':
            camera.ParallelProjectionOn()
            camera.SetParallelScale(image_scale)
            camera.SetViewUp(0, 1.0, 0)
            legendScaleActor = vtk.vtkLegendScaleActor()
            renderer.AddActor(legendScaleActor)
        else:
            camera.SetRoll(roll)
        renderer.SetActiveCamera(camera)
        
        renderWindow.Render()
        
        # Screenshot image to save
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(renderWindow)
        w2if.Update()
    
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(os.path.join(self.project_path, 'snapshots', 
                           self.project_name + '_' + name + '.png'))
        writer.SetInputData(w2if.GetOutput())
        writer.Write()
    
        renderWindow.Finalize()
        del renderWindow
    
    def plot_image(self, z_min, z_max, cmap='inferno', figsize=(15, 15),
                   key=''):
        """
        Plot the image of this project using matplotlib

        Parameters
        ----------
        z_min : float
            Minimum z value in m for color.
        z_max : float
            Maximum z value in m for color.
        cmap : str, optional
            Name of matplotlib colormap to use. The default is 'inferno'.
        figsize : tuple, optional
            Figure size. The default is (15, 15)
        key : str, optional
            Key to index this image and transform in image_dict and
            image_transform_dict. The default is '' (for backward compatability)

        Returns
        -------
        f, ax : matplotlib figure and axes objects

        """
        
        # Use point mask to create array with NaN's where we have no data
        nan_image = copy.deepcopy(self.dsa_image_dict[key]
                                  .PointData['Elevation'])
        nan_image[self.dsa_image_dict[key]
         .PointData['vtkValidPointMask']==0] = np.NaN
        dims = self.image_dict[key].GetDimensions()
        nan_image = nan_image.reshape((dims[1], dims[0]))
        
        # Plot
        f, ax = plt.subplots(1, 1, figsize=figsize)
        cf = ax.imshow(nan_image, cmap=cmap, aspect='equal', origin='lower',
                       vmin=z_min, vmax=z_max)
        f.colorbar(cf, ax=ax)
        ax.set_title(self.project_name)
        
        return f, ax
    
    def get_np_nan_image(self, key=''):
        """
        Convenience function for copying the image to a numpy object.

        Parameters
        ----------
        key : str, optional
            Key to index this image and transform in image_dict and
            image_transform_dict. The default is '' (for backward compatability)

        Returns
        -------
        nan_image : numpy ndarray

        """
        
        # Use point mask to create array with NaN's where we have no data
        nan_image = copy.deepcopy(self.dsa_image_dict[key]
                                  .PointData['Elevation'])
        if 'vtkValidPointMask' in self.dsa_image_dict[key].PointData:
            nan_image[self.dsa_image_dict[key]
             .PointData['vtkValidPointMask']==0] = np.NaN
        dims = self.image_dict[key].GetDimensions()
        nan_image = nan_image.reshape((dims[1], dims[0]))
        
        return nan_image
    
    def merged_points_to_mesh(self, depth=13, min_density=9, x0=None, y0=None,
                              wx=None, wy=None, yaw=0):
        """
        Create mesh from all points in singlescans.
        
        Using Poisson surface reconstruction from Kazhdan 2006 implemented in
        Open3d. This function assumes that our points have normals!

        Parameters
        ----------
        depth : int, optional
            The depth into the octree for the surface reconstruction to go.
            The default is 13.
        min_density : float, optional
            We will filter out triangles that are based on fewer than this
            number of points. The default is 9.
        x0 : float, optional
            x coordinate of the selection box. The default is None
        y0 : float, optional
            y coordinate of the selection box. The default is None
        wx : float, optional
            the selection box width. The default is None
        wy : float, optional
            the selection box height. The default is None
        yaw : float, optional
            yaw of the selection box. The default is 0.

        Returns
        -------
        None.

        """
        
        # First, get merged points
        pdata, history_dict = self.get_merged_points(history_dict=True, x0=x0,
                                                     y0=y0, wx=wx, wy=wy, 
                                                     yaw=yaw)
        
        # Convert to Open3d
        filt_pts_np = vtk_to_numpy(pdata.GetPoints().GetData())
        normals_np = vtk_to_numpy(pdata.GetPointData().GetNormals())
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filt_pts_np)
        pcd.normals = o3d.utility.Vector3dVector(normals_np)
        
        # Get mesh and densities via the Poisson surface reconstruction
        mesh, densities = (o3d.geometry.TriangleMesh
                           .create_from_point_cloud_poisson(pcd, depth=depth, 
                                                            scale=1))
        # Filter out triangles with less than our desired density
        densities = np.asarray(densities)
        vertices_to_remove = densities < min_density
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Store np arrays related to mesh
        if not hasattr(self, 'np_dict'):
            self.np_dict = {}
        self.np_dict['mesh_pts'] = np.array(mesh.vertices, dtype=np.float32)
        self.np_dict['mesh_connectivity'] = np.array(mesh.triangles).ravel()
        self.np_dict['mesh_offsets'] = np.arange(
            self.np_dict['mesh_connectivity'].shape[0]//3 + 1, dtype=
            self.np_dict['mesh_connectivity'].dtype) * 3
        if not self.np_dict['mesh_connectivity'].dtype == np.int32:
            raise RuntimeError('mesh dtype is not int32, update code?')
        
        # Store result in mesh
        self.mesh = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        vtk_arr = numpy_to_vtk(self.np_dict['mesh_pts'], 
                               array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName('mesh_pts')
        pts.SetData(vtk_arr)
        vtk_arr_0 = numpy_to_vtk(self.np_dict['mesh_offsets'], 
                                 array_type=vtk.VTK_TYPE_INT32)
        vtk_arr_0.SetName('mesh_offsets')
        vtk_arr_1 = numpy_to_vtk(self.np_dict['mesh_connectivity'], 
                                 array_type=vtk.VTK_TYPE_INT32)
        vtk_arr_1.SetName('mesh_connectivity')
        polys = vtk.vtkCellArray()
        polys.SetData(vtk_arr_0, vtk_arr_1)
        self.mesh.SetPoints(pts)
        self.mesh.SetPolys(polys)
        
        # Create history dict
        self.mesh_history_dict = {
            "type": "Mesh Generator",
            "git_hash": get_git_hash(),
            "method": "Project.merged_points_to_mesh",
            "input_0": history_dict,
            "params": {"depth": depth, "min_density": min_density}
            }
    
    def transect_points(self, x0, y0, x1, y1, d):
        """
        Get the points within a distance d of transect defined by points.

        Parameters
        ----------
        x0 : float
            Coordinate in project reference frame.
        y0 : float
            Coordinate in project reference frame.
        x1 : float
            Coordinate in project reference frame.
        y1 : float
            Coordinate in project reference frame.
        d : float
            Distance from transect to select points.

        Returns
        -------
        vtkPolyData
            with the selected points in the project reference frame

        """
        
        length = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        # we need a transformation first brings point 0 to the origin, then yaws
        # transect axis onto x-axis
        t_trans = vtk.vtkTransform()
        # set mode to post multiply, so we will first translate and then rotate
        t_trans.PostMultiply()
        t_trans.Translate(-1*x0, -1*y0, 0)
        # Get yaw angle of line
        yaw = np.arctan2(y1 - y0, x1 - x0) * 180/np.pi
        t_trans.RotateZ(-1*yaw)
        
        # Create box
        box = vtk.vtkBox()
        box.SetBounds(-d, length + d, -d, d, -1000, 1000)
        box.SetTransform(t_trans)
        
        # Extract points
        extractPoints = vtk.vtkExtractPoints()
        extractPoints.SetImplicitFunction(box)
        extractPoints.SetInputData(self.get_merged_points())
        extractPoints.Update()
        return extractPoints.GetOutput()
    
    def transect_n_points(self, x0, y0, x1, y1, n_pts, tol=1000, d0=0.5, 
                          dmax=50):
        """
        Get approximately n points around the transect.

        Parameters
        ----------
        x0 : float
            Coordinate in project reference frame.
        y0 : float
            Coordinate in project reference frame.
        x1 : float
            Coordinate in project reference frame.
        y1 : float
            Coordinate in project reference frame.
        n_pts : int
            Minimum number of points to acquire around transect.
        tol : int, optional
            Tolerance for number of points around transect to get. the default
            is 1000.
        d0 : float, optional
            Starting distance. The default is 0.5.
        dmax : float, optional
            Max distance to look for points. the default is 50 m.

        Returns
        -------
        vtkPolyData, d
            with the selected points in the project reference frame. And
            d, the distance from the transect that we selected points.

        """
        
        length = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        # we need a transformation first brings point 0 to the origin, then yaws
        # transect axis onto x-axis
        t_trans = vtk.vtkTransform()
        # set mode to post multiply, so we will first translate and then rotate
        t_trans.PostMultiply()
        t_trans.Translate(-1*x0, -1*y0, 0)
        # Get yaw angle of line
        yaw = np.arctan2(y1 - y0, x1 - x0) * 180/np.pi
        t_trans.RotateZ(-1*yaw)
        
        d = d0
        dmin = 0
        dloopmax = None
        
        # Create box
        box = vtk.vtkBox()
        box.SetBounds(-d, length + d, -d, d, -1000, 1000)
        box.SetTransform(t_trans)
        
        # Extract points
        extractPoints = vtk.vtkExtractPoints()
        extractPoints.SetImplicitFunction(box)
        extractPoints.SetInputData(self.get_merged_points())
        extractPoints.Update()
        
        # While loop for adjusting the size of d
        while np.abs(extractPoints.GetOutput().GetNumberOfPoints()-n_pts)>tol:
            #print(d)
            #print(extractPoints.GetOutput().GetNumberOfPoints())
            if extractPoints.GetOutput().GetNumberOfPoints()<n_pts:
                dmin = d # store current value of d in dmin, we won't search
                # below this distance.
                # if we are still ascending double d
                if dloopmax is None:
                    if (2*d)>dmax:
                        warnings.warn('we have exceed max search distance. '
                                      'increase dmax if this is not a mistake.')
                        break
                    d = 2*d
                else:
                    #otherwise go halfway between where we are and the highest
                    d = (d+dloopmax)/2
            else:
                # If we have too many points go halfway between d and dmin
                dloopmax=d
                d = (d+dmin)/2
            
            # Update box with the new d
            box.SetBounds(-d, length + d, -d, d, -1000, 1000)
            box.Modified()
            extractPoints.Update()
        
        # Return once we are within tolerance
        return extractPoints.GetOutput(), d
    
    def image_transect(self, x0, y0, x1, y1, N, key, image_key=''):
        """
        Sample a transect through the current image and save in profiles.
        
        Parameters:
        -----------
        x0 : float
            Coordinate in project reference frame.
        y0 : float
            Coordinate in project reference frame.
        x1 : float
            Coordinate in project reference frame.
        y1 : float
            Coordinate in project reference frame.
        N : int
            Number of points to put in transect.
        key : const (str, tuple, etc)
            Key to store this profile in profile_dict under.
        image_key : str, optional
            Key for the image in image_dict.
            The default is '' (for backward compatability)

        Returns:
        --------
        None.

        """

        # Get start and end points in image reference frame
        start = self.image_transform_dict[image_key].TransformPoint(x0, y0, 0)
        end = self.image_transform_dict[image_key].TransformPoint(x1, y1, 0)
        
        # Create transect in image coordinates
        pts_np = np.vstack((np.linspace(start[0], end[0], N), 
                            np.linspace(start[1], end[1], N),
                            np.zeros(N))).T
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(pts_np, deep=True, array_type=vtk.VTK_DOUBLE))
        pts_pdata = vtk.vtkPolyData()
        pts_pdata.SetPoints(pts)
        
        # Probe filter
        probeFilter = vtk.vtkProbeFilter()
        probeFilter.SetSourceData(self.image_dict[image_key])
        probeFilter.SetInputData(pts_pdata)
        probeFilter.Update()

        # Transform back into project coordinates
        invTransform = vtk.vtkTransform()
        invTransform.DeepCopy(self.image_transform_dict[image_key])
        invTransform.Inverse()
        tfilter = vtk.vtkTransformPolyDataFilter()
        tfilter.SetTransform(invTransform)
        tfilter.SetInputData(probeFilter.GetOutput())
        tfilter.Update()

        # Store profile and create lines
        self.profile_dict[key] = tfilter.GetOutput()
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(N)
        for i in np.arange(N):
            lines.InsertCellPoint(i)
        self.profile_dict[key].SetLines(lines)
        self.profile_dict[key].Modified()


    def merged_points_transect_gp(self, x0, y0, x1, y1, N, key, 
                                  mx=256, n_neighbors=256, eps=0,
                                  use_z_sigma=True, lengthscale=None, 
                                  outputscale=None, mean=None, nu=0.5,
                                  optimize=False, learning_rate=0.1, 
                                  n_iter=None, max_time=60):
        """
        Use gpytorch to infer a surface transect.

        Parameters
        ----------
        x0 : float
            Coordinate in project reference frame.
        y0 : float
            Coordinate in project reference frame.
        x1 : float
            Coordinate in project reference frame.
        y1 : float
            Coordinate in project reference frame.
        N : int
            Number of points to put in transect.
        key : const (str, tuple, etc)
            Key to store this profile in profile_dict under.
        mx : int, optional
            Number of transect points to estimate in each chunk. The default
            is 256.
        n_neighbors : int, optional
            Number of neighbors around each transect point to use. The default
            is 256
        eps : float, optional
            Eps for nearest neighbors search (see scipy.spatial.cKDTree) for
            more details. The default is 0 (exact nearest neighbors)
        use_z_sigma : bool, optional
            Whether to use the computed pointwise uncertainties or not.
            Generally you should have a reason for not using them. If False we
            will just just a GaussianLikelihood for our likelihood. The default
            is True.
        lengthscale : float, optional
            Lengthscale for kernel in m. See run_gp. The default is None.
        outputscale : float, optional
            Outputscale for kernel. See run_gp. The default is None.
        mean : float, optional
            Initial constant mean value for the GP. If None it will be set to 
            the mean of norm_height. The default is None.
        nu : float, optional
            nu value of Matern kernel. It must be one of [0.5, 1.5, 2.5]. The
            default is 0.5 (exponential kernel)
        optimize : bool, optional
            Whether or not to search for optimal hyperparameters for the GP. The
            default is False.
        learning_rate : float, optional
            The learning rate if we are optimizing (currently just using ADAM). 
            The default is 0.1.
        n_iter : int, optional
            The number of iterations to run optimizer for. Note that we will 
            only ever run optimizer for max_time. The default is None.
        max_time : float, optional
            The maximum amount of time (in seconds) to run an optimization for. 
            The default is 60.


        Returns
        -------
        None.

        """
        
        # Get the points in this rectangular region
        pdata, point_history_dict = self.get_merged_points(history_dict=True)

        # Get the points and the z_sigma as numpy arrays
        pts_np = vtk_to_numpy(pdata.GetPoints().GetData())
        if use_z_sigma:
            z_sigma_np = vtk_to_numpy(pdata.GetPointData()
                                    .GetArray('z_sigma'))
        else:
            z_sigma_np = None
        
        # Create transect points
        grid_points=np.hstack((np.linspace(x0, x1, N, dtype=np.float32)
                               [:, np.newaxis],
                               np.linspace(y0, y1, N, dtype=np.float32)
                               [:, np.newaxis]))
        
        # Indices for stepping through grid_points in mx x my chunks
        i_mx = 0
        ctr = 0

        # Build a cKDTree from the xy coordinates of pts_np
        kdtree = cKDTree(pts_np[:,:2])

        # Output
        grid_mean = np.empty(N, dtype=np.float32) * np.nan
        grid_lower = np.empty(N, dtype=np.float32) * np.nan
        grid_upper = np.empty(N, dtype=np.float32) * np.nan
        while ctr<N:
            # Get start and end indices for x and y values
            x_s = mx*i_mx
            x_e = N if x_s+mx>N else x_s+mx
            step = (x_e-x_s)
            
            # Get the grid points in this chunk and use kdtree to pull point 
            # indices
            ind = np.arange(x_s, x_e)
            m_grid = grid_points[ind, :]
            _, pt_ind = kdtree.query(m_grid, n_neighbors, eps=eps, workers=-1)
            del _
            pt_ind = np.unique(pt_ind.ravel())
            
            # Run our GP on this chunk and estimate values at grid points
            # use indices from above to place output in the right places
            # !!!TODO, use last 3 returns from run_gp
            grid_mean[ind], grid_lower[ind], grid_upper[ind],_,_,_ = run_gp(
                pts_np[pt_ind,:], pts_np[pt_ind,2], m_grid,
                z_sigma=None if z_sigma_np is None else z_sigma_np[pt_ind], 
                lengthscale=lengthscale, outputscale=outputscale, mean=mean,
                nu=nu, optimize=optimize, learning_rate=learning_rate,
                iter=n_iter, max_time=max_time)
            
            # Move i_mx 
            i_mx += 1
            # Increment counter
            ctr += step
        
        # create profile pdata
        self.profile_dict[key] = vtk.vtkPolyData()
        # Create points object and add point data arrays
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(np.hstack((grid_points, 
                                            grid_mean[:,np.newaxis])),
                                 deep=True, array_type=vtk.VTK_FLOAT))
        self.profile_dict[key].SetPoints(pts)
        vtk_arr = numpy_to_vtk(grid_lower, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName('z_lower')
        self.profile_dict[key].GetPointData().AddArray(vtk_arr)
        vtk_arr = numpy_to_vtk(grid_upper, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName('z_upper')
        self.profile_dict[key].GetPointData().AddArray(vtk_arr)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(grid_mean.size)
        for i in np.arange(grid_mean.size):
            lines.InsertCellPoint(i)
        self.profile_dict[key].SetLines(lines)
        self.profile_dict[key].Modified()
        
    def get_profile(self, key):
        """
        Returns the requested profile as a numpy array

        Parameters:
        -----------
        key : const
            The key for the profile in profile_dict.
        
        Returns:
        --------
        ndarray
            array with x coord, y coord, length along transect, mean z lower ci,
            upper ci
        """

        prof = self.profile_dict[key]
        pts = vtk_to_numpy(prof.GetPoints().GetData())

        # Return array with x coord, y coord, length along transect, mean z
        # lower ci, upper ci
        return np.hstack((pts[:,:2], 
                          np.sqrt(np.square(pts[:,:2] - pts[0,:2]).sum(axis=1))
                          [:, np.newaxis],
                          pts[:,2][:, np.newaxis],
                          vtk_to_numpy(prof.GetPointData().GetArray('z_lower'))
                          [:, np.newaxis],
                          vtk_to_numpy(prof.GetPointData().GetArray('z_upper'))
                          [:, np.newaxis]))
    
    def mesh_transect(self, x0, y0, x1, y1, N):
        """
        Cut a transect through the mesh, return the transect

        Parameters
        ----------
        x0 : float
            Coordinate in project reference frame.
        y0 : float
            Coordinate in project reference frame.
        x1 : float
            Coordinate in project reference frame.
        y1 : float
            Coordinate in project reference frame.
        N : int
            Number of points to put in transect.

        Returns
        -------
        ndarray
            N x 4 array with x coord, y coord, length along transect and z

        """
        
        # Elevation filter
        elevFilter = vtk.vtkSimpleElevationFilter()
        elevFilter.SetInputData(self.mesh)
        elevFilter.Update()
        
        # Flatten
        trans = vtk.vtkTransform()
        trans.Scale(1.0, 1.0, 0)
        flattener = vtk.vtkTransformFilter()
        flattener.SetInputConnection(elevFilter.GetOutputPort())
        flattener.SetTransform(trans)
        flattener.Update()
        
        # Line we want to filter on.
        pts_np = np.vstack((np.linspace(x0, x1, N), np.linspace(y0, y1, N),
                            np.zeros(N))).T
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(pts_np, deep=True, array_type=vtk.VTK_DOUBLE))
        pts_pdata = vtk.vtkPolyData()
        pts_pdata.SetPoints(pts)
        
        # Probe filter
        probeFilter = vtk.vtkProbeFilter()
        probeFilter.SetSourceData(flattener.GetOutput())
        probeFilter.SetInputData(pts_pdata)
        probeFilter.Update()
        
        pdata = probeFilter.GetOutput()
        z = vtk_to_numpy(pdata.GetPointData().GetArray('Elevation'))

        output = np.zeros((N, 4))
        output[:,:2] = pts_np[:,:2]
        output[:,2] = np.sqrt((output[:,0]-output[0,0])**2 + 
                              (output[:,1]-output[0,1])**2)
        output[:,3] = z
        return output
    
    def get_merged_points(self, port=False, history_dict=False, x0=None,
                          y0=None, wx=None, wy=None, r=None, yaw=0):
        """
        Returns a polydata with merged points from all single scans
        
        Parameters
        ----------
        port : bool, optional
            Whether to return an output connection instead of a polydata.
            The default is False
        history_dict : bool, optional
            Whether to return the history dict. Note that if port is False,
            we return a deep copy of the history dict that is not linked to
            the SingleScans whereas if port is true we return a linked 
            version. The default is False.
        x0 : float, optional
            x coordinate of the selection box. The default is None
        y0 : float, optional
            y coordinate of the selection box. The default is None
        r : float, optional
            radius of the vertically-aligned cylinder to get points in
            The default is None. Cannot be set if wx/wy is set.
        wx : float, optional
            the selection box width. The default is None
        wy : float, optional
            the selection box height. The default is None
        yaw : float, optional
            yaw of the selection box. The default is 0.

        Returns
        -------
        vtkPolyData.

        """
        git_hash = get_git_hash()
        
        # Create Appending filter and add all data to it
        # Delete old append history dict if it exists
        if hasattr(self, 'append_hist_dict'):
            del self.append_hist_dict
        appendPolyData = vtk.vtkAppendPolyData()
        for key in self.scan_dict:
            self.scan_dict[key].transformFilter.Update()
            connection, temp_hist_dict = self.scan_dict[key].get_polydata(
                port=True, history_dict=True)
            if (wx is not None) or (r is not None):
                if (wx is not None) and (r is not None):
                    raise ValueError("get merged points cannot have x0 and r")
                if wx is not None:
                    # If we are just selecting a box, do so
                    imp = vtk.vtkBox()
                    imp.SetBounds((0, wx, 0, wy, -10, 10))
                    # We need a transform to put the data in the desired location 
                    # relative to our box
                    transform = vtk.vtkTransform()
                    transform.PostMultiply()
                    transform.RotateZ(yaw)
                    transform.Translate(x0, y0, 0)
                    # That transform moves the box relative to the data, so the 
                    # box takes its inverse
                    transform.Inverse()
                    imp.SetTransform(transform)
                elif r is not None:
                    imp = vtk.vtkCylinder()
                    imp.SetCenter(x0, y0, 0)
                    imp.SetRadius(r)
                    imp.SetAxis(0, 0, 1)
                else:
                    raise RuntimeError()
                    
                # vtkExtractPoints does the actual filtering
                extractPoints = vtk.vtkExtractPoints()
                extractPoints.SetImplicitFunction(imp)
                extractPoints.SetInputConnection(connection)
                extractPoints.GetInput().GetPointData().CopyNormalsOn()
                extractPoints.GenerateVerticesOn()
                extractPoints.Update()
                if extractPoints.GetOutput().GetNumberOfPoints()>0:
                    appendPolyData.AddInputConnection(extractPoints
                                                      .GetOutputPort())
            else:
                appendPolyData.AddInputConnection(connection)
            if not hasattr(self, 'append_hist_dict'):
                self.append_hist_dict = {
                    "type": "Pointset Aggregator",
                    "git_hash": git_hash,
                    "method": "Project.get_merged_points",
                    "input_0": temp_hist_dict}
            elif not "input_1" in self.append_hist_dict.keys():
                self.append_hist_dict["input_1"] = temp_hist_dict
            else:
                self.append_hist_dict = {
                    "type": "Pointset Aggregator",
                    "git_hash": git_hash,
                    "method": "Project.get_merged_points",
                    "input_0": temp_hist_dict,
                    "input_1": self.append_hist_dict
                    }
        
        appendPolyData.Update()
        
        if port:
            # If we want to create a connection we need to persist the
            # append polydata object, otherwise it segfaults
            self.appendPolyData = appendPolyData
            if history_dict:
                warnings.warn('History dict may not be fully linked, ' + 
                              'use with caution')
                return (self.appendPolyData.GetOutputPort(), 
                        self.append_hist_dict)
            else:
                return self.appendPolyData.GetOutputPort() 
        else:
            temp_hist_dict = json.loads(json.dumps(self.append_hist_dict))
            del self.append_hist_dict
            if history_dict:
                return (appendPolyData.GetOutput(), temp_hist_dict)
            else:
                return appendPolyData.GetOutput()

    def write_merged_points(self, output_name=None):
        """
        Write the transformed, merged points to a vtp file.
        
        Uses the output of each scans get_polydata so should run
        apply_transforms beforehand.

        Parameters
        ----------
        output_name : str, optional
            Output name for the file, if None use the project_name + 
            '_merged.xyz'. The default is None.

        Returns
        -------
        None.

        """
        
        # Create writer and write data
        writer = vtk.vtkSimplePointsWriter()
        writer.SetInputData(self.get_merged_points())
        if output_name:
            writer.SetFileName(self.project_path + output_name)
        else:
            writer.SetFileName(self.project_path + self.project_name + 
                               '_merged.xyz')
        #writer.SetFileTypeToASCII()
        writer.Write()
    
    def write_las_pdal(self, output_dir=None, filename=None, 
                       mode='transformed', skip_fields=[]):
        """
        Write the data in the project to LAS using pdal

        Parameters
        ----------
        output_dir : str, optional
            Directory to write to. If none defaults to project_path +
            project_name + '\\lasfiles\\pdal_output\\'. The default is None
        filename : str, optional
            Filename, if none uses project name. The default is None.
        mode : str, optional
            Whether to write 'raw' points, 'transformed' points, or 'filtered'
            points. The default is 'transformed'.
        skip_fields : list, optional
            Fields to skip in writing. If this is 'all' then only write x,
            y, z. Otherwise should be a list of field names. The default is []

        Returns
        -------
        None.

        """
        
        # Handle output dir
        if output_dir is None:
            if not os.path.isdir(os.path.join(self.project_path, self.project_name, 
                                 'lasfiles', 'pdal_output')):
                os.mkdir(os.path.join(self.project_path, self.project_name, 
                          'lasfiles', 'pdal_output'))
            output_dir = os.path.join(self.project_path, self.project_name, 
                          'lasfiles', 'pdal_output')
        if filename is None:
            filename = self.project_name
        
        # Write each scan individually to a numpy output
        json_list = []
        for scan_name in self.scan_dict:
            npy_filepath = self.scan_dict[scan_name].write_npy_pdal(
                output_dir, mode=mode, skip_fields=skip_fields)
            if npy_filepath is not None:
                json_list.append({"filename": npy_filepath,
                                  "type": "readers.numpy"})
        
        # Create JSON to instruct merging and conversion
        json_list.append({"type": "filters.merge"})
        json_list.append({"type": "writers.las",
                          "filename": output_dir + filename + '.las',
                          "minor_version": 4,
                          "dataformat_id": 0})
        json_data = json.dumps(json_list, indent=4)
        pipeline = pdal.Pipeline(json_data)
        _ = pipeline.execute()
        
    
    def write_mesh(self, output_path=None, suffix='', name='mesh'):
        """
        Write the mesh out to a file

        Parameters
        ----------
        output_path : str, optional
            Output name for the file, if None use the mesh directory. 
            The default is None.
        suffix : str, optional
            The suffix for the vtkfiles dir. The default is ''.
        name : str, optional
            Name of the file if we're writing to vtkfiles dir. The default 
            is 'mesh'.

        Returns
        -------
        None.

        """
        
        # Create writer and write mesh
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(self.mesh)
        if not output_path:
            # Create the mesh folder if it doesn't already exist
            if not os.path.isdir(os.path.join(self.project_path, 
                                              self.project_name, 
                                              "vtkfiles" + suffix)):
                os.mkdir(os.path.join(self.project_path, self.project_name, 
                                      "vtkfiles" + suffix))
            if not os.path.isdir(os.path.join(self.project_path,   
                                              self.project_name, 
                                              "vtkfiles" + suffix, "meshes")):
                os.mkdir(os.path.join(self.project_path, self.project_name, 
                         "vtkfiles" + suffix, "meshes"))
            output_path = os.path.join(self.project_path, self.project_name, 
                                       "vtkfiles" + suffix, "meshes", 
                                       name + ".vtp")
        
        # if the files already exist, remove them
        for f in os.listdir(os.path.dirname(output_path)):
            if re.match(name, f):
                os.remove(os.path.join(os.path.dirname(output_path), f))
        
        writer.SetFileName(output_path)
        writer.Write()
        
        # Write the history_dict
        f = open(output_path.rsplit('.', maxsplit=1)[0] + '.txt', 'w')
        json.dump(self.mesh_history_dict, f, indent=4)
        f.close()
        
        
        
    def read_mesh(self, mesh_path=None, suffix='', name='mesh'):
        """
        Read in the mesh from a file.

        Parameters
        ----------
        mesh_path : str, optional
            Path to the mesh, if none use the mesh directory. 
            The default is None.
        suffix : str, optional
            The suffix for the vtkfiles dir. The default is ''.
        name : str, optional
            Name of the file if we're reading in vtkfiles dir. The default 
            is 'mesh'.

        Returns
        -------
        None.

        """
        
        # Create reader and read mesh
        reader = vtk.vtkXMLPolyDataReader()
        if not mesh_path:
            mesh_path = os.path.join(self.project_path, self.project_name, 
                                     "vtkfiles" + suffix, "meshes", 
                                     name + ".vtp")
        reader.SetFileName(mesh_path)
        reader.Update()
        self.mesh = reader.GetOutput()
        
        # Read in history dict
        f = open(mesh_path.rsplit('.', maxsplit=1)[0] + '.txt')
        self.raw_history_dict = json.load(f)
        f.close()
    
    def write_image(self, output_path=None, suffix='', name=None, key=''):
        """
        Write the image out to a file.
        
        Can specify a suffix and a name for the image.

        Parameters
        ----------
        output_path : str, optional
            Output name for the file, if None use the mesh directory. 
            The default is None.
        suffix : str, optional
            The suffix for the vtkfiles dir. The default is ''.
        name : str, optional
            Name of the file if we're writing to vtkfiles dir. The default 
            is None, which uses the key unless the key is '' then it uses 
            'image'.
        key : str, optional
            Key to index this image and transform in image_dict and
            image_transform_dict. The default is '' (for backward compatability)

        Returns
        -------
        None.

        """
        
        if name is None:
            if key=='':
                name = 'image'
            else:
                name = key

        # Create writer and write mesh
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetInputData(self.image_dict[key])
        if not output_path:
            # Create the mesh folder if it doesn't already exist
            if not os.path.isdir(os.path.join(self.project_path, 
                                              self.project_name, 
                                              "vtkfiles" + suffix)):
                os.mkdir(os.path.join(self.project_path, self.project_name, 
                                      "vtkfiles" + suffix))
            if not os.path.isdir(os.path.join(self.project_path,   
                                              self.project_name, 
                                              "vtkfiles" + suffix, "images")):
                os.mkdir(os.path.join(self.project_path, self.project_name, 
                         "vtkfiles" + suffix, "images"))
            output_path = os.path.join(self.project_path, self.project_name, 
                                       "vtkfiles" + suffix, "images", 
                                       name + ".vti")
        
        # if the files already exist, remove them
        for f in os.listdir(os.path.dirname(output_path)):
            if re.match(name, f):
                os.remove(os.path.join(os.path.dirname(output_path), f))
        
        writer.SetFileName(output_path)
        writer.Write()

        # Write the imageTransform as a numpy file
        transform_np = np.zeros((4, 4), dtype=np.float64)
        vtk4x4 = self.image_transform_dict[key].GetMatrix()
        for i in range(4):
            for j in range(4):
                transform_np[i, j] = vtk4x4.GetElement(i, j)
        np.save(output_path.rsplit('.', maxsplit=1)[0] + '.npy', transform_np)

        # Write the history_dict
        f = open(output_path.rsplit('.', maxsplit=1)[0] + '.txt', 'w')
        json.dump(self.image_history_dict_dict[key], f, indent=4)
        f.close()
    
    def read_image(self, image_path=None, suffix='', name=None, 
                   overwrite=False):
        """
        Read in the image from a file.

        Parameters
        ----------
        image_path : str, optional
            Path to the image, if none use the image directory. 
            The default is None.
        suffix : str, optional
            The suffix for the vtkfiles dir. The default is ''.
        name : str, optional
            Name of the file if we're reading in vtkfiles dir. If None, name
            is set to 'image' and key is seimage_path=None, suffix='', name=None, overwrite=Falset to ''. Otherwise name and key
            are the the same. The default is None.
        overwrite : bool, optional
            Whether to overwrite preexisting image. If False and key already
            exists in image_dict will raise a warning. The default is False.

        Returns
        -------
        None.

        """
        
        if name is None:
            name = 'image'
            key = ''
        else:
            key = name

        # Check if key already exists
        if key in self.image_dict.keys():
            if overwrite:
                warnings.warn('You are overwriting image: ' + key)
            else:
                warnings.warn(key + 'already exists in image_dict ' +
                                     'set overwrite=True if you want to' + 
                                     'overwrite')
                return

        # Create reader and read image
        reader = vtk.vtkXMLImageDataReader()
        if not image_path:
            image_path = os.path.join(self.project_path, self.project_name, 
                                     "vtkfiles" + suffix, "images", 
                                     name + ".vti")
        reader.SetFileName(image_path)
        reader.Update()
        
        # Read in history dict
        try:
            f = open(image_path.rsplit('.', maxsplit=1)[0] + '.txt')
            self.image_history_dict_dict[key] = json.load(f)
            f.close()
        except:
            warnings.warn('History Dict not read properly!')

        # Read in the imageTransform
        transform_np = np.load(image_path.rsplit('.', maxsplit=1)[0] + '.npy')
        vtk4x4 = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk4x4.SetElement(i, j, transform_np[i, j])
        self.image_transform_dict[key] = vtk.vtkTransform()
        self.image_transform_dict[key].SetMatrix(vtk4x4)

        
        self.image_dict[key] = reader.GetOutput()
        
        # Also wrap with a datasetadaptor for working with numpy
        self.dsa_image_dict[key] = dsa.WrapDataObject(self.image_dict[key])
    
    def create_reflectance(self):
        """
        Create reflectance fielf for each scan.

        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].create_reflectance()
    
    def correct_reflectance_radial(self, mode, r_min=None, r_max=None, 
                                   num=None, base=None):
        """
        Corrects radial artifact in reflectance. result: 'reflectance_radial'
        
        Attempts to correct radial artifact in reflectance. Still developing
        the best way to do this.
        
        If mode is 'median': bin the raw reflectances by radial distance.

        Parameters
        ----------
        mode : str
            Method for correcting radial artifact in reflectances. Currently
            only coded for 'median'.
        r_min : float, optional
            Needed for method 'median' minimum radius to bin
        r_max : float, optional
            Needed for method 'median' maximum radius to bin
        num : int, optional
            Needed for method 'median', number of bins
        base : float, optional
            Needed for method 'median', base for logspaced bins

        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].correct_reflectance_radial(mode,
                                                                 r_min=r_min, 
                                                                 r_max=r_max, 
                                                                 num=num, 
                                                                 base=base)
    
    def areapoints_to_cornercoords(self, areapoints):
        """
        Takes a set of areapoints given as scan_name, PointId pairs. Returns
        the coordinates of these points in the current reference frame

        Parameters
        ----------
        areapoints : List of lists or tuples
            List of area points in the desired order. Each point is a tuple
            in which the zeroth element is the scan_name and the first
            element is the PointId

        Returns
        -------
        ndarray.
            Nx3 array where N is the number of points.

        """
        
        cornercoords = np.empty((len(areapoints[self.project_name]), 3)
                                , dtype=np.float32)
        # Step through areapoints and get each point
        for i in np.arange(len(areapoints[self.project_name])):
            pdata = self.scan_dict[areapoints[self.project_name][i][0]
                                   ].transformFilter.GetOutput()
            PointId = vtk_to_numpy(pdata.GetPointData().GetArray('PointId'))
            pts = vtk_to_numpy(pdata.GetPoints().GetData())
            cornercoords[i,:] = pts[PointId==areapoints[
                self.project_name][i][1],:]
        return cornercoords

    def get_local_max(self, z_threshold, rmax, return_dist=False, 
                      return_zs=False, closest_only=False):
        """
        Get set of locally maxima points above z_threshold

        Parameters
        ----------
        z_threshold : float
            Minimum z value (in current transformation) for all points to 
            consider.
        rmax : float
            Horizontal distance (in m) on which points must be locally maximal
        return_dist : bool, optional
            Whether to return an array with the distance of each point from 
            scanner. The default is False.
        return_zs : bool, optional
            Whether to return the z-sigma of each point. The default is False.
        closest_only : bool, optional
            Whether to only return local max that are closer to their scanner
            than any other scanner. The default is False.

        Returns
        -------
        [local_max, dist, zs]
            ndarrays, Nx3 of locally maximal points in current tranformation.
            optionally distance and z sigma as well.
            
        """

        if closest_only:
            # If we are using only those local max that are closer to their
            # scanner than all others, we need to create distances between 
            # points and scanners
            scanner_pos_dict = {}
            for scan_name in self.scan_dict:
                scanner_pos_dict[scan_name] = np.array(self.scan_dict[scan_name]
                                                       .transform.GetPosition()
                                                       [:2])[np.newaxis,:]
            # For each SingleScan, get local maxima
            maxs_list = []
            for scan_name in self.scan_dict:
                maxs = self.scan_dict[scan_name].get_local_max(z_threshold, 
                        rmax, return_dist=return_dist, return_zs=return_zs)
                # For each point in local max, get the distance to its scanner
                dist_array = np.square(maxs[0][:,:2] 
                                       - scanner_pos_dict[scan_name]
                                       ).sum(axis=1)[:,np.newaxis]
                # Put everything side by side into one array
                if len(maxs)==1:
                    arr = np.hstack((maxs[0], dist_array))
                elif len(maxs)==2:
                    arr = np.hstack((maxs[0], dist_array, 
                                     maxs[1][:,np.newaxis]))
                elif len(maxs)==3:
                    arr = np.hstack((maxs[0], dist_array, 
                                     maxs[1][:,np.newaxis],
                                     maxs[2][:,np.newaxis]))
                maxs_list.append(arr)
            # Combine maxs list into one array
            arr = np.vstack(maxs_list)
            # Now we'll build a kd tree and look for pairs of points 
            # within rmax. Then eliminate points further from scanner
            tree = KDTree(arr[:,:2])
            pair_ind = tree.query_pairs(rmax, output_type='ndarray')
            pair_dist = arr[:,3][pair_ind]
            further_mask = np.empty(pair_ind.shape, dtype=np.bool_)
            further_mask[:,1] = np.argmax(pair_dist, axis=1)
            further_mask[:,0] = np.logical_not(further_mask[:,1])
            further = np.unique(pair_ind[further_mask])
            keep_mask = np.ones(arr.shape[0], dtype=np.bool_)
            keep_mask[further] = False
            arr = arr[keep_mask,:]

            # Get return list
            if arr.shape[1]==4:
                ret_list = [arr[:,:3]]
            elif arr.shape[1]==5:
                ret_list = [arr[:,:3], arr[:,4]]
            elif arr.shape[1]==6:
                ret_list = [arr[:,:3], arr[:,4], arr[:,5]]
        else:
            raise NotImplementedError('Only "closest_only" implemented.')

        return ret_list

    def create_local_max(self, z_threshold, rmax, closest_only=True, 
                         key='local_max'):
        """
        Create pointset of local max points in pdata_dict.

        See get_local_max() for details of local max identification.

        Parameters
        ----------
        z_threshold : float
            Minimum z value (in current transformation) for all points to 
            consider.
        rmax : float
            Horizontal distance (in m) on which points must be locally maximal
        closest_only : bool, optional
            Whether to only return local max that are closer to their scanner
            than any other scanner. The default is True.
        key : str, optional
            Key for this pointset in pdata_dict. The default is 'local_max'

        Returns
        -------
        None.

        """

        pts_np, dist_np, z_sigma_np = self.get_local_max(z_threshold, rmax,
                                                      return_dist=True,
                                                      return_zs=True,
                                                      closest_only=closest_only)
        # Create polydata
        self.pdata_dict[key] = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(pts_np, array_type=vtk.VTK_DOUBLE,
                                 deep=True))
        self.pdata_dict[key].SetPoints(pts)
        dist = numpy_to_vtk(dist_np, array_type=vtk.VTK_DOUBLE, deep=True)
        dist.SetName('dist')
        self.pdata_dict[key].GetPointData().AddArray(dist)
        z_sigma = numpy_to_vtk(z_sigma_np, array_type=vtk.VTK_DOUBLE, deep=True)
        z_sigma.SetName('z_sigma')
        self.pdata_dict[key].GetPointData().AddArray(z_sigma)
        vgf = vtk.vtkVertexGlyphFilter()
        vgf.SetInputData(self.pdata_dict[key])
        vgf.Update()
        self.pdata_dict[key] = vgf.GetOutput()

    def save_local_max(self, suffix='', key='local_max'):
        """
        Saves the local max in the npyfiles directory NOT ROBUST

        Parameters
        ----------
        suffix : str, optional
            Suffix for directory to write to. The default is ''.
        key : str, optional
            Key for this pointset in pdata_dict. The default is 'local_max'

        Returns:
        --------
        None.

        """

        try:
            pdata = self.pdata_dict[key]
        except KeyError:
            raise RuntimeWarning('Key not found in pdata_dict. Nothing saved')
            return
        # Get npy arrays
        pts_np = vtk_to_numpy(pdata.GetPoints().GetData())
        dist_np = vtk_to_numpy(pdata.GetPointData().GetArray('dist'))
        z_sigma_np = vtk_to_numpy(pdata.GetPointData().GetArray('z_sigma'))
        # Save a npz file
        np.savez(os.path.join(self.project_path, self.project_name, 
                              'npyfiles' + suffix, key),
                 pts_np=pts_np, dist_np=dist_np, z_sigma_np=z_sigma_np)

    def load_local_max(self, suffix='', key='local_max'):
        """
        Loads the local max in the npyfiles directory NOT ROBUST

        Parameters
        ----------
        suffix : str, optional
            Suffix for directory to write to. The default is ''.
        key : str, optional
            Key for this pointset in pdata_dict. The default is 'local_max'

        Returns:
        --------
        None.

        """

        warnings.warn('No transform or history information is stored ' +
                             'with this numpy file. Be sure you know where ' +
                             'it came from!')
        with np.load(os.path.join(self.project_path, self.project_name, 
                                    'npyfiles' + suffix, key + '.npz')) as data:
            pts_np = data['pts_np']
            dist_np = data['dist_np']
            z_sigma_np = data['z_sigma_np']
            # Create polydata
            self.pdata_dict[key] = vtk.vtkPolyData()
            pts = vtk.vtkPoints()
            pts.SetData(numpy_to_vtk(pts_np, array_type=vtk.VTK_DOUBLE,
                                     deep=True))
            self.pdata_dict[key].SetPoints(pts)
            dist = numpy_to_vtk(dist_np, array_type=vtk.VTK_DOUBLE, deep=True)
            dist.SetName('dist')
            self.pdata_dict[key].GetPointData().AddArray(dist)
            z_sigma = numpy_to_vtk(z_sigma_np, array_type=vtk.VTK_DOUBLE, deep=True)
            z_sigma.SetName('z_sigma')
            self.pdata_dict[key].GetPointData().AddArray(z_sigma)
            vgf = vtk.vtkVertexGlyphFilter()
            vgf.SetInputData(self.pdata_dict[key])
            vgf.Update()
            self.pdata_dict[key] = vgf.GetOutput()

class ScanArea:
    """
    Manage multiple scans from the same area.
    
    ...
    
    Attributes
    ----------
    project_path : str
        Path to folder containing all Riscan projects
    project_dict : dict
        Dictionary containing each project object keyed on project_name
    registration_list : list
        List of registration instructions, each element is a namedtuple
        (project_name_0, project_name_1, reflector_list, mode, yaw_angle). 
        The order of the list determines the order that actions are performed.
    difference_dict : dict
        Dictionary containing the differences between pairs of scans
    max_difference_dict : dict
        Dictionary containing the differences of local maxes between
        pairs of scans
        
    Methods
    -------
    add_project(project_name, import_mode=None, poly='.1_.1_.01', 
                load_scans=True, read_scans=False, import_las=False, 
                create_id=True, las_fieldnames=None, class_list=[0, 1, 2, 70], 
                suffix='')
        Add a project in project_path directory to project_dict
    compare_reflectors(project_name_0, project_name_1, delaunay=False, 
                       mode='dist')
        Calculate pwdists and plot reflector comparison project 0 to project 1
    add_registration_tuple(registration_tuple, index=None)
        Add a registration tuple to the registration list.
    del_registration_tuple(index)
        Delete registration tuple from registration_list.
    register_all()
        Register all projects according to registration list.
    register_project(project_name_0, project_name_1, reflector_list, mode='lS')
        Register project 1 to project 0 using the reflectors in reflector_list.
    z_align_all(w0=10, w1=10, min_pt_dens=10, max_diff=0.1, 
                frac_exceed_diff_cutoff=0.1, bin_reduc_op='min', 
                diff_mode='mean')
        Align all scans on the basis of their gridded values
    z_alignment(project_name_0, project_name_1, w0=10, w1=10, min_pt_dens=10, 
                max_diff=0.15, frac_exceed_diff_cutoff=0.1, bin_reduc_op='min',
                diff_mode='mean')
        Align successive scans on the basis of their gridded values
    z_alignment_ss(project_name_0, project_name_1, scan_name, w0=10, w1=10, 
                   min_pt_dens=10, max_diff=0.15, bin_reduc_op='min', 
                   return_grid=False, return_history_dict=False)
        Align successive scans on the basis of their gridded minima
    z_tilt_alignment_ss(project_name_0, project_name_1, scan_name, w0=10, w1=10,
                        min_pt_dens=10, max_diff=0.15, bin_reduc_op='mean')
        Align a scan vertically and tilt based upon it's z offsets
    max_alignment_ss(project_name_0, project_name_1, scan_name, w0=5, w1=5, 
                     max_diff=0.1, return_count=False, use_closest=False, 
                     p_thresh=None, az_thresh=None, z_intcpt=None, z_slope=None)
        Align singlescan with project 0 using local maxima as keypoints.
    mesh_to_image(z_min, z_max, nx, ny, dx, dy, x0, y0)
        Interpolate mesh into image.
    difference_projects(project_name_0, project_name_1, 
                        difference_field='Elevation', confidence_interval=False,
                        key='')
        Subtract project_0 from project_1 and store the result in 
        difference_dict.
    get_np_nan_diff_image(project_name_0, project_name_1, key='', 
                          diff_ci_cutoff=np.inf)
        Convenience function for copying the image to a numpy object.
    display_difference(project_name_0, project_name_1, diff_window,
                           cmap='rainbow', profile_list=[], key='')
        Display difference image in vtk interactive window.
    display_warp_difference(project_name_0, project_name_1, diff_window, 
                            field='Elevation_mean_fill', cmap='rainbow', 
                            profile_list=[], show_scanners=False, 
                            scanner_color_0='Yellow', scanner_color_1='Fuchsia',
                            scanner_length=150, key='')
        Display the surface of the image from project_name_1 colored by diff.
    write_plot_warp_difference(project_name_0, project_name_1, 
                                diff_window, camera_position, focal_point,
                                roll=0,
                                field='Elevation',
                                cmap='RdBu_r', filename="", name="",
                                light=None, colorbar=True, profile_list=[],
                                window_size=(2000, 1000), key='')
        Write difference visualization to file.
    write_plot_difference_projects(project_name_0, project_name_1, 
                                   diff_window, filename="", colorbar=True,
                                   key='')
        Display a plot showing the difference between two projects
    difference_maxes(project_name_0, project_name_1, r_pair)
        Compare local maxes in two scans and store result in 
        max_difference_dict.
    
    """
    
    def __init__(self, project_path, project_names=[], registration_list=[],
                 import_mode=None, poly='.1_.1_.01', load_scans=True, 
                 read_scans=False, import_las=False,  create_id=True,
                 las_fieldnames=None, 
                 class_list=[0, 1, 2, 70], suffix=''):
        """
        init stores project_path and initializes project_dict

        Parameters
        ----------
        project_path : str
            Directory location of the projects
        project_names : list, optional
            If given add all of these projects into project_dict. The default
            is [].
        registration_list : list, optional
            If given this is the list of registration actions. Note, we make
            a deep copy of this list to avoid reference issues. The default is
            [].
        import_mode : str, optional
            How to create polydata_raw, the base data for this SingleScan. 
            Options are: 'poly' (read from Riscan generated poly), 'read_scan'
            (read saved vtp file), 'import_las' (use pdal to import from las
            file generate by Riscan), 'empty' (create an empty polydata, 
            useful if we just want to work with transformations). 'import_npy'
            (import from npyfiles directories) If value is None, then code 
            will interpret values of read_scan and import_las
            (deprecated method of specifying which to import) to maintain
            backwards compatibility. The default is None.
        poly : str, optional
            The suffix describing which polydata to load. The default is
            '.1_.1_.01'.
        load_scans : bool, optional
            Whether to actually load the scans. Often if we're just
            aligning successive scans loading all of them causes overhead.
            The default is True.
        read_scans : bool, optional
            If False, each SingleScan object will be initialized to read the
            raw polydata from where RiSCAN saved it. If True, read the saved
            vtp file from in the scan area directory. Useful if we have saved
            already filtered scans. The default is False.
        import_las: bool, optional
            If true (and read_scan is False) read in the las file instead of
            the polydata. The default is False.
        create_id: bool, optional
            If true and PointID's do not exist create PointIDs. The default
            is True.
        las_fieldnames: list, optional
            List of fieldnames to load if we are importing from a las file
            Must include 'Points'. The default is None.
        class_list : list
            List of categories this filter will return, if special value: 
            'all' Then we do not have a selection filter and we pass through 
            all points. The default is [0, 1, 2, 70].
        suffix : str, optional
            Suffix for npyfiles directory if we are reading scans. The default
            is '' which corresponds to the regular npyfiles directory.
        
        Returns
        -------
        None.

        """
        
        self.project_path = project_path
        self.project_dict = {}
        self.difference_dict = {}
        self.difference_dsa_dict = {}
        self.max_difference_dict = {}
        
        for project_name in project_names:
            self.add_project(project_name, import_mode=import_mode, 
                             load_scans=load_scans, read_scans=read_scans,
                             poly=poly, import_las=import_las, 
                             create_id=create_id,
                             las_fieldnames=las_fieldnames,
                             class_list=class_list, suffix=suffix)
            
        self.registration_list = copy.deepcopy(registration_list)
    
    def add_project(self, project_name, import_mode=None, 
                    poly='.1_.1_.01', load_scans=True, 
                    read_scans=False, import_las=False, create_id=True,
                    las_fieldnames=None,
                    class_list=[0, 1, 2, 70], suffix=''):
        """
        Add a new project to the project_dict (or overwrite existing project)

        Parameters
        ----------
        project_name : str
            Name of Riscan project to add
        poly : str, optional
            The suffix describing which polydata to load. The default is
            '.1_.1_.01'.
        import_mode : str, optional
            How to create polydata_raw, the base data for this SingleScan. 
            Options are: 'poly' (read from Riscan generated poly), 'read_scan'
            (read saved vtp file), 'import_las' (use pdal to import from las
            file generate by Riscan), 'empty' (create an empty polydata, 
            useful if we just want to work with transformations). 'import_npy'
            (import from npyfiles directories) If value is None, then code 
            will interpret values of read_scan and import_las
            (deprecated method of specifying which to import) to maintain
            backwards compatibility. The default is None.
        load_scans : bool, optional
            Whether to actually load the scans. Often if we're just
            aligning successive scans loading all of them causes overhead.
            The default is True.
        read_scans : bool, optional
            If False, each SingleScan object will be initialized to read the
            raw polydata from where RiSCAN saved it. If True, read the saved
            vtp file from in the scan area directory. Useful if we have saved
            already filtered scans. The default is False.
        import_las: bool, optional
            If true (and read_scan is False) read in the las file instead of
            the polydata. The default is False.
        create_id: bool, optional
            If true and PointID's do not exist create PointIDs. The default
            is True.
        las_fieldnames: list, optional
            List of fieldnames to load if we are importing from a las file
            Must include 'Points'. The default is None.
        class_list : list
            List of categories this filter will return, if special value: 
            'all' Then we do not have a selection filter and we pass through 
            all points. The default is [0, 1, 2, 70].
        suffix : str, optional
            Suffix for npyfiles directory if we are reading scans. The default
            is '' which corresponds to the regular npyfiles directory.

        Returns
        -------
        None.

        """
        
        self.project_dict[project_name] = Project(self.project_path, 
                                                  project_name, 
                                                  import_mode=import_mode,
                                                  load_scans=
                                                  load_scans, read_scans=
                                                  read_scans, poly=poly,
                                                  import_las=import_las,
                                                  create_id=create_id,
                                                  las_fieldnames=
                                                  las_fieldnames, class_list=
                                                  class_list, suffix=suffix)
    
    def compare_reflectors(self, project_name_0, project_name_1, 
                           delaunay=False, mode='dist', 
                           use_tiepoints_transformed=False):
        """
        Plot the comparison of pwdist changes from project 0 to 1

        Parameters
        ----------
        project_name_0 : str
            Name of project 0. Presumably already transformed to desired loc.
        project_name_1 : str
            Name of project 1. Project we are comparing (usually later)
        delaunay : bool, optional
            Whether to plot just delaunay lines. The default is False.
        mode : str, optional
            Whether to display distance change ('dist') or strain ('strain'). 
            The default is 'dist'.
        use_tiepoints_transformed : bool, optional
            Whether to display tiepoints in their transformed locations.
            The default is False.

        Returns
        -------
        None.

        """
        
        # Add project_0 tiepoints to project_1's dict_compare
        self.project_dict[project_name_1].tiepointlist.compare_pairwise_dist(
            self.project_dict[project_name_0].tiepointlist)
        
        # Display comparison
        (self.project_dict[project_name_1].
         tiepointlist.plot_map(project_name_0, delaunay=delaunay, mode=mode,
                               use_tiepoints_transformed=
                               use_tiepoints_transformed))
    
    def add_registration_tuple(self, registration_tuple, index=None):
        """
        Add registration tuple to registration list, if no index given append.

        Parameters
        ----------
        registration_tuple : tuple
            Registration tuple to add.
        index : int, optional
            Index at which to add tuple or add at end if None. 
            The default is None.

        Returns
        -------
        None.

        """
        
        if index is None:
            self.registration_list.append(registration_tuple)
        else:
            self.registration_list.insert(index, registration_tuple)
        
    def del_registration_tuple(self, index):
        """
        Delete registration tuple from position specified by index.

        Parameters
        ----------
        index : int
            Index of registration tuple to delete.

        Returns
        -------
        None.

        """
        self.registration_list.pop(index)
    
    def register_all(self):
        """
        Register all projects in self according to registration_list.

        Returns
        -------
        None.

        """
        
        for registration_tuple in self.registration_list:
            self.register_project(registration_tuple.project_name_0,
                                  registration_tuple.project_name_1,
                                  registration_tuple.reflector_list,
                                  registration_tuple.mode,
                                  True,
                                  registration_tuple.yaw_angle)
    
    def register_project(self, project_name_0, project_name_1, reflector_list,
                         mode='LS', use_tiepoints_transformed=True,
                         yaw_angle=0):
        """
        Register project_1 to project_0 using reflectors in reflector_list.
        
        Calculates 4x4 transform matrix and adds it to project_1's 
        tiepointlist and to each SingleScan in Project_1's tranform_dict. The
        transform is keyed on the tuple (project_name_0 + '_' + mode, 
        str_reflector_list). Applies the transform to project_1's tiepointlist
        and applies sop and that transform to each SingleScan in project_1.
        
        For the special case where we just want to set a project's
        registration to be its own prcs (leave the tiepoints as is) we set
        the project_name_0 to be the same as project_name_1.

        Parameters
        ----------
        project_name_0 : str
            Name of project to register other project to.
        project_name_1 : str
            Name of project we want to register.
        reflector_list : list
            List of reflectors to use in registration.
        mode : str, optional
            'LS' or 'Yaw", the mode of the registration. See 
            Tiepointlist.calc_transformation for more detail. 
            The default is 'LS'.
        use_tiepoints_transformed : bool, optional
            Whether to register to tiepoints_transformed in project_0.
            The default is True.
        yaw_angle : float, optional
            If the mode is 'Trans' this is the angle (in radians) by which to
            change the heading of the scan. The default is 0.

        Returns
        -------
        None.

        """
        
        if project_name_0 == project_name_1:
            # Apply identity transform to tiepoints
            self.project_dict[project_name_0].tiepointlist.apply_transform(
                ('identity', ''))
            # Apply sop to each single scan
            self.project_dict[project_name_0].apply_transforms(['sop'])
            # Return to skip rest of execution
            return
        
        # Calculate the transformation that aligns project_1 with project_0
        transform_name = (self.project_dict[project_name_1].tiepointlist.
                          calc_transformation(self.project_dict[
                              project_name_0].tiepointlist, reflector_list,
                              mode=mode, use_tiepoints_transformed=
                              use_tiepoints_transformed, yaw_angle=yaw_angle))
        
        # Apply that transform to project_1's tiepointlist
        self.project_dict[project_name_1].tiepointlist.apply_transform(
            transform_name)
        
        # Add that transform to project_1's SingleScans transform_dict
        self.project_dict[project_name_1].add_transform_from_tiepointlist(
            transform_name)
        
        # Apply sop and our new transform to each SingleScan
        self.project_dict[project_name_1].apply_transforms(['sop', 
                                                            transform_name])
    def z_align_all(self, w0=10, w1=10,
                           min_pt_dens=10, max_diff=0.1, 
                           frac_exceed_diff_cutoff=0.1, bin_reduc_op='min',
                           diff_mode='mean'):
        """
        Align all scans on the basis of their gridded values

        !This function does not modify the tiepoint locations so it should 
        only be run after all tiepoint registration steps are done. It also
        requires that there hasn't been ice deformation and will try to not
        run if the fraction that changed by more than the diff cutoff exceeds
        frac_exceed_diff_cutoff.
        
        Parameters
        ----------
        w0 : float, optional
            Grid cell width in x dimension (m). The default is 10.
        w1 : float, optional
            Grid cell width in y dimension (m). The default is 10.
        min_pt_dens : float, optional
            minimum density of points/m^2 for us to compare grid cells from
            projects 0 and 1. The default is 30.
        max_diff : float, optional
            Maximum difference in minima to consider (higher values must be
            noise) in m. The default is 0.1.
        frac_exceed_diff_cutoff : float, optional
            If the max_diff cutoff is causing us to discard greater than this
            fraction of the gridcells that meet the point density requirements
            that probably means the ice has moved, so raise a warning
        bin_reduc_op : str, optional
            What type of gridded reduction to apply. Options are 'min', 'mean'
            and 'mode'. The default is 'min'
        diff_mode : str, optional
            Which property of the difference distribution to set to zero. The
            options are 'mean', 'median' and 'mode'. The default is 'mean'
            
        Returns
        -------
        None.

        """

        for registration_tuple in self.registration_list:
            self.z_alignment(registration_tuple.project_name_0,
                                    registration_tuple.project_name_1,
                                    w0, w1, min_pt_dens, max_diff, 
                                    frac_exceed_diff_cutoff, bin_reduc_op,
                                    diff_mode)

    def z_alignment(self, project_name_0, project_name_1, w0=10, w1=10,
                           min_pt_dens=10, max_diff=0.15, 
                           frac_exceed_diff_cutoff=0.1, bin_reduc_op='min',
                           diff_mode='mean'):
        """
        Align successive scans on the basis of their gridded values

        !This function does not modify the tiepoint locations so it should 
        only be run after all tiepoint registration steps are done. It also
        requires that there hasn't been ice deformation and will try to not
        run if the fraction that changed by more than the diff cutoff exceeds
        frac_exceed_diff_cutoff.
        Parameters
        ----------
        project_name_0 : str
            The reference project we're trying to align project_1 with
        project_name_1 : str
            The project we're aligning with project_0
        w0 : float, optional
            Grid cell width in x dimension (m). The default is 10.
        w1 : float, optional
            Grid cell width in y dimension (m). The default is 10.
        min_pt_dens : float, optional
            minimum density of points/m^2 for us to compare grid cells from
            projects 0 and 1. The default is 30.
        max_diff : float, optional
            Maximum difference in minima to consider (higher values must be
            noise) in m. The default is 0.1.
        frac_exceed_diff_cutoff : float, optional
            If the max_diff cutoff is causing us to discard greater than this
            fraction of the gridcells that meet the point density requirements
            that probably means the ice has moved, so raise a warning
        bin_reduc_op : str, optional
            What type of gridded reduction to apply. Options are 'min', 'mean'
            and 'mode'. The default is 'min'
        diff_mode : str, optional
            Which property of the difference distribution to set to zero. The
            options are 'mean', 'median' and 'mode'. The default is 'mean'
        
        Returns
        -------
        None.

        """
        if project_name_0==project_name_1:
            self.project_dict[project_name_0].add_z_offset(0, history_dict={
                "type": "Transform_Source",
                "git_hash": get_git_hash(),
                "method": "ScanArea.z_alignment",
                "filename": ''
                })
            transform_list = copy.deepcopy(self.project_dict[project_name_0].
                                           current_transform_list)
            transform_list.append('z_offset')
            self.project_dict[project_name_0].apply_transforms(transform_list)
            return
        
        w = [w0, w1]
        min_counts = min_pt_dens * w[0] * w[1]
        
        project_0 = self.project_dict[project_name_0]
        project_1 = self.project_dict[project_name_1]

        # Get the merged points polydata, by not using port we should prevent
        # this memory allocation from persisting
        pdata_merged_project_0, history_dict_project_0 = (project_0
                                                          .get_merged_points(
                                                              history_dict=
                                                              True))
        project_0_points_np = vtk_to_numpy(pdata_merged_project_0
                                           .GetPoints().GetData())
        bounds = pdata_merged_project_0.GetBounds()

        # Create grid
        edges = 2*[None]
        nbin = np.empty(2, np.int_)

        for i in range(2):
            edges[i] = np.arange(int(np.ceil((bounds[2*i + 1] - 
                                              bounds[2*i])/w[i]))
                                 + 1, dtype=np.float32) * w[i] + bounds[2*i]
            # Adjust lower edge so we don't miss lower most point
            edges[i][0] = edges[i][0] - 0.0001
            # Adjust uppermost edge so the bin width is appropriate
            edges[i][-1] = bounds[2*i + 1] + 0.0001
            nbin[i] = len(edges[i]) + 1

        # Get gridded counts and bin reduc
        if bin_reduc_op=='min':
            project_0_counts, project_0_arr = gridded_counts_mins(
                project_0_points_np, edges, np.float32(bounds[5] + 1))
        elif bin_reduc_op=='mean':
            project_0_counts, project_0_arr = gridded_counts_means(
                project_0_points_np, edges)
        elif bin_reduc_op=='mode':
            project_0_counts, project_0_arr = gridded_counts_modes(
                project_0_points_np, edges)
        else:
            raise ValueError('bin_reduc_op must be min, mean or mode')
        
        # Now for each SingleScan in project_1, use the same grid to get minima
        # and compare with project_0, then compute a z_offset and add to
        # transform_dict
        for scan_name in project_1.scan_dict:
            ss = project_1.scan_dict[scan_name]

            # Get points as an array
            ss_points_np = vtk_to_numpy(ss.currentFilter.GetOutput()
                                        .GetPoints().GetData())
            # Get gridded counts and reduc op (using the same grid)
            if bin_reduc_op=='min':
                ss_counts, ss_arr = gridded_counts_mins(
                    ss_points_np, edges, np.float32(bounds[5] + 1))
            elif bin_reduc_op=='mean':
                ss_counts, ss_arr = gridded_counts_means(
                    ss_points_np, edges)
            elif bin_reduc_op=='mode':
                ss_counts, ss_arr = gridded_counts_modes(
                    ss_points_np, edges)

            # Compute differences of gridded minima
            diff = ss_arr - project_0_arr
            diff[project_0_counts < min_counts] = np.nan
            diff[ss_counts < min_counts] = np.nan
            # Create history dict for this transformation
            history_dict = {
                "type": "Transform Computer",
                "git_hash": get_git_hash(),
                "method": "ScanArea.z_alignment",
                "input_0": history_dict_project_0,
                "input_1": json.loads(json.dumps(
                    ss.filt_history_dict)),
                "params": {"w0": w0, "w1": w1,
                   "min_pt_dens": min_pt_dens, "max_diff": max_diff, 
                   "frac_exceed_diff_cutoff": frac_exceed_diff_cutoff,
                   "bin_reduc_op": bin_reduc_op,
                   "diff_mode": diff_mode}
                }
            if np.isnan(diff).all():
                ss.add_z_offset(0, history_dict)
                warnings.warn("No overlap for " + project_name_1 + scan_name +
                              " set z_offset to 0", UserWarning)
            else:
                frac_exceed_max_diff = ((np.abs(diff) > max_diff).sum()
                                        / np.logical_not(np.isnan(diff)).sum())
                if frac_exceed_max_diff > frac_exceed_diff_cutoff:
                    ss.add_z_offset(0, history_dict)
                    num_density = np.logical_not(np.isnan(diff)).sum()
                    warnings.warn("The fraction of the " + str(num_density) +
                                  " cells that meet the " 
                            + "density criteria but exceed the max_diff is " +
                            str(frac_exceed_max_diff) + " assuming ice " +
                            "motion and setting the offset to zero for " +
                            project_name_1 + scan_name, UserWarning)
                else:
                    diff[np.abs(diff) > max_diff] = np.nan
                    diff_notnan = np.ravel(diff)[np.logical_not(np.isnan(
                        np.ravel(diff)))]
                    if diff_mode=='mean':
                        ss.add_z_offset(-1*diff_notnan.mean(), history_dict)
                    elif diff_mode=='median':
                        ss.add_z_offset(-1*np.median(diff_notnan), 
                                        history_dict)
                    elif diff_mode=='mode':
                        m, _ = mode(np.around(diff_notnan, 3))
                        ss.add_z_offset(-1*m, history_dict)
                    else:
                        raise ValueError('diff_mode must be mean, median, or'
                                         + ' mode')

        # Apply the transforms
        transform_list = copy.deepcopy(project_1.current_transform_list)
        transform_list.append('z_offset')
        project_1.apply_transforms(transform_list)
        return
    
    def z_alignment_ss(self, project_name_0, project_name_1, scan_name,
                              w0=10, w1=10, min_pt_dens=10, max_diff=0.15,
                              bin_reduc_op='min', return_grid=False,
                              return_history_dict=False):
        """
        Align successive scans on the basis of their gridded minima
        
        ss version is for looking at the change in just a singlescan

        !This function does not modify the tiepoint locations so it should 
        only be run after all tiepoint registration steps are done. It also
        requires that there hasn't been ice deformation and will try to not
        run if the fraction that changed by more than the diff cutoff exceeds
        frac_exceed_diff_cutoff.
        Parameters
        ----------
        project_name_0 : str
            The reference project we're trying to align project_1 with
        project_name_1 : str
            The project we're aligning with project_0
        w0 : float, optional
            Grid cell width in x dimension (m). The default is 10.
        w1 : float, optional
            Grid cell width in y dimension (m). The default is 10.
        min_pt_dens : float, optional
            minimum density of points/m^2 for us to compare grid cells from
            projects 0 and 1. The default is 30.
        max_diff : float, optional
            Maximum difference in minima to consider (higher values must be
            noise) in m. The default is 0.1.
        bin_reduc_op : str, optional
            What type of gridded reduction to apply. Options are 'min', 'mean'
            and 'mode'. The default is 'min'
        return_history_dict : bool, optional
            Whether to return a history dict with the z offset. The default
            is False.

        Returns
        -------
        diff : ndarray
            Array containing gridded minima differences

        """
        # if project_name_0==project_name_1:
        #     self.project_dict[project_name_0].add_z_offset(0)
        #     transform_list = copy.deepcopy(self.project_dict[project_name_0].
        #                                    current_transform_list)
        #     transform_list.append('z_offset')
        #     self.project_dict[project_name_0].apply_transforms(transform_list)
        #     return
        
        w = [w0, w1]
        min_counts = min_pt_dens * w[0] * w[1]
        
        project_0 = self.project_dict[project_name_0]
        project_1 = self.project_dict[project_name_1]

        # Now get ss
        if project_name_0==project_name_1:
            ss = project_1.scan_dict.pop(scan_name)
        else:
            ss = project_1.scan_dict[scan_name]

        # Get the merged points polydata, by not using port we should prevent
        # this memory allocation from persisting
        pdata_merged_project_0, history_dict_project_0 = (
            project_0.get_merged_points(history_dict=True))
        project_0_points_np = vtk_to_numpy(pdata_merged_project_0
                                           .GetPoints().GetData())
        #bounds = pdata_merged_project_0.GetBounds()
        
        # Get points as an array
        ss_points_np = vtk_to_numpy(ss.currentFilter.GetOutput().GetPoints()
                                    .GetData())
        bounds = ss.currentFilter.GetOutput().GetBounds()
        
        # Create grid
        edges = 2*[None]
        nbin = np.empty(2, np.int_)

        for i in range(2):
            edges[i] = np.arange(int(np.ceil((bounds[2*i + 1] - 
                                              bounds[2*i])/w[i]))
                                 + 1, dtype=np.float32) * w[i] + bounds[2*i]
            # Adjust lower edge so we don't miss lower most point
            edges[i][0] = edges[i][0] - 0.0001
            # Adjust uppermost edge so the bin width is appropriate
            edges[i][-1] = bounds[2*i + 1] + 0.0001
            nbin[i] = len(edges[i]) + 1
        
        # Get gridded counts and reduc op (using the same grid)
        if bin_reduc_op=='min':
            ss_counts, ss_arr = gridded_counts_mins(
                ss_points_np, edges, np.float32(bounds[5] + 1))
        elif bin_reduc_op=='mean':
            ss_counts, ss_arr = gridded_counts_means(
                ss_points_np, edges)
        elif bin_reduc_op=='mode':
            ss_counts, ss_arr = gridded_counts_modes(
                ss_points_np, edges)
        
        # Get gridded counts and bin reduc
        if bin_reduc_op=='min':
            project_0_counts, project_0_arr = gridded_counts_mins(
                project_0_points_np, edges, np.float32(bounds[5] + 1))
        elif bin_reduc_op=='mean':
            project_0_counts, project_0_arr = gridded_counts_means(
                project_0_points_np, edges)
        elif bin_reduc_op=='mode':
            project_0_counts, project_0_arr = gridded_counts_modes(
                project_0_points_np, edges)
        else:
            raise ValueError('bin_reduc_op must be min, mean or mode')
        

        # Compute differences of gridded minima
        diff = ss_arr - project_0_arr
        diff[project_0_counts < min_counts] = np.nan
        diff[ss_counts < min_counts] = np.nan
        if np.isnan(diff).all():
            frac_exceed_max_diff = np.nan
            warnings.warn("No overlap for " + project_name_1 + scan_name +
                          " set z_offset to 0", UserWarning)
        else:
            frac_exceed_max_diff = ((np.abs(diff) > max_diff).sum()
                                    /np.logical_not(np.isnan(diff)).sum())
            diff[np.abs(diff) > max_diff] = np.nan
            if np.isnan(diff).all():
                warnings.warn("All diffs exceed max_diff")

        # return popped scan if the project names are the same
        if project_name_0==project_name_1:
            project_1.scan_dict[scan_name] = ss

        # Create history dict to return if desired
        if return_history_dict:
            history_dict = {
                "type": "Transform Computer",
                "git_hash": get_git_hash(),
                "method": "ScanArea.z_alignment_ss",
                "input_0": history_dict_project_0,
                "input_1": json.loads(json.dumps(
                    ss.filt_history_dict)),
                "params": {"w0": w0, "w1": w1,
                   "min_pt_dens": min_pt_dens, "max_diff": max_diff,
                   "bin_reduc_op": bin_reduc_op}
                }
        
        # If we want to return the grid as a pointcloud compute it here
        if return_grid:
            # Get the grid indices in the current reference frame
            x_trans = (edges[0][:-1] + edges[0][1:])/2
            y_trans = (edges[1][:-1] + edges[1][1:])/2
            X_trans, Y_trans = np.meshgrid(x_trans, y_trans)
            grid_trans = np.hstack((X_trans.ravel()[:,np.newaxis], 
                                    Y_trans.ravel()[:,np.newaxis],
                                    np.zeros((X_trans.size, 1)),
                                    np.ones((X_trans.size, 1))))
            grid_trans = np.hstack((X_trans.ravel()[:,np.newaxis], 
                                    Y_trans.ravel()[:,np.newaxis],
                                    np.zeros((X_trans.size, 1))))
            pts = vtk.vtkPoints()
            pts.SetData(numpy_to_vtk(grid_trans, array_type=vtk.VTK_DOUBLE))
            pdata = vtk.vtkPolyData()
            pdata.SetPoints(pts)
            # Get the inverse of the singlescan's transform
            invTransform = vtk.vtkTransform()
            invTransform.DeepCopy(ss.transform)
            invTransform.Inverse()
            tfilter = vtk.vtkTransformPolyDataFilter()
            tfilter.SetTransform(invTransform)
            tfilter.SetInputData(pdata)
            tfilter.Update()
            grid_ss = vtk_to_numpy(tfilter.GetOutput().GetPoints().GetData())
           
            # Return
            if return_history_dict:
                return (frac_exceed_max_diff, diff.T, grid_ss.copy(), 
                        history_dict)
            else:
                return frac_exceed_max_diff, diff.T, grid_ss.copy()

        else:
            # Return the diff
            if return_history_dict:
                return frac_exceed_max_diff, diff, history_dict
            else:
                return frac_exceed_max_diff, diff

    def z_tilt_alignment_ss(self, project_name_0, project_name_1, scan_name,
                              w0=10, w1=10, min_pt_dens=10, max_diff=0.15,
                              bin_reduc_op='mean'):
        """
        Align a scan vertically and tilt based upon it's z offsets
        
        ss version is for looking at the change in just a singlescan

        !This function does not modify the tiepoint locations so it should 
        only be run after all tiepoint registration steps are done. 

        Parameters
        ----------
        project_name_0 : str
            The reference project we're trying to align project_1 with
        project_name_1 : str
            The project we're aligning with project_0
        scan_name : str
            The scan in project_1 that we are aligning
        w0 : float, optional
            Grid cell width in x dimension (m). The default is 10.
        w1 : float, optional
            Grid cell width in y dimension (m). The default is 10.
        min_pt_dens : float, optional
            minimum density of points/m^2 for us to compare grid cells from
            projects 0 and 1. The default is 30.
        max_diff : float, optional
            Maximum difference in minima to consider (higher values must be
            noise) in m. The default is 0.1.
        bin_reduc_op : str, optional
            What type of gridded reduction to apply. Options are 'min', 'mean'
            and 'mode'. The default is 'min'
        
        Returns
        -------
        A, history_dict :
            A 4x4 matrix containing the resulting transform and history dict

        """

        _, diff, grid, history_dict = self.z_alignment_ss(project_name_0, 
                                                          project_name_1,
                                            scan_name, w0=w0, w1=w1,
                                            min_pt_dens=min_pt_dens,
                                            max_diff=max_diff, bin_reduc_op=
                                            bin_reduc_op, return_grid=True,
                                            return_history_dict=True)
        # Compute the least squares fit 
        ind = np.logical_not(np.isnan(diff.ravel()))
        A = np.hstack((np.ones((ind.sum(),1)), grid[ind,:2]))
        b = diff.ravel()[ind, np.newaxis]
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # Get the current transform applied to ss in translation and rotation
        ss = self.project_dict[project_name_1].scan_dict[scan_name]
        pos = np.float32(ss.transform.GetPosition())
        ori = np.float32(ss.transform.GetOrientation())
        ori = ori * math.pi / 180

        # Add the deltas from the least squares fit (note we make small angle)
        # approximation to go from slope to angles in radians
        u = ori[0] - x[2,0] # negative slope w.r.t. y axis for droll
        v = ori[1] + x[1,0] # positive slope w.r.t. x axis for dpitch
        w = ori[2]
        dx = pos[0]
        dy = pos[1]
        dz = pos[2] - x[0,0] # subtract by z-intercept in regression

        # Create a 4x4 homologous transform from the new parameters
        c = np.cos
        s = np.sin
        
        Rx = np.array([[1, 0, 0, 0],
                      [0, c(u), -s(u), 0],
                      [0, s(u), c(u), 0],
                      [0, 0, 0, 1]])
        Ry = np.array([[c(v), 0, s(v), 0],
                       [0, 1, 0, 0],
                       [-s(v), 0, c(v), 0],
                       [0, 0, 0, 1]])
        Rz = np.array([[c(w), -s(w), 0, 0],
                      [s(w), c(w), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        # Order of rotations in vtk is Pitch, then Roll, then Yaw
        M = Rz @ Rx @ Ry
        # Now add translation components
        M[0, 3] = dx
        M[1, 3] = dy
        M[2, 3] = dz

        # modify history dict
        history_dict['method'] = "ScanArea.z_tilt_alignment_ss"

        return M, history_dict


    def max_alignment_ss(self, project_name_0, project_name_1, scan_name,
                         w0=5, w1=5, max_diff=0.1, return_count=False,
                         use_closest=False, p_thresh=None, az_thresh=None,
                         z_intcpt=None, z_slope=None):
        """
        Align singlescan with project 0 using local maxima as keypoints.
        
        This function breaks each pointcloud up into bins and finds the point
        with the maximum z value in each bin, resulting in a pair of possible
        keypoints for each bin. Then we select only those pairs whose
        euclidean distance is less than max_diff and find the rigid
        transformation which aligns the pairs of points in a least squares
        sense.
        
        This function returns the rigid transform as a numpy 4x4 array and
        the history dict corresponding to this transform.

        Parameters
        ----------
        project_name_0 : str
            The reference project we're aligning project_1 with.
        project_name_1 : str
            The project we are aligning.
        scan_name : str
            The scan we are aligning in the project we're aligning.
        w0 : float, optional
            Bin width in the zeroth dimension. The default is 5.
        w1 : float, optional
            Bin width in the first dimension. The default is 5.
        max_diff : float, optional
            The max distance (in m) between two gridded local maxima for us
            to try to align them. The default is 0.1.
        return_count : bool, optional
            Whether or not to return the number of keypoint pairs as the 3rd
            return. The default is False.
        use_closest : bool, optional
            If True, just align off of the closest singlescan in project_0
            as opposed to all of project_0. The default is False.
        p_thresh : float, optional
            Radial difference threshold for keypoint pairs if using 
            cylindrical divergence. Will only be used if max_diff is None.
            The default is None.
        az_thresh : float, optional
            Azimuthal difference threshold for keypoint pairs if using 
            cylindrical divergence. Will only be used if max_diff is None.
            The default is None.
        z_intcpt : float, optional
            Z intercept difference threshold for keypoint pairs if using 
            cylindrical divergence. Will only be used if max_diff is None.
            The default is None.
        z_slope : float, optional
            Z slope difference threshold for keypoint pairs if using 
            cylindrical divergence. Will only be used if max_diff is None.
            The default is None.

        Returns
        -------
        ndarray, dict, int (optional)
            Returns the 4x4 array that transforms our singlescan's local
            maxima to align with project_0's local maxima. Also returns
            history_dict for the transform. Optionally returns the number of
            keypoint pairs.

        """
        
        # Get pointclouds and history_dicts
        # if we are aligning within the same project need to pop ss
        if project_name_0==project_name_1:
            ss = self.project_dict[project_name_1].scan_dict.pop(scan_name)
        else:
            ss = self.project_dict[project_name_1].scan_dict[scan_name]
        ss_pdata, ss_hist_dict = (ss.get_polydata(history_dict=True))


        if use_closest:
            # Get the position of the scan we're aligning in common ref frame
            pos = np.array(ss.transform.GetPosition())
            # find the closest scan in project 0
            dist = 1000
            for scan_name_0 in self.project_dict[project_name_0].scan_dict:
                p0 = np.array(self.project_dict[project_name_0]
                              .scan_dict[scan_name_0].transform.GetPosition())
                d0 = np.sqrt(np.square(p0 - pos).sum())
                if d0<dist:
                    dist = d0
                    closest_scan = scan_name_0
            project_pdata, project_hist_dict = (
                self.project_dict[project_name_0].scan_dict[closest_scan]
                .get_polydata(history_dict=True))
        else:
            project_pdata, project_hist_dict = (self
                                                .project_dict[project_name_0]
                                                .get_merged_points(
                                                    history_dict=True))
        
        ss_pts = vtk_to_numpy(ss_pdata.GetPoints().GetData())
        project_pts = vtk_to_numpy(project_pdata.GetPoints().GetData())
        
        # Create grid
        w = [w0, w1]
        bounds = ss_pdata.GetBounds()
        edges = 2*[None]
        nbin = np.empty(2, np.int_)
        for i in range(2):
            edges[i] = np.arange(int(np.ceil((bounds[2*i + 1] - 
                                              bounds[2*i])/w[i]))
                                 + 1, dtype=np.float32) * w[i] + bounds[2*i]
            # Adjust lower edge so we don't miss lower most point
            edges[i][0] = edges[i][0] - 0.0001
            # Adjust uppermost edge so the bin width is appropriate
            edges[i][-1] = bounds[2*i + 1] + 0.0001
            nbin[i] = len(edges[i]) + 1
        
        # Get gridded local maxima
        ss_inds = gridded_max_ind(ss_pts, edges)
        project_inds = gridded_max_ind(project_pts, edges)
        
        # We need to mask out cells that don't have points in both pointclouds
        ss_mask = (ss_inds==ss_pts.shape[0])
        project_mask = (project_inds==project_pts.shape[0])
        both_mask = ss_mask | project_mask
        ss_maxs = ss_pts[ss_inds[np.logical_not(both_mask)], :]
        project_maxs = project_pts[project_inds[np.logical_not(both_mask)], :]
        
        # Extract pairs of points that meet our max_diff criteria and name
        # according to Arun et al. (who we'll follow for the rigid alignment)
        # If we are just filtering point on the basis of max diff
        if not (max_diff is None):
            sq_pwdist = np.square(ss_maxs - project_maxs).sum(axis=1)
            ind = sq_pwdist<=(max_diff**2)
            if not (p_thresh is None):
                warnings.warn('You have passed p_thresh without negating' +
                              'max_diff, using max_diff')
        # Otherwise, use the cylindrical divergence around ss
        else:
            pos = np.array(ss.transform.GetPosition())[np.newaxis, :]
            # Transform selected points to cylindrical reference frame
            pos_pts0 = project_maxs[:,:2]-pos[:,:2]
            cyl_pts0 = np.hstack((np.sqrt(np.square(pos_pts0).sum(axis=1))
                                  [:, np.newaxis],
                                  np.arctan2(pos_pts0[:,1], pos_pts0[:,0])
                                  [:, np.newaxis],
                                  (project_maxs[:,2] - pos[0,2])
                                  [:, np.newaxis]))
            pos_pts1 = ss_maxs[:,:2]-pos[:,:2]
            cyl_pts1 = np.hstack((np.sqrt(np.square(pos_pts1).sum(axis=1))
                                  [:, np.newaxis],
                                  np.arctan2(pos_pts1[:,1], pos_pts1[:,0])
                                  [:, np.newaxis],
                                  (ss_maxs[:,2] - pos[0,2])[:, np.newaxis]))
            # Get vectors between paired points in cylindrical coords
            cyl_vec = cyl_pts1 - cyl_pts0
            # handle case when azimuth's are on either side of pi
            atol = 0.5
            cyl_vec[np.isclose(cyl_vec[:,1], 2*np.pi, atol=atol)] -= 2*np.pi
            cyl_vec[np.isclose(cyl_vec[:,1], -2*np.pi, atol=atol)] += 2*np.pi
            # Subset to point pairs that meet radial, azimuthal, and z 
            # tolerances
            ind = ((cyl_vec[:,0]<p_thresh) & (cyl_vec[:,0]>-p_thresh)
                    & (cyl_vec[:,1]<az_thresh) & (cyl_vec[:,1]>-az_thresh)
                    & (cyl_vec[:,2]<(z_intcpt + z_slope*cyl_pts1[:,0])) 
                    & (cyl_vec[:,2]>(-z_intcpt - z_slope*cyl_pts1[:,0])))
        psubi_prime = project_maxs[ind, :].T
        psubi = ss_maxs[ind, :].T
        
        # Compute centroids
        p_prime = psubi_prime.mean(axis=1).reshape((3,1))
        p = psubi.mean(axis=1).reshape((3,1))
        
        # Translate such that centroids are at zero
        qsubi_prime = psubi_prime - p_prime
        qsubi = psubi - p
        
        # Calculate the 3x3 matrix H (Using all 3 axes)
        H = np.matmul(qsubi, qsubi_prime.T)
        # Find it's singular value decomposition
        U, S, Vh = svd(H)
        # Calculate X, the candidate rotation matrix
        X = np.matmul(Vh.T, U.T)
        # Check if the determinant of X is near 1, this should basically 
        # alsways be the case for our data
        if np.isclose(1, np.linalg.det(X)):
            R = X
        elif np.isclose(-1, np.linalg.det(X)):
            V_prime = np.array([Vh[0,:], Vh[1,:], -1*Vh[2,:]]).T
            R = np.matmul(V_prime, U.T)
            print(R)
        else:
            warnings.warn('Determinant of rotation matrix is not close +-1'
                          + ' perhaps we do not have enough points?')
        
        # Now find translation vector to align centroids
        T = p_prime - np.matmul(R, p)
        
        # Combine R and T into 4x4 matrix A, defining the rigid transform
        A = np.eye(4)
        A[:3, :3] = R
        A[:3, 3] = T.squeeze()
        
        # Create history_dict
        history_dict = {
            "type": "Transform Computer",
            "git_hash": get_git_hash(),
            "method": "ScanArea.max_alignment_ss",
            "input_0": project_hist_dict,
            "input_1": ss_hist_dict,
            "params": {"w0": w0, "w1": w1, "max_diff": max_diff}
            }
        
        # return ss to scan_dict if we popped it
        if project_name_0==project_name_1:
            self.project_dict[project_name_1].scan_dict[scan_name] = ss

        if return_count:
            return A, history_dict, psubi.shape[1]
        else:
            return A, history_dict
        
    def mesh_to_image(self, nx, ny, dx, dy, x0, y0, yaw=0, sub_list=[],
                      image_key=''):
        """
        Interpolate mesh at regularly spaced points.
        
        Currently this image can only be axis aligned, if you want a different
        orientation then you need to apply the appropriate transformation to
        the mesh.

        Parameters
        ----------
        nx : int
            Number of gridcells in x direction.
        ny : int
            Number of gridcells in y direction.
        dx : int
            Width of gridcells in x direction.
        dy : int
            Width of gridcells in y direction.
        x0 : float
            x coordinate of the origin in m.
        y0 : float
            y coordinate of the origin in m.
        yaw : float, optional
            yaw angle in degerees of image to create, for generating 
            non-axis aligned image. The default is 0
        sub_list : list, optional
            If given, only apply mesh to image on the projects in sub_list.
            The default is [].

        Returns
        -------
        None.

        """
        
        if len(sub_list)==0:
            for key in self.project_dict:
                self.project_dict[key].mesh_to_image(nx, ny, dx, dy, x0, y0, 
                                                     yaw=yaw, key=image_key)
        else:
            for key in sub_list:
                self.project_dict[key].mesh_to_image(nx, ny, dx, dy, x0, y0,
                                                     yaw=yaw, key=image_key)

    def difference_projects(self, project_name_0, project_name_1, 
                            difference_field='Elevation', 
                            confidence_interval=False, key=''):
        """
        Subtract project_0 from project_1 and store in difference_dict.

        Parameters
        ----------
        project_name_0 : str
            Name of project to subtract (usually older).
        project_name_1 : str
            Name of project to subtract from (usually younger).
        difference_field : str, optional
            Which field in ImageData to use. The default is 'Elevation'
        confidence_interval : bool, optional
            Whether to estimate a confidence interval as well. The default
            is False.
        key : str, optional
            Key for images in image_dicts. Must be the same for both projects.
            The default is ''.

        Returns
        -------
        None.

        """
        
        # Difference projects and copy image to difference dict
        # assume projects have the same sized images covering same extent
        # Create image
        im = vtk.vtkImageData()
        im.SetDimensions(self.project_dict[project_name_0].image_dict[key].
                         GetDimensions())
        im.SetOrigin(self.project_dict[project_name_0].image_dict[key].
                     GetOrigin())
        im.SetSpacing(self.project_dict[project_name_0].image_dict[key].
                      GetSpacing())
        arr = vtk.vtkFloatArray()
        arr.SetNumberOfValues(self.project_dict[project_name_0].image_dict[key].
                              GetNumberOfPoints())
        arr.SetName('Difference')
        im.GetPointData().SetScalars(arr)
        self.difference_dsa_dict[(project_name_0, project_name_1, key)] = (
            dsa.WrapDataObject(im))
        # Difference images
        self.difference_dsa_dict[(project_name_0, project_name_1, key)
         ].PointData[
            'Difference'][:] = (
            self.project_dict[project_name_1].dsa_image_dict[key].PointData[
                difference_field]
            - self.project_dict[project_name_0].dsa_image_dict[key].PointData[
                difference_field])
        
        # Repeat Add confidence interval if requested
        if confidence_interval:
            arr = vtk.vtkFloatArray()
            arr.SetNumberOfValues(self.project_dict[project_name_0]
                                  .image_dict[key].GetNumberOfPoints())
            arr.SetName('diff_ci')
            im.GetPointData().AddArray(arr)
            self.difference_dsa_dict[(project_name_0, project_name_1, key)
                                     ].PointData['diff_ci'][:] = 4*np.sqrt(
            np.square(self.project_dict[project_name_1]
                      .dsa_image_dict[key].PointData['z_ci']/4)
            + np.square(self.project_dict[project_name_0]
                       .dsa_image_dict[key].PointData['z_ci']/4))
        
        # np.ravel(
        #     self.project_dict[project_name_1].get_np_nan_image() -
        #     self.project_dict[project_name_0].get_np_nan_image())
        # # Assign value
        self.difference_dict[(project_name_0, project_name_1, key)] = im
    
    def get_np_nan_diff_image(self, project_name_0, project_name_1,
                              key='', diff_ci_cutoff=np.inf):
        """
        Convenience function for copying the image to a numpy object.

        Parameters
        ----------
        project_name_0 : str
            Name of project to subtract (usually older).
        project_name_1 : str
            Name of project to subtract from (usually younger).
        key : str, optional
            Key to index this image and transform in image_dict and
            image_transform_dict. The default is '' (for backward compatability)
        diff_ci_cutoff : float, optional
            Confidence interval cutoff above which to set value to NaN

        Returns
        -------
        nan_image : numpy ndarray

        """
        
        im_diff = self.difference_dict[(project_name_0, project_name_1, key)]
        
        im_diff_np = vtk_to_numpy(im_diff.GetPointData().GetArray('Difference')
                                  ).reshape((im_diff.GetDimensions()[1], 
                                             im_diff.GetDimensions()[0]), 
                                             order='C')
        im_ci = vtk_to_numpy(im_diff.GetPointData().GetArray('diff_ci')
                                  ).reshape((im_diff.GetDimensions()[1], 
                                             im_diff.GetDimensions()[0]), 
                                             order='C')
        nan_image = copy.deepcopy(im_diff_np)
        nan_image[im_ci>diff_ci_cutoff] = np.NaN
            
        return nan_image

    def display_difference(self, project_name_0, project_name_1, diff_window,
                           cmap='rainbow', profile_list=[], key=''):
        """
        Display image in vtk interactive window.

        Parameters
        ----------
        project_name_0 : str
            Name of project to subtract (usually older).
        project_name_1 : str
            Name of project to subtract from (usually younger).
        diff_window : float
            Scale of differences to display in color in m.
        cmap : str, optional
            Name of matplotlib colormap to use. The default is 'rainbow'
        key : str, optional
            Key for images in image_dicts. Must be the same for both projects.
            The default is ''.

        Returns
        -------
        None.

        """
        # Define function for writing the camera position and focal point to
        # std out when the user presses 'u'
        def cameraCallback(obj, event):
            print("Camera Pos: " + str(obj.GetRenderWindow().
                                           GetRenderers().GetFirstRenderer().
                                           GetActiveCamera().GetPosition()))
            print("Focal Point: " + str(obj.GetRenderWindow().
                                            GetRenderers().GetFirstRenderer().
                                            GetActiveCamera().GetFocalPoint()))
        
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(self.difference_dict[(project_name_0,
                                                  project_name_1, key)])
        mapper.SetLookupTable(mplcmap_to_vtkLUT(-diff_window, diff_window,
                                                name=cmap))
        mapper.SetScalarRange(-diff_window, diff_window)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderer.AddActor(actor)

        # Add requested profiles
        for profile_tup in profile_list:
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(self.project_dict[project_name_1]
                                         .image_transform_dict[key])
            transformFilter.SetInputData(self.project_dict[project_name_1]
                                         .profile_dict[profile_tup[0]])
            transformFilter.Update()
            mapper = vtk.vtkPolyDataMapper()
            flattener = vtk.vtkTransform()
            flattener.Scale(1, 1, 0)
            flatFilter = vtk.vtkTransformPolyDataFilter()
            flatFilter.SetTransform(flattener)
            flatFilter.SetInputData(transformFilter.GetOutput())
            flatFilter.Update()
            mapper.SetInputData(flatFilter.GetOutput())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            if len(profile_tup)>=2:
                actor.GetProperty().SetLineWidth(profile_tup[1])
            else:
                actor.GetProperty().SetLineWidth(5)
            if len(profile_tup)>=5:
                actor.GetProperty().SetColor(profile_tup[2:5])
            else:
                actor.GetProperty().SetColor(0.0, 1.0, 0.0)
            if len(profile_tup)>=6:
                actor.GetProperty().SetOpacity(profile_tup[5])
            else:
                actor.GetProperty().SetOpacity(0.8)
            actor.GetProperty().RenderLinesAsTubesOn()
            renderer.AddActor(actor)
        
        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetLookupTable(mplcmap_to_vtkLUT(-diff_window, diff_window,
                                                   name=cmap))
        renderer.AddActor2D(scalarBar)
        
        renderWindow.AddRenderer(renderer)
        
        # create a renderwindowinteractor
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renderWindow)
        
        iren.Initialize()
        renderWindow.Render()
        iren.AddObserver('UserEvent', cameraCallback)
        iren.Start()
    
    def display_warp_difference(self, project_name_0, project_name_1, 
                                diff_window, field='Elevation_mean_fill',
                                cmap='rainbow', profile_list=[], 
                                show_scanners=False, scanner_color_0='Yellow', 
                                scanner_color_1='Fuchsia', scanner_length=150,
                                key=''):
        """
        Display the surface of the image from project_name_1 colored by diff.

        Parameters
        ----------
        project_name_0 : str
            Name of project to subtract (usually older).
        project_name_1 : str
            Name of project to subtract from (usually younger).
        diff_window : float
            Scale of differences to display in color in m.
        field : str, optional
            The field in PointData of project_name_1 to use as warped scalars.
            The default is 'Elevation_mean_fill'
        cmap : str, optional
            Name of matplotlib colormap to use. The default is 'rainbow'
        profile_list : list, optional
            List of keys of profiles in the profiles dict to display. The 
            default is [].
        show_scanners : bool, optional
            Whether or not to show the scanners. The default is False.
        scanner_color_0 : str, optional
            Name of the color to display project 0 scanners.
             The default is 'Yellow'
        scanner_color_1 : str, optional
            Name of the color to display project 0 scanners.
             The default is 'Fuchsia'
        scanner_length : float, optional
            Length of the ray indicating the scanner's start orientation in m.
            The default is 150
        key : str, optional
            Key for images in image_dicts. Must be the same for both projects.
            The default is ''.

        Returns
        -------
        None.

        """
        # Define function for writing the camera position and focal point to
        # std out when the user presses 'u'
        def cameraCallback(obj, event):
            print("Camera Pos: " + str(obj.GetRenderWindow().
                                           GetRenderers().GetFirstRenderer().
                                           GetActiveCamera().GetPosition()))
            print("Focal Point: " + str(obj.GetRenderWindow().
                                            GetRenderers().GetFirstRenderer().
                                            GetActiveCamera().GetFocalPoint()))
            print("Roll: " + str(obj.GetRenderWindow().
                                            GetRenderers().GetFirstRenderer().
                                            GetActiveCamera().GetRoll()))
        
        # Merge filter combines the geometry from project_name_1 with scalars
        # from difference
        # merge = vtk.vtkMergeFilter()
        # merge.SetGeometryInputData(self.project_dict[project_name_1].
        #                            get_image(field=field, warp_scalars=False,
        #                                      nan_value=-5))
        # im = vtk.vtkImageData()
        # im.DeepCopy(self.difference_dict[(project_name_0, project_name_1)])
        
        im = vtk.vtkImageData()
        im.DeepCopy(self.project_dict[project_name_1].
                                    get_image(field=field, warp_scalars=False,
                                              nan_value=-5, key=key))
        im.GetPointData().SetActiveScalars(field)
        
        im.GetPointData().AddArray(self.difference_dict[(project_name_0, 
                                                         project_name_1, key)]
                                   .GetPointData().GetArray('Difference'))
        field_np = vtk_to_numpy(im.GetPointData().GetArray('Difference'))
        field_np[np.isnan(field_np)] = 0
        field_np = vtk_to_numpy(im.GetPointData().GetArray(field))
        field_np[np.isnan(field_np)] = -5
        im.Modified()
                
        # merge.SetScalarsData(im)
        # merge.Update()
        
        #merge.GetOutput().GetPointData().SetActiveScalars(field)
        geometry = vtk.vtkImageDataGeometryFilter()
        geometry.SetInputData(im)
        geometry.SetThresholdValue(-4.9)
        geometry.ThresholdCellsOn()
        geometry.Update()
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(geometry.GetOutput())
        tri.Update()
        strip = vtk.vtkStripper()
        strip.SetInputData(tri.GetOutput())
        strip.Update()
        warp = vtk.vtkWarpScalar()
        warp.SetScaleFactor(1)
        warp.SetInputData(strip.GetOutput())
        warp.Update()
        #return warp.GetOutput()
        # Compute normals to make the shading look better
        normals = vtk.vtkPPolyDataNormals()
        normals.SetInputData(warp.GetOutput())
        normals.Update()
        normals.GetOutput().GetPointData().SetActiveScalars('Difference')
        
        
        mapper = vtk.vtkPolyDataMapper()
        #mapper = vtk.vtkDataSetMapper()
        #mapper.SetInputData(merge.GetOutput())
        mapper.SetInputData(normals.GetOutput())
        mapper.SetLookupTable(mplcmap_to_vtkLUT(-diff_window, diff_window,
                                                name=cmap))
        mapper.SetScalarRange(-diff_window, diff_window)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderer.AddActor(actor)

        # Add scanners if requested
        if show_scanners:
            for project_name, color in zip([project_name_0, project_name_1],
                                           [scanner_color_0, scanner_color_1]):
                project = self.project_dict[project_name]
                for scan_name in project.scan_dict:
                    project.scan_dict[scan_name].create_scanner_actor(
                        color=color, length=scanner_length)
                    # copy the userTransform and concatenate the imageTransform
                    transform = vtk.vtkTransform()
                    transform.DeepCopy(project.scan_dict[scan_name].scannerActor
                                       .GetUserTransform())
                    transform.Concatenate(project.imageTransform)
                    project.scan_dict[scan_name].scannerActor.SetUserTransform(
                        transform)
                    project.scan_dict[scan_name].scannerText.AddPosition(
                        project.image_transform_dict[key].GetPosition())
                    renderer.AddActor(project.scan_dict[scan_name].scannerActor)
                    renderer.AddActor(project.scan_dict[scan_name].scannerText)

        # Add requested profiles
        for profile_tup in profile_list:
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(self.project_dict[project_name_1]
                                         .image_transform_dict[key])
            transformFilter.SetInputData(self.project_dict[project_name_1]
                                         .profile_dict[profile_tup[0]])
            transformFilter.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(transformFilter.GetOutput())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            if len(profile_tup)>=2:
                actor.GetProperty().SetLineWidth(profile_tup[1])
            else:
                actor.GetProperty().SetLineWidth(5)
            if len(profile_tup)>=5:
                actor.GetProperty().SetColor(profile_tup[2:5])
            else:
                actor.GetProperty().SetColor(0.0, 1.0, 0.0)
            if len(profile_tup)>=6:
                actor.GetProperty().SetOpacity(profile_tup[5])
            else:
                actor.GetProperty().SetOpacity(0.8)
            actor.GetProperty().RenderLinesAsTubesOn()
            renderer.AddActor(actor)
        
        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetLookupTable(mplcmap_to_vtkLUT(-diff_window, diff_window,
                                                   name=cmap))
        renderer.AddActor2D(scalarBar)
        
        renderWindow.AddRenderer(renderer)
        
        # create a renderwindowinteractor
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renderWindow)
        iren.Initialize()
        renderWindow.Render()

        # Set camera for followers
        if show_scanners:
            for project_name in [project_name_0, project_name_1]:
                project = self.project_dict[project_name]
                for scan_name in project.scan_dict:
                    project.scan_dict[scan_name].scannerText.SetCamera(
                        renderer.GetActiveCamera())

        iren.AddObserver('UserEvent', cameraCallback)
        iren.Start()
        
    def write_plot_warp_difference(self, project_name_0, project_name_1, 
                                diff_window, camera_position, focal_point,
                                roll=0,
                                field='Elevation',
                                cmap='RdBu_r', filename="", name="",
                                light=None, colorbar=True, profile_list=[],
                                window_size=(2000, 1000), key=''):
        """
        

        Parameters
        ----------
        project_name_0 : str
            Name of project to subtract (usually older).
        project_name_1 : str
            Name of project to subtract from (usually younger).
        diff_window : float
            Scale of differences to display in color in m.
        camera_position : TYPE
            DESCRIPTION.
        focal_point : TYPE
            DESCRIPTION.
        roll : float, optional
            Camera roll angle in degrees. The default is 0.
        field : str, optional
            DESCRIPTION. The default is 'Elevation_mean_fill'.
        cmap : str, optional
            matplotlib colormap to use for differences. The default is 
            'rainbow'.
        filename : str, optional
            full filename (w/ path) to write to. If empty will be placed
            in snapshots folder. The default is "".
        name : str, optional
            string to append to filename if filename is empty. The default
            is ''.
        light : vtkLight, optional
            If desired add a light to the scene to replace the default vtk
            lighting. The default is None
        colorbar : bool, optional
            Whether to render a colorbar. The default is True
        key : str, optional
            Key for images in image_dicts. Must be the same for both projects.
            The default is ''.

        Returns
        -------
        None.

        """
        # Merge filter combines the geometry from project_name_1 with scalars
        # from difference
        # merge = vtk.vtkMergeFilter()
        # merge.SetGeometryInputData(self.project_dict[project_name_1].
        #                            get_image(field=field, warp_scalars=True))
        # merge.SetScalarsData(self.difference_dict[(project_name_0,
        #                                           project_name_1)])
        # merge.Update()
        
        im = vtk.vtkImageData()
        im.DeepCopy(self.project_dict[project_name_1].
                                    get_image(field=field, warp_scalars=False,
                                              nan_value=-5, key=key))
        im.GetPointData().SetActiveScalars(field)
        
        im.GetPointData().AddArray(self.difference_dict[(project_name_0, 
                                                         project_name_1, key)]
                                   .GetPointData().GetArray('Difference'))
        field_np = vtk_to_numpy(im.GetPointData().GetArray('Difference'))
        field_np[np.isnan(field_np)] = 0
        field_np = vtk_to_numpy(im.GetPointData().GetArray(field))
        field_np[np.isnan(field_np)] = -5
        im.Modified()
                
        # merge.SetScalarsData(im)
        # merge.Update()
        
        #merge.GetOutput().GetPointData().SetActiveScalars(field)
        geometry = vtk.vtkImageDataGeometryFilter()
        geometry.SetInputData(im)
        geometry.SetThresholdValue(-4.9)
        geometry.ThresholdCellsOn()
        geometry.Update()
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(geometry.GetOutput())
        tri.Update()
        strip = vtk.vtkStripper()
        strip.SetInputData(tri.GetOutput())
        strip.Update()
        warp = vtk.vtkWarpScalar()
        warp.SetScaleFactor(1)
        warp.SetInputData(strip.GetOutput())
        warp.Update()
        #return warp.GetOutput()
        # Compute normals to make the shading look better
        normals = vtk.vtkPPolyDataNormals()
        normals.SetInputData(warp.GetOutput())
        normals.Update()
        normals.GetOutput().GetPointData().SetActiveScalars('Difference')
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(normals.GetOutput())
        mapper.SetLookupTable(mplcmap_to_vtkLUT(-diff_window, diff_window,
                                                name=cmap))
        mapper.SetScalarRange(-diff_window, diff_window)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetSize(window_size[0], window_size[1])
        renderer.AddActor(actor)

        # Add requested profiles
        for profile_tup in profile_list:
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(self.project_dict[project_name_1]
                                         .image_transform_dict[key])
            transformFilter.SetInputData(self.project_dict[project_name_1]
                                         .profile_dict[profile_tup[0]])
            transformFilter.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(transformFilter.GetOutput())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            if len(profile_tup)>=2:
                actor.GetProperty().SetLineWidth(profile_tup[1])
            else:
                actor.GetProperty().SetLineWidth(5)
            if len(profile_tup)>=5:
                actor.GetProperty().SetColor(profile_tup[2:5])
            else:
                actor.GetProperty().SetColor(0.0, 1.0, 0.0)
            if len(profile_tup)>=6:
                actor.GetProperty().SetOpacity(profile_tup[5])
            else:
                actor.GetProperty().SetOpacity(0.8)
            actor.GetProperty().RenderLinesAsTubesOn()
            renderer.AddActor(actor)

        if light:
            renderer.AddLight(light)
        
        if colorbar:
            scalarBar = vtk.vtkScalarBarActor()
            scalarBar.SetLookupTable(mplcmap_to_vtkLUT(-diff_window, 
                                                       diff_window, name=cmap))
            renderer.AddActor2D(scalarBar)
        
        renderWindow.AddRenderer(renderer)
        
        # Create Camera
        camera = vtk.vtkCamera()
        camera.SetFocalPoint(focal_point)
        camera.SetPosition(camera_position)
        camera.SetViewUp(0, 0, 1)
        camera.SetRoll(roll)
        renderer.SetActiveCamera(camera)
        
        renderWindow.Render()
        # Screenshot image to save
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(renderWindow)
        w2if.Update()
    
        writer = vtk.vtkPNGWriter()
        if filename=="":
            writer.SetFileName(os.path.join(self.project_path, 'snapshots', 
                               project_name_0 + "_" + project_name_1 + 
                               'warp_difference_' + name + '.png'))
        else:
            writer.SetFileName(filename)
        
        writer.SetInputData(w2if.GetOutput())
        writer.Write()
    
        renderWindow.Finalize()
        del renderWindow
        
    def write_plot_difference_projects(self, project_name_0, project_name_1, 
                                 diff_window, filename="", colorbar=True,
                                 key=''):
        """
        Display a plot showing the difference between two projects

        Parameters
        ----------
        project_name_0 : str
            Name of project to subtract (usually older).
        project_name_1 : str
            Name of project to subtract from (usually younger).
        diff_window : float
            Scale of differences to display in color in m.
        filename : str
            Path and name of file to write, defaults to snapshots folder if ""
        key : str, optional
            Key for images in image_dicts. Must be the same for both projects.
            The default is ''.

        Returns
        -------
        None.

        """
        
        # Create mapper
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(self.difference_dict[(project_name_0, 
                                                  project_name_1, key)])
        mapper.SetLookupTable(mplcmap_to_vtkLUT(-1*diff_window, diff_window))
        mapper.SetScalarRange(-1*diff_window, diff_window)
        
        # Create Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Create Renderer
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        
        # Create Colorbar
        if colorbar:
            scalarBar = vtk.vtkScalarBarActor()
            scalarBar.SetLookupTable(mplcmap_to_vtkLUT(-1*diff_window,
                                                       diff_window))
            renderer.AddActor2D(scalarBar)
            
        # Add TExt
        textActor = vtk.vtkTextActor()
        textActor.SetInput("Change from " + project_name_0 + " to " 
                           + project_name_1)
        textActor.SetPosition2(10, 40)
        textActor.SetTextScaleModeToNone()
        textActor.GetTextProperty().SetFontSize(40)
        textActor.GetTextProperty().SetColor(0.0, 1.0, 0.0)
        renderer.AddActor2D(textActor)
        
        # Create Render window
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetSize(1500, 1500)
        renderWindow.AddRenderer(renderer)
        renderWindow.Render()
        
        # Screenshot image to save
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(renderWindow)
        w2if.Update()
    
        writer = vtk.vtkPNGWriter()
        if filename=="":
            writer.SetFileName(os.path.join(self.project_path, 'snapshots', 
                               project_name_0 + "_" + project_name_1 + '.png'))
        else:
            writer.SetFileName(filename)
        
        writer.SetInputData(w2if.GetOutput())
        writer.Write()
    
        renderWindow.Finalize()
        del renderWindow

    def difference_maxes(self, project_name_0, project_name_1, r_pair):
        """
        Find pairs of local maxes and store their differences.

        Each pair of points that are within r_pair of each other is
        stored in a PolyData as the two points and a line connecting
        them. The project_name_0 point is always first. The z difference
        and z_sigma (quadrature sum of individual z_sigmas) are also stored
        in the CellData attribute.
        
        Parameters
        ----------
        project_name_0 : str
            Name of project to subtract (usually older).
        project_name_1 : str
            Name of project to subtract from (usually younger).
        r_pair : float
            Maximum horizontal distance in m that local maxima may be
            separated to be considered same point.

        Returns
        -------
        None.

        """

        # Get points and relevant arrays
        pts_np_0 = vtk_to_numpy(self.project_dict[project_name_0].pdata_dict
                                ['local_max'].GetPoints().GetData())

        z_sigma_np_0 = vtk_to_numpy(self.project_dict[project_name_0]
                                    .pdata_dict['local_max'].GetPointData()
                                    .GetArray('z_sigma'))

        pts_np_1 = vtk_to_numpy(self.project_dict[project_name_1].pdata_dict
                                ['local_max'].GetPoints().GetData())
        z_sigma_np_1 = vtk_to_numpy(self.project_dict[project_name_1]
                                    .pdata_dict['local_max'].GetPointData()
                                    .GetArray('z_sigma'))
        
        # Get pairs of closest points within r_pair distance
        tree_0 = KDTree(pts_np_0[:,:2])
        tree_1 = KDTree(pts_np_1[:,:2])
        closest = tree_0.query_ball_tree(tree_1, r_pair)
        # Turn closest into pairs of indices, for loop is probably only way
        pairs = []
        for i in range(len(closest)):
            if len(closest[i])==0:
                continue
            elif len(closest[i])==1:
                pairs.append([i, closest[i][0]])
            else:
                raise RuntimeError('Too many closest at: ' + str(i))
        
        # Create a VTK polydata of line segments
        pdata = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        # Allocate arrays
        max_id_np = np.empty(2*len(pairs), dtype=np.uint32)
        z_sigma_np_pts = np.empty(2*len(pairs), dtype=float)
        z_diff_np = np.empty(len(pairs), dtype=float)
        z_sigma_np = np.empty(len(pairs), dtype=float)
        pts.SetNumberOfPoints(2*len(pairs))
        # For each pair of local maxima add points and line to objects
        for i in range(len(pairs)):
            pts.SetPoint(2*i, pts_np_0[pairs[i][0],:])
            pts.SetPoint(2*i + 1, pts_np_1[pairs[i][1],:])
            max_id_np[2*i] = pairs[i][0]
            max_id_np[2*i + 1] = pairs[i][1]
            z_sigma_np_pts[2*i] = z_sigma_np_0[pairs[i][0]]
            z_sigma_np_pts[2*i + 1] = z_sigma_np_1[pairs[i][1]]
            lines.InsertNextCell(2)
            lines.InsertCellPoint(2*i)
            lines.InsertCellPoint(2*i + 1)
            z_diff_np[i] = pts_np_1[pairs[i][1],2] - pts_np_0[pairs[i][0],2]
            z_sigma_np[i] = np.sqrt(z_sigma_np_1[pairs[i][1]]**2 
                                 + z_sigma_np_0[pairs[i][0]]**2)
        # Finish assembling polydata
        pdata.SetPoints(pts)
        pdata.SetLines(lines)
        arr = numpy_to_vtk(max_id_np, deep=True, 
                           array_type=vtk.VTK_UNSIGNED_INT)
        arr.SetName('max_id')
        pdata.GetPointData().AddArray(arr)
        arr = numpy_to_vtk(z_diff_np, deep=True, array_type=vtk.VTK_DOUBLE)
        arr.SetName('z_diff')
        pdata.GetCellData().AddArray(arr)
        pdata.GetCellData().SetScalars(arr)
        arr = numpy_to_vtk(z_sigma_np, deep=True, array_type=vtk.VTK_DOUBLE)
        arr.SetName('z_sigma')
        pdata.GetCellData().AddArray(arr)
        arr = numpy_to_vtk(z_sigma_np_pts, deep=True, array_type=vtk.VTK_DOUBLE)
        arr.SetName('z_sigma')
        pdata.GetPointData().AddArray(arr)
        
        pdata.Modified()
        self.max_difference_dict[(project_name_0, project_name_1)] = pdata

class Classifier:
    """
    Manage multiple scans from the same area.
    
    ...
    
    Attributes
    ----------
    df_labeled : pandas dataframe
        A dataframe where each row is a classified point. Must contain column
        'Classification'.
    classifier : object
        A classifier, initially probably 
        sklearn.ensemble.RandomForestClassifer
    feature_list : list
        List of features used to train classifier. Need to save for prediction
        step.
        
    Methods
    -------
    init_randomforest(n_jobs, **kwargs)
        Creates random forest classifier
    train_classifier()
        Trains classifier, does some decimation of ground category.
    classify_pdata()
        Updates Classification field in pdata.
    
    """
    
    def __init__(self, df_labeled=pd.DataFrame()):
        """
        Create object, can add df_labeled at this point if desired

        Parameters
        ----------
        df_labeled : pandas dataframe, optional
            A dataframe where each row is a classified point. Must contain 
            column 'Classification'. If none is provided an empty one will be 
            created. The default is pd.DataFrame().

        Returns
        -------
        None.

        """
        
        self.df_labeled=df_labeled
    
    def init_randomforest(self, n_jobs=-1, **kwargs):
        """
        Initialize a RandomForestClassifer.

        Parameters
        ----------
        n_jobs : int, optional
            Number of . The default is -1.
        **kwargs : dict
            Additional keyword arguments to RandomForestClassifier.

        Returns
        -------
        None.

        """
        
        self.classifier = RandomForestClassifier(n_jobs=n_jobs, **kwargs)
    
    def init_histgradboost(self, **kwargs):
        """
        Initialize a HistGradientBoostingClassifier

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to HistGradientBoostingClassifer
            

        Returns
        -------
        None.

        """
        self.classifier = HistGradientBoostingClassifier(**kwargs)
    
    def train_classifier(self, feature_list, n_ground_multiplier=3, 
                         strat_by_class=True, reduce_ground=True, 
                         surface_only=False, **kwargs):
        """
        Train classifier, kwargs go to train_test_split

        Parameters
        ----------
        feature_list : list
            List of feature names to use as independent variables. Must be
            columns in df_labeled.
        n_ground_multiplier : float
            We have many more labeled ground points than others. So that
            training doesn't take forever we only take n times as many ground
            points as all other classes combined. The default is 3.
        strat_by_class : bool, optional
            Whether or not to stratify the split by 'Classification'. The 
            default is True.
        reduce_ground : bool, optional
            Whether to reduce the number of ground points, the default
            True.
        surface_only : bool, optional
            If true just perform binary classification into snow surface
            or not (lump roads and ground together). The default is False.
        **kwargs : dict
            Additional kwargs for train_test_split.

        Returns
        -------
        None.

        """
        
        self.feature_list = feature_list
        
        # Downsample ground points
        if reduce_ground:
            n = int(n_ground_multiplier 
                    * (self.df_labeled['Classification']!=2).sum())
            df_sub_list = []
            for category in self.df_labeled['Classification'].unique():
                if category==2:
                    df_sub_list.append(self.df_labeled[self.df_labeled[
                        'Classification']==category].sample(n=n))
                else:
                    df_sub_list.append(self.df_labeled[
                        self.df_labeled['Classification']==category])
            df_sub = pd.concat(df_sub_list)
        else:
            df_sub = copy.deepcopy(self.df_labeled)
        if surface_only:
            df_sub['Classification'] = (1 + np.isin(df_sub['Classification'
                                                           ].values, [2, 70]
                                                    ).astype(np.int8))
        
        # Split labeled data
        if strat_by_class:
            X_train, X_test, y_train, y_test = train_test_split(df_sub[
                feature_list], df_sub.Classification, 
                stratify=df_sub.Classification, **kwargs)
        else:
            X_train, X_test, y_train, y_test = train_test_split(df_sub[
                feature_list], df_sub.Classification, **kwargs)
        
        nonground_weight_multiplier = (y_train==2).sum()/((y_train==1).sum())
        weights = np.ones(y_train.shape)
        weights[y_train==1] = nonground_weight_multiplier
        
        # Train Classifier
        self.classifier.fit(X_train, y_train)
    
    def classify_pdata(self, pdata):
        """
        Apply classifier to classify points in the pdata. Pdata's PointData
        must contain all fields in feature_list

        Parameters
        ----------
        pdata : vtkPolyData
            vtkPolyData object to classify points in.

        Returns
        -------
        None.

        """
        
        X_points = np.zeros((pdata.GetNumberOfPoints(), 
                             len(self.feature_list)), dtype=np.float32)
        dsa_pdata = dsa.WrapDataObject(pdata)
        for i in range(len(self.feature_list)):
            X_points[:, i] = dsa_pdata.PointData[self.feature_list[i]]
        dsa_pdata.PointData['Classification'][:] = self.classifier.predict(
            X_points)
        pdata.Modified()

def mplcmap_to_vtkLUT(vmin, vmax, name='rainbow', N=256, color_under='fuchsia', 
                      color_over='white'):
    """
    Create a vtkLookupTable from a matplotlib colormap.

    Parameters
    ----------
    vmin : float
        Minimum value for colormap.
    vmax : float
        Maximum value for colormap.
    name : str, optional
        Matplotlib name of the colormap. The default is 'rainbow'
    N : int, optional
        Number of levels in colormap. The default is 256.
    color_under : str, optional
        Color for values less than vmin, should be in vtkNamedColors. 
        The default is 'fuchsia'.
    color_over : str, optional
        Color for values greater than vmax, should be in vtkNamedColors. 
        The default is 'white'.

    Returns
    -------
    vtkLookupTable

    """

    # Pull the matplotlib colormap
    mpl_cmap = cm.get_cmap(name, N)
    
    # Create Lookup Table
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(vmin, vmax)
    
    # Add Colors from mpl colormap
    lut.SetNumberOfTableValues(N)
    for (i, c) in zip(np.arange(N), mpl_cmap(range(N))):
        lut.SetTableValue(i, c)

    # Add above and below range colors
    nc = vtk.vtkNamedColors()
    lut.SetBelowRangeColor(nc.GetColor4d(color_under))
    lut.SetAboveRangeColor(nc.GetColor4d(color_over))

    lut.Build()
    
    return lut

def mosaic_date_parser(project_name):
    """
    Parses the project name into a date.

    Parameters
    ----------
    project_name : str
        The project name as a string.

    Returns
    -------
    The date and time (for April 8 b)

    """
    
    # The date is always a sequence of 6 numbers in the filename.
    seq = re.compile('[0-9]{6}')
    seq_match = seq.search(project_name)
    
    if seq_match:
        year = '20' + seq_match[0][-2:]
        if year == '2020':
            date = year + '-' + seq_match[0][2:4] + '-' + seq_match[0][:2]
        elif year == '2019':
            date = year + '-' + seq_match[0][:2] + '-' + seq_match[0][2:4]
        elif year == '2022':
            date = year + '-' + seq_match[0][2:4] + '-' + seq_match[0][:2]
    else:
        return None
    
    # Looks like Dec 6 was an exception to the format...
    if date=='2019-06-12':
        date = '2019-12-06'
    
    # Handle April 8 b scan case.
    if project_name[seq_match.end()]=='b':
        date = date + ' 12:00:00'
        
    return date

def sample_autocorr_2d(samples_in, mode='normal', nanmode='zero-fill'):
    """
    Returns autocorrelation fn for evenly spaced 2D sample
    
    Parameters
    ----------
    samples_in : 2D array
        2D array of samples
    mode : str, optional
        {'normal', 'fourier', 'radial'} specifies whether the indices should
        be normal (0,0 in middle) or fourier 0-> positive lags
        then negative lags -> 0. The default is 'normal'
    nanmode : str, optional
        if 'zero-fill' if the input contains nan's, zero fill them and adjust
        output by multiplicative factor  from Handbook of Spatial Statistics
        pg. 71
    
    Returns
    -------
    ndarray
        empirical autocorrelation, lag spacing is same as sample spacing
        
    Raises
    ------
        TypeError: samples must be 2D
        ValueError: mode must be either 'normal', 'fourier' or 'radial'
    """
    
    # Input checking
    if not len(samples_in.shape)==2:
        raise TypeError("Samples must be 2D")
    
    # Handle nan values
    # Following Handbook of Spatial Statistics, pg. 71
    samples = copy.deepcopy(samples_in)
    if nanmode=='zero-fill':
        n_values = (~np.isnan(samples)).sum()
        samples[np.isnan(samples)] = 0 # zero-fill as it says on the bottom
        # of pg. 71
    else:
        n_values = samples.size
    
    # Compute autocorrelation with fft convolve
    con = fftconvolve(samples, np.flip(samples), mode='full')
    counts = fftconvolve(np.ones(samples.shape), np.ones(samples.shape), 
                         mode='full')
    # Return autocorrelation
    # dividing by n_values normalizes for missing data
    if mode=='normal':
        return (samples.size/n_values) * con/counts
    elif mode=='fourier':
        return (samples.size/n_values) * np.fft.ifftshift(con/counts)
    elif mode=='radial':
        x = np.concatenate((np.arange(-1*samples.shape[1] + 1, 0),
                            np.arange(samples.shape[1])))
        y = np.concatenate((np.arange(-1*samples.shape[0] + 1, 0),
                            np.arange(samples.shape[0])))
        X = np.tile(x[np.newaxis, :], (y.size, 1))
        Y = np.tile(y[:, np.newaxis], (1, x.size))
        r = np.hypot(X, Y)
        rbin = np.around(r, decimals=0)
        corr = (samples.size/n_values) * ndimage.mean(con/counts, labels=rbin, 
                            index=np.arange(samples.shape[0]))
        # In addition to the pairwise correlation in each bin, we also want
        # the centroid of the pairwise distances, this is especially important
        # for close to the origin.
        r_radial = ndimage.mean(r, labels=rbin, 
                                index=np.arange(samples.shape[0]))
        return (r_radial, corr)
    else:
        raise ValueError("mode must be either 'normal', 'fourier' or 'radial'")

def biquad_plot(ax, arr, bins_0, bins_1, dx0=1, dx1=1, mode='inverse',
                vmin=None, vmax=None):
    """Plots center of a 2D array that is fourier indexed.
    
    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        axis object to plot on
    arr : 2D array
        array to plot, can be either full or biquad
    bins_0 : int
        number of positive and negative bins along zeroth axis to 
        show
    bins_1 : int
        number of positive bins along first axis
    dx0 : float, optional
        pixel length along axis 0. The default is 1.
    dx1 : float, optional
        pixel length along axis 1. The default is 1.
    mode : str, optional
        Which mode for bin units. The default is 'inverse'
    vmin, vmax : float, optional
        Limits for color plotting. The default is None.
    
    Returns
    -------
    pyplot mappable
        plot object to generate colorbar with
    """
    # Pull requested bins from arr
    im = np.vstack((arr[-1*bins_0:,:bins_1],
                   arr[:bins_0, :bins_1]))
    if mode=='inverse':
        ax0 = (np.arange(bins_0*2) - bins_0)*1/dx0
        ax1 = (np.arange(bins_1)) * 1/dx1
    elif mode=='direct':
        ax0 = (np.arange(bins_0*2) - bins_0) * dx0
        ax1 = (np.arange(bins_1)) * dx1
    # Plot and return plot object
    return ax.contourf(ax1, ax0, im, levels=10, vmin=vmin, vmax=vmax)

def biquad_difference_plot(ax, arr0, arr1, bins_0, bins_1, dx0=1, dx1=1):
    """Plots difference of 2 2D arrays that are fourier indexed.
    
    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        axis object to plot on
    arr0 : 2D array
        array to difference, can be either full or biquad
    arr1 : 2d array
        array to difference, can be either full or biquad
    bins_0 : int
        number of positive and negative freqency bins along zeroth axis to 
        show
    bins_1 : int
        number of positive bins along first axis
    dx0 : float, optional
        pixel length along axis 0. The default is 1.
    dx1 : float, optional
        pixel length along axis 1. The default is 1.
        
    Returns
    -------
    pyplot mappable
        plot object to generate colorbar with
    """
    # Pull requested bins from arr
    im = (np.vstack((arr1[-1*bins_0:,:bins_1],
                   arr1[:bins_0, :bins_1]))
          - np.vstack((arr0[-1*bins_0:,:bins_1],
                      arr0[:bins_0, :bins_1])))
    nu0 = (np.arange(bins_0*2) - bins_0)*1/dx0
    nu1 = (np.arange(bins_1)) * 1/dx1
    # Plot and return plot object
    print(np.sqrt((im**2).mean()))
    return ax.contourf(nu1, nu0, im)

def radial_spacing(r, dtheta=0.025, dphi=0.025, scanner_height=2.5, slope=0):
    """
    Computes expected spacing between lidar points at distances r.

    Parameters
    ----------
    r : ndarray
        (n,) Array of distances to compute spacing for.
    dtheta : float, optional
        Step size in degrees for lidar inclination angle. The default is 0.025.
    dphi : float, optional
        Step size in degrees for lidar azimuth angle. The default is 0.025.
    scanner_height : float, optional
        Height of the scanner above ground in m. The default is 2.5.
    slope : float or ndarray, optional
        Slope of the surface in degrees in radial direction. If array it must
        have the same dimensions as r. The default is 0.

    Returns
    -------
    (n, 2) Array of spacings at each distance in azimuthal and radial dirs.

    """
    
    # Create output array
    spac = np.zeros((r.size, 2))
    
    # Compute azimuthal spacing
    r_g = np.sqrt(r**2 - scanner_height**2)
    spac[:,0] = r_g * dphi * np.pi / 180
    
    # Compute radial spacing
    dtheta_rad = dtheta * np.pi / 180
    theta = np.arccos(scanner_height/r)
    r_g_prime = scanner_height * np.tan(theta - dtheta_rad)
    if slope==0:
        spac[:,1] = r_g - r_g_prime
    else:
        dx = r_g - r_g_prime
    
    return spac

def matern_semivariance(theta_1, theta_2, h, nu):
    """
    Computes the semivariance at lag h for C(0)=theta_1 and lengthscale theta_2
    
    From Gelfand et al. Handbook of Spatial Statistics pg. 37
    
    Parameters:
    -----------
    theta_1 : array
        Array of variances, must be same size as theta_2.
    theta_2 : array
        Array of lengthscales, must be same size as theta_1.
    h : float
        Lag distance at which to compute semivariance.
    nu : float
        Matern nu.

    Returns:
    --------
        Array of semivariances at lag distance h, for all theta_1, theta_2

    """

    if nu==0.5:
        return theta_1 * (1 - np.exp(-h/theta_2))
    else:
        raise NotImplementedError()

##### Functions that require cython_util

def gridded_counts_mins(points, edges, init_val=np.float32(1000)):
    """
    Grids a point cloud in x and y and returns the cellwise counts and min z

    Parameters
    ----------
    points : float[:, :]
        Pointcloud, Nx3 array of type np.float32
    edges : list
        2 item list containing edges for gridding
    init_val : float
        Initial values in mins array. Pick something larger than largest z
        in Points. In mins output, all bins that don't have any points in them
        will take on this value.

    Returns:
    --------
    counts : long[:, :]
        Gridded array with the counts for each bin
    mins : float[:, :]
        Gridded array with the min z value for each bin.
    """

    Ncount = tuple(np.searchsorted(edges[i], points[:,i], 
                               side='right') for i in range(2))

    nbin = np.empty(2, np.int_)
    nbin[0] = len(edges[0]) + 1
    nbin[1] = len(edges[1]) + 1

    xy = np.ravel_multi_index(Ncount, nbin)

    # Compute gridded mins and counts
    counts, mins = cython_util.create_counts_mins_cy(nbin[0], nbin[1],
                                                     points,
                                                     np.int_(xy), init_val)
    counts = counts.reshape(nbin)
    mins = mins.reshape(nbin)
    core = 2*(slice(1, -1),)
    counts = counts[core]
    mins = mins[core]
    mins[mins==init_val] = np.nan

    return counts, mins

def gridded_counts_means(points, edges):
    """
    Grids a point cloud in x and y and returns the cellwise counts and mean z

    Parameters
    ----------
    points : float[:, :]
        Pointcloud, Nx3 array of type np.float32
    edges : list
        2 item list containing edges for gridding

    Returns:
    --------
    counts : long[:, :]
        Gridded array with the counts for each bin
    means : float[:, :]
        Gridded array with the mean z value for each bin.
    """

    Ncount = tuple(np.searchsorted(edges[i], points[:,i], 
                               side='right') for i in range(2))

    nbin = np.empty(2, np.int_)
    nbin[0] = len(edges[0]) + 1
    nbin[1] = len(edges[1]) + 1

    xy = np.ravel_multi_index(Ncount, nbin)

    # Compute gridded mins and counts
    counts, sums = cython_util.create_counts_sums_cy(nbin[0], nbin[1],
                                                     points,
                                                     np.int_(xy))
    counts = counts.reshape(nbin)
    sums = sums.reshape(nbin)
    core = 2*(slice(1, -1),)
    counts = counts[core]
    sums = sums[core]
    means = sums
    means[counts==0] = np.nan
    means = means/counts

    return counts, means

def gridded_counts_means_vars(points, edges):
    """
    Grids a point could in x and y and returns the cellwise counts, means and
    variances.

    Parameters
    ----------
    points : float[:, :]
        Pointcloud, Nx3 array of type np.float32
    edges : list
        2 item list containing edges for gridding

    Returns:
    --------
    counts : long[:, :]
        Gridded array with the counts for each bin
    means : float[:, :]
        Gridded array with the mean z value for each bin.
    vars : float[:, :]
        Gridded array with the variance in z values for each bin.

    """

    Ncount = tuple(np.searchsorted(edges[i], points[:,i], 
                               side='right') for i in range(2))

    nbin = np.empty(2, np.int_)
    nbin[0] = len(edges[0]) + 1
    nbin[1] = len(edges[1]) + 1

    xy = np.ravel_multi_index(Ncount, nbin)

    # Compute gridded mins and counts
    counts, means, m2s = cython_util.create_counts_means_M2_cy(nbin[0], nbin[1],
                                                     points,
                                                     np.int_(xy))
    counts = counts.reshape(nbin)
    means = means.reshape(nbin)
    m2s = m2s.reshape(nbin)
    core = 2*(slice(1, -1),)
    counts = counts[core]
    means = means[core]
    m2s = m2s[core]
    means[counts==0] = np.nan
    m2s[counts==0] = np.nan
    var = m2s/counts

    return counts, means, var

def gridded_counts_modes(points, edges, dz_hist=0.01):
    """
    Grids a point cloud in x and y and returns the cellwise counts and mode z

    Parameters
    ----------
    points : float[:, :]
        Pointcloud, Nx3 array of type np.float32
    edges : list
        2 item list containing edges for gridding
    dz_hist : float, optional
        Bin width, in m for the z histogram (resolution of modal values)
        The default is 0.01.

    Returns:
    --------
    counts : long[:, :]
        Gridded array with the counts for each bin
    modes : float[:, :]
        Gridded array with the modal z value for each bin.
    """

    Ncount = tuple(np.searchsorted(edges[i], points[:,i], 
                               side='right') for i in range(2))

    nbin = np.empty(2, np.int_)
    nbin[0] = len(edges[0]) + 1
    nbin[1] = len(edges[1]) + 1

    xy = np.ravel_multi_index(Ncount, nbin)

    # Create histogram bins
    flr_min_z = np.floor(points.min(axis=0)[2])
    z_edges = np.arange(int(np.ceil((points.max(axis=0)[2] - flr_min_z)/dz_hist)
                            ) + 1, 
                         dtype=np.float32) * dz_hist + flr_min_z
    nbin_h = np.int_(len(z_edges) + 1)
    # modify z values of points array such that we get the z hist bin index
    h_ind = ((points[:, 2] - flr_min_z)/dz_hist).astype(np.int_)


    # Compute gridded hists and counts
    counts, hists = cython_util.create_counts_hists_cy(nbin[0], nbin[1], h_ind, 
                                                       np.int_(xy), nbin_h)
    counts = counts.reshape(nbin)
    # The argmax of the histogram is the index of the mode, z_edges[that index]
    # is the mode (or at least to the nearest lower edge)
    modes = z_edges[hists.argmax(axis=1)]
    modes = modes.reshape(nbin)
    core = 2*(slice(1, -1),)
    counts = counts[core]
    modes = modes[core]
    modes[counts==0] = np.nan

    return counts, modes

def gridded_max_ind(points, edges):
    """
    Grids a point cloud in x and y and returns the cellwise index of the point
    within that cell with the max z.

    Parameters
    ----------
    points : float[:, :]
        Pointcloud, Nx3 array of type np.float32
    edges : list
        2 item list containing edges for gridding

    Returns:
    --------
    inds : intptr_t[:]
        Array with the index of the maximum point for each bin. 
        Length nbin_0*nbin_1. For bins with no points in them this array will
        contain a value equal to the number of points.
    """

    Ncount = tuple(np.searchsorted(edges[i], points[:,i], 
                               side='right') for i in range(2))

    nbin = np.empty(2, np.int_)
    nbin[0] = len(edges[0]) + 1
    nbin[1] = len(edges[1]) + 1

    xy = np.ravel_multi_index(Ncount, nbin)

    # Compute gridded inds
    inds = cython_util.binwise_max_cy(nbin[0], nbin[1], points, np.int_(xy),
                                      init_val=points[:,2].min()-1)
    return inds

def z_alignment_ss(project_0, ss,
                              w0=10, w1=10, min_pt_dens=10, max_diff=0.15,
                              bin_reduc_op='min', return_grid=False):
        """
        Align successive scans on the basis of their gridded minima
        
        ss version is for looking at the change in just a singlescan.
        This version is for running outside of the scan_area object.

        !This function does not modify the tiepoint locations so it should 
        only be run after all tiepoint registration steps are done. It also
        requires that there hasn't been ice deformation and will try to not
        run if the fraction that changed by more than the diff cutoff exceeds
        frac_exceed_diff_cutoff.
        Parameters
        ----------
        project_0 : pydar.Project
            The project to align to.
        ss : pydar.SingleScan
            The singlescan to align
        w0 : float, optional
            Grid cell width in x dimension (m). The default is 10.
        w1 : float, optional
            Grid cell width in y dimension (m). The default is 10.
        min_pt_dens : float, optional
            minimum density of points/m^2 for us to compare grid cells from
            projects 0 and 1. The default is 30.
        max_diff : float, optional
            Maximum difference in minima to consider (higher values must be
            noise) in m. The default is 0.1.
        bin_reduc_op : str, optional
            What type of gridded reduction to apply. Options are 'min', 'mean'
            and 'mode'. The default is 'min'

        Returns
        -------
        diff : ndarray
            Array containing gridded minima differences

        """
        w = [w0, w1]
        min_counts = min_pt_dens * w[0] * w[1]

        # Get the merged points polydata, by not using port we should prevent
        # this memory allocation from persisting
        pdata_merged_project_0 = project_0.get_merged_points()
        project_0_points_np = vtk_to_numpy(pdata_merged_project_0
                                           .GetPoints().GetData())
        # Get points as an array
        ss_points_np = vtk_to_numpy(ss.currentFilter.GetOutput().GetPoints()
                                    .GetData())
        bounds = ss.currentFilter.GetOutput().GetBounds()
        
        # Create grid
        edges = 2*[None]
        nbin = np.empty(2, np.int_)

        for i in range(2):
            edges[i] = np.arange(int(np.ceil((bounds[2*i + 1] - 
                                              bounds[2*i])/w[i]))
                                 + 1, dtype=np.float32) * w[i] + bounds[2*i]
            # Adjust lower edge so we don't miss lower most point
            edges[i][0] = edges[i][0] - 0.0001
            # Adjust uppermost edge so the bin width is appropriate
            edges[i][-1] = bounds[2*i + 1] + 0.0001
            nbin[i] = len(edges[i]) + 1
        
        # Get gridded counts and reduc op (using the same grid)
        if bin_reduc_op=='min':
            ss_counts, ss_arr = gridded_counts_mins(
                ss_points_np, edges, np.float32(bounds[5] + 1))
        elif bin_reduc_op=='mean':
            ss_counts, ss_arr = gridded_counts_means(
                ss_points_np, edges)
        elif bin_reduc_op=='mode':
            ss_counts, ss_arr = gridded_counts_modes(
                ss_points_np, edges)
        
        # Get gridded counts and bin reduc
        if bin_reduc_op=='min':
            project_0_counts, project_0_arr = gridded_counts_mins(
                project_0_points_np, edges, np.float32(bounds[5] + 1))
        elif bin_reduc_op=='mean':
            project_0_counts, project_0_arr = gridded_counts_means(
                project_0_points_np, edges)
        elif bin_reduc_op=='mode':
            project_0_counts, project_0_arr = gridded_counts_modes(
                project_0_points_np, edges)
        else:
            raise ValueError('bin_reduc_op must be min, mean or mode')
        

        # Compute differences of gridded minima
        diff = ss_arr - project_0_arr
        diff[project_0_counts < min_counts] = np.nan
        diff[ss_counts < min_counts] = np.nan
        if np.isnan(diff).all():
            frac_exceed_max_diff = np.nan
            warnings.warn("No overlap for " + ss.scan_name +
                          " set z_offset to 0", UserWarning)
        else:
            frac_exceed_max_diff = ((np.abs(diff) > max_diff).sum()
                                    /np.logical_not(np.isnan(diff)).sum())
            diff[np.abs(diff) > max_diff] = np.nan
            if np.isnan(diff).all():
                warnings.warn("All diffs exceed max_diff")
        
        # If we want to return the grid as a pointcloud compute it here
        if return_grid:
            # Get the grid indices in the current reference frame
            x_trans = (edges[0][:-1] + edges[0][1:])/2
            y_trans = (edges[1][:-1] + edges[1][1:])/2
            X_trans, Y_trans = np.meshgrid(x_trans, y_trans)
            grid_trans = np.hstack((X_trans.ravel()[:,np.newaxis], 
                                    Y_trans.ravel()[:,np.newaxis],
                                    np.zeros((X_trans.size, 1)),
                                    np.ones((X_trans.size, 1))))
            grid_trans = np.hstack((X_trans.ravel()[:,np.newaxis], 
                                    Y_trans.ravel()[:,np.newaxis],
                                    np.zeros((X_trans.size, 1))))
            pts = vtk.vtkPoints()
            pts.SetData(numpy_to_vtk(grid_trans, array_type=vtk.VTK_DOUBLE))
            pdata = vtk.vtkPolyData()
            pdata.SetPoints(pts)
            # Get the inverse of the singlescan's transform
            invTransform = vtk.vtkTransform()
            invTransform.DeepCopy(ss.transform)
            invTransform.Inverse()
            tfilter = vtk.vtkTransformPolyDataFilter()
            tfilter.SetTransform(invTransform)
            tfilter.SetInputData(pdata)
            tfilter.Update()
            grid_ss = vtk_to_numpy(tfilter.GetOutput().GetPoints().GetData())
           
            # Return
            return frac_exceed_max_diff, diff.T, grid_ss.copy()

        else:
            # Return the diff
            return frac_exceed_max_diff, diff.T

def get_git_hash():
    """
    Get the current git hash string for pydar

    Returns
    -------
    str
        The git hash string for the pydar repository

    """
    
    # Get and store the directory we are currently in
    cwd = os.getcwd()
    # Get the top level pydar directory (where this file resides)
    pydar_path = os.path.dirname(os.path.realpath(__file__))
    # Move to the pydar directory
    os.chdir(pydar_path)
    # Get the git hash as a string
    stream = os.popen('git rev-parse HEAD')
    output = stream.read()
    git_hash = output.split('\n')[0]
    # Return to the directory we started in
    os.chdir(cwd)
    # Return the hash
    return git_hash

# Only if we have imported GPyTorch!
# Define a class and a function to generate gaussian process evaluate 
# at a set of points.
if 'gpytorch' in sys.modules:
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, lengthscale=None,
                     mean=None, outputscale=None, nu=0.5):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            # Let's create the mean module
            self.mean_module = gpytorch.means.ConstantMean()
            if mean is None:
                self.mean_module.initialize(constant=train_y.mean())
            else:
                self.mean_module.initialize(constant=mean)

            # Create the covariance module, we'll use a Matern Kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.keops.MaternKernel(nu=nu))
            # None and 0 evaluate to False so initialize any hyperparameters
            # that the user has set
            if lengthscale:
                self.covar_module.base_kernel.initialize(lengthscale=
                                                         lengthscale)
            if outputscale:
                self.covar_module.initialize(outputscale=outputscale)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, 
                                                             covar_x)

    def run_gp(pts, height, grid_points, z_sigma=None, 
               lengthscale=None, mean=None, outputscale=None, nu=1.5, 
               optimize=False, learning_rate=0.1, iter=None, max_time=60,
               optim_param=None, multiply_outputscale=False):
        """
        Run Gaussian Process to predict surface at grid_points.

        Create and run a gaussian process using GPyTorch to predict the surface
        at the points in grid_points. We can choose whether or not to optimize 
        the hyperparameters of the GP. Currently, we only use Matern kernels and
        exact GP's with keops but that could change in the future.

        Parameters:
        -----------
        pts : Nx2+ ndarray
            Location coordinates of the input data. Only the first two columns 
            will be used.
        height : Nx0 ndarray
            Height (or response variable).
        grid_points : Nx2 ndarray
            Location coordinates where we will predict the surface
        z_sigma : Nx0 ndarray, optional
            Array with 1 st. dev. uncertainties for each point in the input. If 
            it is given we will use a FixedNoiseGaussianLikelihood with this 
            array as the fixed noise. If None, we will use a GaussianLikelihood.
            The default is None.
        lengthscale : float, optional
            The initial lengthscale for the kernel in the same units as pts. If
            None this defaults to the GPyTorch's default (ln(1) I think). The 
            default is None.
        mean : float, optional
            Initial constant mean value for the GP. If None it will be set to 
            the mean of norm_height. The default is None.
        outputscale : float, optional
            Initial outputscale for scale of kernel. If None this defaults to 
            the GPyTorch default. The default is None.
        nu : float, optional
            nu value of Matern kernel. It must be one of [0.5, 1.5, 2.5]. The
            default is 1.5
        optimize : bool, optional
            Whether or not to search for optimal hyperparameters for the GP. The
            default is False.
        learning_rate : float, optional
            The learning rate if we are optimizing (currently just using ADAM). 
            The default is 0.1.
        iter : int, optional
            The number of iterations to run optimizer for. Note that we will 
            only ever run optimizer for max_time. The default is None.
        max_time : float, optional
            The maximum amount of time (in seconds) to run an optimization for. 
            The default is 60.
        optim_param : list, optional
            The parameters to optimize, as a list of dicts. If None, we will
            optimize across all parameters. The default is None.
        multiply_outputscale : bool, optional
            If True, set outputscale for each chunk to be outputscale times
            the variance of the z-values of points in that chunk. The default 
            is False.


        Returns:
        --------
        grid_mean, grid_lower, grid_upper, mean, outputscale, lengthscale
            Tuple of 3 1D arrays with the predicted values and lower and upper
            confidence intervals at each of grid_points.

        """

        # Move arrays to gpu
        device = torch.device('cuda:0')
        t_x = torch.tensor(pts[:,:2], device=device)
        t_y = torch.tensor(height, device=device)

        # Initialize the likelihood
        if z_sigma is None:
            # If no norm_z_sigma then we use a GaussianLikelihood
            likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        else:
            # use a fixedNoiseGaussianLikelihood with our noise.
            t_z_sigma = torch.tensor(z_sigma, device=device)
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=t_z_sigma).cuda()

        # If we have multiply outputscale True adjust accordingly
        if multiply_outputscale:
            outputscale = outputscale * torch.var(t_y)

        # Initialize Model, using our class defined above
        model = ExactGPModel(t_x, t_y, likelihood, lengthscale=lengthscale,
                         mean=mean, outputscale=outputscale, nu=nu).cuda()

        # Optimize Model
        if optimize:
            # Get into train mode
            model.train()
            likelihood.train()

            # Create adam optimizer
            if optim_param is None:
                optimizer = torch.optim.Adam([{'params': model.parameters()},], 
                                             lr=learning_rate)
            else:
                def param_parser(model, p):
                    if p=='mean':
                        return model.mean_module.constant
                    elif p=='lengthscale':
                        return model.covar_module.base_kernel.raw_lengthscale
                    else:
                        raise RuntimeError('Param ' + p + ' not implemented')
                param_list = []
                for param_dict in optim_param:
                    param_list.append({'params': param_parser(model, 
                                                              param_dict['p']),
                                       'lr': param_dict['lr']})
                optimizer = torch.optim.Adam(param_list, lr=learning_rate)

            # 'Loss' for GP is the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            # Run optimization loop either until we've done iter or max_time
            ctr = 0
            t_start = time.perf_counter()
            while (time.perf_counter()-t_start)<max_time:
                if not (iter is None):
                    if ctr>=iter:
                        break
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(t_x)
                # Calc loss and backprop gradients
                loss = -mll(output, t_y)
                loss.backward()
                # Take optimization step
                optimizer.step()

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        # Move grid points to gpu and evaluate gp at grid points
        t_grid_points = torch.tensor(grid_points, device=device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            grid_pred = model(t_grid_points)
            grid_mean = grid_pred.mean.to(torch.device('cpu')).numpy()
            grid_lower = grid_pred.confidence_region()[0].to(torch.device(
                'cpu')).numpy()
            grid_upper = grid_pred.confidence_region()[1].to(torch.device(
                'cpu')).numpy()

        # The lower bound is 2 standard deviations below the mean
        return (grid_mean, grid_lower, grid_upper, 
                model.mean_module.constant.item(),
                model.covar_module.outputscale.item(),
                model.covar_module.base_kernel.lengthscale.item())
