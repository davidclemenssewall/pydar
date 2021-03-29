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
from scipy.spatial import Delaunay
from scipy.special import erf, erfinv
from scipy.signal import fftconvolve
from scipy.stats import mode
from numpy.linalg import svd
import cv2 as cv
import scipy.sparse as sp
from scipy.optimize import minimize, minimize_scalar
import pandas as pd
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import os
import re
import copy
import json
import pdal
import math
import warnings
import time
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tracemalloc
try:
    import pyximport
    pyximport.install(inplace=True, language_level=3)
    import cython_util
except ModuleNotFoundError:
    print('cython_util was not imported, functions relying on it will fail')

class TiePointList:
    """Class to contain tiepointlist object and methods.
    
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
    calc_pairwise_dist()
        Calculates the distances between each unique pair of reflectors.
    compare_pairwise_dist(other_tiepointlist)
        Compares pwdist with other_tiepointlist, stores result in compare_dict
    plot_map(other_project_name, delaunay=False, mode='dist')
        Plots a map of the change in reflector distances.
    calc_transformation(other_tiepointlist, reflector_list, mode='LS')
        Calculates best fitting rigid transformation to align with other.
    add_transform(name, transform, reflector_list=[], std=np.NaN)
        Adds a transform the the transforms dataframe.
    apply_transform(index)
        Applies a transform in transforms to update tiepoints_transformed.
    get_transform(index)
        Returns the requested numpy array.
    """
    
    def __init__(self, project_path, project_name):
        """Stores the project_path and project_name variables and loads the 
        tiepoints into a pandas dataframe"""
        self.project_path = project_path
        self.project_name = project_name
        self.tiepoints = pd.read_csv(os.path.join(project_path, project_name,
                                     'tiepoints.csv'),
                                     index_col=0, usecols=[0,1,2,3])
        self.tiepoints.sort_index(inplace=True)
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
            still translates in all 3 dimensions). The default is 'LS'.
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
        
    
    Methods
    -------
    write_scan()
        Writes the scan to a file to save the filters
    read_scan()
        Reads the scan from a file, replacing the RiSCAN version.
    write_current_transform()
        Writes the current transform and its history_dict to files.
    read_transform()
        Reads a transform from file and places it in current transforms.
    load_man_class()
        Load the manual classification table
    apply_transforms(transform_list)
        updates transform to be concatenation of transforms in transform list.
    add_sop()
        load the appropriate SOP matrix into transform_dict
    add_z_offset(z_offset)
        add a z_offset transformation to transform_dict
    add_transform(key, matrix)
        add a transform to transform_dict
    create_elevation_pipeline(z_min, z_max, lower_threshold=-1000,
                              upper_threshold=1000)
        create mapper and actor for displaying points with colors by elevation
    get_polydata()
        Returns the polydata object for the current settings of transforms
        and filters.
    apply_elevation_filter(z_max)
        Filter out all points above a certain height. Sets the flag in 
        Classification to 64.
    apply_snowflake_filter_2(z_diff, N, r_min):
        Filter snowflakes based on their vertical distance from nearby points.
    apply_snowflake_filter_returnindex(cylinder_rad, radial_precision)
        Filter snowflakes based on their return index and whether they are on
        the border of the visible region.
    random_voxel_downsample_filter(wx, wy, wz, seed=1234)
        Subset the filtered pointcloud randomly by voxels. Replaces Polydata!!
    clear_classification
        Reset all Classification values to 0.
    update_man_class(pdata, classification)
        Update the points in man_class with the points in pdata.
    update_man_class_fields(update_fields='all', update_trans=True)
        Update the man_class table with values from the fields currently in
        polydata_raw. Useful, for example if we've improved the HAG filter and
        don't want to have to repick all points.
    create_normalized_heights(x, cdf)
        Use normalize function to create normalized heights in new PointData
        array.
    create_reflectance()
        Create reflectance field in polydata_raw according to RiSCAN instructs.
    create_reflectance_pipeline(v_min, v_max)
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
    create_dimensionality_pdal(temp_file="", from_current=True, voxel=True,
                               h_voxel=0.1, v_voxel=0.01, threads=8)
        Create the four dimensionality variables from Demantke2011 and
        Guinard2017. Uses pdal to do so.
    create_heightaboveground_pdal(resolution=1, temp_file="",
                                  from_current=True, voxel=True, h_voxel=0.1,
                                  v_voxel=0.1)
        Create height above ground value for each point in scan. To start
        we'll just rasterize all points but we may eventually add csf filter.
    add_dist()
        Add distance from scanner to polydata_raw
    """
    
    def __init__(self, project_path, project_name, scan_name, 
                 import_mode=None, poly='.1_.1_.01',
                 read_scan=False, import_las=False, create_id=True,
                 las_fieldnames=None, 
                 class_list=[0, 1, 2, 70], read_dir=None, suffix=''):
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

        Returns
        -------
        None.

        """
        # Store instance attributes
        self.project_path = project_path
        self.project_name = project_name
        self.scan_name = scan_name
        self.poly = poly
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
                    las_fieldnames[i] = las_fieldnames[i] + '.npy'
            
            pdata = vtk.vtkPolyData()
            self.np_dict = {}
            for k in las_fieldnames:
                try:
                    name = k.split('.')[0]
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
            vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
            vertexGlyphFilter.SetInputData(pdata)
            vertexGlyphFilter.Update()
            self.polydata_raw = vertexGlyphFilter.GetOutput()
            
            # Load in history dict
            f = open(os.path.join(npy_path, 'raw_history_dict.txt'))
            self.raw_history_dict = json.load(f)
            f.close()
            
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
                filename = next(f for f, m in zip(filenames, matches) if m)
                json_list = [os.path.join(self.project_path, self.project_name, 
                             "lasfiles", filename)]
                json_data = json.dumps(json_list, indent=4)
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
                vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
                vertexGlyphFilter.SetInputData(pdata)
                vertexGlyphFilter.Update()
                self.polydata_raw = vertexGlyphFilter.GetOutput()
                # We're importing from LAS (RiSCAN output) so initialize
                # raw_history_dict as a Pointset Source
                self.raw_history_dict = {
                        "type": "Pointset Source",
                        "git_hash": git_hash,
                        "method": "SingleScan.__init__",
                        "filename": os.path.join(self.project_path, 
                                                 self.project_name, 
                                                 "lasfiles", filename),
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
        
        # Create currentFilter
        # self.currentFilter = ClassFilter(self.transformFilter.GetOutputPort(),
        #                                  class_list=class_list)
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
        
    def write_scan(self, write_dir=None, class_list=None, suffix=''):
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
        for f in os.listdir(write_dir):
            os.remove(os.path.join(write_dir, f))

        # If class_list is None just write raw data
        if class_list is None:
            # Save Points
            np.save(os.path.join(write_dir, 'Points.npy'), self.dsa_raw.Points)
            # Save arrays
            for name in self.dsa_raw.PointData.keys():
                np.save(os.path.join(write_dir, name), 
                        self.dsa_raw.PointData[name])
            # Save history_dict
            f = open(os.path.join(write_dir, 'raw_history_dict.txt'), 'w')
            json.dump(self.raw_history_dict, f, indent=4)
            f.close()
        else:
            ind = np.isin(self.dsa_raw.PointData['Classification'], class_list)

            # Save Points
            np.save(os.path.join(write_dir, 'Points.npy'), 
                    self.dsa_raw.Points[ind, :])
            # Save arrays
            for name in self.dsa_raw.PointData.keys():
                np.save(os.path.join(write_dir, name), 
                        self.dsa_raw.PointData[name][ind])
            # Save history_dict
            temp_hist_dict = {
                "type": "Filter",
                "git_hash": get_git_hash(),
                "method": "SingleScan.write_scan",
                "input_0": self.raw_history_dict,
                "params": {"class_list": class_list}
                }
            f = open(os.path.join(write_dir, 'raw_history_dict.txt'), 'w')
            json.dump(temp_hist_dict, f, indent=4)
            f.close()
        
        
    def read_scan(self):
        """
        Reads a scan from a vtp file

        Returns
        -------
        None.

        """
        raise RuntimeError("There's no longer any reason to call SingleScan."
                           + "read_scan. Simply init a new SingleScan object")
        
        # Clear polydata_raw and dsa_raw
        if hasattr(self, 'dsa_raw'):
            del self.dsa_raw
            del self.polydata_raw

            
        # Create Reader, read file
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(os.path.join(self.project_path, self.project_name, 
                           "vtkfiles", "pointclouds",
                           self.scan_name + '.vtp'))
        reader.Update()
        self.polydata_raw = reader.GetOutput()
        self.polydata_raw.Modified()
        
        # Create dsa, link with transform filter
        self.dsa_raw = dsa.WrapDataObject(self.polydata_raw)
        self.transformFilter.SetInputData(self.polydata_raw)
        self.transformFilter.Update()
        self.currentFilter.Update()
    
    def write_current_transform(self, write_dir=None, name='current_transform'
                                , mode='rigid', suffix=''):
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
        # If the files already exist remove them
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
        # Try loading the transform
        transform_np = np.load(os.path.join(read_dir, name + '.npy'))
        
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
    
            # Add to transform dict, include history dict
            f = open(os.path.join(read_dir, name + '.txt'))
            self.add_transform(name, M, json.load(f))
            f.close()
        
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
    
            # Add to transform dict, include history dict
            f = open(os.path.join(read_dir, name + '.txt'))
            self.add_transform(name, M, json.load(f))
            f.close()
        
        else:
            raise RuntimeError('transform does not match known format')
            
    
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
        
        # If the norm_height array exists delete it we will recreate it 
        # if needed
        if 'norm_height' in self.dsa_raw.PointData.keys():
            self.polydata_raw.GetPointData().RemoveArray('norm_height')
            self.polydata_raw.Modified()
            # Update raw_history_dict accordingly
            self.raw_history_dict = {
                "type": "Scalar Modifier",
                "git_hash": git_hash,
                "method": 'SingleScan.apply_transforms',
                "name": "Removed norm_height",
                "input_0": json.loads(json.dumps(self.raw_history_dict))
                }
            self.transformed_history_dict["input_0"] = self.raw_history_dict
            
        self.transformFilter.Update()
        self.currentFilter.Update()
    
    def random_voxel_downsample_filter(self, wx, wy, wz, seed=1234):
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
            Voxel x dimension in m.
        wz : float
            Voxel x dimension in m.
        seed : int, optional
            Random seed for the shuffler. The default is 1234.

        Returns
        -------
        None.

        """
        
        # Step 1, create shuffled points from current polydata
        filt_pts = vtk_to_numpy(self.get_polydata().GetPoints().GetData())
        rng = np.random.default_rng(seed=seed)
        shuff_ind = np.arange(filt_pts.shape[0])
        rng.shuffle(shuff_ind)
        shuff_pts = filt_pts[shuff_ind, :]
        
        # Step 2, bin and downsample
        w = [wx, wy, wz]
        edges = 3*[None]
        nbin = np.empty(3, np.int_)
        
        for i in range(3):
            edges[i] = np.arange(int(np.ceil((shuff_pts.max(axis=0)[i] - 
                                              shuff_pts.min(axis=0)[i])/w[i]))
                                 + 1, dtype=
                                 np.float32) * w[i] + shuff_pts.min(axis=0)[i]
            # needed to avoid min point falling out of bounds
            edges[i][0] = edges[i][0] - 0.0001 
            nbin[i] = len(edges[i]) + 1
        
        Ncount = tuple(np.searchsorted(edges[i], shuff_pts[:,i], side='right')
                      for i in range(3))
        
        xyz = np.ravel_multi_index(Ncount, nbin)
        
        # We want to take just one random point from each bin. Since we've shuffled the points,
        # the first point we find in each bin will suffice. Thus we use the unique function
        _, inds = np.unique(xyz, return_index=True)
        
        print('here')
        
    
    def clear_classification(self):
        """
        Reset Classification for all points to 0

        Returns
        -------
        None.

        """
        
        self.dsa_raw.PointData['Classification'][:] = 0
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
            "input_0": json.loads(json.dumps(self.raw_history_dict))
            }
        self.transformed_history_dict["input_0"] = self.raw_history_dict
    
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
        
        # Get the points of the currentTransform as a numpy array
        dsa_current = dsa.WrapDataObject(self.transformFilter.GetOutput())
        # Set the in Classification for points whose z-value is above z_max to 1
        self.dsa_raw.PointData['Classification'][dsa_current.Points[:,2] 
                                              > z_max] = 64
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
            "input_0": json.loads(json.dumps(self.raw_history_dict)),
            "params": {"z_max": z_max}
            }
        self.transformed_history_dict["input_0"] = self.raw_history_dict
    
    def apply_snowflake_filter_2(self, z_diff, N, r_min):
        """
        Filter snowflakes based on their vertical distance from nearby points.
        
        Here we exploit the fact that snowflakes (and for that matter power
        cables and some other human objects) are higher than their nearby
        points. The filter steps through each point in the transformed
        dataset and compares its z value with the mean of the z-values of
        the N closest points. If the difference exceeds z_diff then set the
        Classification for that point to be 2. Also, there is a shadow around the
        base of the scanner so all points within there must be spurious. We
        filter all points within r_min
    
        Parameters
        ----------
        z_diff : float
            Maximum vertical difference in m a point can have from its 
            neighborhood.
        N : int
            Number of neighbors to find.
        r_min : float
            Radius of scanner in m within which to filter all points.
    
        Returns
        -------
        None.
    
        """
        
        # Move z-values to scalars 
        elevFilter = vtk.vtkSimpleElevationFilter()
        elevFilter.SetInputConnection(self.transformFilter.GetOutputPort())
        elevFilter.Update()
        # Flatten points
        flattener = vtk.vtkTransformPolyDataFilter()
        trans = vtk.vtkTransform()
        trans.Scale(1, 1, 0)
        flattener.SetTransform(trans)
        flattener.SetInputConnection(elevFilter.GetOutputPort())
        flattener.Update()
        
        # Create pdata and locator
        pdata = flattener.GetOutput()
        locator = vtk.vtkOctreePointLocator()
        pdata.SetPointLocator(locator)
        locator.SetDataSet(pdata)
        pdata.BuildLocator()
        
        # Create temporary arrays for holding points
        output = vtk.vtkFloatArray()
        output.SetNumberOfValues(N)
        output_np = vtk_to_numpy(output)
        pt_ids = vtk.vtkIdList()
        pt_ids.SetNumberOfIds(N)
        pt = np.zeros((3))
        scan_pos = np.array(self.transform.GetPosition())
        
        for m in np.arange(pdata.GetNumberOfPoints()):
            # Get the point
            pdata.GetPoint(m, pt)
            # Check if the point is within our exclusion zone
            r = np.linalg.norm(pt[:2]-scan_pos[:2])
            if r < r_min:
                self.dsa_raw.PointData['Classification'][m] = 65
                continue
            
            # Get N closest points
            locator.FindClosestNPoints(N, pt, pt_ids)
            # now using the list of point_ids set the values in output to be the z
            # values of the found points
            pdata.GetPointData().GetScalars().GetTuples(pt_ids, output)
            # If we exceed z_diff set Classification to 2
            if (pdata.GetPointData().GetScalars().GetTuple(m)
                - output_np.mean())>z_diff:
                self.dsa_raw.PointData['Classification'][m] = 65
        
        # Update currentTransform
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()
    
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

    def create_dimensionality_pdal(self, temp_file="", from_current=True, 
                                   voxel=True, h_voxel=0.1, v_voxel=0.01, 
                                   threads=8):
        """
        Uses pdal functions to voxel downsample and create dimensionality
        variables from Demantke2011 and Guinard2017.
        
        WARNING: This method replaces polydata_raw but does not modify
        transforms
        
        If from_current, we write the current polydata-points with 
        Classification of 0, 1, or 2 and transformed by the current transform-
        to a pdal readable numpy file in temp_file. We then create a json
        with the instructions to voxelize that data, run 
        filters.optimalneighborhood and filters.covariancefeatures and make
        the results available as numpy arrays. Finally, we replace the points
        in polydata_raw with the matching PointId's and eliminate other points
        in polydata_raw.

        Parameters
        ----------
        temp_file : str, optional
            Location to write numpy file to. If "" use self.project_path +
            '\\temp\\temp_pdal.npy'. The default is "".
        from_current : bool, optional
            Whether to write the current polydata to file, if False will use
            whichever file is currently in the temp directory. False should 
            only be used for debugging. The default is True.
        voxel : bool, optional
            Whether to voxel nearest centroid downsample. The default is True.
        h_voxel : float, optional
            Horizontal voxel dimension in m. The default is 0.1.
        v_voxel : float, optional
            Vertical voxel dimension in m. The default is 0.01.
        threads : int, optional
            Number of threads to use in covariancefeatures

        Returns
        -------
        None.

        """
        warnings.warn("History tracking not implemented yet")
        
        # Parse temp_file
        if not temp_file:
            temp_file = os.path.join(self.project_path, 'temp', 'temp_pdal.npy')
         
        if from_current:
            # Write to temp_file
            self.write_npy_pdal(temp_file, filename='', mode='filtered')
        else:
            warnings.warn("create_dimensionality_pdal is reading whichever " 
                          + "file is in the temp directory, make sure this "
                          + "is the desired behavior")
        
        # Create dimensionality in pdal and import back into python
        json_list = []
        # Stage for reading data
        json_list.append(
            {"filename": os.path.join(self.project_path, 'temp', 'temp_pdal.npy'),
             "type": "readers.numpy"})
        # Add voxel filter stages if desired
        if voxel:
            json_list.append(
                {"type": "filters.transformation",
                 "matrix": str(1/h_voxel) + " 0 0 0 0 " + str(1/h_voxel) + 
                 " 0 0 0 0 " + str(1/v_voxel) + " 0 0 0 0 1"})
            json_list.append(
                {"type": "filters.voxelcentroidnearestneighbor",
                 "cell": 1})
            json_list.append({"type": "filters.transformation",
                 "matrix": str(h_voxel) + " 0 0 0 0 " + str(h_voxel) + 
                 " 0 0 0 0 " + str(v_voxel) + " 0 0 0 0 1"})
        # Stage to calculate optimal NN
        json_list.append(
            {"type": "filters.optimalneighborhood",
            "min_k": 8,
            "max_k": 50})
        # Stage to calculate dimensionality
        json_list.append(
            {"type": "filters.covariancefeatures",
             "threads": threads,
             "optimized": True,
             "feature_set": "all"})
        # Create and execute pdal pipeline
        json_data = json.dumps(json_list, indent=4)
        pipeline = pdal.Pipeline(json_data)
        _ = pipeline.execute()
        
        # Update Contents of PolydataRaw with the filter output
        arr = pipeline.get_arrays()
        # If we downsampled then we will change polydata raw
        if voxel:
            # Create PedigreeIds array to do selection
            pedigreeIds = vtk.vtkTypeUInt32Array()
            pedigreeIds.SetNumberOfComponents(1)
            pedigreeIds.SetNumberOfTuples(arr[0].size)
            np_pedigreeIds = vtk_to_numpy(pedigreeIds)
            np_pedigreeIds[:] = arr[0]['PointId']
            pedigreeIds.Modified()
            
            # Selection points from original polydata_raw by PedigreeId
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
            
            # vtk automatically sorts points by pedigree id, so apply same 
            # sorting to our new dimensions and update
            sort_inds = np.argsort(arr[0]['PointId'])
            dims = ['Linearity', 'Planarity', 'Scattering', 'Verticality',
                    'Density', 'Anisotropy']
            for dim in dims:
                arr_vtk = vtk.vtkFloatArray()
                arr_vtk.SetName(dim)
                arr_vtk.SetNumberOfComponents(1)
                arr_vtk.SetNumberOfTuples(pdata.GetNumberOfPoints())
                pdata.GetPointData().AddArray(arr_vtk)
                dsa_pdata.PointData[dim][:] = np.float32(arr[0][dim]
                                                         [sort_inds])
                
            pdata.Modified()
            
            # Update polydata_raw
            vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
            vertexGlyphFilter.SetInputData(pdata)
            vertexGlyphFilter.Update()
            self.polydata_raw = vertexGlyphFilter.GetOutput()
            self.dsa_raw = dsa.WrapDataObject(self.polydata_raw)
            
            # Update filters
            self.transformFilter.SetInputData(self.polydata_raw)
            self.transformFilter.Update()
            self.currentFilter.Update()
        else:
            raise NotImplementedError("create_dimensionality_pdal must have"
                                      + " voxel=True for now")
    
    def create_heightaboveground_pdal(self, temp_file="",
                                      temp_file_tif="", from_current=True, 
                                      create_dem=True):
        """
        Create height above ground value for each point in scan.

        Parameters
        ----------
        resolution : float, optional
            DEM resolution in m. The default is 1.
        temp_file : str, optional
            Location to write numpy file to. If "" use self.project_path +
            '\\temp\\temp_pdal.npy'. The default is "".
        temp_file_tif : str, optional
            tempfile for storing dem, if "" defaults to temp_dir/project_name
            _scan_name.tif. The default is "".
        from_current : bool, optional
            Whether to write the current polydata to file, if False will use
            whichever file is currently in the temp directory. False should 
            only be used for debugging. The default is True.
        create_dem : bool, optional
            Whether to create a dem from this SingleScan. If not you should
            probably supply a temp_file_tif string to this function. The 
            default is True.

        Returns
        -------
        None.

        """
        warnings.warn("History tracking not implemented yet")
        
        # Parse temp_file
        if not temp_file:
            temp_file = os.path.join(self.project_path, 'temp', self.project_name + 
                         '_' + self.scan_name + '.npy')
         
        if from_current:
            # Write to temp_file
            self.write_npy_pdal(temp_file, filename='', mode='transformed',
                                skip_fields='all')
        else:
            warnings.warn("create_heightaboveground_pdal is reading whichever" 
                          + " file is in the temp directory, make sure this "
                          + "is the desired behavior")
        
        # Parse temp_file_tif
        if not temp_file_tif:
            if not create_dem:
                warnings.warn("create_heightaboveground_pdal is not creating"
                              + " a dem but is reading an auto-generated " 
                              "filename! Make sure this is desired")
            temp_file_tif = os.path.join(self.project_path, 'temp', self.project_name 
                             + '_' + self.scan_name + '.tif')
        
        if create_dem:
            # Create DEM
            json_list = []
    
            # Numpy reader
            json_list.append({"filename": temp_file,
                              "type": "readers.numpy"})
            
            json_list.append({"type": "filters.smrf",
                              "slope": 0.667,
                              "scalar": 2.18,
                              "threshold": 0.894,
                              "window": 14.8,
                              "cell": 0.624})
            
            json_list.append({"type": "filters.range",
                              "limits": "Classification[2:2]"})
            
            json_list.append({"type": "writers.gdal",
                              "filename": temp_file_tif,
                              "resolution": 0.5,
                              "data_type": "float32",
                              "output_type": "idw",
                              "window_size": 7})
            
            json_data = json.dumps(json_list, indent=4)
            
            pipeline = pdal.Pipeline(json_data)
            _ = pipeline.execute()
            del _
            del pipeline
        
        # Apply HeightAboveGround filter
        json_list = []

        # Numpy reader
        json_list.append({"filename": temp_file,
                          "type": "readers.numpy"})
        
        # Height above ground stage
        json_list.append({"type": "filters.hag_dem",
                          "raster": temp_file_tif,
                          "zero_ground": 'false'})
        
        json_data = json.dumps(json_list, indent=4)

        pipeline = pdal.Pipeline(json_data)
        _ = pipeline.execute()
        
        # Update Contents of polydata_raw with the filter output
        arr = pipeline.get_arrays()
        sort_inds = np.argsort(arr[0]['PointId'])
        if not 'HeightAboveGround' in self.dsa_raw.PointData.keys():
            arr_vtk = vtk.vtkFloatArray()
            arr_vtk.SetName('HeightAboveGround')
            arr_vtk.SetNumberOfComponents(1)
            arr_vtk.SetNumberOfTuples(self.polydata_raw.GetNumberOfPoints())
            self.polydata_raw.GetPointData().AddArray(arr_vtk)
        self.dsa_raw.PointData['HeightAboveGround'][:] = np.float32(
            arr[0]['HeightAboveGround'][sort_inds])
        
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()
        
        # Delete temporary files to prevent bloat
        del pipeline
        os.remove(temp_file)
        if create_dem:
            os.remove(temp_file_tif)
        
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
                                            72: (253/255, 191/255, 111/255, 1)
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
        self.transformFilter.GetOutput().GetPointData().SetActiveScalars(
            'Classification')
        
        # Create Lookuptable
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(max(colors) + 1)
        lut.SetTableRange(0, max(colors))
        for key in colors:
            lut.SetTableValue(key, colors[key])
        lut.Build()
        
        # Create mapper
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.transformFilter.GetOutputPort())
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
        
        # Create mapper
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.currentFilter.GetOutputPort())
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
                                  upper_threshold=1000):
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

        Returns
        -------
        None.

        """
        
        # # Create elevation filter
        elevFilter = vtk.vtkSimpleElevationFilter()
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()
        elevFilter.SetInputConnection(self.currentFilter.GetOutputPort())
        elevFilter.Update()
        
        # Create Threshold filter
        thresholdFilter = vtk.vtkThresholdPoints()
        thresholdFilter.SetInputConnection(elevFilter.GetOutputPort())
        thresholdFilter.ThresholdBetween(lower_threshold, upper_threshold)
        thresholdFilter.Update()
        
        # Create mapper, hardcode LUT for now
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(thresholdFilter.GetOutputPort())
        self.mapper.SetLookupTable(mplcmap_to_vtkLUT(z_min, z_max))
        self.mapper.SetScalarRange(z_min, z_max)
        self.mapper.SetScalarVisibility(1)
        
        # Create subsampled for LOD rendering
        maskPoints = vtk.vtkPMaskPoints()
        maskPoints.ProportionalMaximumNumberOfPointsOn()
        maskPoints.SetOnRatio(10)
        maskPoints.GenerateVerticesOn()
        maskPoints.SetInputConnection(thresholdFilter.GetOutputPort())
        maskPoints.Update()
        self.mapper_sub = vtk.vtkPolyDataMapper()
        self.mapper_sub.SetInputConnection(maskPoints.GetOutputPort())
        self.mapper_sub.SetLookupTable(mplcmap_to_vtkLUT(z_min, z_max))
        self.mapper_sub.SetScalarRange(z_min, z_max)
        self.mapper_sub.SetScalarVisibility(1)
        
        
        # Create actor
        self.actor = vtk.vtkLODProp3D()
        self.actor.AddLOD(self.mapper, 0.0)
        self.actor.AddLOD(self.mapper_sub, 0.0)
    
    def create_normalized_heights(self, x, cdf):
        """
        Use normalize function to create normalized heights in new PointData
        array.
        
        Here we use the normalize function (defined below in Pydar) to
        transform the z values from the output of transformFilter to a normal
        distribution assuming they were drawn from the distribution specified
        by x and cdf.

        Parameters
        ----------
        x : 1d-array
            Bin values in empirical cdf.
        cdf : 1d-array
            Values of empirical cdf.

        Returns
        -------
        None.

        """
        
        # If normalized height array doesn't exist, create it.
        if not 'norm_height' in self.dsa_raw.PointData.keys():
            arr = vtk.vtkFloatArray()
            arr.SetName('norm_height')
            arr.SetNumberOfComponents(1)
            arr.SetNumberOfTuples(self.polydata_raw.GetNumberOfPoints())
            arr.FillComponent(0, 0)
            self.polydata_raw.GetPointData().AddArray(arr)
        
        # Create a temporary dataset_adaptor for the output of currentTransform
        dsa_transform = dsa.WrapDataObject(self.transformFilter.GetOutput())
        # Normalize
        self.dsa_raw.PointData['norm_height'][:] = normalize(
            dsa_transform.Points[:, 2].squeeze(), x, cdf)
        
        # Update
        self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()
        
        # Update raw_history_dict
        self.raw_history_dict = {
            "type": "Scalar Modifier",
            "git_hash": get_git_hash(),
            "method": "SingleScan.create_normalized_heights",
            "name": "Create normalized heights field",
            "input_0": json.loads(json.dumps(self.raw_history_dict)),
            "input_1": json.loads(json.dumps(self.transformed_history_dict)),
            "params": {"x": x.tolist(),
                       "cdf": cdf.tolist()}
            }
        self.transformed_history_dict["input_0"] = self.raw_history_dict
    
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
        # Create mapper, hardcode LUT for now
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.currentFilter.GetOutputPort())
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
        
        np.save(os.path.join(output_dir, filename), output_npy)
    
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
    image : vtkImageData
        Image data containing gridded height information over specified region
    imageTransform : vtkTransform
        Transform for going from mesh reference frame to image ref
        frame. Needed because vtkImageData can only be axis aligned.
    dsa_image : datasetadapter
        Wrapper to interact with image via numpy
    empirical_cdf : tuple
        (bounds, x, cdf) tuple with distribution of snow surface heights for 
        use with transforming and normalizing z-values.
    theta : 1D array or list
        Theta parameters for the GMRF used in pixel infilling. See Rue and
        Held 2005 Chapter 5.1
    theta1 : float
        Scaling parameter for GMRF used in pixel infilling. See Rue and Held
        2005 Chapter 5.1
    mu_agivenb : 1D array
        Expectation of missing pixel values conditioned on observed pixels.
    sparse_LAA : sparse matrix
        Lower triangular Cholesky factor of the conditional precision matrix
        (QAA) of the missing pixels conditioned on the observed pixels.
    
    Methods
    -------
    add_z_offset(z_offset)
        Add z offset to all scans in project.
    apply_transforms(transform_list)
        Update transform for each scan and update current_transform_list.
    display_project(z_min, z_max, lower_threshold=-1000, upper_threshold=1000)
        Display project in a vtk interactive window.
    display_image(z_min, z_max)
        Display project image in a vtk interactive window.
    write_merged_points(output_name=self.project_name + '_merged.vtp')
        Merge all transformed and filtered pointclouds and write to file.
    write_las_pdal(output_dir, filename)
        Merge all points and write to a las formatted output.
    write_mesh(output_name=self.project_name + '_mesh.vtp')
        Write mesh to vtp file.
    read_mesh(mesh_name=self.project_name + '_mesh.vtp')
        Read mesh from file.
    write_scans()
        Write all singlescans to files.
    read_scans()
        Read all singlescans from files.
    merged_points_to_mesh(subgrid_x, subgrid_y, min_pts=100, alpha=0,
                          overlap=0.1)
        Merge all transformed pointclouds and convert to mesh.
    project_to_image(z_min, z_max, focal_point, camera_position,
                         image_scale=500, lower_threshold=-1000, 
                         upper_threshold=1000, mode='map', colorbar=True,
                         name='')
        Write out an image of the project (in point cloud) to the snapshot
        folder.
    add_transforms(key, matrix)
        Add the provided transform to each SingleScan
    apply_snowflake_filter(shells)
        Apply the snowflake filter.
    apply_snowflake_filter_2(z_diff, N, r_min)
        Apply a snowflake filter based on z difference with nearby points.
    apply_snowflake_filter_returnindex(cylinder_rad, radial_precision)
        Filter snowflakes based on their return index and whether they are on
        the border of the visible region.
    create_scanwise_closest_point()
        For each point in each scan, find the vertical and horizontal
        distances to the closest point in all other scans. Useful for
        filtering out snowmobiles and humans and things that move.
    create_heightaboveground_pdal(resolution=1, voxel=true, h_voxel=0.1,
                                  v_voxel=0.1, project_dem=True)
        Create height above ground value for each point in scan.
    update_man_class_fields(update_fields='all', update_trans=True)
        Update the man_class table with values from the fields currently in
        polydata_raw. Useful, for example if we've improved the HAG filter and
        don't want to have to repick all points.
    get_merged_points()
        Get the merged points as a polydata
    mesh_to_image(z_min, z_max, nx, ny, dx, dy, x0, y0)
        Interpolate mesh into image.
    plot_image(z_min, z_max, cmap='inferno')
        Plots the image using matplotlib
    get_np_nan_image()
        Convenience function for copying the image to a numpy object.
    create_empirical_cdf(bounds)
        Creates an empirical cdf from z-values of all points within bounds.
    create_empirical_cdf_image(z_min, z_max)
        Creates an empirical cdf from z-values of image.
    create_im_gaus()
        Create normalized image based on the empirical cdf
    add_theta(theta1, theta)
        Adds the GMRF parameters theta1 and theta to attributes.
    create_im_nan_border(buffer)
        Creates a mask for which missing pixels not to infill.
    write_image(output_name=None)
        Write vtkImageData to file. Useful for saving im_nan_border.
    read_image(image_path=None)
        Read image from file.
    create_gmrf()
        Create GMRF for pixel infilling. Generates mu_agivenb and sparse_LAA.
    create_reflectance()
        Create reflectance for each scan.
    correct_reflectance_radial()
        Attempt to correct for reflectance bias due to distance from scanner.
    """
    
    def __init__(self, project_path, project_name, poly='.1_.1_.01', 
                 import_mode=None, load_scans=True, read_scans=False, 
                 import_las=False, create_id=True, 
                 las_fieldnames=None, 
                 class_list=[0, 1, 2, 70], suffix=''):
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
        scan_names = os.listdir(os.path.join(project_path, project_name, 
                                                 'SCANS'))
        for scan_name in scan_names:
            if import_mode=='read_scan':
                if os.path.isdir(os.path.join(self.project_path, 
                                              self.project_name, 'npyfiles' +
                                              suffix, scan_name)):
                    scan = SingleScan(self.project_path, self.project_name,
                                      scan_name, import_mode=import_mode,
                                      poly=poly, create_id=create_id,
                                      las_fieldnames=las_fieldnames,
                                      class_list=class_list, suffix=suffix)
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
    
    def write_scans(self, project_write_dir=None, suffix=''):
        """
        Write all single scans to files.
        
        Parameters
        ----------
        A directory to write all scans for this project to. If none write
        to default npyfiles location. The default is None.
        
        suffix : str, optional
            Suffix for npyfiles directory if we are reading scans. The default
            is '' which corresponds to the regular npyfiles directory.

        Returns
        -------
        None.

        """
        
        if project_write_dir is None:
            for scan_name in self.scan_dict:
                self.scan_dict[scan_name].write_scan(suffix=suffix)
        else:
            # For each scan name create a directory under project_write_dir
            # if it does not already exist.
            for scan_name in self.scan_dict:
                if not os.path.isdir(os.path.join(project_write_dir, 
                                                  scan_name)):
                    os.mkdir(os.path.join(project_write_dir, scan_name))
                self.scan_dict[scan_name].write_scan(os.path.join(
                    project_write_dir, scan_name), suffix=suffix)
    
    def read_scans(self):
        """
        Read all single scans from files.

        Returns
        -------
        None.

        """
        raise RuntimeError('Do not use, just init a new Project object')
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].read_scan()
    
    def write_current_transforms(self, suffix=''):
        """
        Have each SingleScan write its current transform to a file.

        suffix : str, optional
            Suffix for transforms directory if we are reading scans. 
            The default is '' which corresponds to the regular transforms
            directory.
            
        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].write_current_transform(suffix=suffix)
    
    def read_transforms(self, suffix=''):
        """
        Have each SingleScan read a transform from file

        suffix : str, optional
            Suffix for transforms directory if we are reading scans. 
            The default is '' which corresponds to the regular transforms
            directory.
            
        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].read_transform(suffix=suffix)
    
    def load_man_class(self):
        """
        Direct each single scan to load it's man_class table

        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].load_man_class()
    
    def apply_snowflake_filter(self, shells):
        """
        Apply a snowflake filter to each SingleScan

        Parameters
        ----------
        shells : array-like of tuples
            shells is an array-like set of tuples where each tuple is four
            elements long (inner_r, outer_r, point_radius, neighbors). *_r
            define the inner and outer radius of a halo defining shell. 
            point_radius is radius for vtkRadiusOutlierRemoval to look at
            (if 0, remove all points). Neighbors is number of neighbors that
            must be within point_radius (if 0, keep all points)

        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].apply_snowflake_filter(shells)
        self.filterName = "snowflake"
    
    def apply_snowflake_filter_2(self, z_diff, N, r_min):
        """
        Filter snowflakes based on their vertical distance from nearby points.
        
        Here we exploit the fact that snowflakes (and for that matter power
        cables and some other human objects) are higher than their nearby
        points. The filter steps through each point in the transformed
        dataset and compares it's z value with the mean of the z-values of
        the N closest points. If the difference exceeds z_diff then set the
        Classification for that point to be 2. Also, there is a shadow around the
        base of the scanner so all points within there must be spurious. We
        filter all points within r_min

        Parameters
        ----------
        z_diff : float
            Maximum vertical difference in m a point can have from its 
            neighborhood.
        N : int
            Number of neighbors to find.
        r_min : float
            Radius of scanner in m within which to filter all points.

        Returns
        -------
        None.
        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].apply_snowflake_filter_2(z_diff, N,
                                                               r_min)
        self.filterName = "snowflake_2"
    
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
    
    def create_scanwise_closest_point(self):
        """
        Create VerticalClosestPoint and HorizontalClosestPoint fields.
        
        For each point in each scan, find the vertical and horizontal
        distances to the closest point in all other scans. Useful for
        filtering out snowmobiles and humans and things that move.

        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            print(scan_name)
            pdata = self.scan_dict[scan_name].transformFilter.GetOutput()
            
            # Combine all other scans into a single polydata and build
            n_other_points = 0
            for s in self.scan_dict:
                if s==scan_name:
                    continue
                else:
                    n_other_points += (self.scan_dict[s].transformFilter
                                       .GetOutput().GetNumberOfPoints())
            
            otherPoints = vtk.vtkPoints()
            otherPoints.SetDataTypeToFloat()
            otherPoints.SetNumberOfPoints(n_other_points)
            ctr = 0
            for s in self.scan_dict:
                if s==scan_name:
                    continue
                else:
                    n = (self.scan_dict[s].transformFilter.GetOutput()
                         .GetNumberOfPoints())
                    otherPoints.InsertPoints(ctr, n, 0, self.scan_dict[s]
                                             .transformFilter.GetOutput()
                                             .GetPoints())
                    ctr += n
            
            otherPData = vtk.vtkPolyData()
            otherPData.SetPoints(otherPoints)
            locator = vtk.vtkOctreePointLocator()
            locator.SetDataSet(otherPData)
            otherPData.SetPointLocator(locator)
            otherPData.BuildLocator()
            
            
            # Create numpy array to hold points
            closest_points_np = np.zeros((pdata.GetNumberOfPoints(), 3), 
                                         dtype=np.float32)
            for i in np.arange(pdata.GetNumberOfPoints()):
                otherPData.GetPoint(locator.FindClosestPoint(pdata
                                                             .GetPoint(i)),
                                    closest_points_np[i,:])
            
            # Create arrays to hold horizontal and vertical distances
            hArr = vtk.vtkFloatArray()
            hArr.SetNumberOfComponents(1)
            hArr.SetNumberOfTuples(pdata.GetNumberOfPoints())
            hArr.SetName('HorizontalClosestPoint')
            vArr = vtk.vtkFloatArray()
            vArr.SetNumberOfComponents(1)
            vArr.SetNumberOfTuples(pdata.GetNumberOfPoints())
            vArr.SetName('VerticalClosestPoint')
            
            # Add arrays to polydata_raw
            self.scan_dict[scan_name].polydata_raw.GetPointData().AddArray(
                hArr)
            self.scan_dict[scan_name].polydata_raw.GetPointData().AddArray(
                vArr)
            self.scan_dict[scan_name].polydata_raw.Modified()
            
            # Populate with vertical and horizontal distances
            dsa_pdata = dsa.WrapDataObject(pdata)
            self.scan_dict[scan_name].dsa_raw.PointData['VerticalClosestPoint'
                                                        ][:] = (
                dsa_pdata.Points[:,2] - closest_points_np[:,2]).squeeze()
            self.scan_dict[scan_name].dsa_raw.PointData[
                'HorizontalClosestPoint'][:] = (
                np.sqrt(np.sum(np.square(dsa_pdata.Points[:,:2] 
                                         - closest_points_np[:,:2]),
                               axis=1)).squeeze())
            
            self.scan_dict[scan_name].polydata_raw.Modified()
            self.scan_dict[scan_name].transformFilter.Update()
            self.scan_dict[scan_name].currentFilter.Update()
        
    def create_heightaboveground_pdal(self, 
                                      project_dem=True):
        """
        Create height above ground for each point in each scan.
        
        For now we are just building a dem from inverse distance weighting
        of all points. Will change to csf filter eventually.

        Parameters
        ----------
        project_dem : bool, optional
            Whether to create a dem from all scans in the project. 
            The default is True.

        Returns
        -------
        None.

        """
        
        if project_dem:
            filenames = []
            dem_name = os.path.join(self.project_path, 'temp', self.project_name 
                        + '.tif')
            for scan_name in self.scan_dict:
                # Write scan to numpy
                filenames.append(os.path.join(self.project_path, 'temp', 
                                 self.project_name + '_' + scan_name 
                                 + '.npy'))
                self.scan_dict[scan_name].write_npy_pdal(filenames[-1],
                                                         filename='',
                                                         mode='transformed',
                                                         skip_fields='all')
            
            # Now create dem from all scans
            json_list = []
            for filename in filenames:
                json_list.append({"filename": filename,
                                  "type": "readers.numpy"})
            json_list.append({"type": "filters.merge"})
            json_list.append({"type": "filters.smrf",
                              "slope": 0.667,
                              "scalar": 2.18,
                              "threshold": 0.894,
                              "window": 14.8,
                              "cell": 0.624})

            json_list.append({"type": "filters.range",
                              "limits": "Classification[2:2]"})
            
            json_list.append({"type": "writers.gdal",
                              "filename": dem_name,
                              "resolution": 0.5,
                              "data_type": "float32",
                              "output_type": "idw",
                              "window_size": 7})
            
            json_data = json.dumps(json_list, indent=4)
            
            pipeline = pdal.Pipeline(json_data)
            _ = pipeline.execute()
            del _
            del pipeline
            
            # Now create heightaboveground for each scan
            for filename, scan_name in zip(filenames, self.scan_dict):
                self.scan_dict[scan_name].create_heightaboveground_pdal(
                    create_dem=False, from_current=False, temp_file=filename,
                    temp_file_tif=dem_name)
            os.remove(dem_name)
        else:
            for scan_name in self.scan_dict:
                self.scan_dict[scan_name].create_heightaboveground_pdal()
    
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
            
    def display_project(self, z_min, z_max, lower_threshold=-1000, 
                        upper_threshold=1000, colorbar=True, field='Elevation',
                        mapview=False):
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
        renderWindow.SetSize(2000, 1500)
        renderWindow.AddRenderer(renderer)
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renderWindow)

        style = vtk.vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(style)
            
        iren.Initialize()
        renderWindow.Render()
        iren.AddObserver('UserEvent', cameraCallback)
        iren.Start()
        
        # Needed to get window to close on linux
        renderWindow.Finalize()
        iren.TerminateApp()
        
        del renderWindow, iren
    
    def project_to_image(self, z_min, z_max, focal_point, camera_position,
                         roll=0, image_scale=500, lower_threshold=-1000, 
                         upper_threshold=1000, mode='map', colorbar=True,
                         name=''):
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
            What kind of projection system to use. 'Map' indicates parallel
            or orthorectified projection. The default is 'map'.
        colorbar : bool, optional
            Whether to display a colorbar.
        name : str, optional
            Name to append to this snapshot. The default is ''.

        Returns
        -------
        None.

        """
        
        # Create renderer
        renderer = vtk.vtkRenderer()
        
        # Run create elevation pipeline for each scan and add each actor
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].create_elevation_pipeline(z_min, z_max, 
                                                            lower_threshold, 
                                                            upper_threshold)
            renderer.AddActor(self.scan_dict[scan_name].actor)
            
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
        renderWindow.SetSize(2000, 1000)
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
    
    def mesh_to_image(self, nx, ny, dx, dy, x0, y0, yaw=0):
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

        Returns
        -------
        None.

        """
        
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
        self.imageTransform = vtk.vtkTransform()
        self.imageTransform.PostMultiply()
        # Translate origin to be at (x0, y0)
        self.imageTransform.Translate(-x0, -y0, 0)
        # Rotate around this origin
        self.imageTransform.RotateZ(-yaw)
        
        # Create transform filter and apply
        imageTransformFilter = vtk.vtkTransformPolyDataFilter()
        imageTransformFilter.SetTransform(self.imageTransform)
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
        # If we've already created an image delete and make way for this one
        if hasattr(self, 'image'):
            del(self.image)
        self.image = probe.GetOutput()
        
        # Also wrap with a datasetadaptor for working with numpy
        self.dsa_image = dsa.WrapDataObject(self.image)
        
        # and use dsa to set NaN values where we have no data
        bool_arr = self.dsa_image.PointData['vtkValidPointMask']==0
        self.dsa_image.PointData['Elevation'][bool_arr] = np.NaN
    
    def get_image(self, field='Elevation', warp_scalars=False):
        """
        Return image as vtkImageData or vtkPolyData depending on warp_scalars

        Parameters
        ----------
        field : str, optional
            Which field in PointData to set active. The default is 'Elevation'
        warp_scalars : bool, optional
            Whether to warp the scalars in the image to create 3D surface

        Returns
        -------
        image: vtkImageData or vtkPolyData

        """
        
        self.image.GetPointData().SetActiveScalars(field)
        if warp_scalars:
            geometry = vtk.vtkImageDataGeometryFilter()
            geometry.SetInputData(self.image)
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
            
            return warp.GetOutput()
        
        else:
            return self.image
    
    def display_image(self, z_min, z_max, field='Elevation',
                      warp_scalars=False):
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
            mapper.SetInputData(self.get_image(field, warp_scalars))
            
        else:
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(self.get_image(field, warp_scalars))
            
        mapper.SetLookupTable(mplcmap_to_vtkLUT(z_min, z_max))
        mapper.SetScalarRange(z_min, z_max)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetSize(2000, 1000)
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
        iren.AddObserver('UserEvent', cameraCallback)
        iren.Start()
    
    def write_plot_image(self, z_min, z_max, focal_point, camera_position,
                         field='Elevation', warp_scalars=False,
                         roll=0, image_scale=500, lower_threshold=-1000, 
                         upper_threshold=1000, mode='map', colorbar=True,
                         name=''):
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

        Returns
        -------
        None.

        """
        
        # Get image
        if warp_scalars:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.get_image(field, warp_scalars))
        else:
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(self.get_image(field, warp_scalars))
        mapper.SetLookupTable(mplcmap_to_vtkLUT(z_min, z_max))
        mapper.SetScalarRange(z_min, z_max)
        
        # Create actor and renderer        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer = vtk.vtkRenderer()
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
        renderWindow.SetSize(2000, 1000)
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
    
    def plot_image(self, z_min, z_max, cmap='inferno', figsize=(15, 15)):
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

        Returns
        -------
        f, ax : matplotlib figure and axes objects

        """
        
        # Use point mask to create array with NaN's where we have no data
        nan_image = copy.deepcopy(self.dsa_image.PointData['Elevation'])
        nan_image[self.dsa_image.PointData['vtkValidPointMask']==0] = np.NaN
        dims = self.image.GetDimensions()
        nan_image = nan_image.reshape((dims[1], dims[0]))
        
        # Plot
        f, ax = plt.subplots(1, 1, figsize=figsize)
        cf = ax.imshow(nan_image, cmap=cmap, aspect='equal', origin='lower',
                       vmin=z_min, vmax=z_max)
        f.colorbar(cf, ax=ax)
        ax.set_title(self.project_name)
        
        return f, ax
    
    def get_np_nan_image(self):
        """
        Convenience function for copying the image to a numpy object.

        Returns
        -------
        nan_image : numpy ndarray

        """
        
        # Use point mask to create array with NaN's where we have no data
        nan_image = copy.deepcopy(self.dsa_image.PointData['Elevation'])
        nan_image[self.dsa_image.PointData['vtkValidPointMask']==0] = np.NaN
        dims = self.image.GetDimensions()
        nan_image = nan_image.reshape((dims[1], dims[0]))
        
        return nan_image
    
    def merged_points_to_mesh(self, subgrid_x, subgrid_y, min_pts=100, 
                              alpha=0, overlap=0.1, x0=None, y0=None, wx=None,
                              wy=None, yaw=0):
        """
        Create mesh from all points in singlescans.
        
        Note, we use a delaunay2D filter to create this mesh. The filter
        encounters memory issues for large numbers of input points. So before
        the mesh creation step, we break the project up into a subgrid of 
        smaller chunks and then we apply the delaunay2D filter to each of 
        these and save the output in the mesh object.

        Parameters
        ----------
        subgrid_x : float
            x spacing, in m for the subgrid.
        subgrid_y : float
            y spacing, in m for the subgrid.
        min_pts : int, optional
            Minimum number of points for a subgrid region below which dont 
            include data from this region. The default is 100.
        alpha : float, optional
            Alpha value for vtkDelaunay2D filter. Inverse of how large of data
            gaps to interpolate over in m. The default is 0.
        overlap : float, optional
            Overlap value indicates how much overlap to permit between subgrid
            chunks in meters. The default is 0.1

        Returns
        -------
        None.

        """
        
        # Create kDTree
        kDTree = vtk.vtkKdTree()
        # If we only want to build mesh from a region of data:
        if x0 is not None:
            # Create box object
            box = vtk.vtkBox()
            box.SetBounds((0, wx, 0, wy, -10, 10))
            # We need a transform to put the data in the desired location relative to our
            # box
            transform = vtk.vtkTransform()
            transform.PostMultiply()
            transform.RotateZ(yaw)
            transform.Translate(x0, y0, 0)
            # That transform moves the box relative to the data, so the box takes its
            # inverse
            transform.Inverse()
            box.SetTransform(transform)
            
            # vtkExtractPoints does the actual filtering
            extractPoints = vtk.vtkExtractPoints()
            extractPoints.SetImplicitFunction(box)
            extractPoints.SetInputData(self.get_merged_points())
            extractPoints.Update()
            pdata_merged = extractPoints.GetOutput()
        else:
            pdata_merged = self.get_merged_points()
        kDTree.BuildLocatorFromPoints(pdata_merged.GetPoints())
        
        # Create Appending filter
        appendPolyData = vtk.vtkAppendPolyData()
        
        # Step through grid
        # Get the overall bounds of the data
        data_bounds = np.zeros(6)
        kDTree.GetBounds(data_bounds)
        x_min = data_bounds[0]
        x_max = data_bounds[1]
        y_min = data_bounds[2]
        y_max = data_bounds[3]
        z_min = data_bounds[4]
        z_max = data_bounds[5]
        while x_min < x_max:
            while y_min < y_max:
                # Create bounds and find points in area
                bounds = (x_min - overlap, x_min + subgrid_x + overlap, 
                          y_min - overlap, y_min + subgrid_y + overlap,
                          z_min, z_max)
                #print(bounds)
                ids = vtk.vtkIdTypeArray()
                kDTree.FindPointsInArea(bounds, ids)
                
                if (ids.GetNumberOfValues() < min_pts):
                    y_min = y_min + subgrid_y
                    continue
                
                # Create polydata with the found points
                pts = vtk.vtkPoints()
                pts.SetNumberOfPoints(ids.GetNumberOfValues())
                for i in np.arange(ids.GetNumberOfValues()):
                    pts.InsertPoint(i, pdata_merged.
                                        GetPoint(ids.GetValue(i)))
                pdata = vtk.vtkPolyData()
                pdata.SetPoints(pts)
                vertexFilter = vtk.vtkVertexGlyphFilter()
                vertexFilter.SetInputData(pdata)
                vertexFilter.Update()
                
                # Apply delaunay triangulation to this subgrid
                delaunay2D = vtk.vtkDelaunay2D()
                delaunay2D.SetAlpha(alpha)
                delaunay2D.SetInputData(vertexFilter.GetOutput())
                delaunay2D.Update()
                
                # Append resulting mesh
                appendPolyData.AddInputData(delaunay2D.GetOutput())
                appendPolyData.Update()
                
                # Update y_min
                y_min = y_min + subgrid_y
                
            # Return y_min to start again
            y_min = data_bounds[2]
            # Increment x_min by one
            x_min = x_min + subgrid_x
        
        # Store result in mesh
        self.mesh = appendPolyData.GetOutput()
    
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
    
    def get_merged_points(self, port=False, history_dict=False):
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

        Returns
        -------
        vtkPolyData.

        """
        git_hash = get_git_hash()
        
        # Create Appending filter and add all data to it
        appendPolyData = vtk.vtkAppendPolyData()
        for key in self.scan_dict:
            self.scan_dict[key].transformFilter.Update()
            connection, temp_hist_dict = self.scan_dict[key].get_polydata(
                port=True, history_dict=True)
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
            self.scan_dict[scan_name].write_npy_pdal(output_dir, mode=mode,
                                                     skip_fields=skip_fields)
            json_list.append({"filename": output_dir + self.project_name + 
                              '_' + scan_name + '.npy',
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
        if output_path:
            writer.SetFileName(output_path)
        else:
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
            writer.SetFileName(os.path.join(self.project_path, 
                                            self.project_name, 
                                            "vtkfiles" + suffix, 
                                            "meshes", name + ".vtp"))
        writer.Write()
        
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
        if mesh_path:
            reader.SetFileName(mesh_path)
        else:
            reader.SetFileName(os.path.join(self.project_path, 
                                            self.project_name, 
                                            "vtkfiles" + suffix, 
                                            "meshes", name + ".vtp"))
        reader.Update()
        self.mesh = reader.GetOutput()
    
    def create_empirical_cdf(self, bounds):
        """
        Creates an empirical distribution of heights from histogram.
        
        Currently sets the resolution of distribution to 1 mm vertically but
        could change that if needed.

        Parameters
        ----------
        bounds : six element tuple
            The boundaries of box of points to create distribution from.
            Format is (xmin, xmax, ymin, ymax, zmin, zmax)

        Returns
        -------
        None.

        """
        
        # Get all points within bounds
        box = vtk.vtkBox()
        box.SetBounds(bounds)
        appendPolyData = vtk.vtkAppendPolyData()
        for key in self.scan_dict:
            self.scan_dict[key].currentFilter.Update()
            extractPoints = vtk.vtkExtractPoints()
            extractPoints.SetImplicitFunction(box)
            extractPoints.SetExtractInside(1)
            extractPoints.SetInputData(self.scan_dict[key].get_polydata())
            extractPoints.Update()
            appendPolyData.AddInputData(extractPoints.GetOutput())
        
        appendPolyData.Update()
        
        # Get z-values of points and create cdf
        dsa_points = dsa.WrapDataObject(appendPolyData.GetOutput())
        z = dsa_points.Points[:, 2].squeeze()
        minh = np.min(z)
        maxh = np.max(z)
        nbins = int(1000*(maxh - minh))
        pdf, bin_edges = np.histogram(z,
                                  density=True, bins=nbins)
        x = (bin_edges[:-1] + bin_edges[1:])/2
        cdf = np.cumsum(pdf)/1000
        
        # Store result in empirical_cdf attribute
        self.empirical_cdf = (bounds, x, cdf)
    
    def create_empirical_cdf_image(self, z_min, z_max):
        """
        Creates an empirical distribution of heights from the image

        Parameters
        ----------
        z_min : float
            Minimum height value to consider.
        z_max : float
            Maximum height value to consider.

        Returns
        -------
        None.

        """
        
        # Check that image has been created
        if not hasattr(self, 'image'):
            raise RuntimeError('Need to create an image for project: ' + 
                               self.project_name)
        
        z = np.ravel(self.get_np_nan_image())
        z[z<z_min] = np.NaN
        z[z>z_max] = np.NaN
        minh = np.nanmin(z)
        maxh = np.nanmax(z)
        nbins = int(1000*(maxh - minh))
        pdf, bin_edges = np.histogram(z[~np.isnan(z)],
                                  density=True, bins=nbins)
        x = (bin_edges[:-1] + bin_edges[1:])/2
        cdf = np.cumsum(pdf)/1000
        
        # Store result in empirical_cdf attribute
        bounds_image = self.image.GetBounds()
        bounds = (bounds_image[0], bounds_image[1], bounds_image[2], 
                  bounds_image[3], z_min, z_max)
        self.empirical_cdf = (bounds, x, cdf)
    
    def create_im_gaus(self):
        """
        Transform image to gaussian using current value of empirical cdf

        Returns
        -------
        None.

        """
        
        # Check that image  and empirical cdf has been created
        if not hasattr(self, 'empirical_cdf'):
            raise RuntimeError('Need to create an empirical_cdf for project: '
                               + self.project_name)
        if not hasattr(self, 'image'):
            raise RuntimeError('Need to create an image for project: ' + 
                               self.project_name)
        
        # Create im_gaus field if it doesn't exist.
        if not 'im_gaus' in self.dsa_image.PointData.keys():
            arr = vtk.vtkFloatArray()
            arr.SetName('im_gaus')
            arr.SetNumberOfComponents(1)
            arr.SetNumberOfTuples(self.image.GetNumberOfPoints())
            arr.FillComponent(0, 0)
            self.image.GetPointData().AddArray(arr)
        
        # Fill im_gaus
        self.dsa_image.PointData['im_gaus'][:] = normalize(
            self.dsa_image.PointData['Elevation'],
            self.empirical_cdf[1], self.empirical_cdf[2])
    
    def add_theta(self, theta1, theta):
        """
        Adds theta attributes for specifying GMRF, see Rue and Held 2005 ch. 5

        Parameters
        ----------
        theta1 : float
            Scaling parameter theta1.
        theta : array
            Conditional covariances of neighbors, see Rue and Held Ch. 5.

        Returns
        -------
        None.

        """
        
        self.theta1 = theta1
        self.theta = theta
    
    def create_im_nan_border(self, buffer=2):
        """
        Creates a mask of which missing pixels not to infill.
        
        We don't want to infill certain pixels because either, they border the 
        boundary of our domain or more generally are conditionally dependent
        on a pixel outside the border of our domain (which we cannot know).
        We create this mask recursively starting at the boundary of the domain
        and then iteratively finding all missing pixels that are in contact
        with the boundary.

        Parameters
        ----------
        buffer : int, optional
            The width of the neighborhood around a pixel, same as m in the
            specification of theta. The default is 2.

        Returns
        -------
        None.

        """
        
        # Get logical array of missing pixels
        im_nan = np.array(np.isnan(self.dsa_image.PointData['im_gaus']).
                          reshape((self.image.GetDimensions()[1],
                                   self.image.GetDimensions()[0])), 
                          dtype='uint8')
        
        # Connected components filter
        retval, labels = cv.connectedComponents(im_nan)
        
        # First find missing blobs on boundary
        border = np.zeros(im_nan.shape, dtype='bool')
        border[:buffer,:] = True
        border[:,:buffer] = True
        border[-1*buffer:,:] = True
        border[:,-1*buffer:] = True
        
        im_nan_border = np.zeros(im_nan.shape, dtype='bool')
        
        for i in np.arange(labels.max()):
            if (np.logical_and((labels==i), border)).any():
                im_nan_border[(labels==i)] = True
        
        im_nan_border = np.logical_and(im_nan_border, im_nan, dtype='uint8')
        
        # Now repeatedly dilate missing area by the buffer, see if any missing
        # areas are within the neighborhood, add those that are, and repeat
        # until we've gotten all areas conditionally dependent on pixels
        # outside of boundary.
        kernel = np.ones((2*buffer+1, 2*buffer+1), dtype='uint8')
        while True:
            im_nan_border_dilate = cv.dilate(im_nan_border.astype(np.uint8), 
                                             kernel, iterations=1)
            im_nan_border2 = np.zeros(im_nan.shape, dtype='bool')
            for i in np.arange(labels.max()):
                if (np.logical_and((labels==i), im_nan_border_dilate)).any():
                    im_nan_border2[(labels==i)] = True
            im_nan_border2 = np.logical_and(im_nan_border2, im_nan, 
                                            dtype='uint8')
            
            if np.equal(im_nan_border, im_nan_border2).all():
                break
            else:
                im_nan_border = copy.deepcopy(im_nan_border2)
        
        # Create im_nan_border field if it doesn't exist.
        if not 'im_nan_border' in self.dsa_image.PointData.keys():
            arr = vtk.vtkUnsignedCharArray()
            arr.SetName('im_nan_border')
            arr.SetNumberOfComponents(1)
            arr.SetNumberOfTuples(self.image.GetNumberOfPoints())
            arr.FillComponent(0, 0)
            self.image.GetPointData().AddArray(arr)
        self.dsa_image.PointData['im_nan_border'][:] = np.ravel(im_nan_border)
        
    def dummy_im_nan_border(self):
        """
        Create an all false im_nan_border.
        
        In case we want to do pixel infilling on all pixels.

        Returns
        -------
        None.

        """
        
        # Create im_nan_border field if it doesn't exist.
        if not 'im_nan_border' in self.dsa_image.PointData.keys():
            arr = vtk.vtkUnsignedCharArray()
            arr.SetName('im_nan_border')
            arr.SetNumberOfComponents(1)
            arr.SetNumberOfTuples(self.image.GetNumberOfPoints())
            arr.FillComponent(0, 0)
            self.image.GetPointData().AddArray(arr)
        self.dsa_image.PointData['im_nan_border'][:] = 0
    
    def write_image(self, output_name=None):
        """
        Write the image out to a file.
        
        Particularly useful for saving im_nan_border

        Parameters
        ----------
        output_name : str, optional
            Output name for the file, if None use the project_name + 
            '_image.vti'. The default is None.

        Returns
        -------
        None.

        """
        
        # Create writer and write mesh
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetInputData(self.image)
        if output_name:
            writer.SetFileName(self.project_path + output_name)
        else:
            writer.SetFileName(self.project_path + self.project_name + 
                               '_image.vti')
        writer.Write()
    
    def read_image(self, image_path=None):
        """
        Read in the image from a file.

        Parameters
        ----------
        mesh_path : str, optional
            Path to the mesh, if none use project_path + project_name + 
            '_image.vti'. The default is None.

        Returns
        -------
        None.

        """
        
        # Create reader and read mesh
        reader = vtk.vtkXMLImageDataReader()
        if image_path:
            reader.SetFileName(image_path)
        else:
            reader.SetFileName(self.project_path + self.project_name + 
                               '_image.vti')
        reader.Update()
        
        # If we've already created an image delete and make way for this one
        if hasattr(self, 'image'):
            del(self.image)
        self.image = reader.GetOutput()
        
        # Also wrap with a datasetadaptor for working with numpy
        self.dsa_image = dsa.WrapDataObject(self.image)
        
    def create_gmrf(self):
        """
        Creates a GMRF for the missing data in im_gaus using theta1 and theta.
        
        This method implements our pixel infilling scheme, which is:
        represent the gaussian transformed image as a Gaussian Markov Random
        Field with a known, sparse precision matrix Q. The neighbors of each
        node are the points in a square around it (usually 5 pixels wide). If
        the whole image is a GMRF, then any subset A is also a GMRF with a 
        mean function that is conditional on the points not in the subset (
        we'll label these points B) and the precision matrix of A (QAA) is
        just a subset of Q. Specifically, we choose the missing pixels to be
        the subset A and the known ones to be our subset B. Then, following
        Rue and Held 2005 ch. 2 we can find the expectation of this GMRF and
        simulate draws from it.
        
        This function generates the attributes mu_agivenb and sparse_LAA

        Returns
        -------
        None.

        """
        
        if not (len(self.theta)==5):
            raise RuntimeError("Currently we assume m = 2 only.")
        # Create sparse_Q, precision matrix for the entire image
        sparse_Q = theta_to_Q(self.image.GetDimensions()[1], 
                              self.image.GetDimensions()[0], 
                              2, self.theta) * self.theta1
        sparse_Q = sparse_Q.tocsr()
        
        # SHOULD REALLY CHECK THAT ALL OF THIS INDEXING IS WORKING AS PLANNED
        # Find indices of missing pixels and subset Q accordingly
        if not 'im_nan_border' in self.dsa_image.PointData.keys():
            raise RuntimeError('Must create im_nan_border before gmrf')
        ind_a = np.logical_and(np.isnan(self.dsa_image.PointData['im_gaus']),
                       ~np.array(self.dsa_image.PointData['im_nan_border'],
                                 dtype='bool'))
        ind_b = ~np.isnan(self.dsa_image.PointData['im_gaus'])
        i_b = np.argwhere(ind_b)
        #i_a = np.argwhere(ind_missing) not needed
        i_b = np.argwhere(ind_b)
        sparse_QAA = sparse_Q[ind_a,:][:,ind_a]
        sparse_QAB = sparse_Q[ind_a,:][:,ind_b]
        
        # Compute the b a given b parameter of the canonical conditional dist
        b_agivenb = -1*sparse_QAB.dot(self.dsa_image.PointData['im_gaus'][i_b])
        # And conditional mean Q*mu = b, so for the conditional distribution 
        # Q=QAA
        self.mu_agivenb = sp.linalg.spsolve(sparse_QAA, b_agivenb)
        
        # create sparse_LAA to sample from the posterior
        # In order to sample from the posterior we need to find the cholesky 
        # factor of QAA, let's see if we can do this with SuperLU
        options = dict(SymmetricMode=True)
        # set permute colum specification to natural to prevent it from 
        # permuting matrix, mild computation hit but makes coding easier
        splu_QAA = sp.linalg.splu(sparse_QAA.tocsc(), permc_spec='NATURAL', 
                        options=options)
        
        # The L factor in SuperLU is normalized with principal diagonal elements 
        # equal to 1.
        # Follow this reference for computing L (cholesky)
        # https://people.eecs.berkeley.edu/~demmel/ma221_Fall11/Lectures/
        # Lecture_13.html
        d = np.sqrt(splu_QAA.U.diagonal(0))
        D = sp.diags(d)
        self.sparse_LAA = splu_QAA.L.dot(D)
    
    def create_im_gaus_mean_fill(self):
        """
        Fill missing pixels in im_gaus with expectation (mu_agivenb)

        Returns
        -------
        None.

        """
        
        # Create im_gaus_mean_fill field if it doesn't exist.
        if not 'im_gaus_mean_fill' in self.dsa_image.PointData.keys():
            arr = vtk.vtkFloatArray()
            arr.SetName('im_gaus_mean_fill')
            arr.SetNumberOfComponents(1)
            arr.SetNumberOfTuples(self.image.GetNumberOfPoints())
            arr.FillComponent(0, 0)
            self.image.GetPointData().AddArray(arr)
        
        # Fill im_gaus_mean_fill
        self.dsa_image.PointData['im_gaus_mean_fill'][:] = copy.deepcopy(
            self.dsa_image.PointData['im_gaus'])
        ind_a = np.logical_and(np.isnan(self.dsa_image.PointData['im_gaus']),
                       ~np.array(self.dsa_image.PointData['im_nan_border'],
                                 dtype='bool'))
        print(ind_a.sum())
        i_a = np.argwhere(ind_a)
        print(self.dsa_image.PointData['im_gaus_mean_fill'][i_a].shape)
        self.dsa_image.PointData['im_gaus_mean_fill'][i_a] = self.mu_agivenb[
            :, np.newaxis]
    
    def create_elevation_mean_fill(self):
        """
        Fill missing pixels in Elevation with transformed expectation

        Returns
        -------
        None.

        """
        # Create im_gaus_mean_fill field if it doesn't exist.
        if not 'Elevation_mean_fill' in self.dsa_image.PointData.keys():
            arr = vtk.vtkFloatArray()
            arr.SetName('Elevation_mean_fill')
            arr.SetNumberOfComponents(1)
            arr.SetNumberOfTuples(self.image.GetNumberOfPoints())
            arr.FillComponent(0, 0)
            self.image.GetPointData().AddArray(arr)
        
        # Fill Elevation_mean_fill
        self.dsa_image.PointData['Elevation_mean_fill'][:] = copy.deepcopy(
            self.dsa_image.PointData['Elevation'])
        ind_a = np.logical_and(np.isnan(self.dsa_image.PointData['im_gaus']),
                       ~np.array(self.dsa_image.PointData['im_nan_border'],
                                 dtype='bool'))
        i_a = np.argwhere(ind_a)
        #print(self.dsa_image.PointData['im_gaus_mean_fill'][i_a].shape)
        self.dsa_image.PointData['Elevation_mean_fill'][i_a] = inormalize(
            self.mu_agivenb[:, np.newaxis], 
            self.empirical_cdf[1],
            self.empirical_cdf[2])
    
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
        
    Methods
    -------
    add_project(project_name)
        Add a project in project_path directory to project_dict
    compare_reflectors(project_name_0, project_name_1, delaunay=False, 
                       mode='dist')
        Calculate pwdists and plot reflector comparison project 0 to project 1
    register_project(project_name_0, project_name_1, reflector_list, mode='lS')
        Register project 1 to project 0 using the reflectors in reflector_list.
    add_registration_tuple(registration_tuple, index=None)
        Add a registration tuple to the registration list.
    del_registration_tuple(index)
        Delete registration tuple from registration_list.
    register_all()
        Register all projects according to registration list.
    apply_snowflake_filter(shells)
        Apply a snowflake filter to each scan in each project.
    apply_snowflake_filter_2(z_diff, N, r_min)
        Apply a snowflake filter based on z difference with nearby points.
    merged_points_to_mesh(self, subgrid_x, subgrid_y, min_pts=100, 
                          alpha=0, overlap=0.1)
        For each project convert pointcloud to mesh.
    mesh_to_image(z_min, z_max, nx, ny, dx, dy, x0, y0)
        Interpolate mesh into image.
    difference_projects(project_name_0, project_name_1)
        Subtract project_0 from project_1 and store the result in 
        difference_dict.
    
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
        Align successive scans on the basis of their gridded minima

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
                              bin_reduc_op='min'):
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

        Returns
        -------
        diff : ndarray
            Array containing gridded minima differences

        """
        if project_name_0==project_name_1:
            self.project_dict[project_name_0].add_z_offset(0)
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
        pdata_merged_project_0 = project_0.get_merged_points()
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
        
        # Now use the same grid to get on ss
        ss = project_1.scan_dict[scan_name]

        # Get points as an array
        ss_points_np = vtk_to_numpy(ss.currentFilter.GetOutput().GetPoints()
                                    .GetData())
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

        # Return the diff
        return frac_exceed_max_diff, diff
    
    def kernel_alignment(self, project_name_0, project_name_1, bin_width=0.15, 
                         max_points=800000, max_dist=250, blur=0.005,
                         max_steps=100, cutoff_t=0.0005, cutoff_r=0.000005,
                         plot_optimization=False):
        """
        Align each single scan in project 1 with project 0 using geomloss

        This function requires you to be running python from the docker group
        and have the jupyter-keops container running. We will first subset
        the two scans to areas with high overlapping point density. Then,
        we apply compute_kernel_alignment.py to find the best fitting rigid
        transformation. Finally, we load in this transformation.

        This function updates the current_transform for each single scan and
        applies current_transform to each scan.
        
        Parameters:
        -----------
        project_name_0 : str
            The reference project we're trying to align project_1 with
        project_name_1 : str
            The project we're aligning with project_0
        bin_width : float
            Bin side length (in m) for when we downsample to areas with shared
            high point density. The default is 0.15.
        max_points : int
            The maximum number of points from either scan to use, larger numbers
            will increase runtimes. The default is 800000.
        max_dist : float
            The maximum distance from the scanner position of the scan we're
            trying to align to look for points. The default is 250.
        blur : float
            Gaussian blur sigma for point alignment in m. The default is 0.005.
        max_steps : int
            The maximum number of steps that the optimization should take. The
            default is 100.
        cutoff_t : float
            Threshold by which if all translations are less than cutoff_t and 
            all rotations are less than cutoff_r for the current step stop
            optimization. The default is 0.0005.
        cutoff_r : float
            Threshold by which if all translations are less than cutoff_t and 
            all rotations are less than cutoff_r for the current step stop
            optimization. The default is 0.000005.
        plot_optimization : bool
            Whether to plot the optimization. The default is False.

        Returns
        -------
        None.

        """

        for scan_name in self.project_dict[project_name_1].scan_dict:
            print(scan_name)
            t0 = time.perf_counter()
            self.kernel_alignment_ss(project_name_0, project_name_1, scan_name,
                                     bin_width=bin_width, max_points=max_points,
                                     max_dist=max_dist, blur=blur, max_steps=
                                     max_steps, cutoff_t=cutoff_t, cutoff_r=
                                     cutoff_r, plot_optimization=
                                     plot_optimization)
            t1 = time.perf_counter()
            print(t1-t0)
        # Apply current transform
        self.project_dict[project_name_1].apply_transforms([
                                                           'current_transform'])
    def kernel_alignment_ss(self, project_name_0, project_name_1, scan_name,
                            bin_width=0.15, max_points=800000, max_dist=250, 
                            blur=0.005, max_steps=100, cutoff_t=0.0005, 
                            cutoff_r=0.000005, plot_optimization=False):
        """
        Align single scan in project 1 with project 0 using geomloss

        This function requires you to be running python from the docker group
        and have the jupyter-keops container running. We will first subset
        the two scans to areas with high overlapping point density. Then,
        we apply compute_kernel_alignment.py to find the best fitting rigid
        transformation. Finally, we load in this transformation.

        This function will load the 
        
        Parameters:
        -----------
        project_name_0 : str
            The reference project we're trying to align project_1 with
        project_name_1 : str
            The project we're aligning with project_0
        bin_width : float
            Bin side length (in m) for when we downsample to areas with shared
            high point density. The default is 0.15.
        max_points : int
            The maximum number of points from either scan to use, larger numbers
            will increase runtimes. The default is 800000.
        max_dist : float
            The maximum distance from the scanner position of the scan we're
            trying to align to look for points. The default is 250.
        blur : float
            Gaussian blur sigma for point alignment in m. The default is 0.005.
        max_steps : int
            The maximum number of steps that the optimization should take. The
            default is 100.
        cutoff_t : float
            Threshold by which if all translations are less than cutoff_t and 
            all rotations are less than cutoff_r for the current step stop
            optimization. The default is 0.0005.
        cutoff_r : float
            Threshold by which if all translations are less than cutoff_t and 
            all rotations are less than cutoff_r for the current step stop
            optimization. The default is 0.000005.
        plot_optimization : bool
            Whether to plot the optimization. The default is False.

        Returns
        -------
        None.

        """

        git_hash = get_git_hash()
        
        # Create docker project path
        docker_project_path = os.path.join('/mosaic_lidar/', 
                                           self.project_path.rsplit('/',2)[1])

        # Get pointclouds as numpy files and their history dicts
        project_0 = self.project_dict[project_name_0]
        ss = self.project_dict[project_name_1].scan_dict[scan_name]

        pdata_merged_project_0, history_dict_project_0 = (project_0
            .get_merged_points(history_dict=True))
        project_0_points_np = vtk_to_numpy(pdata_merged_project_0
                                           .GetPoints().GetData())
        project_0_class_np = vtk_to_numpy(pdata_merged_project_0.GetPointData()
                                          .GetArray('Classification'))
        ss_points_np = vtk_to_numpy(ss.currentFilter.GetOutput().GetPoints()
                                            .GetData())
        ss_PointId = vtk_to_numpy(ss.currentFilter.GetOutput().GetPointData().
                                  GetArray('PointId'))
        history_dict_ss = json.loads(json.dumps(ss.filt_history_dict))

        # Create Grid
        w = [bin_width, bin_width]
        bounds = pdata_merged_project_0.GetBounds()
        edges = 2*[None]
        nbin = np.empty(2, np.int_)
        for i in range(2):
            edges[i] = np.arange(int(np.ceil((bounds[2*i + 1] - 
                                              bounds[2*i])/w[i]))
                                 + 1, dtype=np.float32) * w[i] + bounds[2*i]
            # Adjust lower edge so we don't miss lower most point
            edges[i][0] = edges[i][0] - 0.00001
            # Adjust uppermost edge so the bin width is appropriate
            edges[i][-1] = bounds[2*i + 1] + 0.00001
            nbin[i] = len(edges[i]) + 1

        # Bin single scan
        ss_Ncount = tuple(np.searchsorted(edges[i], ss_points_np[:,i], 
                                          side='right') for i in range(2))
        ss_xy = np.ravel_multi_index(ss_Ncount, nbin)
        del ss_Ncount
        ss_counts = np.bincount(ss_xy, minlength=nbin.prod())

        # Bin Project
        project_0_Ncount = tuple(np.searchsorted(edges[i], 
                                                 project_0_points_np[:,i], 
                                                 side='right') 
                                 for i in range(2))
        project_0_xy = np.ravel_multi_index(project_0_Ncount, nbin)
        del project_0_Ncount
        project_0_counts = np.bincount(project_0_xy, minlength=nbin.prod())

        # Create array with min_counts per bucket
        min_counts = np.minimum(project_0_counts, ss_counts)

        # Sort by min_counts
        sort_ind = np.argsort(min_counts)[::-1]

        # Now starting at the bucket with the largest min number of points, 
        # step through the buckets until we've gathered as many points as we 
        # can fit under max_points
        project_0_N = 0
        ss_N = 0
        for i in range(len(sort_ind)):
            if ((project_0_N+project_0_counts[sort_ind[i]]>max_points) 
                or (ss_N+ss_counts[sort_ind[i]]>max_points)):
                break
            else:
                project_0_N += project_0_counts[sort_ind[i]]
                ss_N += ss_counts[sort_ind[i]]

        bin_ind = sort_ind[0:i]

        # Use isin to get points
        ss_bool = np.isin(ss_xy, bin_ind)
        project_0_bool = np.isin(project_0_xy, bin_ind)

        # The project points can be written directly to files, first we check
        # if a folder already exists in the temp directory.
        scan_0_path = os.path.join(self.project_path, 'temp', 'kernel_scan_0')
        d_scan_0_path = os.path.join(docker_project_path, 'temp', 
                                     'kernel_scan_0')
        if not os.path.isdir(scan_0_path):
            os.mkdir(scan_0_path)
        # Delete old files
        for f in os.listdir(scan_0_path):
            os.remove(os.path.join(scan_0_path, f))
        # Now save project_0
        # Points
        np.save(os.path.join(scan_0_path, 'Points.npy'), project_0_points_np[
                project_0_bool, :])
        # Classification
        np.save(os.path.join(scan_0_path, 'Classification.npy'), 
                project_0_class_np[project_0_bool])
        # Create and save the history dict
        history_dict = {
            "type": "Filter",
            "git_hash": git_hash,
            "method": "ScanArea.kernel_alignment_ss",
            "input_0": history_dict_project_0,
            "params": {
                "bin_width": bin_width,
                "max_points": max_points,
                "other_pointset": history_dict_ss
                }
            }
        f = open(os.path.join(scan_0_path, 'raw_history_dict.txt'), 'w')
        json.dump(history_dict, f, indent=4)
        f.close()
        # Save a blank transform
        trans = np.array([(0, 0, 0, 
                          0, 0, 0)],
                          dtype=[('x0', '<f8'), ('y0', '<f8'), 
                                 ('z0', '<f8'), ('u0', '<f8'),
                                 ('v0', '<f8'), ('w0', '<f8')])
        np.save(os.path.join(scan_0_path, 'current_transform.npy'), trans)
        # Now create a blank transform source (indicates identity trans), save
        history_dict = {
            "type": "Transform Source",
            "git_hash": git_hash,
            "method": "ScanArea.kernel_alignment_ss",
            "filename": ''
            }
        f = open(os.path.join(scan_0_path, 'current_transform.txt'), 'w')
        json.dump(history_dict, f, indent=4)
        f.close()

        # For scan 1 we want to save untransformed points. So first we use
        # pedigree id selection to get these points
        ss_subset_pointid = ss_PointId[ss_bool]
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(1) # we want to select points
        selectionNode.SetContentType(2) # 2 corresponds to pedigreeIds
        selectionNode.SetSelectionList(numpy_to_vtk(ss_subset_pointid, 
                                                    array_type=
                                                    vtk.VTK_UNSIGNED_INT))
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)
        extractSelection = vtk.vtkExtractSelection()
        extractSelection.SetInputData(0, ss.polydata_raw)
        extractSelection.SetInputData(1, selection)
        extractSelection.Update()
        pdata = extractSelection.GetOutput()

        ss_points_write = vtk_to_numpy(pdata.GetPoints().GetData())
        ss_classification_write = vtk_to_numpy(pdata.GetPointData()
                                               .GetArray('Classification'))
        # Now write selected points to files
        # if a folder already exists in the temp directory.
        scan_1_path = os.path.join(self.project_path, 'temp', 'kernel_scan_1')
        d_scan_1_path = os.path.join(docker_project_path, 'temp', 
                                     'kernel_scan_1')
        if not os.path.isdir(scan_1_path):
            os.mkdir(scan_1_path)
        # Delete old files
        for f in os.listdir(scan_1_path):
            os.remove(os.path.join(scan_1_path, f))
        # Now save scan 1
        # Points
        np.save(os.path.join(scan_1_path, 'Points.npy'), ss_points_write)
        # Classification
        np.save(os.path.join(scan_1_path, 'Classification.npy'), 
                ss_classification_write)
        # Create and save the history dict
        history_dict = {
            "type": "Transformer",
            "git_hash": git_hash,
            "method": "ScanArea.kernel_alignment_ss",
            "input_0": {
                "type": "Filter",
                "git_hash": git_hash,
                "method": "ScanArea.kernel_alignment_ss",
                "input_0": history_dict_ss,
                "params": {
                    "bin_width": bin_width,
                    "max_points": max_points,
                    "other_pointset": history_dict_project_0
                    }
                },
            "input_1": {
                "type": "Invert Transform",
                "input_0": ss.transformed_history_dict["input_1"]
                }
            }
        f = open(os.path.join(scan_1_path, 'raw_history_dict.txt'), 'w')
        json.dump(history_dict, f, indent=4)
        f.close()

        # Now we can create our command for the kernel optimization
        cmd = ("docker exec -w /code/pydar/keops jupyter-keops python " +
               "compute_kernel_alignment.py --set_paths_directly --scan_1_path "
               + d_scan_1_path + " --trans_1_path " + os.path.join(
                '/mosaic_lidar/', docker_project_path, 
                project_name_1, 'transforms', scan_name) + 
               " --scan_0_paths " + d_scan_0_path + " --trans_0_paths " +
               d_scan_0_path + " --max_dist " + str(max_dist) + " --max_pts " 
               + str(max_points) + " --blur " + str(blur) + " --max_steps " + 
               str(max_steps) + " --cutoff_t " + str(cutoff_t) + " --cutoff_r "
               + str(cutoff_r) + " --git_hash " + git_hash + 
               " --plot_optimization --plot_output_path " + os.path.join(
               docker_project_path, 'snapshots'))

        # Delete all unneeded objects to free up some memory
        del history_dict, ss_classification_write, ss_points_write
        del pdata, extractSelection, selection, selectionNode, ss_subset_pointid
        del project_0_bool, ss_bool, bin_ind, sort_ind, min_counts
        del project_0_counts, project_0_xy, ss_counts, ss_xy, nbin, edges
        del history_dict_ss, ss_PointId, ss_points_np, project_0_class_np
        del project_0_points_np, pdata_merged_project_0, history_dict_project_0

        # Execute command
        os.system(cmd)

        # Update the current transform in our single scan
        ss.read_transform()

    def apply_snowflake_filter(self, shells):
        """
        Apply a snowflake filter to each project.

        Parameters
        ----------
        shells : array-like of tuples
            shells is an array-like set of tuples where each tuple is four
            elements long (inner_r, outer_r, point_radius, neighbors). *_r
            define the inner and outer radius of a halo defining shell. 
            point_radius is radius for vtkRadiusOutlierRemoval to look at
            (if 0, remove all points). Neighbors is number of neighbors that
            must be within point_radius (if 0, keep all points)

        Returns
        -------
        None.

        """
        
        for key in self.project_dict:
            self.project_dict[key].apply_snowflake_filter(shells)
        
    def apply_snowflake_filter_2(self, z_diff, N, r_min):
        """
        Filter snowflakes based on their vertical distance from nearby points.
        
        Here we exploit the fact that snowflakes (and for that matter power
        cables and some other human objects) are higher than their nearby
        points. The filter steps through each point in the transformed
        dataset and compares it's z value with the mean of the z-values of
        the N closest points. If the difference exceeds z_diff then set the
        Classification for that point to be 65. Also, there is a shadow around the
        base of the scanner so all points within there must be spurious. We
        filter all points within r_min

        Parameters
        ----------
        z_diff : float
            Maximum vertical difference in m a point can have from its 
            neighborhood.
        N : int
            Number of neighbors to find.
        r_min : float
            Radius of scanner in m within which to filter all points.

        Returns
        -------
        None.
        """
        
        for key in self.project_dict:
            print(key)
            self.project_dict[key].apply_snowflake_filter_2(z_diff, N, r_min)
    
    def merged_points_to_mesh(self, subgrid_x, subgrid_y, min_pts=100, 
                              alpha=0, overlap=0.1, sub_list=[]):
        """
        Create mesh from all points in singlescans.
        
        Note, we use a delaunay2D filter to create this mesh. The filter
        encounters memory issues for large numbers of input points. So before
        the mesh creation step, we break the project up into a subgrid of 
        smaller chunks and then we apply the delaunay2D filter to each of 
        these and save the output in the mesh object.

        Parameters
        ----------
        subgrid_x : float
            x spacing, in m for the subgrid.
        subgrid_y : float
            y spacing, in m for the subgrid.
        min_pts : int, optional
            Minimum number of points for a subgrid region below which dont 
            include data from this region. The default is 100.
        alpha : float, optional
            Alpha value for vtkDelaunay2D filter. Inverse of how large of data
            gaps to interpolate over in m. The default is 0.
        overlap : float, optional
            Overlap value indicates how much overlap to permit between subgrid
            chunks in meters. The default is 0.1
        sub_list : list, optional
            List of project names to apply to if not whole project.

        Returns
        -------
        None.

        """
        
        if len(sub_list)==0:
            for key in self.project_dict:
                print(key)
                self.project_dict[key].merged_points_to_mesh(subgrid_x, 
                                                             subgrid_y,
                                                             min_pts, alpha, 
                                                             overlap)
        else:
            for key in sub_list:
                print(key)
                self.project_dict[key].merged_points_to_mesh(subgrid_x, 
                                                             subgrid_y,
                                                             min_pts, alpha, 
                                                             overlap)
        
    def mesh_to_image(self, nx, ny, dx, dy, x0, y0, yaw=0, sub_list=[]):
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
                                                     yaw=yaw)
        else:
            for key in sub_list:
                self.project_dict[key].mesh_to_image(nx, ny, dx, dy, x0, y0,
                                                     yaw=yaw)
            
    def difference_projects(self, project_name_0, project_name_1, 
                            difference_field='Elevation'):
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

        Returns
        -------
        None.

        """
        
        # Difference projects and copy image to difference dict
        # assume projects have the same sized images covering same extent
        # Create image
        im = vtk.vtkImageData()
        im.SetDimensions(self.project_dict[project_name_0].image.
                         GetDimensions())
        im.SetOrigin(self.project_dict[project_name_0].image.
                     GetOrigin())
        im.SetSpacing(self.project_dict[project_name_0].image.
                      GetSpacing())
        arr = vtk.vtkFloatArray()
        arr.SetNumberOfValues(self.project_dict[project_name_0].image.
                              GetNumberOfPoints())
        arr.SetName('Difference')
        im.GetPointData().SetScalars(arr)
        self.difference_dsa_dict[(project_name_0, project_name_1)] = (
            dsa.WrapDataObject(im))
        # Difference images
        self.difference_dsa_dict[(project_name_0, project_name_1)].PointData[
            'Difference'][:] = (
            self.project_dict[project_name_1].dsa_image.PointData[
                difference_field]
            - self.project_dict[project_name_0].dsa_image.PointData[
                difference_field])
        
        # np.ravel(
        #     self.project_dict[project_name_1].get_np_nan_image() -
        #     self.project_dict[project_name_0].get_np_nan_image())
        # # Assign value
        self.difference_dict[(project_name_0, project_name_1)] = im
    
    def display_difference(self, project_name_0, project_name_1, diff_window,
                           cmap='rainbow'):
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
                                                  project_name_1)])
        mapper.SetLookupTable(mplcmap_to_vtkLUT(-diff_window, diff_window,
                                                name=cmap))
        mapper.SetScalarRange(-diff_window, diff_window)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
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
                                cmap='rainbow'):
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
        merge = vtk.vtkMergeFilter()
        merge.SetGeometryInputData(self.project_dict[project_name_1].
                                   get_image(field=field, warp_scalars=True))
        merge.SetScalarsData(self.difference_dict[(project_name_0,
                                                  project_name_1)])
        merge.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(merge.GetOutput())
        mapper.SetLookupTable(mplcmap_to_vtkLUT(-diff_window, diff_window,
                                                name=cmap))
        mapper.SetScalarRange(-diff_window, diff_window)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
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
        
    def write_plot_warp_difference(self, project_name_0, project_name_1, 
                                diff_window, camera_position, focal_point,
                                roll=0,
                                field='Elevation_mean_fill',
                                cmap='rainbow', filename="", name="",
                                light=None, colorbar=True):
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

        Returns
        -------
        None.

        """
        # Merge filter combines the geometry from project_name_1 with scalars
        # from difference
        merge = vtk.vtkMergeFilter()
        merge.SetGeometryInputData(self.project_dict[project_name_1].
                                   get_image(field=field, warp_scalars=True))
        merge.SetScalarsData(self.difference_dict[(project_name_0,
                                                  project_name_1)])
        merge.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(merge.GetOutput())
        mapper.SetLookupTable(mplcmap_to_vtkLUT(-diff_window, diff_window,
                                                name=cmap))
        mapper.SetScalarRange(-diff_window, diff_window)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetSize(2000, 1000)
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
                                 diff_window, filename="", colorbar=True):
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

        Returns
        -------
        None.

        """
        
        # Create mapper
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(self.difference_dict[(project_name_0, 
                                                  project_name_1)])
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
    
    def difference_histogram(self, ax, project_name_0, project_name_1, 
                            difference_field='Elevation'):
        """
        

        Parameters
        ----------
        ax : matplotlib axis object
            Axis to plot on.
        project_name_0 : str
            Name of project to subtract (usually older).
        project_name_1 : str
            Name of project to subtract from (usually younger).
        difference_field : str, optional
            Which field in ImageData to use. The default is 'Elevation'
        
        Returns
        -------
        None.

        """
    
class ClassFilter:
    """
    Filter points according to the classification field
    
    Attributes
    ----------
    input_connection : vtkAlgorithmOutput
    class_list : list
        List of categories this filter will return, if special value: 'all'
        Then we do not have a selection filter and we pass through all points
    filter : vtkPolyDataAlgorithm
        The filter
    selectionNode : vtkSelectionNode
        Selection node object for selecting points
    selection : vtkSelection
        selection object
    
    Methods
    -------
    CreateFilter(class_list)
        Creates the desired filter. Overwrites existing if present.
    Update()
        update filters (e.g. call after upstream Modified)
    GetOutputPort()
        Returns the output port
    GetOutput()
        Returns the output as a PolyData
        
    """
    
    def __init__(self, input_connection, class_list=[0, 1, 2, 70]):
        """
        Create objects and pipeline

        Parameters
        ----------
        input_connection : vtkAlgorithmOutput
            vtk input connection to this filter
        class_list : list, optional
            List of categories to include in the output. If 'all' then we will
            do no filtering. The default is [0, 1, 2, 70].

        Returns
        -------
        None.

        """
        
        self.input_connection = input_connection
        self.CreateFilter(class_list)
    
    def CreateFilter(self, class_list):
        """
        Create desired filter replaces existing if one exists.

        Parameters
        ----------
        class_list : list, optional
            List of categories to include in the output. If 'all' then we will
            do no filtering. The default is [0, 1, 2, 70].

        Returns
        -------
        None.

        """
        
        self.class_list = class_list
        if class_list=='all':
            self.filter = vtk.vtkTransformPolyDataFilter()
            self.filter.SetTransform(vtk.vtkTransform())
        else:
            self.selectionList = vtk.vtkUnsignedCharArray()
            for v in self.class_list:
                self.selectionList.InsertNextValue(v)
            self.selectionNode = vtk.vtkSelectionNode()
            self.selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
            self.selectionNode.SetContentType(vtk.vtkSelectionNode.VALUES)
            self.selectionNode.SetSelectionList(self.selectionList)
            
            self.selection = vtk.vtkSelection()
            self.selection.AddNode(self.selectionNode)
            
            self.filter = vtk.vtkExtractSelection()
            self.filter.SetInputData(1, self.selection)
        self.filter.SetInputConnection(0, self.input_connection)
        self.filter.Update()
    
    def Update(self):
        """
        Updates pipeline

        Returns
        -------
        None.

        """
        
        self.filter.Update()
    
    def GetOutput(self):
        """
        Returns polydata

        Returns
        -------
        vtkPolyData

        """
        
        return self.filter.GetOutput()
    
    def GetOutputPort(self):
        """
        Returns output port.

        Returns
        -------
        vtkAlgorithmOutput

        """
        
        return self.filter.GetOutputPort()
    
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
    else:
        return None
    
    # Looks like Dec 6 was an exception to the format...
    if date=='2019-06-12':
        date = '2019-12-06'
    
    # Handle April 8 b scan case.
    if project_name[seq_match.end()]=='b':
        date = date + ' 12:00:00'
        
    return date

def get_man_class(project_tuples):
    """
    Returns a dataframe with all of the requested manual classifications.

    Parameters
    ----------
    project_tuples : list
        List of tuples (project_path, project_name) to try loading manual
        classifications from.

    Returns
    -------
    pandas dataframe of manual classifications.

    """
    
    df_list = []
    
    for project_tuple in project_tuples:
        # if the manual classification folder exists
        man_class_path = os.path.join(project_tuple[0], project_tuple[1],
                          'manualclassification')
        if os.path.isdir(man_class_path):
            filenames = os.listdir(man_class_path)
            for filename in filenames:
                df_list.append(pd.read_parquet(os.path.join(man_class_path, filename),
                                               engine="pyarrow"))
    return pd.concat(df_list, ignore_index=True)

def normalize(xx, x, cdf):
    """
    Use the inverse-cdf approach to transform a dataset to normal distribution

    Parameters
    ----------
    xx : ndarray
        The data to transform (should be in empirical distribution).
    x : 1d-array
        Bin values in empirical cdf.
    cdf : 1d-array
        Values of empirical cdf.

    Returns
    -------
    ndarray
        The gaussian transformed data, same shape as xx.

    """
    # Flatten xx
    shp = xx.shape
    xx = xx.reshape(xx.size)
    
    # First adjust CDF to meet erfinv convention
    c = 2*(cdf-0.5)
    
    # Next find the left nearest indices in array 
    sidx = np.argsort(xx)
    xxsort = xx[sidx]
    xidx = np.searchsorted(x, xxsort, side='left')
    # Now get the values transformed to a uniform distribution
    u = c[xidx-1][np.argsort(sidx)]
    u[np.isnan(xx)] = np.nan
    u = u.reshape(shp)
    # And return values transfored to standard normal distribution
    return np.sqrt(2)*erfinv(u)

def inormalize(yy, x, cdf):
    """
    Use the inverse-cdf approach to transform normal data to an empirical dist.

    Parameters
    ----------
    yy : ndarray
        The normally distributed data to transform.
    x : 1d-array
        Bin values in empirical cdf.
    cdf : 1d-array
        Values of empirical cdf.

    Returns
    -------
    ndarray
        The transformed data, same shape as yy.

    """
    # Flatten yy
    shp = yy.shape
    yy = yy.reshape(yy.size)
    
    # transform to uniform dist
    u = 1/2*(1 + erf(yy/np.sqrt(2)))
    
    # Next use cdf as mapping into x
    sidx = np.argsort(u)
    usort = u[sidx]
    cidx = np.searchsorted(cdf, usort, side='left')
    xx = x[cidx-1][np.argsort(sidx)]
    xx[np.isnan(yy)] = np.NaN
    return xx.reshape(shp)

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
        ValueError: mode must be either 'normal' or 'fourier'
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
        raise ValueError("mode must be either 'normal' or 'fourier'")

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

def block_circ_base(n0, n1, m, theta, c=None):
    """
    Generates a block circulant base with parameters given by theta
    
    See Rue and Held 2005. p. 196

    Parameters
    ----------
    n0 : int
        Size of the domain in the row direction.
    n1 : int
        Size of the domain in the column direction.
    m : int
        Usually 2 or 3, half width of neighborhood.
    theta : 1d-array
        Parameter values of precision matrix, ordered in the same manner as 
        Rue and Held 2005 p. 196.
    c : ndarray, optional
        If given, this is the array to place the block cirulant base into.

    Returns
    -------
    n0xn1 array or none.
        If c is not given then return block circulant base. If c is given 
        modify in-place and don't return anything.

    """
    
    # Create or clear c
    if c is None:
        return_c = True
        c = np.zeros((n0, n1))
    else:
        return_c = False
        c[:] = 0
    
    # Set the origin precision equal to 1
    c[0, 0] = 1
    
    if m>=1:
        # theta2/theta1
        c[0, 1] = theta[0]
        c[1, 0] = theta[0]
        c[0, -1] = theta[0]
        c[-1, 0] = theta[0]
        
        # theta3/theta1
        c[1, 1] = theta[1]
        c[1, -1] = theta[1]
        c[-1, 1] = theta[1]
        c[-1, -1] = theta[1]
    
        if m>=2:
            # theta4/theta1
            c[0, 2] = theta[2]
            c[2, 0] = theta[2]
            c[0, -2] = theta[2]
            c[-2, 0] = theta[2]
            
            # theta5/theta1
            c[1, 2] = theta[3]
            c[2, 1] = theta[3]
            c[-1, -2] = theta[3]
            c[-2, -1] = theta[3]
            c[-1, 2] = theta[3]
            c[2, -1] = theta[3]
            c[1, -2] = theta[3]
            c[-2, 1] = theta[3]
            
            # theta6/theta1
            c[2, 2] = theta[4]
            c[2, -2] = theta[4]
            c[-2, 2] = theta[4]
            c[-2, -2] = theta[4]
            
            if m>=3:
                # theta7/theta1
                c[0, 3] = theta[5]
                c[3, 0] = theta[5]
                c[0, -3] = theta[5]
                c[-3, 0] = theta[5]
                
                # theta8/theta1
                c[1, 3] = theta[6]
                c[3, 1] = theta[6]
                c[-1, -3] = theta[6]
                c[-3, -1] = theta[6]
                c[-1, 3] = theta[6]
                c[3, -1] = theta[6]
                c[1, -3] = theta[6]
                c[-3, 1] = theta[6]
                
                # theta9/theta1
                c[2, 3] = theta[7]
                c[3, 2] = theta[7]
                c[-2, -3] = theta[7]
                c[-3, -2] = theta[7]
                c[-2, 3] = theta[7]
                c[3, -2] = theta[7]
                c[2, -3] = theta[7]
                c[-3, 2] = theta[7]
                
                # theta10/theta1
                c[3, 3] = theta[8]
                c[3, -3] = theta[8]
                c[-3, 3] = theta[8]
                c[-3, -3] = theta[8]
                
    
    # Now return c if appropriate
    if return_c:
        return c
    else:
        return

def objective_fun(theta, args):
    """
    Implements objective function for misfit of GMRF from GF.
    
    See Rue and Held 2005, pg 197 eqn 5.10. This objective function returns
    the weighted 2 norm between ro (the correlation of the desired GF) and
    ro(theta) the correlation function of the GMRF approximation parameterized
    by theta
    
    !!! Note that Rue and Held use orthonormalized DFT !!!

    Parameters
    ----------
    theta : 1d array
        Parameters for GMRF, see block_circ_base.
    args : tuple
        Tuple with necessary arguments: (ro, w) where ro is the target
        correlation function and w is an array of weights (same size as ro)
        The size of ro and w is the same that will be used for GMRF base.

    Returns
    -------
    float
        Weighted 2-norm between ro and ro(theta)

    """
    
    # Create base of precision matrix (block circulant)
    if len(theta)==5:
        m = 2
    elif len(theta)==9:
        m = 3
    else:
        raise ValueError("Theta must be length corresponding to valid " +
                         "neighborhood (e.g. len(theta)=5 if m=2")
    q = block_circ_base(args[0].shape[0], args[1].shape[1], m, theta)
    
    # Invert and scale to get base of correlation matrix
    sigma = (1/args[0].size)*np.fft.irfft2(np.fft.rfft2(q, norm='ortho')**(-1), norm='ortho')
    ro_theta = sigma/sigma[0, 0]
    
    # Return weighted 2-norm
    return (((args[0]-ro_theta)**2)*args[1]).sum()

def objective_fun_theta1(log_theta1, args):
    """
    Objective function for finding theta1 that gives unit marginal variance.

    See Rue and Held 2005, p. 188. They don't specify how they compute the
    value of theta1 to give unit marginal precision so we'll use scalar
    optimization.
    
    !!! Note that Rue and Held use orthonormalized DFT !!!
    Parameters
    ----------
    log_theta1 : float
        Natural log of value of theta1 to test. Note that since by definition
        theta1 must be positive we take the exp of this value so that we 
        don't have to deal with bounds.
    args : list, [theta, n0, n1]
        Theta is the list of parameters for the GMRF, see block_circ_base
        n0, n1 are dimensions of the domain

    Returns
    -------
    float
        absolute value of difference between marginal variance and 1

    """
    
    # Create base of precision matrix q
    q = block_circ_base(args[1], args[2], 2, args[0]) * np.exp(log_theta1)
    # Invert to get base of precision matrix
    sigma = (1/(args[1]*args[2]))*np.fft.irfft2(
        np.fft.rfft2(q, norm='ortho')**(-1), norm='ortho')
    # sigma[0,0] is marginal variance
    return np.abs(1 - sigma[0, 0])

def theta_to_Q(n0, n1, m, theta, circulant=False):
    """
    Create a sparse matrix Q from the parameters in theta.
    
    If circulant is false we'll ignore the wrap around terms.

    Parameters
    ----------
    n0 : int
        Size of the domain in the row direction.
    n1 : int
        Size of the domain in the column direction.
    m : int
        Usually 2 or 3, half width of neighborhood.
    theta : 1d-array
        Parameter values of precision matrix, ordered in the same manner as 
        Rue and Held 2005 p. 196.
    circulant : bool, optional
        Whether to include the wrap-around terms. The default is False.

    Returns
    -------
    (n0*n1)x(n0*n1) sparse matrix
        The sparse precision matrix Q

    """
    if circulant==False:
        diags = [np.ones(n0*n1), # zeroth spot is always 1
                 theta[0] * np.ones(n0*n1 - 1), # one step sideways
                 theta[0] * np.ones(n0*n1 - 1), # one step sideways
                 theta[0] * np.ones(n0*n1 - n1), # one step up or down
                 theta[0] * np.ones(n0*n1 - n1), # one step up or down
                 theta[1] * np.ones(n0*n1 - n1 - 1), # Down to the right
                 theta[1] * np.ones(n0*n1 - n1 + 1), # Down to the left
                 theta[1] * np.ones(n0*n1 - n1 + 1), # Up to the right
                 theta[1] * np.ones(n0*n1 - n1 - 1), # Up to the left
                 theta[2] * np.ones(n0*n1 - 2), # two to the right
                 theta[2] * np.ones(n0*n1 - 2), # two to the left
                 theta[2] * np.ones(n0*n1 - 2*n1), # two down
                 theta[2] * np.ones(n0*n1 - 2*n1), # two up
                 theta[3] * np.ones(n0*n1 - n1 - 2), # Down 1, right 2
                 theta[3] * np.ones(n0*n1 - n1 + 2), # Down 1, left 2
                 theta[3] * np.ones(n0*n1 - n1 + 2), # up 1, right 2
                 theta[3] * np.ones(n0*n1 - n1 - 2), # up 1, left 2
                 theta[3] * np.ones(n0*n1 - 2*n1 - 1), # down 2, right 1
                 theta[3] * np.ones(n0*n1 - 2*n1 + 1), # down 2, left 1
                 theta[3] * np.ones(n0*n1 - 2*n1 + 1), # up 2, right 1
                 theta[3] * np.ones(n0*n1 - 2*n1 - 1), # up 2, left 1
                 theta[4] * np.ones(n0*n1 - 2*n1 - 2), # down 2, right 2
                 theta[4] * np.ones(n0*n1 - 2*n1 + 2), # down 2, left 2
                 theta[4] * np.ones(n0*n1 - 2*n1 + 2), # up 2, right 2
                 theta[4] * np.ones(n0*n1 - 2*n1 - 2) # up 2, left 2
                 ]
        
        ks = [0,
              1,
              -1,
              n1, 
              -n1,
              n1 + 1,
              n1 - 1,
              -n1 + 1,
              -n1 - 1,
              2, 
              -2,
              2*n1,
              -2*n1,
              n1 + 2,
              n1 - 2,
              -n1 + 2,
              -n1 - 2,
              2*n1 + 1,
              2*n1 - 1,
              -2*n1 + 1,
              -2*n1 - 1,
              2*n1 + 2,
              2*n1 - 2, 
              -2*n1 + 2,
              -2*n1 - 2]
    else:
        print("haven't written functionality for circulant yet")
        diags = [np.ones(n0*n1), # zeroth spot is always 1
                 theta[0] * np.ones(n0*n1 - 1), # one step sideways
                 theta[0] * np.ones(n0*n1 - 1), # one step sideways
                 theta[0], # 
                 theta[0] * np.ones(n0*n1 - n1), # one step up or down
                 theta[0] * np.ones(n0*n1 - n1), # one step up or down
                 theta[1] * np.ones(n0*n1 - n1 - 1), # Down to the right
                 theta[1] * np.ones(n0*n1 - n1 + 1), # Down to the left
                 theta[1] * np.ones(n0*n1 - n1 + 1), # Up to the right
                 theta[1] * np.ones(n0*n1 - n1 - 1), # Up to the left
                 theta[2] * np.ones(n0*n1 - 2), # two to the right
                 theta[2] * np.ones(n0*n1 - 2), # two to the left
                 theta[2] * np.ones(n0*n1 - 2*n1), # two down
                 theta[2] * np.ones(n0*n1 - 2*n1), # two up
                 theta[3] * np.ones(n0*n1 - n1 - 2), # Down 1, right 2
                 theta[3] * np.ones(n0*n1 - n1 + 2), # Down 1, left 2
                 theta[3] * np.ones(n0*n1 - n1 + 2), # up 1, right 2
                 theta[3] * np.ones(n0*n1 - n1 - 2), # up 1, left 2
                 theta[3] * np.ones(n0*n1 - 2*n1 - 1), # down 2, right 1
                 theta[3] * np.ones(n0*n1 - 2*n1 + 1), # down 2, left 1
                 theta[3] * np.ones(n0*n1 - 2*n1 + 1), # up 2, right 1
                 theta[3] * np.ones(n0*n1 - 2*n1 - 1), # up 2, left 1
                 theta[4] * np.ones(n0*n1 - 2*n1 - 2), # down 2, right 2
                 theta[4] * np.ones(n0*n1 - 2*n1 + 2), # down 2, left 2
                 theta[4] * np.ones(n0*n1 - 2*n1 + 2), # up 2, right 2
                 theta[4] * np.ones(n0*n1 - 2*n1 - 2) # up 2, left 2
                 ]
        
        ks = [0,
              1,
              -1,
              n1, 
              -n1,
              n1 + 1,
              n1 - 1,
              -n1 + 1,
              -n1 - 1,
              2, 
              -2,
              2*n1,
              -2*n1,
              n1 + 2,
              n1 - 2,
              -n1 + 2,
              -n1 - 2,
              2*n1 + 1,
              2*n1 - 1,
              -2*n1 + 1,
              -2*n1 - 1,
              2*n1 + 2,
              2*n1 - 2, 
              -2*n1 + 2,
              -2*n1 - 2]
    
    return sp.diags(diags, ks)

def fit_theta_to_ro(ro, wts, p, m=2):
    """
    Fits parameters of GMRF precision matrix Q to given correlation matrix ro.
    
    Assumes we want unit marginal variance. See Rue and Held 2005, ch. 5 for
    details.

    Parameters
    ----------
    ro : array
        Target autocorrelation function. Same size as domain we'll simulate
    wts : array
        Weights to use for loss function. Same size as ro.
    p : float
        Guess at range parameter, used in starting point of optimization
    m : int, optional
        Size of the neighborhood to use in GMRF. The default is 2.

    Returns
    -------
    theta1, theta
        Tuple with theta1 scaling factor and theta neighbor weights.

    """
    
    if not (m==2):
        raise RuntimeWarning('fit_theta_to_ro may not work for m not equal 2')
    
    # Create initial guess for theta.
    k = np.sqrt(8)/p
    a = 4 + k**2
    if m==2:
        theta =  [-2*a, 2, 1, 0, 0]
        theta = theta/(4 + a**2)
    elif m==3:
        theta =  [-2*a, 2, 1, 0, 0, 0, 0, 0]
        theta = theta/(4 + a**2)
    else:
        raise RuntimeError('m must be 2 or 3')
    
    # Use scipy.optimize minimize functions to fit theta and theta1
    res = minimize(objective_fun, theta, args=[ro, wts], method='Nelder-Mead')
    res_log_theta1 = minimize_scalar(objective_fun_theta1, args=[res.x, 
                                                                 ro.shape[0], 
                                                                 ro.shape[1]])
    
    # Return parameters
    return(np.exp(res_log_theta1.x), res.x)

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

