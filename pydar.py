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
from vtk.util.numpy_support import vtk_to_numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
        self.tiepoints = pd.read_csv(project_path + project_name +
                                     '\\tiepoints.csv',
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
        
    def add_transform(self, name, transform, reflector_list=[], std=np.NaN):
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
        
        # Return key (index) of transform
        return (name, str_reflector_list)
    
    def get_transform(self, index):
        """
        Return the requested transform's array.

        Parameters
        ----------
        index : tuple
            Key for desired transform in self.transforms.

        Returns
        -------
        None.

        """
        
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
        
        # extract point lists and name as in Arun et al.
        if use_tiepoints_transformed:
            psubi_prime = other_tiepointlist.tiepoints_transformed.loc[
                reflector_list].to_numpy().T
        else:
            psubi_prime = other_tiepointlist.tiepoints.loc[
                reflector_list].to_numpy().T
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
        
        # Add matrix to transforms
        key = self.add_transform(other_tiepointlist.project_name + '_' + mode,
                                 A, reflector_list, std=std)
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
    
    Methods
    -------
    write_scan()
        Writes the scan to a file to save the filters
    read_scan()
        Reads the scan from a file, replacing the RiSCAN version.
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
    clear_classification
        Reset all Classification values to 0.
    update_man_class(pdata, classification)
        Update the points in man_class with the points in pdata.
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
    create_dimensionality_pdal(temp_dir="", from_current=True, voxel=True,
                               h_voxel=0.1, v_voxel=0.01, threads=8)
        Create the four dimensionality variables from Demantke2011 and
        Guinard2017. Uses pdal to do so.
    add_dist()
        Add distance from scanner to polydata_raw
    """
    
    def __init__(self, project_path, project_name, scan_name, poly='.1_.1_.01',
                 read_scan=False, import_las=False):
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
        poly : str, optional
            The suffix describing which polydata to load. The default is
            '.1_.1_.01'.
        read_scan : bool, optional
            Whether to read a saved scan from file. Typically useful for
            handling filtered scans. The default is False
        import_las: bool, optional
            If true (and read_scan is False) read in the las file instead of
            the polydata. The default is False.

        Returns
        -------
        None.

        """
        
        # Store instance attributes
        self.project_path = project_path
        self.project_name = project_name
        self.scan_name = scan_name
        self.poly = poly
        
        # Create reader, transformFilter
        reader = vtk.vtkXMLPolyDataReader()
        polys = os.listdir(self.project_path + self.project_name + '\\SCANS\\'
                           + self.scan_name + '\\POLYDATA\\')
        if read_scan:
            reader.SetFileName(self.project_path + self.project_name + 
                           "\\vtkfiles\\pointclouds\\" +
                           self.scan_name + '.vtp')
            reader.Update()
            self.polydata_raw = reader.GetOutput()
        elif not import_las:
            # Match poly with polys
            for name in polys:
                if re.search(poly + '$', name):
                    reader.SetFileName(self.project_path + self.project_name 
                                            + '\\SCANS\\' + self.scan_name 
                                            + '\\POLYDATA\\' + name + 
                                            '\\1.vtp')
                    reader.Update()
                    self.polydata_raw = reader.GetOutput()
                    break
        elif import_las:
            # import las file from lasfiles directory in project_path
            filenames = os.listdir(self.project_path + self.project_name + 
                                   "\\lasfiles\\")
            pattern = re.compile(self.scan_name + '.*las')
            matches = [pattern.fullmatch(filename) for filename in filenames]
            if any(matches):
                # Create filename input
                filename = next(f for f, m in zip(filenames, matches) if m)
                json_list = [self.project_path + self.project_name + 
                             "\\lasfiles\\" + filename]
                json_data = json.dumps(json_list, indent=4)
                # Load scan into numpy array
                pipeline = pdal.Pipeline(json_data)
                _ = pipeline.execute()
                
                # Create pdata and populate with points from las file
                pdata = vtk.vtkPolyData()
                points = vtk.vtkPoints()
                points.SetDataTypeToFloat()
                points.SetNumberOfPoints(pipeline.arrays[0].shape[0])
                pdata.SetPoints(points)
                arr_nret = vtk.vtkUnsignedCharArray()
                arr_nret.SetName('NumberOfReturns')
                arr_nret.SetNumberOfComponents(1)
                arr_nret.SetNumberOfTuples(pipeline.arrays[0].shape[0])
                pdata.GetPointData().AddArray(arr_nret)
                arr_reti = vtk.vtkSignedCharArray()
                arr_reti.SetName('ReturnIndex')
                arr_reti.SetNumberOfComponents(1)
                arr_reti.SetNumberOfTuples(pipeline.arrays[0].shape[0])
                pdata.GetPointData().AddArray(arr_reti)
                arr_refl = vtk.vtkFloatArray()
                arr_refl.SetName('Reflectance')
                arr_refl.SetNumberOfComponents(1)
                arr_refl.SetNumberOfTuples(pipeline.arrays[0].shape[0])
                pdata.GetPointData().AddArray(arr_refl)
                arr_amp = vtk.vtkFloatArray()
                arr_amp.SetName('Amplitude')
                arr_amp.SetNumberOfComponents(1)
                arr_amp.SetNumberOfTuples(pipeline.arrays[0].shape[0])
                pdata.GetPointData().AddArray(arr_amp)
                dsa_pdata = dsa.WrapDataObject(pdata)
                dsa_pdata.Points[:,0] = np.float32(pipeline.arrays[0]['X'])
                dsa_pdata.Points[:,1] = np.float32(pipeline.arrays[0]['Y'])
                dsa_pdata.Points[:,2] = np.float32(pipeline.arrays[0]['Z'])
                dsa_pdata.PointData['NumberOfReturns'][:] = (pipeline.arrays
                                                    [0]['NumberOfReturns'])
                ReturnIndex = np.int16(pipeline.arrays[0]['ReturnNumber'])
                ReturnIndex[ReturnIndex==7] = 0
                ReturnIndex = (ReturnIndex - 
                               pipeline.arrays[0]['NumberOfReturns'])
                dsa_pdata.PointData['ReturnIndex'][:] = ReturnIndex
                dsa_pdata.PointData['Reflectance'][:] = np.float32(
                    pipeline.arrays[0]['Reflectance'])
                dsa_pdata.PointData['Amplitude'][:] = np.float32(
                    pipeline.arrays[0]['Amplitude'])
                pdata.Modified()
                
                # Create VertexGlyphFilter so that we have vertices for
                # displaying
                vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
                vertexGlyphFilter.SetInputData(pdata)
                vertexGlyphFilter.Update()
                self.polydata_raw = vertexGlyphFilter.GetOutput()
        
        # Create dataset adaptor for interacting with polydata_raw
        self.dsa_raw = dsa.WrapDataObject(self.polydata_raw)
        
        # Add Classification array to polydata_raw if it's not present
        if not 'Classification' in self.dsa_raw.PointData.keys():
            arr = vtk.vtkUnsignedCharArray()
            arr.SetName('Classification')
            arr.SetNumberOfComponents(1)
            arr.SetNumberOfTuples(self.polydata_raw.GetNumberOfPoints())
            arr.FillComponent(0, 0)
            self.polydata_raw.GetPointData().AddArray(arr)
            self.polydata_raw.GetPointData().SetActiveScalars('Classification')
        
        # Add PedigreeIds if they are not already present
        if not 'PointId' in self.dsa_raw.PointData.keys():
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
        
        self.polydata_raw.Modified()
        
        self.transform = vtk.vtkTransform()
        # Set mode to post-multiply, so concatenation is successive transforms
        self.transform.PostMultiply()
        self.transformFilter = vtk.vtkTransformPolyDataFilter()
        self.transformFilter.SetTransform(self.transform)
        self.transformFilter.SetInputData(self.polydata_raw)
        
        # Create other attributes
        self.transform_dict = {}
        self.filterName = 'None'
        self.filterDict = {}
        
        # Create currentFilter
        self.currentFilter = vtk.vtkThresholdPoints()
        self.currentFilter.ThresholdBetween(-.5, 2.5)
        self.currentFilter.AddInputConnection(
            self.transformFilter.GetOutputPort())
        self.currentFilter.Update()
    
    def write_scan(self):
        """
        Write the scan to a vtp file. Thus storing Classification.

        Returns
        -------
        None.

        """
        
        # If the write directory doesn't exist, create it
        if not os.path.isdir(self.project_path + self.project_name + 
                             "\\vtkfiles"):
            os.mkdir(self.project_path + self.project_name + 
                     "\\vtkfiles")
        if not os.path.isdir(self.project_path + self.project_name + 
                             "\\vtkfiles\\pointclouds"):
            os.mkdir(self.project_path + self.project_name + 
                     "\\vtkfiles\\pointclouds")
        
        # Create writer and write mesh
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(self.polydata_raw)
        writer.SetFileName(self.project_path + self.project_name + 
                           "\\vtkfiles\\pointclouds\\" +
                               self.scan_name + '.vtp')
        writer.Write()
        
    def read_scan(self):
        """
        Reads a scan from a vtp file

        Returns
        -------
        None.

        """
        
        # Clear polydata_raw and dsa_raw
        if hasattr(self, 'dsa_raw'):
            del self.dsa_raw
            del self.polydata_raw

            
        # Create Reader, read file
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.project_path + self.project_name + 
                           "\\vtkfiles\\pointclouds\\" +
                           self.scan_name + '.vtp')
        reader.Update()
        self.polydata_raw = reader.GetOutput()
        self.polydata_raw.Modified()
        
        # Create dsa, link with transform filter
        self.dsa_raw = dsa.WrapDataObject(self.polydata_raw)
        self.transformFilter.SetInputData(self.polydata_raw)
        self.transformFilter.Update()
        self.currentFilter.Update()
    
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
        if os.path.isdir(self.project_path + self.project_name + 
                         '\\manualclassification'):
            # Check if file exists
            if os.path.isfile(self.project_path + self.project_name + 
                              '\\manualclassification\\' + self.scan_name +
                              '.parquet'):
                self.man_class = pd.read_parquet(self.project_path + 
                                                 self.project_name + 
                                                 '\\manualclassification\\' + 
                                                 self.scan_name + '.parquet',
                                                 engine="pyarrow")
            # otherwise create dataframe
            else:
                create_df = True
        else:
            # Create directory and dataframe
            create_df = True
            os.mkdir(self.project_path + self.project_name + 
                     '\\manualclassification')
        
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
                                           'Elevation':
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
        
    
    def add_transform(self, key, matrix):
        """
        Adds a new transform to the transform_dict

        Parameters
        ----------
        key : str
            Name of the tranform (e.g. 'sop')
        matrix : 4x4 array-like
            4x4 matrix of transformation in homologous coordinates.

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
        
    def add_sop(self):
        """
        Add the sop matrix to transform_dict. Must have exported from RiSCAN

        Returns
        -------
        None.

        """
        
        trans = np.genfromtxt(self.project_path + self.project_name + '\\' 
                              + self.scan_name + '.DAT', delimiter=' ')
        self.add_transform('sop', trans)
        
    def add_z_offset(self, z_offset):
        """
        Adds a uniform z offset to the scan

        Parameters
        ----------
        z_offset : float
            z offset to add in meters.

        Returns
        -------
        None.

        """
        
        trans = np.eye(4)
        trans[2, 3] = z_offset
        self.add_transform('z_offset', trans)
        
    def get_polydata(self):
        """
        Returns vtkPolyData of scan with current transforms and filters.

        Returns
        -------
        vtkPolyData.

        """
        
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
        
        for key in transform_list:
            try:
                self.transform.Concatenate(self.transform_dict[key])
            except Exception as e:
                print("Requested transform " + key + " is not in " +
                      "transform_dict")
                print(e)
        
        # If the norm_height array exists delete it we will recreate it 
        # if needed
        if 'norm_height' in self.dsa_raw.PointData.keys():
            self.polydata_raw.GetPointData().RemoveArray('norm_height')
            self.polydata_raw.Modified()
        self.transformFilter.Update()
        self.currentFilter.Update()
    
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
                                 'Elevation' : (dsa_pdata
                                                .PointData['Elevation']),
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
        self.man_class.to_parquet(self.project_path + 
                                                 self.project_name + 
                                                 '\\manualclassification\\' + 
                                                 self.scan_name + '.parquet',
                                                 engine="pyarrow", 
                                                 compression=None)
        
    
    def apply_elevation_filter(self, z_max):
        """
        Set Classification for all points above z_max to be 1. 

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
        
        # Parse temp_file
        if not temp_file:
            temp_file = self.project_path + 'temp\\temp_pdal.npy'
         
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
            {"filename": self.project_path + 'temp\\temp_pdal.npy',
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
                                            71: (255/255, 127/255, 0/255, 1)}):
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
        
        # Create actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        
    
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
        
        # Create actor
        #self.actor = vtk.vtkLODActor()
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
    
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
        
        # Create actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
    
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
    
    def write_npy_pdal(self, output_dir, filename=None, mode='transformed'):
        """
        Write scan to structured numpy array that can be read by PDAL.

        Parameters
        ----------
        output_dir : str
            Directory to write to.
        filename : str, optional
            Filename to write, if None will write PROJECT_NAME_SCAN_NAME. 
            The default is None.
        mode : str, optional
            Whether to write 'raw' points, 'transformed' points, or 'filtered'
            points. The default is 'transformed'.

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
        names = tuple(dsa_pdata.PointData.keys() + ['X', 'Y', 'Z'])
        formats = []
        for value in dsa_pdata.PointData:
            formats.append(value.dtype)
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
        
        # if 'Reflectance' in dsa_pdata.PointData.keys():
        #     output_npy = np.zeros(n_pts, dtype={'names':('X', 'Y', 'Z', 
        #                                                  'Reflectance'),
        #                             'formats':(np.float32, np.float32, 
        #                                        np.float32, np.float32)})
        #     output_npy['X'] = dsa_pdata.Points[:,0]
        #     output_npy['Y'] = dsa_pdata.Points[:,1]
        #     output_npy['Z'] = dsa_pdata.Points[:,2]
        #     output_npy['Reflectance'] = dsa_pdata.PointData['Reflectance']
        # else:
        #     output_npy = np.zeros(n_pts, dtype={'names':('X', 'Y', 'Z'),
        #                             'formats':(np.float32, np.float32, 
        #                                        np.float32)})
        #     output_npy['X'] = dsa_pdata.Points[:,0]
        #     output_npy['Y'] = dsa_pdata.Points[:,1]
        #     output_npy['Z'] = dsa_pdata.Points[:,2]
        
        if filename is None:
            filename = self.project_name + '_' + self.scan_name
        
        np.save(output_dir + filename, output_npy)
    
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
                 load_scans=True, read_scans=False, import_las=False):
        """
        Generates project, also inits singlescan objects

        Parameters
        ----------
        project_path : str
            Directory location of the project.
        project_name : str
            Filename of the RiSCAN project.
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
        
        # Add SingleScans, including their SOPs, 
        # we will only add a singlescan if it has an SOP and a polydata that
        # matches the poly suffix
        if load_scans:
            scan_names = os.listdir(project_path + project_name + '\\SCANS\\')
            for scan_name in scan_names:
                if os.path.isfile(self.project_path + self.project_name + '\\' 
                                  + scan_name + '.DAT'):
                    if read_scans:
                        scan = SingleScan(self.project_path, self.project_name,
                                          scan_name, poly=poly,
                                          read_scan=read_scans)
                        scan.add_sop()
                        
                        self.scan_dict.update({scan_name : scan})
                    
                    elif import_las:
                        scan = SingleScan(self.project_path, self.project_name,
                                          scan_name, poly=poly,
                                          import_las=import_las)
                        scan.add_sop()
                        
                        self.scan_dict.update({scan_name : scan})
                    
                    else:
                        polys = os.listdir(self.project_path + self.project_name +
                                           '\\SCANS\\'+scan_name + '\\POLYDATA\\')
                        match = False
                        for name in polys:
                            if re.search(poly + '$', name):
                                match = True
                                break
                        if match:
                            scan = SingleScan(self.project_path, 
                                              self.project_name,
                                              scan_name, poly=poly,
                                              read_scan=read_scans)
                            scan.add_sop()
                            
                            self.scan_dict.update({scan_name : scan})
        
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
    
    def write_scans(self):
        """
        Write all single scans to files.

        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].write_scan()
    
    def read_scans(self):
        """
        Read all single scans from files.

        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].read_scan()
    
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
        
    def add_transform(self, key, matrix):
        """
        Add the provided transform to each single scan

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
            self.scan_dict[scan_name].add_transform(key, matrix)
    
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
            self.scan_dict[scan_name].add_transform(key, 
                                                   self.tiepointlist.
                                                   get_transform(key))
            
    def add_z_offset(self, z_offset):
        """
        Add z_offset transform to each single scan in scan_dict

        Parameters
        ----------
        z_offset : float.
            z offset to add in meters.

        Returns
        -------
        None.

        """
        
        for scan_name in self.scan_dict:
            self.scan_dict[scan_name].add_z_offset(z_offset)
            
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
        writer.SetFileName(self.project_path + 'snapshots\\' + 
                           self.project_name + '_' + name + '.png')
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
        writer.SetFileName(self.project_path + 'snapshots\\' + 
                           self.project_name + '_' + name + '.png')
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
        
    def get_merged_points(self):
        """
        Returns a polydata with merged points from all single scans

        Returns
        -------
        vtkPolyData.

        """
        
        # Create Appending filter and add all data to it
        appendPolyData = vtk.vtkAppendPolyData()
        for key in self.scan_dict:
            self.scan_dict[key].transformFilter.Update()
            appendPolyData.AddInputData(self.scan_dict[key].
                                        get_polydata())
        
        appendPolyData.Update()
        
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
            '_merged.vtp'. The default is None.

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
    
    def write_mesh(self, output_name=None):
        """
        Write the mesh out to a file

        Parameters
        ----------
        output_name : str, optional
            Output name for the file, if None use the project_name + 
            '_mesh.vtp'. The default is None.

        Returns
        -------
        None.

        """
        
        # Create writer and write mesh
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(self.mesh)
        if output_name:
            writer.SetFileName(self.project_path + output_name)
        else:
            writer.SetFileName(self.project_path + self.project_name + 
                               '_mesh.vtp')
        writer.Write()
        
    def read_mesh(self, mesh_path=None):
        """
        Read in the mesh from a file.

        Parameters
        ----------
        mesh_path : str, optional
            Path to the mesh, if none use project_path + project_name + 
            '_mesh.vtp'. The default is None.

        Returns
        -------
        None.

        """
        
        # Create reader and read mesh
        reader = vtk.vtkXMLPolyDataReader()
        if mesh_path:
            reader.SetFileName(mesh_path)
        else:
            reader.SetFileName(self.project_path + self.project_name + 
                               '_mesh.vtp')
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
                 poly='.1_.1_.01', load_scans=True, read_scans=False,
                 import_las=False):
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
        
        Returns
        -------
        None.

        """
        
        self.project_path = project_path
        self.project_dict = {}
        self.difference_dict = {}
        self.difference_dsa_dict = {}
        
        for project_name in project_names:
            self.add_project(project_name, load_scans=load_scans,
                             read_scans=read_scans, poly=poly, 
                             import_las=import_las)
            
        self.registration_list = copy.deepcopy(registration_list)
    
    def add_project(self, project_name, poly='.1_.1_.01', load_scans=True, 
                    read_scans=False, import_las=False):
        """
        Add a new project to the project_dict (or overwrite existing project)

        Parameters
        ----------
        project_name : str
            Name of Riscan project to add
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

        Returns
        -------
        None.

        """
        
        self.project_dict[project_name] = Project(self.project_path, 
                                                  project_name, load_scans=
                                                  load_scans, read_scans=
                                                  read_scans, poly=poly,
                                                  import_las=import_las)
    
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
                              alpha=0, overlap=0.1):
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
        
        for key in self.project_dict:
            print(key)
            self.project_dict[key].merged_points_to_mesh(subgrid_x, subgrid_y,
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
            writer.SetFileName(self.project_path + 'snapshots\\' + 
                               project_name_0 + "_" + project_name_1 + 
                               'warp_difference_' + name + '.png')
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
            writer.SetFileName(self.project_path + 'snapshots\\' + 
                               project_name_0 + "_" + project_name_1 + '.png')
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
    
    def train_classifier(self, feature_list, n_ground_multiplier=3, 
                         strat_by_class=True, **kwargs):
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
        **kwargs : dict
            Additional kwargs for train_test_split.

        Returns
        -------
        None.

        """
        
        self.feature_list = feature_list
        
        # Downsample ground points
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
        
        # Split labeled data
        if strat_by_class:
            X_train, X_test, y_train, y_test = train_test_split(df_sub[
                feature_list], df_sub.Classification, 
                stratify=df_sub.Classification, **kwargs)
        else:
            X_train, X_test, y_train, y_test = train_test_split(df_sub[
                feature_list], df_sub.Classification, **kwargs)
        
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