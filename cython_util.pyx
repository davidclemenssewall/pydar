import numpy as np

def create_counts_mins_cy(long nbin_0, long nbin_1, float[:, :] Points, 
                          long[:] xy, float init_val):
    """
    Given a bin point cloud, return the binwise number of items and min z val.

    Parameters
    ----------
    nbin_0 : long
        Number of bins along the zeroth axis
    nbin_1 : long
        Number of bins along the first axis
    Points : float[:, :]
        Pointcloud, Nx3 array of type np.float32
    xy : long[:]
        Bin index for each point, must be same as numbe rof points.
    init_val : float
        Initial values in mins array. Pick something larger than largest z
        in Points. In mins output, all bins that don't have any points in them
        will take on this value.

    Returns:
    --------
    counts : long[:]
        Array with the counts for each bin. Same length as xy
    mins : float[:]
        Array with the min z value for each bin. Same length as xy
    """
    counts = np.zeros(nbin_0 * nbin_1, dtype=np.int64)
    cdef long[:] counts_view = counts
    
    mins = init_val * np.ones(nbin_0 * nbin_1, dtype=np.float32)
    cdef float[:] mins_view = mins
    
    for i in range(len(xy)):
        counts_view[xy[i]] += 1
        if mins_view[xy[i]] > Points[i, 2]:
            mins_view[xy[i]] = Points[i, 2]
    
    return counts, mins

def create_counts_sums_cy(long nbin_0, long nbin_1, float[:, :] Points, 
                          long[:] xy):
    """
    Return the binwise number of points and sum of the z values
    
    Parameters
    ----------
    nbin_0 : long
        Number of bins along the zeroth axis
    nbin_1 : long
        Number of bins along the first axis
    Points : float[:, :]
        Pointcloud, Nx3 array of type np.float32
    xy : long[:]
        Bin index for each point, must be same as numbe rof points.

    Returns:
    --------
    counts : long[:]
        Array with the counts for each bin. Same length as xy
    sums : float[:]
        Array with the sum of z values for each bin. Same length as xy
    """

    counts = np.zeros(nbin_0 * nbin_1, dtype=np.int64)
    cdef long[:] counts_view = counts
    
    sums = np.zeros(nbin_0 * nbin_1, dtype=np.float32)
    cdef float[:] sums_view = sums
    
    for i in range(len(xy)):
        counts_view[xy[i]] += 1
        sums_view[xy[i]] += Points[i, 2]
    
    return counts, sums

def create_counts_hists_cy(long nbin_0, long nbin_1, long[:] h_ind, long[:] xy, 
                           long nbin_h):
    """
    Return the binwise number of points and sum of the z values
    
    Parameters
    ----------
    nbin_0 : long
        Number of bins along the zeroth axis
    nbin_1 : long
        Number of bins along the first axis
    h_ind : long[:]
        Z value histogram index for each point. Smallest value must be >=0 and
        largest value must be <= nbin_h-1.
    xy : long[:]
        Bin index for each point, must be same as number of points.
    nbin_h : long
        The number of z histogram bins there are.

    Returns:
    --------
    counts : long[:]
        Array with the counts for each bin. Same length as xy
    hists : long[:, :]
        Array with the historam of z values for each bin. Zeroth dim same 
        length as xy. 1st dim length nbin_h
    """
    counts = np.zeros(nbin_0 * nbin_1, dtype=np.int64)
    cdef long[:] counts_view = counts
    
    hists = np.zeros((nbin_0 * nbin_1, nbin_h), dtype=np.int64)
    cdef long[:, :] hists_view = hists
    
    for i in range(len(xy)):
        counts_view[xy[i]] += 1
        hists_view[xy[i], h_ind[i]] += 1
    
    return counts, hists
