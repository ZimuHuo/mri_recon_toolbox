import numpy as np
import sys
sys.path.insert(1, '../')
from tqdm import tqdm
from scipy.spatial import Voronoi, ConvexHull, convex_hull_plot_2d
'''
Formula from Jack et al.(1991)
According to Jackson 1991, on selection of the colvolution kernel page 3
u is only defined for |u|<w/2  
The convolution kernel is symmetrical, so only half part is computed, and it is also 
presampled with oversampling ratio of 2 for faster computation, check Betty's paper for lower oversampling ratio. 
'''
# import NUFFT class
from pynufft import NUFFT
import pkg_resources
import scipy
from scipy.spatial import KDTree
def get_nearest_points(traj, cur_location, nk = 5):
    kdtree=KDTree(traj)
    dist,points = kdtree.query( cur_location, k = nk)
    return points 
def l2(a, b):
    return np.linalg.norm(np.abs(a.flatten()) - np.abs(np.abs(b.flatten())))
import scipy.io
def design_trajectory(n):
    nTR = int(np.floor(n * n * np.pi ))
    nRO = n
    trajectory= radial([nTR, nRO , 3], [n,n,n])
    [ntr, nro, ndim] = trajectory.shape
    trajectory = trajectory.reshape([ntr*nro, ndim])
    trajectory = trajectory /n*2 * np.pi 
    return trajectory
    
def nufft_forward(images, traj):
    [ny, nx, nz, nc] = images.shape
    NufftObj = NUFFT()
    Nd = (ny, nx, nz)  # image size
    print('setting image dimension Nd...', Nd)
    Kd = (ny, nx, nz)  # k-space saize
    print('setting spectrum dimension Kd...', Kd)
    Jd = (3, 3, 3)  # interpolation size
    print('setting interpolation size Jd...', Jd)
    NufftObj.plan(traj, Nd, Kd, Jd)
    data = multi_coil_NUFFT_forward(NufftObj,  images)
    return data
def nufft_adjoint(shape, data, traj):
    [ny, nx, nz, nc] = shape
    NufftObj = NUFFT()
    Nd = (ny, nx, nz)  # image size
    Kd = (ny, nx, nz)  # k-space saize
    Jd = (3, 3, 3)  # interpolation size
    NufftObj.plan(traj, Nd, Kd, Jd)
    images = multi_coil_NUFFT_adjoint(NufftObj,  data)
    return images 
def multi_coil_NUFFT_forward(NufftObj,  images):
    [ny, nx, nz, nc] = images.shape
    trajectory = NufftObj.st["om"]
    nl = trajectory.shape[0]
    data = np.zeros([nl, nc], dtype = complex)
    for c in range(nc): 
        data[:,c] = NufftObj.forward(images[:,:,:,c])
    return data
def multi_coil_NUFFT_adjoint(NufftObj,  data):
    [ny, nx, nz] = NufftObj.st["Nd"]
    nc = data.shape[-1]
    trajectory = NufftObj.st["om"]
    nl = trajectory.shape[0]
    images = np.zeros([ny, nx, nz, nc], dtype = complex)
    for c in range(nc): 
        images[:,:,:,c] = NufftObj.solve(data[:,c], solver='cg', maxiter= 50)
    return images
    
def get_brain_images():
    tissuetype = ['graymatter', 'deep_graymatter', 'whitematter', 'csf']
    T2 = [110, 100, 60, 1500]
    T2s = [40, 45, 50, 1000] 
    mat = scipy.io.loadmat('../lib/tissue_images.mat')
    tissues = mat.get("tissue_images")
    return np.squeeze(tissues), tissuetype, T2


# Generic kb kernel 
def kb(u, width, beta):
    u = beta*np.sqrt(1-(2*u/width)**2)
    u = np.i0(u)/width
    return u 



def KaiserBesselwindow(width, length,overgridfactor):
    w = width
    l = length
    alpha = overgridfactor
    beta = np.pi*np.sqrt(w**2/alpha**2*(alpha-1/2)**2-0.8)
    # from betty, 2005, on rapid griding algorithms 
    
    u = np.arange(0,l,1)/(l-1)*w/2
    #According to Jackson 1991, on selection of the colvolution kernel page 3
    #u is only defined for |u|<w/2

    window = kb(u, w, beta)
    window = window/window[0]

    return window




'''
standard griding
'''
def gridding(mat, data, traj, dcf,kernalwidth=5):
    gridsize = mat.shape[0]
    Kernallength = 32
    kernalwidth = kernalwidth
    window = KaiserBesselwindow(kernalwidth, Kernallength, 1.375)  
    kwidth = kernalwidth / 2 / gridsize
    gridcenter = gridsize / 2
    for n, weight in enumerate(dcf):
        kx = traj[n,0]
        ky = traj[n,1]
        xmin = int((kx - kwidth) * gridsize + gridcenter)
        xmax = int((kx + kwidth) * gridsize + gridcenter) + 1
        ymin = int((ky - kwidth) * gridsize + gridcenter)
        ymax = int((ky + kwidth) * gridsize + gridcenter) + 1
        if (xmin < 0):
            xmin = 0
        if (xmax >= gridsize):
            xmax = gridsize
        if (ymin < 0):
            ymin = 0
        if (ymax >= gridsize):
            ymax = gridsize
        for x in range(xmin, xmax):
            dx = (x - gridcenter) / gridsize - kx
            for y in range(ymin, ymax):
                dy = (y - gridcenter) / gridsize - ky
                d = np.sqrt(dx ** 2 + dy ** 2)
                if (d < kwidth):
                    idx = d / kwidth * (Kernallength - 1)
                    idxint = int(idx)
                    frac = idx - idxint
                    kernal = window[idxint] * (1 - frac) + window[idxint + 1] * frac
                    mat[x, y] += kernal * weight * data[n]
    return mat



'''
Equation 19
(W * phi) * R from  Jim Pipe et al. (1999)
It simply means convolve weight with kernel on to R, which is a cartesian grid
Complexity O(2pi L^2 N) 
'''
    
def grid(traj, dcf, gridsize = 256):
    mat = np.zeros([gridsize, gridsize], dtype=complex)
    gridsize = mat.shape[0]
    Kernallength = 32
    kernalwidth = 5 
    window = KaiserBesselwindow(kernalwidth, Kernallength, 1.375)  
    kwidth = kernalwidth / 2 / gridsize
    gridcenter = gridsize / 2
    for n, weight in enumerate(dcf):
        kx = traj[n,0]
        ky = traj[n,1]
        xmin = int((kx - kwidth) * gridsize + gridcenter)
        xmax = int((kx + kwidth) * gridsize + gridcenter) + 1
        ymin = int((ky - kwidth) * gridsize + gridcenter)
        ymax = int((ky + kwidth) * gridsize + gridcenter) + 1
        if (xmin < 0):
            xmin = 0
        if (xmax >= gridsize):
            xmax = gridsize
        if (ymin < 0):
            ymin = 0
        if (ymax >= gridsize):
            ymax = gridsize
        for x in range(xmin, xmax):
            dx = (x - gridcenter) / gridsize - kx
            for y in range(ymin, ymax):
                dy = (y - gridcenter) / gridsize - ky
                d = np.sqrt(dx ** 2 + dy ** 2)
                if (d < kwidth):
                    idx = d / kwidth * (Kernallength - 1)
                    idxint = int(idx)
                    frac = idx - idxint
                    kernal = window[idxint] * (1 - frac) + window[idxint + 1] * frac
                    mat[x, y] += kernal * weight
    return mat

'''
Equation 19
(((W * phi) * R) *phi) * S = w from  Jim Pipe et al. (1999)
It simply means convolve weight with kernel on to R, which is a cartesian grid
then re-sample back to the weigth vector w using the kernel phi from trajectory S
Complexity also O(2pi L^2 N) 
'''
def degrid(mat,traj):
    gridsize = mat.shape[0]
    gridcenter = (gridsize / 2)
    weight = np.zeros(traj.shape[0])
    Kernallength = 32
    kernalwidth = 5 
    window = KaiserBesselwindow(kernalwidth, Kernallength, 1.375)  
    kwidth = kernalwidth / 2 / gridsize
    for n, loc in enumerate(traj):
        kx = loc[0]
        ky = loc[1]
        xmin = int((kx - kwidth) * gridsize + gridcenter)
        xmax = int((kx + kwidth) * gridsize + gridcenter) + 1
        ymin = int((ky - kwidth) * gridsize + gridcenter)
        ymax = int((ky + kwidth) * gridsize + gridcenter) + 1
        if (xmin < 0):
            xmin = 0
        if (xmax >= gridsize):
            xmax = gridsize 
        if (ymin < 0):
            ymin = 0
        if (ymax >= gridsize):
            ymax = gridsize 
        for x in range(xmin, xmax):
            dx = (x - gridcenter) / gridsize - kx
            for y in range(ymin, ymax):
                dy = (y - gridcenter) / gridsize - ky
                d = np.sqrt(dx ** 2 + dy ** 2)
                if (d < kwidth):
                    idx = d / kwidth * (Kernallength - 1)
                    idxint = int(idx)
                    frac = idx - idxint
                    kernal = window[idxint] * (1 - frac) + window[idxint + 1] * frac
                    weight[n] += np.abs(mat[x, y]) * (kernal)
    return weight

'''
The mean loop:
Equation 19
(((W * phi) * R) *phi) * S = w from  Jim Pipe et al. (1999)
'''
def pipedcf(traj, ns):
    dcf = np.ones(ns)
    for i in tqdm(range(10)):
        mat = grid(traj, dcf)
        newdcf = degrid(mat, traj)
        dcf = dcf / newdcf
    return dcf

def voronoidcf(points, threshold = 95):
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = 0
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    vol[vol > np.percentile(vol,threshold)] = 0
    return vol
def radial(coord_shape, img_shape, golden=True, dtype=float):
    """Generate radial trajectory.

    Args:
        coord_shape (tuple of ints): coordinates of shape [ntr, nro, ndim],
            where ntr is the number of TRs, nro is the number of readout,
            and ndim is the number of dimensions.
        img_shape (tuple of ints): image shape.
        golden (bool): golden angle ordering.
        dtype (Dtype): data type.

    Returns:
        array: radial coordinates.

    References:
        1. An Optimal Radial Profile Order Based on the Golden
        Ratio for Time-Resolved MRI
        Stefanie Winkelmann, Tobias Schaeffter, Thomas Koehler,
        Holger Eggers, and Olaf Doessel. TMI 2007.
        2. Temporal stability of adaptive 3D radial MRI using
        multidimensional golden means
        Rachel W. Chan, Elizabeth A. Ramsay, Charles H. Cunningham,
        and Donald B. Plewes. MRM 2009.

    """
    if len(img_shape) != coord_shape[-1]:
        raise ValueError(
            "coord_shape[-1] must match len(img_shape), "
            "got {} and {}".format(coord_shape[-1], len(img_shape))
        )
    ntr, nro, ndim = coord_shape
    if ndim == 2:
        if golden:
            phi = np.pi * (3 - 5**0.5)
        else:
            phi = 2 * np.pi / ntr

        n, r = np.mgrid[:ntr, : 0.5 : 0.5 / nro]

        theta = n * phi
        coord = np.zeros((ntr, nro, 2))
        coord[:, :, -1] = r * np.cos(theta)
        coord[:, :, -2] = r * np.sin(theta)

    elif ndim == 3:
        if golden:
            phi1 = 0.465571231876768
            phi2 = 0.682327803828019
        else:
            raise NotImplementedError

        n, r = np.mgrid[:ntr, : 0.5 : 0.5 / nro]
        beta = np.arccos(2 * ((n * phi1) % 1) - 1)
        alpha = 2 * np.pi * ((n * phi2) % 1)

        coord = np.zeros((ntr, nro, 3))
        coord[:, :, -1] = r * np.sin(beta) * np.cos(alpha)
        coord[:, :, -2] = r * np.sin(beta) * np.sin(alpha)
        coord[:, :, -3] = r * np.cos(beta)
    else:
        raise ValueError("coord_shape[-1] must be 2 or 3, got {}".format(ndim))

    return (coord * img_shape[-ndim:]).astype(dtype)