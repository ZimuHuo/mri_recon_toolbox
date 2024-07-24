import sys
sys.path.insert(1, '../')
sys.path.insert(1, '../..')
import numpy as np
import matplotlib.pyplot as plt
from nufft import *
from util.fft import *
from util.phantom import * 
from scipy import interpolate
from tqdm import tqdm
from trajectory import * 
from reg import * 
from nufft import * 
from util.zpad import * 

def coilmap(
        size = (64,64,64), ncoils=4, array_cent=None, coil_width=8, n_rings=None, phi=0):
    """ Apply simulated sensitivity maps. Based on a script by Florian Knoll.

        Args:
            size (tuple): Size of the image array for the sensitivity coils.
            nc_range (int, default: 8): Number of coils to simulate.
            array_cent (tuple, default: 0): Location of the center of the coil
                array
            coil_width (double, default: 2): Parameter governing the width of
                the coil, multiplied by actual image dimension.
            n_rings (int, default: ncoils // 4): Number of rings for a
                cylindrical hardware set-up.
            phi (double, default: 0): Parameter for rotating coil geometry.

        Returns:
            coil_array (array_like): An array of dimensions (ncoils (N)),
                specifying spatially-varying sensitivity maps for each coil.
    """
    if array_cent is None:
        c_shift = [0, 0, 0]
    elif len(array_cent) < 3:
        c_shift = array_cent + (0,)
    else:
        c_shift = array_cent

    c_width = coil_width * min(size)

    if (len(size) > 2):
        if n_rings is None:
            n_rings = ncoils // 4

    c_rad = min(size[0:1]) / 2
    smap = []
    if (len(size) > 2):
        zz, yy, xx = np.meshgrid(range(size[2]), range(size[1]),
                                 range(size[0]), indexing='ij')
    else:
        yy, xx = np.meshgrid(range(size[1]), range(size[0]),
                             indexing='ij')

    if ncoils > 1:
        x0 = np.zeros((ncoils,))
        y0 = np.zeros((ncoils,))
        z0 = np.zeros((ncoils,))

        for i in range(ncoils):
            if (len(size) > 2):
                theta = np.radians((i-1)*360/(ncoils + n_rings) + phi)
            else:
                theta = np.radians((i-1)*360/ncoils + phi)
            x0[i] = c_rad * np.cos(theta) + size[0]/2
            y0[i] = c_rad * np.sin(theta) + size[1]/2
            if (len(size) > 2):
                z0[i] = (size[2]/(n_rings+1)) * (i//n_rings)
                smap.append(np.exp(-1*((xx-x0[i])**2 + (yy-y0[i])**2 +
                                       (zz-z0[i])**2) / (2*c_width)))
            else:
                smap.append(np.exp(-1*((xx-x0[i])**2 + (yy-y0[i])**2) /
                                   (2*c_width)))
    else:
        x0 = c_shift[0]
        y0 = c_shift[1]
        z0 = c_shift[2]
        if (len(size) > 2):
            smap = np.exp(-1*((xx-x0)**2 + (yy-y0)**2 +
                              (zz-z0)**2) / (2*c_width))
        else:
            smap = np.exp(-1*((xx-x0)**2 + (yy-y0)**2) / (2*c_width))

    side_mat = np.arange(int(size[0]//2)-20, 1, -1)
    side_mat = (np.reshape(side_mat, (1,) + side_mat.shape) *
                np.ones(shape=(size[1], 1)))
    cent_zeros = np.zeros(shape=(size[1], size[0]-side_mat.shape[1]*2))

    ph = np.concatenate((side_mat, cent_zeros, side_mat), axis=1) / 5
    if (len(size) > 2):
        ph = np.reshape(ph, (1,) + ph.shape)

    for i, s in enumerate(smap):
        smap[i] = s * np.exp(i*1j*ph*np.pi/180)
    smap = np.moveaxis(smap, 0, -1)
    return smap


def PLOARKS_C(data, k):
    [ny, nx, nz, nc] = data.shape
    data = np.moveaxis(data, -1, 0)
    mat = np.zeros([(ny-k+1)*(nx-k+1)*(nz-k+1), k * k * k * nc], dtype = complex)
    idx = 0
    for y in range(max(1, ny - k + 1)):
        for x in range(max(1, nx - k + 1)):
            for z in range(max(1, nz - k + 1)):
                mat[idx, :] = data[:,y:y+k, x:x+k, z:z+k].reshape(1,-1)
                idx += 1
    return mat

def PLOARKS_Cinv(data, k, shape):
    [ny, nx, nz, nc] = shape
    [nt, ks] = data.shape
    data = data.reshape(nt, nc, k , k, k)
    mat = np.zeros([nc, ny, nx, nz], dtype = complex)
    count = np.zeros([nc, ny, nx, nz])
    idx = 0
    for y in range(max(1, ny - k + 1)):
        for x in range(max(1, nx - k + 1)):
            for z in range(max(1, nz - k + 1)):
                mat[:,y:y+k, x:x+k, z:z+k] += data[idx]
                count[:,y:y+k, x:x+k, z:z+k] += 1 
                idx += 1
    mat = mat/count
    return np.moveaxis(mat, 0, -1) 
def inspect_rank(data, thres = 0.01):
    U, S, VT = np.linalg.svd(data,full_matrices=False)
    s = np.copy(S)
    s = s / np.max(s)
    index = [ n for n,i in enumerate(s) if i > thres ][-1]
    yval = s[index] 
    S = np.diag(S)/np.max(np.diag(S))
    
    print("rank:"+str(index))
    
    S = np.diag(S)
    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(2,1,1)
    plt.plot(S)
    plt.axhline(y =yval, color = 'r', linestyle = '-')
    plt.title('Singular Values')
    plt.subplot(2,1,2)
    plt.imshow(np.abs(np.transpose(VT)),aspect='auto', cmap = "gray")
    plt.show()
from numpy import linalg 
def rank_approx(data, rank):
    U, S, VT = np.linalg.svd(data,full_matrices=False)
    return U[:,:rank] @ np.diag(S)[:rank, :rank] @  VT[:rank]
def l2(a, b):
    return np.linalg.norm(np.abs(a.flatten()) - np.abs(np.abs(b.flatten())))


tissues,tissuetype, t2 = get_brain_images()
ny, nx, nz, nt = tissues.shape
TE = 100
ideal_image = np.zeros([ny, nx, nz], dtype = complex)
for t in range(nt):
    ideal_image += tissues[...,t] * np.exp(TE/t2[t])
images = ideal_image
images = zpad(images, [256, 256, 256], (0, 1, 2))
images = images[::4,::4,::4] * 5e3
[ny, nx, nz] = images.shape
nc = 4
sensMap = coilmap(size = (64,64,64), ncoils=nc, coil_width=16)
images = np.repeat(images[:, :, :,np.newaxis], nc, axis=-1)
images = images * sensMap
images = np.squeeze(images)