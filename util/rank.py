# Zimu Huo
# 07/07/2022

import numpy as np
import math   
import matplotlib.pyplot as plt
from util.fft import *
from util.zpad import *
from tqdm.notebook import tqdm
from util.coil import * 
def PLOARKS_VCC(data, kx, ky):
    data = np.concatenate((data, np.conj(data)), axis=-1)
    [ny, nx, nc] = data.shape
    data = np.moveaxis(data, -1, 0)
    mat = np.zeros([(ny-ky+1)*(nx-kx+1), kx * ky * nc], dtype = complex)
    idx = 0
    for y in range(max(1, ny - ky + 1)):
        for x in range(max(1, nx - kx + 1)):
            mat[idx, :] = data[:,y:y+ky, x:x+kx].reshape(1,-1)
            idx += 1
    return mat

def PLOARKS_VCCinv(data, kx, ky, shape):
    [ny, nx, nc] = shape
    [nt, ks] = data.shape
    data = data.reshape(nt, nc * 2, ky , kx)
    mat = np.zeros([nc*2, ny, nx], dtype = complex)
    count = np.zeros([nc*2, ny, nx])
    idx = 0
    for y in range(max(1, ny - ky + 1)):
        for x in range(max(1, nx - kx + 1)):
            mat[:,y:y+ky, x:x+kx] += data[idx]
            count[:,y:y+ky, x:x+kx] += 1 
            idx += 1
    mat = mat/count
    mat = (mat[:nc,...] + np.conj(mat[nc:,...]))/2
    return np.moveaxis(mat, 0, -1) 

def var_den_mask(shape, R):
    start = 1e-7
    end = 0.1
    for i in range(20):    
        cur = (start + end)/2
        mask = var_dens_mask(shape, cur)
        curR = undersampling_rate(mask)
        if curR < 1/R: 
            end = (start + end) /2
        else: 
            start = (start + end) /2
    return mask

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

#LOARKS operators
def LOARKS_C(data, k):
    [ny, nx] = data.shape
    mat = np.zeros([(ny-k+1)*(nx-k+1), k * k], dtype = complex)
    idx = 0
    for y in range(max(1, ny - k + 1)):
        for x in range(max(1, nx - k + 1)):
            mat[idx, :] = data[y:y+k, x:x+k].reshape(1,-1)
            idx += 1
    return mat

def LOARKS_Cinv(data, k, shape):
    [ny, nx] = shape
    [nt, ks] = data.shape
    data = data.reshape(nt, k , k)
    mat = np.zeros([ny, nx], dtype = complex)
    count = np.zeros([ny, nx])
    idx = 0
    for y in range(max(1, ny - k + 1)):
        for x in range(max(1, nx - k + 1)):
            mat[y:y+k, x:x+k] += data[idx]
            count[y:y+k, x:x+k] += 1 
            idx += 1
    mat = mat/count
    return np.moveaxis(mat, 0, -1) 

def LOARKS_S(data, k):
    [ny, nx] = data.shape
    ks = k * k 
    nt = (ny-k+1)*(nx-k+1)
    cdata = data[::-1, ::-1]
    mat = np.zeros([nt * 2, ks * 2])
    n = 0
    for y in range(max(1, ny - k + 1)):
        for x in range(max(1, nx - k + 1)):
            Sp = data[y:y+k, x:x+k].reshape(1,-1)
            Sm = cdata[y:y+k, x:x+k].reshape(1,-1)
            mat[n   , :ks] = Sp.real - Sm.real
            mat[n   ,-ks:] = Sp.imag + Sm.imag
            mat[n+nt, :ks] = -Sp.imag + Sm.imag
            mat[n+nt,-ks:] = Sp.real + Sm.real
            n += 1
    return mat

def LOARKS_Sinv(mat, k, shape):
    [ny, nx] = shape
    ks = k * k 
    nt = (ny-k+1)*(nx-k+1)
    data = np.zeros([ny, nx], dtype = complex)
    cdata = np.zeros([ny, nx], dtype = complex)
    count = np.zeros([ny, nx], dtype = complex)
    ccount = np.zeros([ny, nx], dtype = complex)
    n = 0
    for y in range(max(1, ny - k + 1)):
        for x in range(max(1, nx - k + 1)):
            data[y:y+k, x:x+k] = ((mat[n, :ks] + mat[n+nt,-ks:])/2 + 1j*(mat[n,-ks:]-mat[n+nt, :ks])/2).reshape(k, k)
            cdata[y:y+k, x:x+k] = (-(mat[n, :ks]-mat[n+nt,-ks:])/2 +1j*(mat[n,-ks:]+mat[n+nt, :ks])/2).reshape(k, k)
            n += 1
    data = data + cdata[::-1,::-1]
    return data/2
#PLOARKS operators
def PLOARKS_C(data, k):
    [ny, nx, nc] = data.shape
    data = np.moveaxis(data, -1, 0)
    mat = np.zeros([(ny-k+1)*(nx-k+1), k * k * nc], dtype = complex)
    idx = 0
    for y in range(max(1, ny - k + 1)):
        for x in range(max(1, nx - k + 1)):
            mat[idx, :] = data[:,y:y+k, x:x+k].reshape(1,-1)
            idx += 1
    return mat

def PLOARKS_Cinv(data, k, shape):
    [ny, nx, nc] = shape
    [nt, ks] = data.shape
    data = data.reshape(nt, nc, k , k)
    mat = np.zeros([nc, ny, nx], dtype = complex)
    count = np.zeros([nc, ny, nx])
    idx = 0
    for y in range(max(1, ny - k + 1)):
        for x in range(max(1, nx - k + 1)):
            mat[:,y:y+k, x:x+k] += data[idx]
            count[:,y:y+k, x:x+k] += 1 
            idx += 1
    mat = mat/count
    return np.moveaxis(mat, 0, -1) 
def PLOARKS_S(data, k):
    [ny, nx, nc] = data.shape
    data = np.moveaxis(data, -1, 0)
    ks = k * k * nc
    nt = (ny-k+1)*(nx-k+1)
    cdata = data[:,::-1, ::-1]
    mat = np.zeros([nt * 2, ks * 2])
    n = 0
    for y in range(max(1, ny - k + 1)):
        for x in range(max(1, nx - k + 1)):
            Sp = data[:,y:y+k, x:x+k].reshape(1,-1)
            Sm = cdata[:,y:y+k, x:x+k].reshape(1,-1)
            mat[n   , :ks] = Sp.real - Sm.real
            mat[n   ,-ks:] = Sp.imag + Sm.imag
            mat[n+nt, :ks] = -Sp.imag + Sm.imag
            mat[n+nt,-ks:] = Sp.real + Sm.real
            n += 1
    return mat

def PLOARKS_Sinv(mat, k, shape):
    [ny, nx, nc] = shape
    ks = nc * k * k 
    nt = (ny-k+1)*(nx-k+1)
    data = np.zeros([nc, ny, nx], dtype = complex)
    cdata = np.zeros([nc, ny, nx], dtype = complex)
    count = np.zeros([nc, ny, nx], dtype = complex)
    ccount = np.zeros([nc, ny, nx], dtype = complex)
    n = 0
    for y in range(max(1, ny - k + 1)):
        for x in range(max(1, nx - k + 1)):
            data[:, y:y+k, x:x+k] = ((mat[n, :ks] + mat[n+nt,-ks:])/2 + 1j*(mat[n,-ks:]-mat[n+nt, :ks])/2).reshape(nc, k, k)
            cdata[:, y:y+k, x:x+k] = (-(mat[n, :ks]-mat[n+nt,-ks:])/2 +1j*(mat[n,-ks:]+mat[n+nt, :ks])/2).reshape(nc, k, k)
            n += 1
    data = data + cdata[:,::-1,::-1]
    data = np.moveaxis(data, 0, -1)
    return data/2


# ESPIRiT operators 
def ESPIRiT_forward(data, kh, kw):
    [ny, nx, nc] = data.shape
    data = np.moveaxis(data, -1, 0)
    mat = np.zeros([(ny-kh+1)*(nx-kw+1), kh * kw * nc], dtype = complex)
    idx = 0
    for y in range(max(1, ny - kh + 1)):
        for x in range(max(1, nx - kw + 1)):
            mat[idx, :] = data[:,y:y+kh, x:x+kw].flatten()
            idx += 1
    return mat
# PRUNO operators 
def PRUNO_forward(data, kh, kw):
    [ny, nx, nc] = data.shape
    data = np.moveaxis(data, -1, 0)
    mat = np.zeros([(ny-kh+1)*(nx-kw+1), kh * kw * nc], dtype = complex)
    idx = 0
    for y in range(max(1, ny - kh + 1)):
        for x in range(max(1, nx - kw + 1)):
            mat[idx, :] = data[:,y:y+kh, x:x+kw].flatten()
            idx += 1
    return mat
# SAKE operators 
def SAKE_forward(data, k):
    [ny, nx, nc] = data.shape
    data = np.moveaxis(data, -1, 0)
    mat = np.zeros([(ny-k+1)*(nx-k+1), k * k * nc], dtype = complex)
    idx = 0
    for y in range(max(1, ny - k + 1)):
        for x in range(max(1, nx - k + 1)):
            mat[idx, :] = data[:,y:y+k, x:x+k].reshape(1,-1)
            idx += 1
    return mat

def SAKE_adjoint(data, k, shape):
    [ny, nx, nc] = shape
    [nt, ks] = data.shape
    data = data.reshape(nt, nc, k , k)
    mat = np.zeros([nc, ny, nx], dtype = complex)
    count = np.zeros([nc, ny, nx])
    idx = 0
    for y in range(max(1, ny - k + 1)):
        for x in range(max(1, nx - k + 1)):
            mat[:,y:y+k, x:x+k] += data[idx]
            count[:,y:y+k, x:x+k] += 1 
            idx += 1
    mat = mat/count
    return np.moveaxis(mat, 0, -1) 

def sake(dataR, niter= 50, mask=None, thres = 0.05, threshold=None, k = 5, v=0):
    if threshold is None: 
        U, S, VT = np.linalg.svd(SAKE_forward(dataR, k) ,full_matrices=False)
        S = S/np.max(S)
        threshold =  [i for i,v in enumerate(S) if v > thres][-1]
    out = np.copy(dataR)
    for i in (range(niter)):
        patch = SAKE_forward(out, k)
        U, S, VT = np.linalg.svd(patch ,full_matrices=False)
        S = np.diag(S)
        patch = U[:,:threshold] @ S[:threshold, :threshold] @  VT[:threshold]
        out = SAKE_adjoint(patch, k, (dataR.shape))
        if mask is not None: 
            out = out * (1-mask) + dataR
        else: 
            out = out
    if v: 
        plt.figure(figsize = (12, 8))
        plt.subplot(121)
        plt.title("before")
        plt.imshow(np.abs(rsos(ifft2c(dataR))), cmap ="gray")
        plt.subplot(122)
        plt.title("after")
        plt.imshow(np.abs(rsos(ifft2c(out))), cmap ="gray")
        plt.show()
    return out
# def espirit(data, kh = 6, kw = 6, threshold = 0.01, eigenval = 0.9925, v = 0):
#     [ny, nx, nc] = data.shape
#     kern = ESPIRiT_forward(data, kh, kw)
#     U, S, VT = np.linalg.svd(kern, full_matrices=False)
#     V = VT.conj().T
#     S = S/np.max(S)
#     n = np.sum(S >= threshold)
#     V = V[:, 0:n]
#     if v : 
#         S = np.diag(S)
#         plt.figure()
#         plt.subplot(2,1,1)
#         tmp = np.diag(S)/np.max(np.diag(S))
#         plt.plot(tmp)
#         plt.axhline(y=tmp[n-1], color='r', linestyle='-')
#         plt.title('Singular Values')
#         plt.subplot(2,1,2)
#         plt.imshow(np.abs(np.transpose(VT)),aspect='auto', cmap = "gray")
#         plt.show()
#     kern = V.reshape(nc, kh, kw, n)
#     kern = np.moveaxis(kern, 0, -2)
#     kern = kern[::-1,::-1,:,:]
#     kern = zpad(kern, (ny, nx), (0,1))
#     kern = ifft2c(kern)/ np.sqrt(kh * kw)
#     sensmap = np.zeros([ny, nx, nc, nc], dtype = complex)
#     for y in tqdm(range(ny)):
#         for x in range(nx):
#             u, s, vT = np.linalg.svd(kern[y, x, ...], full_matrices=True)
#             for c in range(nc):
#                 if (s[c]**2 > eigenval):
#                     sensmap[y, x,:,c] = u[:,c]
#     return sensmap[...,0]
# the next three functions are copied from github https://github.com/marijavella
import numpy as np
from numpy.lib.stride_tricks import as_strided
def soft_thresh(u, lmda):
    """Soft-threshing operator for complex valued input"""
    Su = (abs(u) - lmda) / abs(u) * u
    Su[abs(u) < lmda] = 0
    return Su

def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)

def var_dens_mask(shape, ivar, sample_high_freq=True):
    """Variable Density Mask (2D undersampling)"""
    if len(shape) == 3:
        Nt, Nx, Ny = shape
    else:
        Nx, Ny = shape
        Nt = 1

    pdf_x = normal_pdf(Nx, ivar)
    pdf_y = normal_pdf(Ny, ivar)
    pdf = np.outer(pdf_x, pdf_y)

    size = pdf.itemsize
    strided_pdf = as_strided(pdf, (Nt, Nx, Ny), (0, Ny * size, size))
    # this must be false if undersampling rate is very low (around 90%~ish)
    if sample_high_freq:
        strided_pdf = strided_pdf / 1.25 + 0.02
    mask = np.random.binomial(1, strided_pdf)

    xc = Nx // 2
    yc = Ny // 2
    mask[xc - 4:xc + 5, yc - 4:yc + 5] = True

    if Nt == 1:
        return mask.reshape((Ny, Nx))

    return mask
def undersampling_rate(mask):
    return float(mask.sum()) / mask.size
def get_phase(x):
    xr = np.real(x)
    xi = np.imag(x)
    phase = np.arctan(xi / (xr + 1e-12))
    return phase