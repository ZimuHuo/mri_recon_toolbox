import numpy as np
from util.fft import *
from util.coil import * 
import math
from scipy.linalg import pinv
from util.fft import *
from tqdm.notebook import tqdm
from util.zpad import * 
'''
-------------------------------------------------------------------------
Parameters

dataR: array_like
undersampled k space data [height, width, coils]

-------------------------------------------------------------------------
Returns
image : array like
reconstructed image for each coil 

-------------------------------------------------------------------------
Author: Zimu Huo
-------------------------------------------------------------------------
References

[1] 
Author: Mark A. Griswold et al. 
Title: Generalized Autocalibrating Partially Parallel Acquisitions (GRAPPA)
Link: https://pubmed.ncbi.nlm.nih.gov/12111967/
'''

import numpy as np
import matplotlib.pyplot as plt

def plot_singular_values(matrix):
    # Compute the singular values
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    print(singular_values[1]/singular_values[-1])
    # # Plotting the singular values
    # plt.figure(figsize=(8, 4))
    # plt.plot(singular_values, 'o-')
    # plt.title('Singular Values')
    # plt.xlabel('Index')
    # plt.ylabel('Singular Value')
    # plt.grid(True)
    # plt.show()



def grappa(data, acs, method = "conv", kh = 4, kw = 5, lamda = 1e-5,combine =True, w= None):
    if method == "conv":
        return grappa2(data, acs, kh , kw , lamda, combine, w)
    else:
        return grappa1(data, acs, kh , kw , lamda, combine, w)


def grappa2_weights(calib, R, kh = 4, kw = 5, lamda = 0.01):
    calib = np.moveaxis(calib, -1, 0) # move the coil to the front -> fft in the axis 3 and 4
    #train
    [nc, ncy, ncx] = calib.shape
    ks = nc*kh*kw
    nt = (ncy-(kh-1)*R)*(ncx-(kw-1))
    inMat=np.zeros([ks,nt], dtype = complex)
    outMat=np.zeros([nc*(R-1),nt], dtype = complex)
    if ks < nc * R: 
        print("underdetermined system")
    n = 0
    for x in ((np.arange(np.floor(kw/2),ncx-np.floor(kw/2), dtype=int))):
        for y in (np.arange(ncy-(kh-1)*R)):
            inMat[...,n] = calib[:,y:y+kh*R:R, int(x-np.floor(kw/2)):int(x+np.floor(kw/2))+1].reshape(1,-1)
            outMat[...,n] = calib[:,int(y+np.floor((R*(kh-1)+1)/2) - np.floor(R/2))+1:int(y+np.floor((R*(kh-1)+1)/2)-np.floor(R/2)+R),x].reshape(1,-1)
            n = n + 1  
    inMat = inMat.T
    plot_singular_values(inMat)
    outMat = outMat.T
    AHA = inMat.conj().T @ inMat
    S = np.sqrt(max(np.abs(np.linalg.svd(AHA,compute_uv=False))))
    w = np.linalg.solve(
        AHA + (lamda*S)*np.eye(AHA.shape[0]), inMat.conj().T @ outMat).T
    return w

def grappa2(dataR, calib, kh = 4, kw = 5, lamda = 0.01,combine =True, w= None, R = None):
    if R is None:
        mask = np.where(dataR[:,0,0] == 0, 0, 1).flatten()
        R = int(np.ceil(mask.shape[0]/np.sum(mask)))

    acs = calib
    calib = np.moveaxis(calib, -1, 0) # move the coil to the front -> fft in the axis 3 and 4
    dataR = np.moveaxis(dataR, -1, 0)
    [nc, ny, nx] = dataR.shape
    if w is None: 
        #train
        w = grappa2_weights(acs, R, kh, kw, lamda)

    data = np.zeros([nc,ny,nx], dtype = complex)
    for x in range(nx):
        xs = get_circ_xidx(x, kw, nx)
        for y in range (0,ny,R):
            ys = np.mod(np.arange(y, y+(kh)*R, R), ny)
            yf = get_circ_yidx(ys, R, kh, ny)
            kernel = dataR[:,ys][:,:,xs].reshape(-1,1)
            data[:,yf, x] = np.matmul(w, kernel).reshape(nc,R-1)
    data += dataR
    data = np.moveaxis(data, 0, -1)
    
    images = ifft2c(data) 
    if combine:
        pat = ifft2c(zpad(acs, (ny, nx), (0,1)))
        cmap = inati_cmap(pat)
        return  np.sum(images * cmap.conj(), -1)
    else: 
        return images

def get_circ_xidx(x, kw, nx):
    return np.mod(np.linspace(x-np.floor(kw/2), x+np.floor(kw/2), kw,dtype = int),nx)
def get_circ_yidx(ys, R, kh, ny):
    return np.mod(np.linspace(ys[kh//2-1]+1, np.mod(ys[kh//2]-1,ny), R-1, dtype = int), ny) 
    
    
    


import numpy as np
import math
from scipy.linalg import pinv
from util.fft import *
from tqdm.notebook import tqdm
from util.fft import * 
from util.coil import * 
from util.zpad import * 
# this is grappa but image based recon 
def grappa1_weights(calib, R, kh = 4, kw = 5, lamda = 0.01):
    calib = np.moveaxis(calib, -1, 0) # move the coil to the front -> fft in the axis 3 and 4
    [nc, ncy, ncx] = calib.shape
    ks = nc*kh*kw
    nt = (ncy-(kh-1)*R)*(ncx-(kw-1))
    inMat=np.zeros([ks,nt], dtype = complex)
    outMat=np.zeros([nc*R,nt], dtype = complex)
    n = 0
    for x in (np.arange(np.floor(kw/2),ncx-np.floor(kw/2), dtype=int)):
        for y in (np.arange(ncy-(kh-1)*R)):
            inMat[...,n] = calib[:,y:y+kh*R:R, int(x-np.floor(kw/2)):int(x+np.floor(kw/2))+1].reshape(1,-1)
            outMat[...,n] = calib[:,int(y+np.floor((R*(kh-1)+1)/2) - np.floor(R/2)):int(y+np.floor((R*(kh-1)+1)/2)-np.floor(R/2)+R),x].reshape(1,-1)
            n = n + 1  
    #wt =  outMat@pinv(inMat, 1E-4)
    inMat = inMat.T
    outMat = outMat.T
    plot_singular_values(inMat)
    AHA = inMat.conj().T @ inMat
    print(np.abs(np.linalg.cond(AHA, p='fro')))
    S = np.sqrt(max(np.abs(np.linalg.svd(AHA,compute_uv=False))))
    wt = np.linalg.solve(
        AHA + (lamda*S)*np.eye(AHA.shape[0]), inMat.conj().T @ outMat).T
    wt = wt.reshape(nc,R,nc,kh,kw)
    return wt

def grappa1(data, acs, kh = 4, kw = 5, lamda = 0.1,combine = True, wt= None, v= 0, R=None):
    if R is None:
        mask = np.where(data[:,0,0] == 0, 0, 1).flatten()
        R = int(np.ceil(mask.shape[0]/np.sum(mask)))
    if v:
        print("undersample factor of "+str(R))
    [ny, nx, nc] = data.shape
    
    #train
    if wt is None:
        wt = grappa1_weights(acs,R,kh,kw, lamda)
    wt = np.flip(np.flip(wt.reshape(nc,R,nc,kh,kw),axis=3),axis=4)
    w = np.zeros([nc, nc, kh*R, kw], dtype = complex)
    for r in range(R):
        w[:,:,r:R*kh:R,:] = wt[:,r,:,:,:]
    del wt
    ws_k = np.zeros([nc, nc, ny, nx], dtype = complex)
    ws_k[:,:,math.ceil((ny-kh*R)/2):math.ceil((ny+kh*R)/2),math.ceil((nx-kw)/2):math.ceil((nx+kw)/2)] = w
    wim = ifft2c(ws_k, axis = (2,3))
    

    aliased_image = ifft2c(data)
    recon = np.zeros([ny,nx, nc], dtype = complex) 
    for c in range (nc):
        tmp = wim[c,:,:,:]
        tmp = np.moveaxis(tmp, 0,-1)
        recon[:,:,c] = np.sum(tmp*aliased_image, axis = 2)  
    if combine: 
        pat = ifft2c(zpad(acs, (ny, nx), (0,1)))
        cmap = inati_cmap(pat)
        return  np.sum(recon * cmap.conj(), -1)
    else:
        return recon

