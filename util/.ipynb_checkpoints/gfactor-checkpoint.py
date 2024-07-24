from scipy import signal
import numpy as np
from util.coil import *
from util.fft import *
from tqdm.notebook import tqdm
from util.zpad import *
import math
from scipy.linalg import pinv

def gfactor_grappa(calib, ny, nx, R, kh = 2, kw = 3):
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
    wt =  outMat@pinv(inMat, 1E-4)
    del inMat, outMat
    
    wt = wt.reshape(nc,R,nc,kh,kw)
    wt = np.flip(np.flip(wt.reshape(nc,R,nc,kh,kw),axis=3),axis=4)
    w = np.zeros([nc, nc, kh*R, kw], dtype = complex)
    for r in range(R):
        w[:,:,r:R*kh:R,:] = wt[:,r,:,:,:]
    del wt

    ws_k = np.zeros([nc, nc, ny, nx], dtype = complex)
    ws_k[:,:,math.ceil((ny-kh*R)/2):math.ceil((ny+kh*R)/2),math.ceil((nx-kw)/2):math.ceil((nx+kw)/2)] = w
    wim = ifft2c(ws_k, axis = (2,3))
    
    g = np.zeros([ny, nx])
    calib = np.moveaxis(calib, 0, -1)
    if ncy == ny and ncx == nx:
        acsIm = ifft2c(calib)
    else:
        acsIm = ifft2c(zpad(calib,(ny, nx),(0,1))) 
        h = np.hamming(ncy) 
        ham2d = np.sqrt(np.outer(h,h))
        for c in range(nc):
            acsIm[...,c] = signal.convolve(acsIm[...,c], ham2d, mode="same")
        
    for y in  range (ny):
        for x in range (nx):  
            w = wim[:,:,y,x] / R
            p = acsIm[y,x,:] / np.sqrt(np.sum(np.abs(acsIm[y,x,:])**2))
            g[y,x] = np.abs(np.sqrt(np.abs((p@w)@((p@w).conj().T)))/ np.sqrt(np.abs(p@np.eye(nc))@((p@np.eye(nc)).conj().T)))
    return g

def gfactor_sense(cmap, R , lamda = 0.01):
    [ny, nx, coil] = cmap.shape
    image = np.zeros([ny, nx])
    readny = int(ny/R)
    for x in tqdm(range(nx)):
        for y in range(readny):
            yidx = np.arange(y,ny,readny)
            S = cmap[yidx,x,:]
            STS = S @ S.conj().T  
            #M = np.linalg.inv(STS+np.eye(STS.shape[0])*lamda*np.linalg.norm(STS)/STS.shape[0])
            M = np.linalg.pinv(STS) 
            image[yidx,x] = np.abs(np.sqrt(np.diag(STS)* np.diag(M)))
    return image    