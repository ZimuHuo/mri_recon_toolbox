import math
import scipy
from scipy import sparse 
from util.fft import * 
from util.coil import * 
from util.zpad import * 
from util.coil import *
import scipy.sparse.linalg
def spirit_weights(dataR, calib, R, kh = 4, kw = 5, lamda = 0.01):
    [ny, nx, nc] = dataR.shape
    ny = ny * R
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
    AHA = inMat.conj().T @ inMat
    S = np.sqrt(max(np.abs(np.linalg.svd(AHA,compute_uv=False))))
    wt = np.linalg.solve(
        AHA + (lamda*S)**2*np.eye(AHA.shape[0]), inMat.conj().T @ outMat).T
    wt = wt.reshape(nc,R,nc,kh,kw)
     #wt = wt.reshape(nc,R,nc,kh,kw)
    wt = np.flip(np.flip(wt.reshape(nc,R,nc,kh,kw),axis=3),axis=4)
    w = np.zeros([nc, nc, kh*R, kw], dtype = complex)
    for r in range(R):
        w[:,:,r:R*kh:R,:] = wt[:,r,:,:,:]
    del wt

    ws_k = np.zeros([nc, nc, ny, nx], dtype = complex)
    ws_k[:,:,math.ceil((ny-kh*R)/2):math.ceil((ny+kh*R)/2),math.ceil((nx-kw)/2):math.ceil((nx+kw)/2)] = w
    return ws_k
import scipy
def spirit_forward(data, kernel):
    image = ifft2c(data)
    wim = ifft2c(kernel, axis = (0,1))
    ny, nx, nc = data.shape
    out = np.zeros([ny, nx, nc], dtype = complex) 
    for c in range (nc):
        out[:,:,c] = np.sum(image*wim[...,c], axis =2)  
    return fft2c(out)

def spirit(dataR, calib, kh = 4, kw = 5, lamda = 0.01,combine = True, w= None):
    # mask = np.where(data[:,0,0] == 0, 0, 1).flatten()
    # R = mask.shape[0]//np.sum(mask)
    acs = calib
    [ny, nx, nc] = dataR.shape
    ny = ny * 2
    R = 2
    
    #train
    if w is None:
        w = spirit_weights(dataR, calib, R, kh, kw, lamda)
        w = np.moveaxis(w, 1, -1)
        w = np.moveaxis(w, 0, -1)
    
    data = np.zeros([ny, nx,nc], dtype = complex)
    data[::R] = dataR
    mask  = np.where(data == 0, 0, 1)
    
    
    def Ax(x):
        x = x.reshape(ny, nx, nc)
        out = x * mask
        return out.flatten()
    def Atb(b):
        b = b.reshape(ny, nx, nc)
        out = spirit_forward(b, w) 
        return out.flatten()
    A = scipy.sparse.linalg.LinearOperator((ny*nx*nc, ny*nx*nc), matvec=Ax, rmatvec=Atb)
    result = scipy.sparse.linalg.lsqr(A, data.flatten())
    images = ifft2c(result[0].reshape(ny, nx, nc))
    if combine:
        pat = ifft2c(zpad(acs, (ny, nx), (0,1)))
        cmap = inati_cmap(pat)
        return  np.sum(images * cmap.conj(), -1)
    else: 
        return images
    