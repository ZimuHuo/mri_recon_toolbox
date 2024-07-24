import numpy as np
from util.fft import *
from util.coil import * 
import math
from scipy.linalg import pinv
from util.fft import *
from tqdm.notebook import tqdm
from util.zpad import * 
def plot_singular_values(matrix):
    # Compute the singular values
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    # print(singular_values[1]/singular_values[-1])
    
    # Plotting the singular values
    plt.figure(figsize=(8, 4))
    plt.plot(singular_values, 'o-')
    plt.title('Singular Values')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.grid(True)
    plt.show()
    
def joint_grappa_weights(calib, R, kh = 4, kw = 5, lamda = 3e-3):
    #train
    [ncy, ncx, pc, nc] = calib.shape
    ks = pc*nc*kh*kw
    nt = (ncy-(kh-1)*R)*(ncx-(kw-1))
    inMat=np.zeros([nt, ks], dtype = complex)
    outMat=np.zeros([nt,(R-1),pc, nc], dtype = complex)
    n = 0
    for x in ((np.arange(np.floor(kw/2),ncx-np.floor(kw/2), dtype=int))):
        for y in (np.arange(ncy-(kh-1)*R)):
            inMat[n,...] = calib[y:y+kh*R:R, int(x-np.floor(kw/2)):int(x+np.floor(kw/2))+1,:,:].reshape(1,-1)
            outMat[n,...] = calib[int(y+np.floor((R*(kh-1)+1)/2) - np.floor(R/2))+1:int(y+np.floor((R*(kh-1)+1)/2)-np.floor(R/2)+R),x,:,:]
            n = n + 1  
    weight = np.zeros([(R-1), ks, pc, nc], dtype = complex)
    # plot_singular_values(inMat)
    if lamda:
        [u,s,vh] = np.linalg.svd(inMat,full_matrices=False)
        s_inv = np.conj(s) / (np.abs(s)**2 + lamda);
        inMat_inv = vh.conj().T @ np.diag(s_inv) @ u.conj().T;
        for c in range(nc):
            for p in range(pc):
                weight[:,:,p,c] = (inMat_inv @ outMat[:,:,p,c]).T;
    else:
        for c in range(nc): 
            for p in range(pc):
                weight[:,:,p,c] = (np.linalg.pinv(inMat) @ outMat[:,:,p,c]).T;
    return weight

def joint_grappa(dataR, calib, kh = 4, kw = 5, lamda = 0,combine =True, w= None, R = None):
    if R is None:
        mask = np.where(dataR[:,0,0] == 0, 0, 1).flatten()
        R = int(np.ceil(mask.shape[0]/np.sum(mask)))

    acs = calib
    [ny, nx, pc, nc] = dataR.shape
    if w is None: 
        w = joint_grappa_weights(acs, R, kh, kw, lamda)

    data = np.zeros([ny, nx, pc, nc], dtype = complex)
    for x in (range(nx)):
        xs = get_circ_xidx(x, kw, nx)
        for y in range (0,ny,R):
            ys = np.mod(np.arange(y, y+(kh)*R, R), ny)
            yf = get_circ_yidx(ys, R, kh, ny)
            kernel = dataR[ys, :, :,:][:, xs,:,:].reshape(-1,1)
            for c in range(nc):
                for p in range(pc):
                    data[yf, x, p,c] = np.matmul(w[:,:,p,c], kernel).flatten()
    data += dataR
    
    images = ifft2c(data) 
    if combine:
        return rsos(rsos(images,-1),-1)
    else: 
        return images

def get_circ_xidx(x, kw, nx):
    return np.mod(np.linspace(x-np.floor(kw/2), x+np.floor(kw/2), kw,dtype = int),nx)
def get_circ_yidx(ys, R, kh, ny):
    return np.mod(np.linspace(ys[kh//2-1]+1, np.mod(ys[kh//2]-1,ny), R-1, dtype = int), ny) 
    

    
    
    
    
    
# def disp_condition(matrix):
#     # Compute the singular values
#     singular_values = np.linalg.svd(matrix, compute_uv=False)
#     print(singular_values[1]/singular_values[-1])
#     # # Plotting the singular values
#     # plt.figure(figsize=(8, 4))
#     # plt.plot(singular_values, 'o-')
#     # plt.title('Singular Values')
#     # plt.xlabel('Index')
#     # plt.ylabel('Singular Value')
#     # plt.grid(True)
#     # plt.show()

    
    
    
    
    
def grappa_weights(calib, R, kh = 4, kw = 5, lamda = 1e-3):
    #train
    [ncy, ncx, nc] = calib.shape
    ks = nc*kh*kw
    nt = (ncy-(kh-1)*R)*(ncx-(kw-1))
    inMat=np.zeros([nt, ks], dtype = complex)
    outMat=np.zeros([nt,(R-1),nc], dtype = complex)
    n = 0
    for x in ((np.arange(np.floor(kw/2),ncx-np.floor(kw/2), dtype=int))):
        for y in (np.arange(ncy-(kh-1)*R)):
            inMat[n,...] = calib[y:y+kh*R:R, int(x-np.floor(kw/2)):int(x+np.floor(kw/2))+1,:].reshape(1,-1)
            outMat[n,...] = calib[int(y+np.floor((R*(kh-1)+1)/2) - np.floor(R/2))+1:int(y+np.floor((R*(kh-1)+1)/2)-np.floor(R/2)+R),x,:]
            n = n + 1  
    weight = np.zeros([(R-1), ks, nc], dtype = complex)
    # disp_condition(inMat)
    if lamda:
        [u,s,vh] = np.linalg.svd(inMat,full_matrices=False)
        s_inv = np.conj(s) / (np.abs(s)**2 + lamda);
        inMat_inv = vh.conj().T @ np.diag(s_inv) @ u.conj().T;
        for c in range(nc): 
            weight[:,:,c] = (inMat_inv @ outMat[:,:,c]).T;
    else:
        for c in range(nc): 
            weight[:,:,c] = np.linalg.pinv(inMat) @ outMat[:,:,c];
    return weight #[34, 680]

def grappa(dataR, calib, kh = 4, kw = 5, lamda = 1e-3,combine =True, w= None, R = None):
    if R is None:
        mask = np.where(dataR[:,0,0] == 0, 0, 1).flatten()
        R = int(np.ceil(mask.shape[0]/np.sum(mask)))

    acs = calib
    [ny, nx, nc] = dataR.shape
    if w is None: 
        w = grappa_weights(acs, R, kh, kw, lamda)

    data = np.zeros([ny, nx, nc], dtype = complex)
    for x in range(nx):
        xs = get_circ_xidx(x, kw, nx)
        for y in range (0,ny,R):
            ys = np.mod(np.arange(y, y+(kh)*R, R), ny)
            yf = get_circ_yidx(ys, R, kh, ny)
            kernel = dataR[ys, :, :][:, xs,:].reshape(-1,1)
            for c in range(nc):
                data[yf, x, c] = np.matmul(w[:,:,c], kernel).flatten()
    data += dataR
    
    images = ifft2c(data) 
    if combine:
        return rsos(images)
    else: 
        return images

def get_circ_xidx(x, kw, nx):
    return np.mod(np.linspace(x-np.floor(kw/2), x+np.floor(kw/2), kw,dtype = int),nx)
def get_circ_yidx(ys, R, kh, ny):
    return np.mod(np.linspace(ys[kh//2-1]+1, np.mod(ys[kh//2]-1,ny), R-1, dtype = int), ny) 
    
    