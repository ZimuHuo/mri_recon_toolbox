import numpy as np
import matplotlib.pyplot as plt
from util.coil import *
import time


from math import log10, sqrt
import numpy as np
def l2(a, b):
    return np.linalg.norm(np.abs(a.flatten()) - np.abs(np.abs(b.flatten())))
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
def rescale(ref, data):
    from scipy import optimize
    ref = np.abs(ref)
    data = np.abs(data)
    def minl2(x, ref, data):
        return np.sum(np.abs(ref - data/x)**2)
    x = optimize.fmin(minl2, 1, args= (ref, data), disp=0)
    data = data / x 
    return ref, data 


def stitch(images, row = 4):
    ny, nx, ns = images.shape
    col = int(np.floor(ns/row))
    slices = int(row * col)
    images = images[:,: ,:slices]
    images = images.reshape(ny, nx, row, col)
    out = np.zeros([ny * row, nx * col], dtype =images.dtype)
    for r in range(row): 
        out[r*ny:(r+1)*ny] = images[...,r,:].reshape(ny,-1, order = "F")
    return out

from scipy import ndimage
def object_mask(image, threshold, ds = 7, es = 7):
    image = image / np.max(image)
    mask = np.zeros(image.shape)
    mask[image > threshold] = 1
    mask = ndimage.binary_dilation(mask, structure=np.ones((ds,ds)))
    mask = ndimage.binary_erosion(mask, structure=np.ones((es,es)))
    return mask

def factorise(num):
    l = []
    for i in range(1, num):
        if num % i == 0:
            l.append(i)
    length = len(l)
    return int(l[int(length//2)])

def mosaic(images):
    fig = plt.figure(figsize=(12, 12), dpi=80)
    fig.subplots_adjust(wspace=0, hspace=0)
    l = images.shape[-1]
    w = factorise(l)
    for idx in range(l):
        ax = fig.add_subplot(w, l//w, idx+1)
        ax.imshow(np.abs(images[...,idx]), cmap = "gray")
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    plt.show()
        
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import clear_output
def slider(images):
    images = np.abs(images)
    if len(images.shape) > 3:
            images = rsos(images)
    def show(idx):
        _, ax = plt.subplots(1,1)
        plt.imshow(images[...,int(idx)], cmap="gray")
        plt.axis('off')
    interact(show, idx = widgets.FloatSlider(value=0,
                                                   min=0,
                                                   max=images.shape[-1]-1,
                                                   step=1))
    
def plot1d(traj):
    val = 0. # this is the value where you want the data to appear on the y-axis.
    ar = traj # just as an example array
    plt.plot(ar, np.zeros_like(ar) + val, '.')
    plt.axis('off')
    plt.show()
    
def show(image, row = 1):
    if len(image.shape) > 2: 
        # mosaic(image)
        [ny, nx, nz] = image.shape
        image = image[:,:,:nz//row *row]
        plt.imshow(np.abs(stitch(image, row)), cmap="gray") 
        plt.axis('off')
        plt.show()
        
    else: 
        plt.imshow(np.abs(image), cmap="gray") 
        plt.axis('off')
        plt.show()
    
def showrsos(image):
    if len(image.shape) > 3: 
        mosaic(rsos(image))
    else: 
        plt.figure(figsize = (12, 8))
        plt.imshow(np.abs(rsos(image)), cmap="gray")
        plt.axis('off')
    
    

def conjugate(data):
    return fft2c(np.conj(ifft2c(data)))

def showc(image, row = 4):
    plt.figure()
    if len(image.shape) > 2: 
        [ny, nx, nz] = image.shape
        image = image[:,:,:nz//row *row]
        image = stitch(image, row)
        tf = plt.imshow(np.abs(image),cmap='jet')    
    else:
        tf = plt.imshow(np.abs(image),cmap='jet')
    plt.colorbar(tf, fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.show()
    
def ifft(F, axis = (0)):
    x = (axis)
    tmp0 = np.fft.ifftshift(F, axes=(x,))
    tmp1 = np.fft.ifft(tmp0, axis = x)
    f = np.fft.fftshift(tmp1, axes=(x,))
    return f * F.shape[x]

def fft(f, axis = (0)):
    x = (axis)
    tmp0 = np.fft.fftshift(f, axes=(x,))
    tmp1 = np.fft.fft(tmp0, axis = x)
    F = np.fft.ifftshift(tmp1, axes=(x,))
    return F / f.shape[x]
def fft1c(f, axis = (0)):
    x = (axis)
    tmp0 = np.fft.fftshift(f, axes=(x,))
    tmp1 = np.fft.fft(tmp0, axis = x)
    F = np.fft.ifftshift(tmp1, axes=(x,))
    return F / f.shape[x]
def ifft2c(F, axis = (0,1)):
    x,y = (axis)
    tmp0 = np.fft.ifftshift(np.fft.ifftshift(F, axes=(x,)), axes=(y,))
    tmp1 = np.fft.ifft(np.fft.ifft(tmp0, axis = x), axis = y)
    f = np.fft.fftshift(np.fft.fftshift(tmp1, axes=(x,)), axes=(y,))
    return f * F.shape[x]* F.shape[y] 

def fft2c(f, axis = (0,1)):
    x,y = (axis)
    tmp0 = np.fft.fftshift(np.fft.fftshift(f, axes=(x,)), axes=(y,))
    tmp1 = np.fft.fft(np.fft.fft(tmp0, axis = x), axis = y)
    F = np.fft.ifftshift(np.fft.ifftshift(tmp1, axes=(x,)), axes=(y,))
    return F / f.shape[x]/ f.shape[y]

def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

def siemens_readout_trim(F):
    [ny,nx,] = F.shape[:2]
    f = ifft(F,1)
    f = f[:,nx//4:nx-nx//4,...]
    return fft(f, 1)

def srsos(images,coilaxis = 2):
    images = fft2c(images)
    images = siemens_readout_trim(images)
    images = ifft2c(images)
    return np.sqrt(np.sum(np.square(np.abs(images)),axis = coilaxis))

def process_reference(acs):
    l = len(acs.shape)
    tmp = acs
    for i in range(l-1):
        tmp = np.sum(tmp, -1)
    index = np.where(np.abs(tmp)!=0)[0]
    acs = acs[index,:,:]
    return acs

def show_mask(dataR):
    
    mask = np.where(dataR == 0, 0, 1)
    if len(mask.shape) > 2: 
        return mask[...,0]
    return mask
    