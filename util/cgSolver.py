import numpy as np
from util.fft import *
from tqdm.notebook import tqdm
import time

def cg_sense(dataR, sensMap, numIter = 50):
    mask = np.where(dataR == 0, 0, 1)
    imagesR = ifft2c(dataR)
    [height, width, coil] = imagesR.shape
    sconj = np.conj(sensMap)
    B = np.sum(imagesR*sconj,axis = 2)
    B = B.flatten()
    x = 0*B
    r = B 
    d = r 
    for j in range(numIter):
        Ad = np.zeros([height,width],dtype = complex)
        for i in range(coil):  
            Ad += ifft2c(fft2c(d.reshape([height,width])*sensMap[:,:,i])*mask[:,:,i])*sconj[:,:,i] 
            # this is intuitive, the correct image is sensivity encoded, then undersampled in K space, but this took me quite a while :(
        # see equations from 45 to 49 in [4].
        Ad = Ad.flatten()
        a = np.dot(r,r)/(np.dot(d,Ad))
        '''
        This is the core idea behind steepest descent, where the new direction is orthogonal to the previous search. 
        This also very intuitive, in some sense you only maximise the information obatined in this search when it can no longer 
        improve your next search. However, this will lead to zigzag-ish path, which is highly inefficient.     
        '''
        x = x + a*d
        '''
        This corresponds to a collection of images, like layers upon layers staring from the initial guess. 
        '''
        rn = r - a*Ad
        beta = np.dot(rn,rn)/np.dot(r,r)
        r=rn
        d = r + beta*d
        '''
        This is where conjugate gradient kicks in, it uses the previous hardwork along with the current search direction. 
        As a result, this is more efficient. Like climbing a hill but carries some weight, or having inertia. 
        This idea is also quite commonly used in machine learning -> momentum
        '''
    return x.reshape([height,width])