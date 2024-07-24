import numpy as np
from util.fft import *
from sklearn import linear_model, datasets
from scipy.integrate import cumtrapz
#process trimmed epi phase correction with ref 
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
from tqdm import tqdm
import scipy.signal
from scipy import linalg
from scipy.signal import savgol_filter
from scipy.optimize import fmin

def phase_difference(sig1, sig2, refflag):
    mode = "avg_coil"
    if len(sig1.shape) == 1:
        mode = "per_coil"
    phasep = np.angle(ifft(sig1))
    phasem = np.angle(ifft(sig2))
    l2p = np.exp(1j*phasep)
    l2m = np.exp(1j*phasem)
    if refflag[0]:
        if mode == "per_coil":
            return np.angle(l2m/l2p)
        else:
            return np.average(np.angle(l2m/l2p),1)     
    else:
        if mode == "per_coil":
            return np.angle(l2p/l2m)
        else:
            return np.average(np.angle(l2p/l2m),1)       
def object_projection1d(ref):
    if len(ref.shape)>2:
        obj = (np.sqrt(np.sum(np.abs(ifft(ref[0]))**4,1))+np.sqrt(np.sum(np.abs(ifft(ref[2]))**4,1)) )/2
    else:
        obj = (np.sqrt((np.abs(ifft(ref[0]))**4))+np.sqrt((np.abs(ifft(ref[2]))**4)))/2
    obj *= 1/obj.max()
    return obj

def polyfit(x, y, w, order):
    coef = np.polyfit(x,y,order, w=w)
    phi = np.poly1d(coef) 
    return phi(x.reshape(-1,1)).flatten()

def epi_phasecorrection_percoil(data, ref, dataflag, refflag, dork_corr = False, residual = None, order = 8, v = 0):
    [ny, nx, nc] = data.shape
    x = np.arange(nx) - nx//2 
    recon = np.zeros(data.shape, dtype = complex)
    if v: 
        plt.figure(), plt.title("EPI navigator signals")
    for c in range(nc):
        obj = object_projection1d(ref[...,c])
        
        tvec_epi = np.ones(x.shape)
        tvec_dork = np.ones(x.shape)
                
        if dork_corr: 
            dif = phase_difference(ref[0,...,c], ref[2,...,c], refflag)
            dif = savgol_filter(dif, 11, 4)
            intercept = np.mean(dif)
            tvec = x * 0 + intercept
            tvec_b = np.exp(-1j*tvec*0.5)
            
        
        dif = phase_difference((ref[0,...,c]+ref[2,...,c])/2, ref[1,...,c], refflag)
        if residual is not None: 
            dif -= phase_difference((residual[0,...,c]+residual[2,...,c])/2, residual[1,...,c], refflag)
            dif = savgol_filter(dif, 11, 4)
            tvec = polyfit(x, dif, w = obj, order = order)

        else: 
            dif = savgol_filter(dif, 11, 4)
            tvec = polyfit(x, dif, w = obj, order = order)
        if v and c < 3:
            plt.plot(x, dif), plt.plot(x, tvec), plt.ylim([-np.pi, np.pi])
        tvec_epi = np.exp(1j*tvec)
        for y in range(data.shape[0]):
            if(dataflag[y]):
                recon[y,...,c] = ifft(data[y,...,c],0) * tvec_epi  * tvec_dork**(y+1)
            else: 
                recon[y,...,c] = ifft(data[y,...,c],0) * tvec_dork**(y+1)
        recon[...,c] = fft(recon[...,c],1)

    return recon

def epi_phasecorrection_avgcoil(data, ref, dataflag, refflag,dork_corr = False, residual = None,  order = 8, v = 0):
    [ny, nx, nc] = data.shape
    x = np.arange(nx) - nx//2 
    recon = np.zeros(data.shape, dtype = complex)
    if v: 
        plt.figure(), plt.title("EPI navigator signals")
    obj = object_projection1d(ref)
    tvec_dork = np.ones(x.shape)
    tvec_epi = np.ones(x.shape)

    if dork_corr: 
        dif = phase_difference(ref[0], ref[2], refflag)
        dif = savgol_filter(dif, 11, 4)
        intercept = np.mean(dif)
        tvec = x * 0 + intercept
        tvec_b = np.exp(-1j*tvec*0.5)

    dif = phase_difference((ref[0]+ref[2])/2, ref[1], refflag)
    if residual is not None: 
        dif -= phase_difference((residual[0]+residual[2])/2, residual[1], refflag)
        dif = savgol_filter(dif, 11, order)
        tvec = polyfit(x, dif, w = obj, order = order)
    else: 
        dif = savgol_filter(dif, 11, order)
        tvec = polyfit(x, dif, w = obj, order = order)
    if v:
        plt.plot(x, dif), plt.plot(x, tvec), plt.ylim([-np.pi, np.pi])
    tvec_epi = np.exp(1j*tvec)
    for y in range(data.shape[0]):
        if(dataflag[y]):
            recon[y] = ifft(data[y],0) * np.tile((tvec_epi  * tvec_dork**y).reshape(-1,1), data.shape[2])
        else: 
            recon[y] = ifft(data[y],0) * np.tile((tvec_dork**y).reshape(-1,1), data.shape[2])
    recon = fft(recon,1)
    return recon

def epi_phasecorrection_avgcoil_parameter(data, ref, dataflag, refflag,  order = 1, v = 0):
    [ny, nx, nc] = data.shape
    x = np.arange(nx) - nx//2 
    if v: 
        plt.figure(), plt.title("EPI navigator signals")
    obj = object_projection1d(ref)
    tvec_epi = np.ones(x.shape)
    dif = phase_difference((ref[0]), ref[1], refflag)
    # filter_length = 11
    # if order > 10:
    filter_length = 11
    dif = savgol_filter(dif, filter_length, 4)
    tvec = polyfit(x, dif, w = obj, order = order)
    if v:
        plt.plot(x, dif), plt.plot(x, tvec), plt.ylim([-np.pi, np.pi])
    tvec_epi = np.exp(1j*tvec)
    return tvec_epi

def apply_epi_phasecorrection_avgcoil_parameter(data, dataflag, tvec_epi):
    [ny, nx, nc] = data.shape
    recon = np.zeros(data.shape, dtype = complex)
    for y in range(data.shape[0]):
        if(dataflag[y]):
            recon[y] = ifft(data[y],0) * np.tile((tvec_epi ).reshape(-1,1), data.shape[2])
        else: 
            recon[y] = ifft(data[y],0) 
    recon = fft(recon,1)
    return recon