import numpy as np
from util.fft import *





# ESPIRiT mask
# mask = np.zeros([ny, nx, nc])
# start = np.floor((ny - ACS)/2)
# end = start + ACS
# for i in range(ny):
#     for j in range(nx):
#         if (i >= start) and (i < end) and (j >= start) and (j < end):
#             mask[i, j, :] = 1  
#         if (i % R == 1):
#             mask[i, :, :] = 1  
#         if (j % R == 1):
#             mask[:, j, :] = 1  




def partialFourier(shape, ptg):
    [ny, nx] = shape[:2]
    readny = int(ny * ptg)
    mask = np.ones(shape)
    mask[-(ny-readny):,...] = 0 
    return mask
'''
---------------------------------------------------
SENSE mask
---------------------------------------------------
'''
def sense(shape, R):
    [fovHeight, fovWidth, numCoil] = shape
    mask = np.zeros([fovHeight, fovWidth, numCoil])
    mask[::R,:,:] = 1
    return mask






'''
---------------------------------------------------
GRAPPA mask
---------------------------------------------------
'''
def grappa(shape, ACS, R):
    [phase, frequency, coil] = shape
    mask = np.zeros([phase, frequency, coil])
    start = np.floor((phase - ACS)/2)
    end = start + ACS
    for i in range(phase):
        if (i >= start) and (i < end):
            mask[i, :, :] = 1  # middle region
        if (i % R == 1):
            mask[i, :, :] = 1  # outside region
    return mask


def igrappa(data, undersampling_ratio, height= 5):
    [phase, frequency, coil] = data.shape
    mask = np.zeros([phase, frequency, coil])
    ACS = 2 * undersampling_ratio*(height-1)  # assume the minimum
    start = np.floor((phase - ACS)/2)
    end = start + ACS
    for i in range(phase):
        if (i >= start) and (i < end):
            mask[i, :, :] = 1  # middle region
        if (i % undersampling_ratio == 1):
            mask[i, :, :] = 1  # outside region
    data_R = data * mask
    return data_R

'''
---------------------------------------------------
compressed sensing / sparse
---------------------------------------------------
'''
def genPDF(imSize, p, pctg, radius):
    minval = 0
    maxval = 1
    sx = imSize[0]
    sy = imSize[1]
    PCTG = np.floor(pctg*sx*sy)
    x = np.linspace(-1, 1, sx)
    y = np.linspace(-1, 1, sy)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2+Y**2)
    r = r/np.max(np.abs(r))
    idx = np.zeros(imSize)
    idx[r < radius] = 1
    pdf = (np.ones(imSize)-r)**p
    pdf[idx == 1] = 1

    assert np.floor(np.sum(np.sum(pdf))) < PCTG, "increase power"

    while True:
        val = minval/2 + maxval/2
        pdf = (1-r)**p + val
        pdf[pdf > 1] = 1
        pdf[idx == 1] = 1
        N = np.floor(np.sum(np.sum(pdf)))
        if (N > PCTG):
            maxval = val
        elif (N < PCTG):
            minval = val
        else:
            break
    return pdf

def genMask(pdf, tol, numIter):
    [width, height] = pdf.shape
    K = np.floor(np.sum(np.sum(pdf)))
    minIntr = 1e99
    mask = np.zeros(pdf.shape)
    for n in range(numIter):
        temp = np.zeros(pdf.shape)
        while((np.abs(np.sum(np.sum(temp))-K)) > tol):
            uniform = np.random.uniform(-1, 1,
                                        int(width)**2).reshape(pdf.shape)
            temp = uniform < pdf
        TEMP = ifft2c(temp/pdf)
        if (np.max(np.abs(TEMP[2:])) < minIntr):
            minIntr = np.max(np.abs(TEMP[2:]))
            mask = temp
            N = n
    return mask

