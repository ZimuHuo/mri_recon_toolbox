import numpy as np
from tqdm.notebook import tqdm
from util.fft import * 
from util.zpad import * 
import util.coil as coil
# def sense(images, sensmap, R = 2):
#     '''
#     -------------------------------------------------------------------------
#     Parameters
    
#     sensmap: array_like
#     sensivity maps for each coils [height, width, coils]
    
#     images: array_like
#     images for each coils [height, width, coils]
    
#     R: scalar 
#     under sampling ratio 
    
#     -------------------------------------------------------------------------
#     Returns
#     image : array like
#     reconstructed image
    
#     -------------------------------------------------------------------------
#     Notes
#     Intuitive implementation
    
#     -------------------------------------------------------------------------
#     References
    
#     [1] 
#     Author: Klaas P. Pruessmann et al. 
#     Title: SENSE: Sensitivity Encoding for Fast MRI
#     Link: https://pubmed.ncbi.nlm.nih.gov/10542355/
#     '''
#     [height, width, coil] = sensmap.shape
#     image = np.zeros([height, width], dtype= complex)
#     for y in range(int(height/R)):
#         index = np.arange(y,height,int(height/R))
#         for x in range(width):
#             s = np.transpose(sensmap[index,x,:].reshape(R,-1))
#             M = np.matmul(np.linalg.pinv(s),images[y,x,:].reshape(-1,1))    
#             image[index,x] = M[:,0]
#     return image

# strictly using the formulation from the papar, coil correlation can be mixed into the play. 
# from tqdm.notebook import tqdm
# import util.coil as coil 
# from util.zpad import * 
# from util.fft import * 

def sense(dataR, acs, lamda = 1E-4):
    mask = np.where(dataR[:,0,0] == 0, 0, 1).flatten()
    R = int(np.ceil(mask.shape[0]/np.sum(mask)))
    [ny, nx, nc] = dataR.shape
    images = ifft2c(dataR)
    readny = int(ny/R)
    pat = ifft2c(zpad(acs, (ny, nx), (0,1)))
    coilmaps = coil.inati_cmap(pat) 
    coilmaps = coilmaps / np.max(coil.rsos(coilmaps))
    recon = np.zeros([ny,nx], dtype = complex)
    for x in (range(nx)):
        for y in range(readny):
            yidx = np.arange(y,ny,readny)
            S = coilmaps[yidx,x,:]
            STS = S.T @ S     
            #M = np.linalg.inv(STS+np.eye(STS.shape[0])*lamda*np.linalg.norm(STS)/STS.shape[0])@S.T 
            M = np.linalg.pinv(STS)@S.T 
            recon[yidx,x] = M.T@images[y,x,:]
    return recon