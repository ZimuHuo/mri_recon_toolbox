import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
def PLOARKS_C(data, kx, ky):
    [ny, nx, nc] = data.shape
    data = np.moveaxis(data, -1, 0)
    mat = np.zeros([(ny-ky+1)*(nx-kx+1), kx * ky * nc], dtype = complex)
    idx = 0
    for y in range(max(1, ny - ky + 1)):
        for x in range(max(1, nx - kx + 1)):
            mat[idx, :] = data[:,y:y+ky, x:x+kx].reshape(1,-1)
            idx += 1
    return mat
'''Coil compression using principal component analysis.'''

import logging

import numpy as np
from sklearn.decomposition import PCA

def python_pca(X, n_components=False):
    '''Python implementation of principal component analysis.

    To verify I know what sklearn's PCA is doing.

    Parameters
    ----------
    X : array_like
        Matrix to perform PCA on.
    n_components : int, optional
        Number of components to keep.

    Returns
    -------
    P : array_like
        n_component principal components of X.
    '''

    M = np.mean(X.T, axis=1)
    C = X - M
    V = np.cov(C.T)
    _values, vectors = np.linalg.eig(V)
    P = vectors.T.dot(C.T)[:n_components, :].T

    return P

def coil_pca(
        coil_ims,
        coil_dim=-1,
        n_components=4,
        give_explained_var=False,
        real_imag=True,
        debug_level=logging.WARNING):
    '''Reduce the dimensionality of the coil dimension using PCA.

    Parameters
    ----------
    coil_ims : array_like
        Coil images.
    coil_dim : int, optional
        Coil axis, default is last axis.
    n_components : int, optional
        How many principal components to keep.
    give_explained_var : bool, optional
        Return explained variance for real,imag decomposition
    real_imag : bool, optional
        Perform PCA on real/imag parts separately or mag/phase.
    debug_level : logging_level, optional
        Verbosity level to set logging module.

    Returns
    -------
    coil_ims_pca : array_like
        Compressed coil images representing n_components principal
        components.
    expl_var : array_like, optional
        complex valued 1D vector representing explained variance.  Is
        returned if `give_explained_var=True`
    '''

    # Every day I'm logging...
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=debug_level)
    logging.info(
        'Starting coil_pca: initial size: %s', str(coil_ims.shape))

    # Get data in form (n_samples,n_features)
    coil_ims = np.moveaxis(coil_ims, coil_dim, -1)
    n_features = coil_ims.shape[-1]
    im_shape = coil_ims.shape[:-1]
    coil_ims = np.reshape(coil_ims, (-1, n_features))
    logging.info('Number of features: %d', n_features)

    # Do PCA on both real/imag parts
    if real_imag:
        logging.info('Performing PCA on real/imag parts...')
        pca_real = PCA(n_components=n_components)
        pca_imag = PCA(n_components=n_components)
        coil_ims_real = pca_real.fit_transform(coil_ims.real)
        coil_ims_imag = pca_imag.fit_transform(coil_ims.imag)

        coil_ims_pca = (coil_ims_real + 1j*coil_ims_imag).reshape(
            (*im_shape, n_components))
    else:
        # Do PCA on magnitude and phase
        logging.info('Performing PCA on mag/phase...')
        pca_mag = PCA(n_components=n_components)
        pca_phase = PCA(n_components=n_components)
        coil_ims_mag = pca_mag.fit_transform(np.abs(coil_ims))
        coil_ims_phase = pca_phase.fit_transform(np.angle(coil_ims))

        coil_ims_pca = (
            coil_ims_mag*np.exp(1j*coil_ims_phase)).reshape(
                (*im_shape, n_components))

    # Move coil dim back to where it was
    coil_ims_pca = np.moveaxis(coil_ims_pca, -1, coil_dim)

    logging.info('Resulting size: %s', str(coil_ims_pca.shape))
    logging.info('Number of components: %d', n_components)

    if give_explained_var:
        logging.info((
            'Returning explained_variance_ratio for both real and imag PCA'
            ' decompositions.'))
        logging.info((
            'Do mr_utils.view(expl_var.real) to see the plot for the real'
            'part.'))
        expl_var = (np.cumsum(pca_real.explained_variance_ratio_)
                    + 1j*np.cumsum(pca_imag.explained_variance_ratio_))
        return(coil_ims_pca, expl_var)

    # else...
    return coil_ims_pca

if __name__ == '__main__':
    pass
def PLOARKS_Cinv(data, kx, ky, shape):
    [ny, nx, nc] = shape
    [nt, ks] = data.shape
    data = data.reshape(nt, nc, ky , kx)
    mat = np.zeros([nc, ny, nx], dtype = complex)
    count = np.zeros([nc, ny, nx])
    idx = 0
    for y in range(max(1, ny - ky + 1)):
        for x in range(max(1, nx - kx + 1)):
            mat[:,y:y+ky, x:x+kx] += data[idx]
            count[:,y:y+ky, x:x+kx] += 1 
            idx += 1
    mat = mat/count
    return np.moveaxis(mat, 0, -1) 
    
def cg_sense(NufftObj, dataR, sensMap, gap, maxit = 10, tol = 1e-6, lambd = 0):
    nk = 5
    nl = 20
    [ny, nx, nc] = sensMap.shape
    [nTR, nRO, ndim] = [1000,200,2]
    traj = NufftObj.st["om"].reshape(1000, 200, ndim)
    B = multi_coil_NUFFT_adjoint(NufftObj,  dataR, sensMap)
    ground_truth = np.copy(dataR.reshape(nTR, nRO, nc))
    
    B = B.flatten()
    x = 0*B
    r = B 
    d = r 
    for ii in range(maxit):
        tmp = multi_coil_NUFFT_forward(NufftObj, d.reshape(ny, nx), sensMap)
        tmp = tmp.reshape(nTR, nRO, nc)
        for RO in reversed(range(gap)):
            for TR in tqdm(reversed(range(nTR))):
                cur_location = traj[TR, RO+1,:]
                index = get_nearest_points(traj[:,RO+1,:], cur_location, nk)
                calib_data = dataR.reshape(nTR, nRO, nc)[index, RO:RO+nl, :]
                rank = 7
                kx = 5
                ky = 2
                kc = np.copy(calib_data)
                kc[:,1:,:] = calib_data[:,1:,:]
                tmp[index,RO,:] = kc[:,0,:]
                
        tmp = tmp.reshape(nTR*nRO, nc)
        Ad = multi_coil_NUFFT_adjoint(NufftObj,  tmp, sensMap) +lambd *d.reshape(ny, nx)
        Ad = Ad.flatten()
        a = np.dot(r,r)/(np.dot(d,Ad))
        x = x + a*d
        plt.figure()
        plt.imshow(np.abs(x.reshape(ny, nx)), cmap = "gray")
        plt.show()
        rn = r - a*Ad
        beta = np.dot(rn,rn)/np.dot(r,r)
        r=rn
        d = r + beta*d
    return x.reshape([ny, nx])
def get_nearest_points(traj, cur_location, nk = 5):
    kdtree=KDTree(traj)
    dist,points =kdtree.query( cur_location, k = nk)
    # print(points.shape)
    # plt.figure()
    # plt.scatter( traj[:,0], traj [:,1], alpha = 0.3, s = 1)
    # plt.scatter(cur_location[0], cur_location[1], marker = "x", color = 'red', alpha = 0.8, s= 80)
    # plt.scatter(traj[points,0], traj[points,1], marker = "x", color = 'black', alpha = 0.3, s= 40)
    # # plt.ylim([-0.15, -0.1])
    # # plt.xlim([-0.15, 0.06])
    # plt.show()
    return points 
def multi_coil_NUFFT_forward(NufftObj,  images):
    [ny, nx, nz, nc] = images.shape
    trajectory = NufftObj.st["om"]
    nl = trajectory.shape[0]
    data = np.zeros([nl, nc], dtype = complex)
    for c in range(nc): 
        data[:,c] = NufftObj.forward(images[:,:,:,c])
    return data
def multi_coil_NUFFT_adjoint(NufftObj,  data):
    [ny, nx, nz] = NufftObj.st["Nd"]
    nc = data.shape[-1]
    trajectory = NufftObj.st["om"]
    nl = trajectory.shape[0]
    images = np.zeros([ny, nx, nz, nc], dtype = complex)
    for c in range(nc): 
        images[:,:,:,c] = NufftObj.solve(data[:,c], solver='cg', maxiter= 50)
    return images


def multi_coil_NUFFT_forward_sens(NufftObj,  images, sensMap):
    [ny, nx, nc] = sensMap.shape
    images = np.repeat(images[:, :,  np.newaxis], nc, axis=-1)
    images = images * sensMap 
    trajectory = NufftObj.st["om"]
    nl = trajectory.shape[0]
    data = np.zeros([nl, nc], dtype = complex)
    for c in range(nc): 
        data[:,c] = NufftObj.forward(images[:,:,c])
    return data

def multi_coil_NUFFT_adjoint_sens(NufftObj,  data, sensMap):
    [ny, nx, nc] = sensMap.shape
    trajectory = NufftObj.st["om"]
    nl = trajectory.shape[0]
    images = np.zeros([ny, nx, nc], dtype = complex)
    for c in range(nc): 
        images[:,:,c] = NufftObj.solve(data[:,c], solver='dc', maxiter= 50)
    image = np.sum(images * np.conj(sensMap), -1)
    return image

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
def l2(a, b):
    return np.linalg.norm(np.abs(a.flatten()) - np.abs(np.abs(b.flatten())))