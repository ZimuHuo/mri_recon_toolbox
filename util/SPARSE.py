
import pywt
def wavelet_denoising(x, wavelet='db6', level=4):
    '''
    -------------------------------------------------------------------------
    Parameters
    
    x: array_like
    undersampled image
    
    -------------------------------------------------------------------------
    Returns
    image : array like
    reconstructed image
    
    -------------------------------------------------------------------------
    Notes
    python is just too good 
    
    -------------------------------------------------------------------------
    References
    
    [1] 
    Author: Michael Lustig et al. 
    Title: Sparse MRI: The Application of Compressed Sensing for Rapid MR Imaging
    Link: https://pubmed.ncbi.nlm.nih.gov/17969013/
    '''
    coeff = pywt.wavedec(x, wavelet, mode="per")
    coeff[1:] = (pywt.threshold(i, value=0.05, mode='soft') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')