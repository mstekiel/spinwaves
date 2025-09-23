import numpy as np
from numpy.typing import NDArray
from typing import Sequence, Tuple



# Fitting functions and other
def gauss_bkg(x,x0,A,sigma,bkg):
    '''
    Gaussian with constant background.
    
    :math:`f(x) = A exp(-(x-x_0)^2/(2 \\sigma^2)) + bkg`
    
    To convert to intensity :math:`I = \\sqrt{2 \\pi} A \\sigma`
    
    To convert to FWHM :math:`FWHM = 2\\sqrt{2 \\ln 2} \\sigma \\approx 2.355 \\sigma`
    '''
    return A*np.exp(-(x-x0)**2/(2*sigma**2)) + bkg
    
def lorentz_bkg(x,x0,A,gamma,bkg):
    '''
    Lorentzian with constant background.
    
    :math:`f(x) = \\frac{A}{(1+(x-x_0)^2/\\gamma^2))} + bkg`
    
    To convert to intensity of the peak :math:`I = \\pi A \\gamma`
    '''
    return A/(1+np.power((x-x0)/gamma,2)) + bkg
    
def pseudoVoigt_bkg(x,x0,I,f,eta,bkg):
    '''
    Pseudo-Voigt function.
    '''
    
    return eta*I*gauss_bkg(x,x0,1/(np.sqrt(2*np.pi)*f),f,0) + (1-eta)*I*lorentz_bkg(x,x0,1/(np.pi*f),f,0) + bkg

def gauss_satellites_bkg(x,x0,xs,As,sigmas,bkg):
    '''
    Gaussian satellites
    
    :math:`f(x) = A ( exp(-(x-x_0-x_s)^2/(2 \\sigma^2)) + exp(-(x-x_0+x_s)^2/(2 \\sigma^2)) ) + bkg`
    
    To convert to intensity of the peak :math:`I = \\sqrt{2 \\pi} A \\sigma`
    '''
    return As*np.exp(-(x-x0-xs)**2/(2*sigmas**2)) + As*np.exp(-(x-x0+xs)**2/(2*sigmas**2)) + bkg

