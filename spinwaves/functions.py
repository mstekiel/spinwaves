import numpy as np
from typing import Sequence, Tuple

# Exchange interaction matrices
def DMI(v: tuple[float,float,float]) -> np.ndarray:
    '''
    Antisymmetric exchange interaction = Dzialoshinskii-Moriya interaction matrix.

    Defined by the DM vector D = [dx, dy, dz] as:
    D = [[  0,  dz,-dy],
         [-dz,   0, dx],
         [ dy, -dx,  0] ]
    '''
    dx, dy, dz = v

    return np.array([ [  0,  dz,-dy], [-dz,   0, dx], [dy, -dx,  0] ], dtype=float)

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

# Rotations
# All of them are right-handed
def rotate(n, angle):
    '''
    Return a matrix representing the rotation around vector `n` by `angle` radians.
    Length of the `n` vector does not matter.
    '''
    _, theta, phi = cartesian2spherical(n)

    return np.matmul(Rz(phi), np.matmul(Ry(theta), np.matmul(Rz(angle), np.matmul(Ry(-theta), Rz(-phi) ))))
    
def Rx(alpha: float) -> np.ndarray:
    '''Matrix of right-handed rotation around x-axis [1,0,0] by angle alpha in radians.'''
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
    
def Ry(alpha: float) -> np.ndarray:
    '''Matrix of right-handed rotation around y-axis [0,1,0] by angle alpha in radians.'''
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])

def Rz(alpha: float) -> np.ndarray:
    '''Matrix of right-handed rotation around z-axis [0,0,1] by angle alpha in radians.'''
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])

def rot_Rn(n_uvw: Tuple[int,int,int], 
           modulation_vector: Tuple[float,float,float], 
           global_rotation_axis: Tuple[float,float,float]) -> np.ndarray:
    '''Rotation matrix corresponding to modulation of magnetic moments in a crystal.

    Rn rotates the magnetic moment in the unit cell `n_uvw`, according to the phase
    of the `modulation_vector` around the `global_rotation_axis`.

    Symbolically:

    S_nj = R_n S_0j

    S_nj is magnetic moment of j-th atom in the n-th unit cell, where the n-th unit cell is indexed by triple-int `n_uvw`.

    Notes
    -----
    As for eq. (6) of [SpinW]
    '''
    phi = 2*np.pi*np.dot(modulation_vector, n_uvw)
    Rn = rotate(global_rotation_axis, phi)
    return Rn

def rot_Rprime(v: Tuple[float,float,float]) -> np.ndarray:
    '''Rotation matrix that rotates vector `v` to be along the `z` axis

    Notes
    -----
    As for eq. (7) [SpinW]
    Rn' is the rotation that puts the magnetic moment along z axis.
        S'_nj = R'_n S''_nj
        S'_nj=S_0j : magnetic moment of j-th atom in the 0-th unit cell, independent on unit cell
        S''_nj : spin oriented along the ferromagnetic axis
    '''
    _, theta, phi = cartesian2spherical(v)
    return Rz(phi) @ Ry(-theta) @ Rz(-phi) 

def rot_Rodrigues_complex(n: tuple[float,float,float]):
    """Return Rodrigues matrices of rotation R1, R2 according to complex formulation:

    R(phi) = e^(i*phi)*R1 + e^(-i*phi)*R1.conj + R2

    as in (39) [SpinW]
    """
    R2 = np.outer(n, n)
    R1 = (np.eye(3,3) - 1j*DMI(n).T - R2) / 2.0

    return R1, R2

# Vectors
def cartesian2spherical(xyz) -> list[float, float, float]:
    '''
    Return the spherical [r, theta, phi] coordinates of the cartesian vector [x,y,z]

    Conventions
    -----------
    r > 0

    theta in (0 : pi)

    phi in (-pi : pi)
    '''
    xy = xyz[0]**2 + xyz[1]**2
    r = norm(xyz)
    theta = np.arctan2(np.sqrt(xy), xyz[2])
    phi = np.arctan2(xyz[1], xyz[0])
    return [r,theta,phi]
    
def norm(x, **kwargs):
    '''
    Take the Euclidean norm of the n-dimensional vector x.
    Wrapper for the `np.linalg.norm` function.
    '''
    return np.linalg.norm(x, **kwargs)
    
    
def angle(v1: list[float], v2: list[float]) -> float:
    '''
    Return the angle in radians between two vectors
    '''
    # Clip is required to deal with floating points.
    return np.arccos(np.clip( np.dot(v1,v2)/norm(v1)/norm(v2), -1, 1))


def perp_matrix(q: list[float]):
    '''
    Return the matrix representing projection on the plane perpendicular to the given vector q
    '''
    
    # For the sake of speed the matrix is given explicitly based on calculations on paper
    _, theta, phi = cartesian2spherical(q)
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    
    return np.array([   [1-st*st*cp*cp, -st*st*sp*cp,   -ct*st*cp],
                        [-st*st*sp*cp,  1-st*st*sp*sp,  -ct*st*sp],
                        [-st*ct*sp,     -st*ct*sp,      1-ct*ct]])
    
def perp_part(m,q):
    '''
    Return the part of vector m that is perpendicular to the vector q
    '''
    # eq = np.array(q)/norm(q)
    # return np.cross(np.cross(eq,m), eq)
    
    return np.dot(perp_matrix(q), m)