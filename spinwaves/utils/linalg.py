import numpy as np

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


# Rotations
# All of them are right-handed
def rotate(n, angle: np.ndarray[float]):
    '''
    Return a matrix representing the rotation around vector `n` by `angle` radians.
    Length of the `n` vector does not matter.
    '''
    _, theta, phi = cartesian2spherical(n)

    return np.matmul(Rz(phi), np.matmul(Ry(theta), np.matmul(Rz(angle), np.matmul(Ry(-theta), Rz(-phi) ))))
    
def Rx(alpha: np.ndarray[float]) -> np.ndarray:
    '''Matrix of right-handed rotation around x-axis [1,0,0] by angle alpha in radians.
    >>> Ry = [ [          1,          0,           0],
               [          0, cos(alpha), -sin(alpha)]
               [          0, sin(alpha),  cos(alpha)]  ]
               
    Parameters
    ----------
    alpha: array_like
        Array of rotation angles.
    '''
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    R = np.zeros(np.shape(alpha)+(3,3))
    R[...,0,0] =   1
    R[...,1,1] =  ca
    R[...,1,2] = -sa
    R[...,2,1] =  sa
    R[...,2,2] =  ca
    return R
    
def Ry(alpha: np.ndarray[float]) -> np.ndarray:
    '''Matrix of right-handed rotation around y-axis [0,1,0] by angle alpha in radians.
    >>> Ry = [ [ cos(alpha), 0, sin(alpha)],
               [          0, 1,          0]
               [-sin(alpha), 1, cos(alpha)]  ]
               
    Parameters
    ----------
    alpha: array_like
        Array of rotation angles.
    '''
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    R = np.zeros(np.shape(alpha)+(3,3))
    R[...,0,0] =  ca
    R[...,0,2] =  sa
    R[...,1,1] =   1
    R[...,2,0] = -sa
    R[...,2,2] =  ca
    return R

def Rz(alpha: np.ndarray[float]) -> np.ndarray:
    '''Matrix of right-handed rotation around z-axis [0,0,1] by angle alpha in radians.
    >>> Rz = [ [cos(alpha), -sin(alpha), 0],
               [sin(alpha),  cos(alpha), 0],
               [         0,           0, 1]  ]

    Parameters
    ----------
    alpha: array_like
        Array of rotation angles.
    '''
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    R = np.zeros(np.shape(alpha)+(3,3))
    R[...,0,0] =  ca
    R[...,0,1] = -sa
    R[...,1,0] =  sa
    R[...,1,1] =  ca
    R[...,2,2] =   1
    return R

def rot_Rn(n_uvw: tuple[int,int,int], 
           modulation_vector: np.ndarray[float], 
           global_rotation_axis: tuple[float,float,float]) -> np.ndarray:
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
    Rn = rotate(global_rotation_axis, -phi)
    return Rn

def RtoZ(v: tuple[float,float,float]) -> np.ndarray:
    '''Rotation matrix that rotates directly vector `v` to be along the `z` axis,
    by rotating around the normal to `vz` plane.
    '''
    _, theta, phi = cartesian2spherical(v)
    return Rz(phi) @ Ry(-theta) @ Rz(-phi) 

def RfromZ(v: tuple[float,float,float]) -> np.ndarray:
    '''Rotation matrix that rotates directly from the `z` axis to the vector `v`.'''
    _, theta, phi = cartesian2spherical(v)
    return Rz(phi) @ Ry(theta) @ Rz(-phi) 


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
    Return the matrix representing projection on the plane perpendicular to the given vector q.
    Also known as perpendicular projection operator P_ij = 1 - q_i*q_j/|q|
    '''
    
    # For the sake of speed the matrix is given explicitly based on calculations on paper
    _, theta, phi = cartesian2spherical(q)
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    
    return np.array([   [1-st*st*cp*cp,   -st*st*sp*cp,    -ct*st*cp],
                        [ -st*st*sp*cp,  1-st*st*sp*sp,    -st*ct*sp],
                        [ -ct*st*cp,      -st*ct*sp,      1-ct*ct]])
    
def perp_part(m,q):
    '''
    Return the part of vector m that is perpendicular to the vector q
    '''
    # eq = np.array(q)/norm(q)
    # return np.cross(np.cross(eq,m), eq)
    
    return np.dot(perp_matrix(q), m)