'''Linear algebra utilities of the `spinwaves` package.

Checks against numerical noise use `spinwaves.config.tolerances.numeric`,
read at call time; see `spinwaves.config`.
'''
import logging

import numpy as np

from .. import config
from ..config import NumericalToleranceError

logger = logging.getLogger('Numerics')


### Matrix helpers
def dagger(A: np.ndarray) -> np.ndarray:
    '''Hermitian conjugate over the last two axes of `A` with shape (...,N,N).'''
    return A.conj().swapaxes(-2,-1)

def offdiagonal(T: np.ndarray) -> np.ndarray:
    '''Copy of `T` with shape (...,N,N) with the diagonal over the last two axes set to zero.'''
    return T - np.diagonal(T, axis1=-2, axis2=-1)[...,None] * np.eye(T.shape[-1])

def cross_matrix(n) -> np.ndarray:
    '''[n]_x such that [n]_x @ v = n x v'''
    nx, ny, nz = n
    return np.array([[0, -nz, ny], [nz, 0, -nx], [-ny, nx, 0]], dtype=float)


### Checks against the numerical tolerance
def _relative_norm(D: np.ndarray, A: np.ndarray) -> np.ndarray:
    '''Per-matrix ||D||_F / ||A||_F over the last two axes, 0 where A is zero.'''
    norm_D = np.linalg.norm(D, axis=(-2,-1))
    norm_A = np.linalg.norm(A, axis=(-2,-1))
    return np.divide(norm_D, norm_A, out=np.zeros_like(norm_D), where=norm_A > 0)

def _batch_indices(mask: np.ndarray) -> list[tuple]:
    return [tuple(int(i) for i in idx) for idx in np.argwhere(mask)]

def _describe_indices(indices: list[tuple], max_shown: int=10) -> str:
    if indices == [()]:
        return ''
    shown = ', '.join(str(idx) for idx in indices[:max_shown])
    more = f' and {len(indices)-max_shown} more' if len(indices) > max_shown else ''
    return f' at batch indices {shown}{more}'

def ensure_hermitian(A: np.ndarray, rtol: float=None, name: str='matrix') -> np.ndarray:
    '''Hermitian part of `A`, after checking that `A` is Hermitian up to numerical noise.

    Parameters
    ----------
    A: (...,N,N) ndarray
        Matrix, or batch of matrices, Hermitian by construction.
    rtol: float, optional
        Tolerance of ||A - A^dagger||_F / ||A||_F, per matrix.
        Default `config.tolerances.numeric`.
    name: str, optional
        Name of the matrix used in messages.

    Returns
    -------
    (A + A^dagger)/2: (...,N,N) ndarray

    Raises
    ------
    NumericalToleranceError
        If any matrix deviates from hermiticity by more than `rtol`.
    '''
    rtol = config.tolerances.numeric if rtol is None else rtol
    A = np.asarray(A)
    deviation = _relative_norm(A - dagger(A), A)

    failing = deviation > rtol
    if np.any(failing):
        indices = _batch_indices(failing)
        raise NumericalToleranceError(
            f'`{name}` is not Hermitian: relative deviation {deviation.max():.1e} '
            f'> tolerance {rtol:.1e}{_describe_indices(indices)}',
            deviation=float(deviation.max()), tolerance=rtol, indices=indices)

    if np.any(deviation > 0):
        logger.debug(f'Symmetrized `{name}`, largest relative non-hermiticity {deviation.max():.1e}')

    return (A + dagger(A)) / 2

def regularized_cholesky(h: np.ndarray, rtol: float=None, name: str='h') -> np.ndarray:
    '''Lower triangular Cholesky factor K of positive (semi)definite `h` = K @ K^dagger.

    Eigenvalues `l` of `h` are compared to the scale `max|l|` of each matrix:

    - `min(l) < -rtol*scale`: `h` is not positive semidefinite, raise.
    - `|min(l)| <= rtol*scale`: zero modes, e.g. Goldstone modes, are regularized
      by adding `rtol*scale` to the diagonal. For spin waves this opens a gap at
      the zero modes of `rtol*scale` for quadratic (ferromagnetic) dispersion,
      up to about `sqrt(rtol)*scale` for linear (antiferromagnetic) dispersion.

    Parameters
    ----------
    h: (...,N,N) ndarray
        Hermitian matrix, or batch of matrices.
    rtol: float, optional
        Relative tolerance, default `config.tolerances.numeric`.
    name: str, optional
        Name of the matrix used in messages.

    Returns
    -------
    K: (...,N,N) ndarray
        Lower triangular Cholesky factor of the (regularized) `h`.

    Raises
    ------
    NumericalToleranceError
        If `h` is not Hermitian or not positive semidefinite within `rtol`.
    '''
    rtol = config.tolerances.numeric if rtol is None else rtol
    h = ensure_hermitian(h, rtol=rtol, name=name)

    eigenvalues = np.linalg.eigvalsh(h)
    lowest = eigenvalues[...,0]
    scale = np.max(np.abs(eigenvalues), axis=-1)
    scale = np.where(scale > 0, scale, 1.0)     # zero matrix: absolute shift by `rtol`

    negative = lowest < -rtol*scale
    if np.any(negative):
        indices = _batch_indices(negative)
        deviation = float(np.max(-lowest/scale))
        raise NumericalToleranceError(
            f'`{name}` is not positive definite: smallest eigenvalue relative to the largest '
            f'is {-deviation:.1e} < -{rtol:.1e}{_describe_indices(indices)}',
            deviation=deviation, tolerance=rtol, indices=indices)

    zero_modes = lowest <= rtol*scale
    if np.any(zero_modes):
        logger.debug(f'Regularized zero modes of `{name}` in {np.count_nonzero(zero_modes)} matrices')
        h = h + (rtol*scale*zero_modes)[...,None,None] * np.eye(h.shape[-1])

    try:
        return np.linalg.cholesky(h)
    except np.linalg.LinAlgError as e:
        raise NumericalToleranceError(f'Cholesky decomposition of `{name}` failed after regularization',
                                      deviation=np.nan, tolerance=rtol) from e

def hermitian_eig(A: np.ndarray, rtol: float=None, name: str='matrix') -> tuple[np.ndarray, np.ndarray]:
    '''Eigen-decomposition A = U @ diag(E) @ U^dagger of a matrix Hermitian by construction.

    `A` is checked and symmetrized with `ensure_hermitian`, then decomposed
    with `numpy.linalg.eigh`, which returns unitary `U` also for degenerate
    eigenvalues.

    Parameters
    ----------
    A: (...,N,N) ndarray
        Matrix, or batch of matrices, Hermitian by construction.
    rtol: float, optional
        Tolerance of non-hermiticity, default `config.tolerances.numeric`.
    name: str, optional
        Name of the matrix used in messages.

    Returns
    -------
    E: (...,N) ndarray
        Real eigenvalues in ascending order.
    U: (...,N,N) ndarray
        Unitary matrix with eigenvectors in columns.
    '''
    return np.linalg.eigh(ensure_hermitian(A, rtol=rtol, name=name))

### Exchange interaction matrices
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

def dipolar(r: np.ndarray[float]) -> np.ndarray:
    '''Interaction matrix for the dipolar interaction between two magnetic moments.
    [ref] https://en.wikipedia.org/wiki/Magnetic_dipole-dipole_interaction

    Magnetic dipolar interaction:
    H_dip = -mu_0*g_i*g_j*hbar^2/(4pi) [ 3(S_i . r) * (S_j . r) - S_i.S_j * |r|^2 ] / |r|^5

    Can be rewritten as:
    H_dip = S_i . Jdip_ij . S_j

    In case ions are the same their spin numbers S, and gyromagnetic factors g are the same
    and can be factored out
    
    Jdip_ij = [3 r_i r_j - delta_ij |r|^2 ] / |r|^5

    Returns
    -------
    Jdip_ij : ndarray
        3x3 interaction matrix for dipolar interaction between two magnetic moments
    '''
    Jdd =  3 * r[:, None] * r[None, :] - np.eye(3)*np.linalg.norm(r)**2
    return Jdd/ np.linalg.norm(r)**5


#### Rotations
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
    xyz = np.array(xyz)
    xy = xyz[...,0]**2 + xyz[...,1]**2
    r = np.linalg.norm(xyz, axis=-1)
    theta = np.arctan2(np.sqrt(xy), xyz[...,2])
    phi = np.arctan2(xyz[...,1], xyz[...,0])
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
    # Trick to deal with numerics is using arctan2
    # Use arctan2(|v1 x v2|, v1.v2)
    # it is effectively arctan( v1*v2*sin(alpha) , v2*v2*cos(alpha) ) = alpha

    return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))


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