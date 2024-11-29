import numpy as np

from typing import Union, List, Tuple

from . import functions as funs_sw

import warnings
# warnings.filterwarnings("error")

# -*- coding: utf-8 -*-
r"""Handles lattice geometries to find rotations and transformations

"""
from functools import cached_property
import numpy as np


class Lattice2(object):
    """Class to describe a generic lattice system defined by lattice six
    parameters, (three constants and three angles).

    Parameters
    ----------
    a : float
        Unit cell length in angstroms

    b : float
        Unit cell length in angstroms

    c : float
        Unit cell length in angstroms

    alpha : float
        Angle between b and c in degrees

    beta : float
        Angle between a and c in degrees

    gamma : float
        Angle between a and b in degrees

    Attributes
    ----------
    a
    b
    c
    alpha
    beta
    gamma
    abc
    abg
    abg_rad
    lattice_type
    Amatrix
    Bmatrix
    G
    Gstar
    volume
    reciprocal_volume

    Conventions
    -----------
    1. Lattice vectors are positioned in the Cartesian coordinates as:
        - a || x
        - b* || y
        - c to complete RHS
    2. The phase factors are (i) exp(i k_xyz r_xyz) and (ii) exp(i Q_hkl r_uvw).
    3. All matrices, so for both real and reciprocal space, are represented for column vectors.
    4. `B` matrix contains the 2pi factor. Consequently A.T @ B = 2pi eye(3,3). Transposition due to pt 3.
    5. Implemented methods should work with arrays assuming the last index representing the `h, k, l` coordinates.

        
    Transformation notes
    --------------------
    A: ndarray((3,3))

        Transforms a real lattice point into an orthonormal coordinates system. Upper triangle matrix.
        [u,v,w] -> [x,y,z] (Angstroems)


    B: ndarray((3,3))

        Transforms a reciprocal lattice point into an orthonormal coordinates system.
        (h,k,l) -> [kx,ky,kz] (1/Angstroem)

    """

    def __init__(self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float):
        self._lattice_parameters = [a,b,c, alpha,beta,gamma]
        self._update_lattice()

    #################################################################################################
    # Core methods

    def _update_lattice(self):
        '''Master function recalculating all matrices involving the lattice parameters.'''
        a,b,c, alpha,beta,gamma = self._lattice_parameters
        # A matrix follows convention 1.
        self._Amatrix = self._constructA(a,b,c, alpha,beta,gamma)

        # B matrix based on the perpendicularity condition to A
        # To get column representation it needs to be transposed
        self._Bmatrix = 2*np.pi* np.linalg.inv(self.Amatrix).T

        # Metric tensor of real space
        self._G = self.Amatrix.T @ self.Amatrix

        # Metric tensor of reciprocal space
        self._Gstar = self.Bmatrix.T @ self.Bmatrix

    def _constructA(self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
        '''
        Construct the `A` matrix as crystal axes in orthonormal system, ie a||x, b in xy plane, c accordingly.

        Transforms a real lattice point into an orthonormal coordinates system. Upper triangle matrix.
        A * [u,v,w] -> [x,y,z] (Angstroems)

        Shortcut to define lattice vectors:
        >>> a, b, c = A.T
        '''
        
        alpha, beta, gamma = self.abg_rad
        bx = b*np.cos(gamma)
        by = b*np.sin(gamma)
        
        cx = c*np.cos(beta)
        cy = c*(np.cos(alpha)-np.cos(gamma)*np.cos(beta))/np.sin(gamma)
        cz  = np.sqrt(c*c-cx*cx-cy*cy)
        
        return np.array([[a,bx,cx],[0,by,cy],[0,0,cz]])

    def __repr__(self):
        return "<Lattice {0}, {1}, {2}, {3}, {4}, {5}>".format(self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

    #################################################################################################
    # Defining properties
    @property
    def Amatrix(self):
        '''Crystal axes in orthonormal system, ie a||x, b in xy plane, c accordingly.
        Upper triangle matrix.

        Transforms a real lattice point into an orthonormal coordinates system.
        A * [u,v,w] -> [x,y,z] (Angstroems)


        Shortcut to define lattice vectors:
        >>> a, b, c = A.T
        '''
        return self._Amatrix

    @property
    def Bmatrix(self):
        '''Reciprocal crystal axes in orthonormal system, perpendicular to real axes.
        By definition `b* || y`.

        Transforms a reciprocal lattice point into an orthonormal coordinates system.
        B*(h,k,l) -> [kx,ky,kz]_{crystal} (1/Angstroem)

        Shortcut to define reciprocal lattice vectors:
        >>> astar, bstar, cstar = B.T
        '''
        return self._Bmatrix

    @property
    def G(self):
        '''Metric tensor of the real lattice.
        G = A @ A.T
        '''
        return self._G


    @property
    def Gstar(self):
        '''Metric tensor of the reciprocal lattice.
        Gstar = B @ B.T

        Allows to calculate products of vector in hkl base.
        '''
        return self._Gstar

    #################################################################################################
    # Properties
    @property
    def lattice_parameters(self) -> list[float]:
        '''Lattice parameters `[a,b,c, alpha,beta,gamma]`, in Angstroms and degrees.'''
        return self._lattice_parameters

    @property
    def a(self) -> float:
        """First lattice parameter `a` in Angstrom."""
        return self._lattice_parameters[0]

    @a.setter
    def a(self, new_a: float):
        self._lattice_parameters[0] = new_a
        self._update_lattice()

    @property
    def b(self) -> float:
        """Second lattice parameter `b` in Angstrom."""
        return self._lattice_parameters[1]

    @b.setter
    def b(self, new_b: float):
        self._lattice_parameters[1] = new_b
        self._update_lattice()

    @property
    def c(self) -> float:
        """Third lattice parameter `c` in Angstrom."""
        return self._lattice_parameters[2]
    
    @c.setter
    def c(self, new_c: float):
        self._lattice_parameters[2] = new_c
        self._update_lattice()


    @property
    def alpha(self) -> float:
        """First lattice angle `alpha` in degrees."""
        return self._lattice_parameters[3]

    @alpha.setter
    def alpha(self, new_alpha):
        self._lattice_parameters[3] = new_alpha
        self._update_lattice()

    @property
    def beta(self) -> float:
        """Second lattice angle `beta` in degrees."""
        return self._lattice_parameters[4]

    @beta.setter
    def beta(self, new_beta: float):
        self._lattice_parameters[4] = new_beta
        self._update_lattice()

    @property
    def gamma(self) -> float:
        """Third lattice angle `gamma` in degrees."""
        return self._lattice_parameters[5]

    @gamma.setter
    def gamma(self, new_gamma: float):
        self._lattice_parameters[5] = new_gamma
        self._update_lattice()

    # Properties without setters

    @property
    def abc(self):
        """Lattice parameters in Angstroem"""
        return self._lattice_parameters[:3]
    
    @property
    def abg(self):
        """Lattice angles in degrees."""
        return self._lattice_parameters[3:]
    
    @property
    def abg_rad(self):
        """Lattice angles in radians."""
        return np.radians(self._lattice_parameters[3:])

    @property
    def lattice_type(self):
        """Type of lattice determined by the provided lattice constants and angles"""

        if len(np.unique(self.abc)) == 3 and len(np.unique(self.abg)) == 3:
            return 'triclinic'
        elif len(np.unique(self.abc)) == 3 and self.abg[1] != 90 and np.all(np.array(self.abg)[:3:2] == 90):
            return 'monoclinic'
        elif len(np.unique(self.abc)) == 3 and np.all(np.array(self.abg) == 90):
            return 'orthorhombic'
        elif len(np.unique(self.abc)) == 1 and len(np.unique(self.abg)) == 1 and np.all(
                        np.array(self.abg) < 120) and np.all(np.array(self.abg) != 90):
            return 'rhombohedral'
        elif len(np.unique(self.abc)) == 2 and self.abc[0] == self.abc[1] and np.all(np.array(self.abg) == 90):
            return 'tetragonal'
        elif len(np.unique(self.abc)) == 2 and self.abc[0] == self.abc[1] and np.all(np.array(self.abg)[0:2] == 90) and \
                        self.abg[2] == 120:
            return 'hexagonal'
        elif len(np.unique(self.abc)) == 1 and np.all(np.array(self.abg) == 90):
            return 'cubic'
        else:
            raise ValueError('Provided lattice constants and angles do not resolve to a valid Bravais lattice')

    @property
    def volume(self) -> float:
        """Volume of the unit cell in [A^3]."""
        return np.sqrt(np.linalg.det(self.G))

    @property
    def reciprocal_volume(self) -> float:
        """Volume of the reciprocal unit cell in [1/A^3]. What about the pi factor?"""
        return np.sqrt(np.linalg.det(self.Gstar))

    #################################################################################################
    # Functionalities
    def get_scalar_product(self, hkl1: np.ndarray, hkl2: np.ndarray):
        """Returns the scalar product between two lists of vectors.
        
        Parameters
        ----------
        hkl1 : array_like (3) or (...,3)
            Vector or array of vectors in reciprocal space.
        hkl2 : array_like (...,3)
            List of vectors in reciprocal space.

        Returns
        -------
        ret : array_like (...)
            List of calculated scalar products between vectors.

        Notes
        -----
        Takes advantage of the `Gstar=B.T @ B` matrix. Simply does:
        `Q_hkl1 @ Gstar @ Q_hkl2 == (B @ Q_hkl1).T @ (B @ Q_hkl2) == Q_xyz1 @ Q_xyz2`.
        Where the last one is in the orthonormal coordinate frame and can be 
        directly computed.
        """
        v1v2_cosine = np.einsum('...i,ij,...j->...', hkl1, self.Gstar, hkl2)

        return v1v2_cosine
    
    def get_Q(self, hkl: np.ndarray) -> np.ndarray:
        '''Returns the magnitude |Q| [1/A] of reciprocal lattice vectors `hkl`.
                
        Parameters
        ----------
        hkl : array_like (3,...)
            Reciprocal lattice vector in r.l.u. Signature: `h,k,l = hkl`

        Returns
        -------
        Q : array_like (,...)
            The magnitude of the reciprocal lattice vectors in [1/A].
            Shape follows the input signature with reduced first dimension.


        Notes
        -----
        Calculates the Q vector from the inverse metric tensor: `Q = sqrt(hkl.T @ Gstar @ hkl)`.
        Alternative method of calculating from B matrix proved to be slower: `Q = norm(B @ hkl)`
        '''
        # return np.sqrt(np.einsum('i...,ij,j...->...', hkl, self.Gstar, hkl))
        return np.sqrt(self.get_scalar_product(hkl, hkl))
    
    def get_dspacing(self,hkl: np.ndarray) -> np.ndarray:
        u"""Returns the d-spacing of a given reciprocal lattice vector.

        Parameters
        ----------
        hkl : array_like (3,...)
            Reciprocal lattice vector in r.l.u. Signature: `h,k,l = hkl`

        Returns
        -------
        d : float (,...)
            The d-spacing in A.
        """
        # DEV NOTES
        # Method with metric tensor proves to be the fastest.
        # Alternative tested was determining Q from `norm(B @ hkl)`

        return 2*np.pi / self.get_Q(hkl)

    def get_tth(self, hkl: np.ndarray, wavelength: float) -> np.ndarray:
        u"""Returns the detector angle two-theta [rad] for a given reciprocal
        lattice vector [rlu] and incident wavelength [A].

        Parameters
        ----------
        hkl : array_like (3,...)
            Reciprocal lattice vector in r.l.u. Signature: `h,k,l = hkl`

        wavelength : float
            Wavelength of the incident beam in Angstroem.

        Returns
        -------
        two_theta : array_like (,...)
            The scattering angle two-theta i nradians.
            Shape follows the input signature with reduced first dimension.

        """

        return 2*np.arcsin( wavelength*self.get_Q(hkl)/4/np.pi )
    
    
    def get_angle_between_planes(self, v1, v2) -> float:
        r"""Returns the angle :math:`\phi` [rad] between two reciprocal lattice
        vectors (or planes as defined by the vectors normal to the plane).

        Parameters
        ----------
        v1 : array_like (3)
            First reciprocal lattice vector in units r.l.u. 

        v2 : array_like (3,...)
            Second reciprocal lattice vector in units r.l.u.

        Returns
        -------
        phi : float (...)
            The angle between v1 and v2 in radians.

        Notes
        -----
        Uses the `Gstar` matrix again and the fact that `Gstar=B B.T` such that
        `v1.Gstar.v2` is the cosine between v1-v2.
        Due to rounding errors the cosine(v1,v2) is clipped to [-1,1].
        """
        v1v2_cosine = np.einsum('i,ij,...j->...', v1, self.Gstar, v2)
        v1 = self.get_Q(v1)
        v2 = self.get_Q(v2)

        return np.arccos( np.clip(v1v2_cosine / (v1*v2), -1, 1) )
    
class Lattice:
    '''
    Object representing the lattice of a crystal. Facilitates transformations between Cartesian and lattice
    coordinates as well as real and reciprocal space.
    
    Conventions:
        1. Lattice vectors are positioned in the Cartesian coordinates as:
            - a || x
            - b* || y
            - c to complete RHS
        2. The phase factors are (i) exp(i k r_xyz) and (ii) exp(2 pi i Q r_uvw).
        3. All matrices, so for both real and reciprocal space, are represented for column vectors.
        4. `B` matrix contains the 2pi factor. Consequently A @ B.T = 2pi eye(3,3). Transposition due to pt 3.
    
    The main idea is that the orientation can be changed, but the lattice type and parameters no.
    
    Attirbutes:
        lattice_parameters: ndarray((6))
            List containing `a,b,c,alpha, beta, gamma` lattice parameters. Lengths in angstroems, angles in degrees.
        A : ndarray((3,3))
            Transforms a real lattice point into an orthonormal coordinates system. Upper triangle matrix.
            [u,v,w] -> [x,y,z] (Angstroems)
        B : ndarray((3,3))
            Transforms a reciprocal lattice point into an orthonormal coordinates system. Upper triangle matrix.
            (h,k,l) -> [kx,ky,kz] (1/Angstroem)
        U : ndarray((3,3))
            Orientation matrix that relates the orthonormal, reciprocal lattice coordinate system into the diffractometer/lab coordinates
            [kx,ky,kz]_{crystal} -> [kx,ky,kz]_{lab}
        UA : ndarray((3,3))
            Transforms a real lattice point into lab coordinate system.
            (u,v,w) -> [x,y,z]_{lab} (Angstroem)
        UB : ndarray((3,3))
            Transforms a reciprocal lattice point into lab coordinate system.
            (h,k,l) -> [kx,ky,kz]_{lab} (1/Angstroem)
    '''

    def __init__(self, lattice_parameters: list[float], orientation: Union[None, tuple, np.ndarray]=None):
        '''
        Object representing crystal lattice and coordinate system in cartesian space.

        Parameters:
        -----------

        lattice parameters: [a,b,c,alpha,beta,gamma]
        
        orientation : None
            Identity matrix
        orientation : hkl_tuple
            The chosen hkl is put perpendicular to the scattering plane, i.e. along the `z` axis.
        orientation : (hkl1_tuple, hkl2_tuple)
            hkl1 is put along the `x` axis and hkl2 in the `xy` plane.
        orientation : ndarray(3,3)
            U is given directly as an argument.
        '''
        self.lattice_parameters = lattice_parameters
        #self.G = self.metricTensor(a,b,c,alpha,beta,gamma)
        
        # Transforms real lattice points into orthonormal coordinate system.
        self.A = self.constructA(lattice_parameters)
        
        # Transforms reciprocal lattice points into orthonormal coordinate system.
        self.B = self.constructB(lattice_parameters)
        
        # Initialize the orientation, U and the UB matrix within the wrapper.
        self.updateOrientation(orientation)

                
        
    # def __str__(self):
    #     return str(self.lattice_parameters)
    
    def __repr__(self):
        rr = f'Lattice('
        rr += ', '.join(f'{name}={value}' for name,value in zip(['a','b','c','alpha','beta','gamma'], 
                                                                self.lattice_parameters))
        rr += f', orientation={self._current_orientation})'

        return rr
    
    def constructA(self, lattice_parameters: list[float]) -> np.ndarray:
        '''
        Construct the `A` matrix as crystal axes in orthonormal system, ie a||x, b in xy plane, c accordingly.

        Transforms a real lattice point into an orthonormal coordinates system. Upper triangle matrix.
        A * [u,v,w] -> [x,y,z] (Angstroems)

        Shortcut to define lattice vectors:
        >>> a, b, c = A.T
        '''
        a,b,c,alpha,beta,gamma = lattice_parameters
        
        bx = b*np.cos(np.radians(gamma))
        by = b*np.sin(np.radians(gamma))
        
        cx = c*np.cos(np.radians(beta))
        cy = c*(np.cos(np.radians(alpha))-np.cos(np.radians(gamma))*np.cos(np.radians(beta)))/np.sin(np.radians(gamma))
        cz  = np.sqrt(c*c-cx*cx-cy*cy)
        
        return np.array([[a,bx,cx],[0,by,cy],[0,0,cz]])
    
    def constructB(self, lattice_parameters: list[float]) -> np.ndarray:
        '''
        Construct the `B` matrix as reciprocal lattice base vectors in orthonormal system.

        Construction is based on the perpendicularity to the `A` matix.
        '''
        # Construction based on the perpendicularity.
        A = self.constructA(lattice_parameters)

        # Include pt 3 and 4 from conventions.
        # 3. Transpose to get column vector representation.
        # 4. Include the 2pi factor.
        B = 2*np.pi*np.linalg.inv(A).T

        # NOTE
        # Previous convention tried to keep a* || x, which requires following rotations. 
        
        # _, B = np.linalg.qr(B.T)
        
        # # Align a* along x
        # if B[0,0]<0:
        #     B = np.dot(ms.Rz(np.pi), B)
            
        # # Make c* have positive z-component
        # if B[2,2]<0:
        #     B = np.dot(ms.Rx(np.pi), B)
            
        return B
        
    def constructU(self, orientation: Union[None, tuple, np.ndarray]) -> np.ndarray:
        '''
        Construct the orientation matrix U. Different schemes are allowed depending on the type of the `orientation` argument.
        
        orientation : None
            Identity matrix
        orientation : hkl_tuple
            The chosen hkl is put perpendicular to the scattering plane, i.e. along the `z` axis.
        orientation : (hkl1_tuple, hkl2_tuple)
            hkl1 is put along the `x` axis and hkl2 in the `xy` plane.
        orientation : ndarray(3,3)
            U is given directly as an argument.
        '''
                
        # If the orientation is not None the U matrix heas to be updated
        if orientation == None:
            # Initial orientation is as given by B
            U = np.eye(3,3)
        elif np.shape(orientation) == (3,):
            # Single vector
            hkl = orientation
            n = np.dot(self.B, hkl)
            U = funs_sw.rotate( np.cross(n, [0,0,1]), funs_sw.angle(n, [0,0,1]) )
        elif np.array(orientation).shape == (2,3):
            # Two vectors
            hkl1, hkl2 = orientation
            n1 = np.dot(self.B, hkl1)
            n2 = np.dot(self.B, hkl2)
            
            # This rotation puts hkl1 along `x`
            R1 = funs_sw.rotate( np.cross(n1, [1,0,0]), funs_sw.angle([1,0,0], n1) )

            # Find the angle necessary to put hkl2 in `xy` plane
            n3 = np.dot(R1, n2)
            beta2 = funs_sw.angle(n3, [n3[0],n3[1],0]) * np.sign(-n3[2])
            R2 = funs_sw.rotate( [1,0,0], beta2 )
            
            U = np.dot(R2, R1)
        elif np.array(orientation).shape == (3,3):
            U = np.array(orientation)
        else:
            raise ValueError('Wrong orientation argument for initializing the Lattice object.')
            

        return U
        
    def updateOrientation(self, orientation: Union[None, tuple, np.ndarray]):
        '''
        Update the orientation matrix of the Lattice, together with the underlying UA and UB matrices.
        
        Raises Warning if the new matrix is not orthonormal
        '''
        
        newU = self.constructU(orientation)
        
        assert np.shape(newU) == (3,3)
        
        try:
            np.testing.assert_almost_equal(np.dot(newU[0],newU[0]), 1)
            np.testing.assert_almost_equal(np.dot(newU[1],newU[1]), 1)
            np.testing.assert_almost_equal(np.dot(newU[2],newU[2]), 1)
        except AssertionError:
            raise Warning('The new orientation matrix does not seem to be row-normalized')
            
        try:
            np.testing.assert_almost_equal(np.dot(newU[0],newU[1]), 0)
            np.testing.assert_almost_equal(np.dot(newU[0],newU[2]), 0)
            np.testing.assert_almost_equal(np.dot(newU[1],newU[2]), 0)
        except AssertionError:
            raise Warning('The new orientation matrix does not seem to be orthogonal')
            
        self.U = newU
        self.UA = np.dot(newU, self.A)
        self.UB = np.dot(newU, self.B)
        self._current_orientation = orientation
        
        return
    
    def uvw2xyz(self, uvw: Union[tuple, list]) -> np.ndarray:
        '''
        Calculate real space coordinates [x,y,z] based on the crystal coordinates [u,v,w].
        
        Parameters:
            uvw : array_like
                Crystal coordinates or list of crystal coordinates
                
        Returns: ndarray
            Vector in real space or list of vectors in real space.
        '''
        
        _uvw = np.array(uvw)
        
        # hkl is a single vector
        if _uvw.shape == (3,):
            out = np.dot(self.UA, _uvw)
        elif _uvw.shape[1] == 3:
            out = np.einsum('kj,ij', self.UA, _uvw)
        else:
            raise IndexError('Incompatible dimension of the uvw array. Should be (3,) or (N,3).')
        
        return out
    
    def xyz2uvw(self, xyz: Union[tuple, list]) -> Union[tuple, list]:
        '''
        Calculate the Miller indices (h,k,l) based on the reciprocal space coordinates (kx,ky,kz).
        
        Parameters:
            Q : array_like
                Reciprocal space coordinates or list of thereof.
                
        Returns: ndarray
            Miller indices or list of Miller indices.
        '''
        
        _xyz = np.array(xyz)
        
        if _xyz.shape == (3,):
            out = np.dot(np.linalg.inv(self.UA), _xyz)
        elif _xyz.shape[1] == 3:
            out = np.einsum('kj,ij', np.linalg.inv(self.UA), _xyz)
        else:
            raise IndexError(f'Incompatible dimension of the `xyz` array. Should be (3,) or (N,3) is: {_xyz.shape}.')
        
        return out


    def hkl2xyz(self, hkl: Union[tuple, list]) -> np.ndarray:
        '''
        Calculate reciprocal space coordinates (kx,ky,kz) based on the Miller indices (h,k,l).
                     
        Parameters:
            hkl : array_like
                Miller indices or list of Miller indices.
                
        Returns: ndarray
            Vector in reciprocal space or list of vectors in reciprocal space.
        '''
        
        _hkl = np.array(hkl)
        
        # hkl is a single vector
        if _hkl.shape == (3,):
            out = np.dot(self.UB, _hkl)
        elif _hkl.shape[1] == 3:
            out = np.einsum('kj,ij', self.UB, _hkl)
        else:
            raise IndexError('Incompatible dimension of the hkl array. Should be (3,) or (N,3).')
        
        return out
        
    def xyz2hkl(self, Q: Union[tuple, list]) -> Union[tuple, list]:
        '''
        Calculate the Miller indices (h,k,l) based on the reciprocal space coordinates (kx,ky,kz).
        
        Parameters:
            Q : array_like
                Reciprocal space coordinates or list of thereof.
                
        Returns: ndarray
            Miller indices or list of Miller indices.
        '''
        
        _Q = np.array(Q)
        
        # Q is a single vector
        if _Q.shape == (3,):
            out = np.dot(np.linalg.inv(self.UB), _Q)
        elif _Q.shape[1] == 3:
            out = np.einsum('kj,ij', np.linalg.inv(self.UB), _Q)
        else:
            raise IndexError('Incompatible dimension of the Q array. Should be (3,) or (N,3).')
        
        return out
        
    def scattering_angle(self, hkl: Union[tuple, list], wavelength: float) -> np.ndarray:
        '''
        Calculate the scattering angle otherwise known as two-theta from the Miller indices.
        
        Parameters:
            hkl : array_like
                Miller indices or list of Miller indices
            wavelength : float
                Wavelength of the incoming wave in Angstroems.
                
        Returns: ndarray
            Scattering angle or a list of scattering angles.
        '''
        
        _hkl = np.array(hkl)
        Q = self.hkl2xyz(hkl)

        # hkl is a single vector
        if _hkl.shape == (3,):
            Q_lengths = np.linalg.norm(Q)
        elif _hkl.shape[1] == 3:
            Q_lengths = np.linalg.norm(Q, axis=1)
        else:
            raise IndexError(f'Incompatible dimension of the hkl array. Should be (3,) or (N,3), but is: {_hkl.shape}')
        
        y = wavelength*Q_lengths/(4*np.pi)
        try:
            theta = np.arcsin(y)
        except RuntimeWarning:
            raise ValueError('Wavelength too long to be able to reach the selected hkl.')
            
        return 2*theta
        
    def is_in_scattering_plane(self, hkl: tuple) -> bool:
        '''
        Test whether the given hkl is in the scattering plane i.e. `xy` plane.
        '''
        # XY is the scattering plane, to be in the scattering plane the z component must be small.
        v = self.hkl2xyz(hkl)
        v = v/funs_sw.norm(v)
        
        return v[2]<1e-7
    
    def make_qPath(self, main_qs: list, Nqs: list[int]) -> np.ndarray:
        '''
        Make a list of q-points along the `main_qs` with spacing defined by `Nqs`.

        main_qs:
            List of consequtive q-points along which the path is construted.
        Nqs:
            Number of q-points in total, or list of numbers that define 
            numbers of q-points in between `main_qs`.
        '''

        _main_qs = np.asarray(main_qs)
        _Nqs = np.asarray(Nqs)

        assert _main_qs.shape[1] == 3 # `main_qs` is list of 3d vectors
        assert _Nqs.shape[0] == _main_qs.shape[0]-1 # `main_qs` is list of 3d vectors

        qPath = []
        for qstart, qend, Nq in zip(_main_qs[:-1], _main_qs[1:], Nqs):
            qPath.append(np.linspace(qstart, qend, Nq))

        return np.vstack(qPath)