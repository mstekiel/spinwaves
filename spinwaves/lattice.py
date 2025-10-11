"""Lattice implementation.
Main functionality is transformation between coordinate systems
of direct and reciprocal lattices, as well as their cartesian bases.

Lattice
---------------------
Holder and validator of lattice parameters with functionalities
of coordinate system transformations.

LatticeOriented
-------------------
`Lattice` equipped with orientation. All coordinate transformations
are now including the orientation of the lattice.
"""
import numpy as np

from .utils import linalg
from .utils.arrays import ensure_shape

from typing import Sequence, Union


class Lattice(object):
    """Class to describe a generic lattice system defined by six lattice
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
    A
    B
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
    2. The phase factors are (i) exp(i k_xyz r_xyz) and (ii) exp(2pi i Q_hkl r_uvw).
    3. All matrices, so for both direct and reciprocal space, are represented for column vectors.
    4. `B` matrix contains the 2pi factor. Consequently A.T @ B = 2pi eye(3,3). Transposition due to pt 3.
    5. Implemented methods should work with arrays assuming the last index representing the `h, k, l` coordinates.

        
    Transformation notes
    --------------------
    A: ndarray((3,3))

        Transforms a direct lattice coordinates into an cartesian coordinates system. Upper triangle matrix.
        [u,v,w] -> [x,y,z] (Angstroems)


    B: ndarray((3,3))

        Transforms a reciprocal lattice point into an orthonormal coordinates system.
        (h,k,l) -> [kx,ky,kz] (1/Angstroem)

    """

    def __init__(self, lattice_pars: Sequence[float]):
        # Initiate the main internal container
        self._lattice_parameters = [1,1,1, 90,90,90]
        # Setters will run validation checks and update lattice automatically
        a,b,c, alpha,beta,gamma = lattice_pars
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    #################################################################################################
    # Core methods

    def _update_lattice_parameters(self):
        '''Master function recalculating all matrices involving the lattice parameters.'''
        a,b,c, alpha,beta,gamma = self._lattice_parameters
        # A matrix follows convention 1.
        self._A = self._constructA(a,b,c, alpha,beta,gamma)

        # B matrix based on the perpendicularity condition to A
        # with the 2pi factor
        # To get column representation it needs to be transposed
        self._B = 2*np.pi* np.linalg.inv(self.A).T

        # Metric tensor of real space
        self._G = self.A.T @ self.A

        # Metric tensor of reciprocal space
        self._Gstar = self.B.T @ self.B

    @staticmethod
    def _constructA(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
        '''
        Construct the `A` matrix as crystal axes in orthonormal system, ie a||x, b in xy plane, c accordingly.

        With the convention of the `Lattice` it is an upper triangular matrix with columns representing
        lattice vectors in cartesian coordinates.'''
        
        alpha, beta, gamma = np.radians([alpha, beta, gamma])
        bx = b*np.cos(gamma)
        by = b*np.sin(gamma)
        
        cx = c*np.cos(beta)
        cy = c*(np.cos(alpha)-np.cos(gamma)*np.cos(beta))/np.sin(gamma)
        cz  = np.sqrt(c*c-cx*cx-cy*cy)
        
        return np.array([[a,bx,cx],[0,by,cy],[0,0,cz]])

    def __repr__(self):
        return f"{self.__class__.__name__}(a={self.a}, b={self.b}, c={self.c}, alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}>"
    
    
    #################################################################################################
    # Defining properties
    @property
    def A(self):
        '''Crystal axes in orthonormal system, ie a||x, b in xy plane, c accordingly.
        Upper triangle matrix.

        Transforms a real lattice point into an orthonormal coordinates system.
        A * [u,v,w] -> [x,y,z] (Angstroems)


        Shortcut to define lattice vectors:
        >>> a, b, c = A.T
        '''
        return self._A

    @property
    def B(self):
        '''Reciprocal crystal axes in orthonormal system, perpendicular to real axes.
        By definition `b* || y`.

        Transforms a reciprocal lattice point into an orthonormal coordinates system.
        B*(h,k,l) -> [kx,ky,kz]_{crystal} (1/Angstroem)

        Shortcut to define reciprocal lattice vectors:
        >>> astar, bstar, cstar = B.T
        '''
        return self._B

    @property
    def G(self):
        '''Metric tensor of the real lattice.
        G = A.T @ A
        '''
        return self._G


    @property
    def Gstar(self):
        '''Metric tensor of the reciprocal lattice.
        Gstar = B.T @ B

        Allows to calculate products of vector in hkl base.
        Note, that due to 2pi factor in B matrix, it contains the 4pi**2 factor.
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
        assert new_a > 0, f'Lattice length must be larger than zero: {new_a=}'
        self._lattice_parameters[0] = new_a
        self._update_lattice_parameters()

    @property
    def b(self) -> float:
        """Second lattice parameter `b` in Angstrom."""
        return self._lattice_parameters[1]

    @b.setter
    def b(self, new_b: float):
        assert new_b > 0, f'Lattice length must be larger than zero: {new_b=}'
        self._lattice_parameters[1] = new_b
        self._update_lattice_parameters()

    @property
    def c(self) -> float:
        """Third lattice parameter `c` in Angstrom."""
        return self._lattice_parameters[2]
    
    @c.setter
    def c(self, new_c: float):
        assert new_c > 0, f'Lattice length must be larger than zero: {new_c=}'
        self._lattice_parameters[2] = new_c
        self._update_lattice_parameters()


    @property
    def alpha(self) -> float:
        """First lattice angle `alpha` in degrees."""
        return self._lattice_parameters[3]

    @alpha.setter
    def alpha(self, new_alpha):
        assert 180 > new_alpha > 0, f'Lattice angle must be 180>...>0: {new_alpha=}'
        self._lattice_parameters[3] = new_alpha
        self._update_lattice_parameters()

    @property
    def beta(self) -> float:
        """Second lattice angle `beta` in degrees."""
        return self._lattice_parameters[4]

    @beta.setter
    def beta(self, new_beta: float):
        assert 180 > new_beta > 0, f'Lattice angle must be 180>...>0: {new_beta=}'
        self._lattice_parameters[4] = new_beta
        self._update_lattice_parameters()

    @property
    def gamma(self) -> float:
        """Third lattice angle `gamma` in degrees."""
        return self._lattice_parameters[5]

    @gamma.setter
    def gamma(self, new_gamma: float):
        assert 180 > new_gamma > 0, f'Lattice angle must be 180>...>0: {new_gamma=}'
        self._lattice_parameters[5] = new_gamma
        self._update_lattice_parameters()

    ### Properties without setters

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
    # Coordinate transforms
    
    @ensure_shape(uvw=(...,3))
    def uvw2xyz(self, uvw: Sequence[float]) -> np.ndarray[float]:
        '''
        Calculate direct cartesian coordinates [x,y,z] from direct lattice coordinates [u,v,w].
        
        Parameters:
            uvw : array_like
                Crystal coordinates or list of crystal coordinates
                
        Returns: ndarray
            Vector in real space or list of vectors in real space.
        '''       
        return np.einsum('ij,...j', self.A, uvw)
    
    @ensure_shape(xyz=(...,3))
    def xyz2uvw(self, xyz: Sequence[float]) -> np.ndarray[float]:
        '''
        Calculate the direct lattice coordinates [u,v,w] based on the direct cartesian coordinates [x,y,z].
        
        Parameters:
            xyz: array_like
                Direct space coordinates or list of thereof.
                
        Returns: ndarray
            Crystal coordinates.
        '''       
        return np.einsum('ij,...j', np.linalg.inv(self.A), xyz)

    @ensure_shape(hkl=(...,3))
    def hkl2xyz(self, hkl: Sequence[float]) -> np.ndarray[float]:
        '''
        Calculate reciprocal cartesian coordinates (kx,ky,kz) based on the reciprocal lattice coordinates (h,k,l).
                     
        Parameters
        ----------
        hkl : array_like, (...,3)
            Reciprocal lattice coordinates.
                
        Returns
        ------- 
        Qxyz: ndarray, (...,3)
            Vector in reciprocal space or list of vectors in reciprocal space.
        '''        
        return  np.einsum('ij,...j', self.B, hkl)
        
    @ensure_shape(xyz=(...,3))
    def xyz2hkl(self, xyz: Sequence[float]) -> np.ndarray[float]:
        '''
        Calculate the reciprocal lattice coordinates (h,k,l) from the reciprocal cartesian coordinates (kx,ky,kz).
        
        Parameters
        ----------
        xyz : array_like, (...,3)
            Reciprocal space coordinates or list of thereof.
                
        Returns
        -------
        Qhkl: ndarray, (...,3)
            Miller indices or list of Miller indices.
        '''
        return np.einsum('ij,...j', np.linalg.inv(self.B), xyz)
    
    #################################################################################################
    # Functionalities
    @ensure_shape(hkl1=(...,3), hkl2=(...,3))
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

    @ensure_shape(hkl=(...,3))
    def get_Q(self, hkl: np.ndarray) -> np.ndarray:
        '''Returns the magnitude |Q| [1/A] of reciprocal lattice vectors `hkl`.
                
        Parameters
        ----------
        hkl : array_like (...,3)
            Reciprocal lattice vector in r.l.u. Signature: `h,k,l = hkl`

        Returns
        -------
        Q : array_like (...)
            The magnitude of the reciprocal lattice vectors in [1/A].
            Shape follows the input signature with reduced last dimension.


        Notes
        -----
        Calculates the Q vector from the inverse metric tensor: `Q = sqrt(hkl.T @ Gstar @ hkl)`.
        Alternative method of calculating from B matrix proved to be slower: `Q = norm(B @ hkl)`
        '''
        # return np.sqrt(np.einsum('i...,ij,j...->...', hkl, self.Gstar, hkl))
        return np.sqrt(self.get_scalar_product(hkl, hkl))
    
    @ensure_shape(hkl=(...,3))
    def get_dspacing(self, hkl: np.ndarray) -> np.ndarray:
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

    @ensure_shape(hkl=(...,3))
    def get_tth(self, hkl: np.ndarray, wavelength: float) -> np.ndarray:
        u"""Returns the detector angle two-theta [rad] for a given reciprocal
        lattice vector [rlu] and incident wavelength [A].

        Parameters
        ----------
        hkl : array_like (...,3)
            Reciprocal lattice vector in r.l.u. Signature: `h,k,l = hkl`

        wavelength : float
            Wavelength of the incident beam in Angstroem.

        Returns
        -------
        two_theta : array_like (...)
            The scattering angle two-theta i nradians.
            Shape follows the input signature with reduced last dimension.

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
    
    @ensure_shape(main_qs=(...,3))
    def make_qPath(self, main_qs: list, Nqs: Union[int, list[int]], return_Qinc: bool=False) -> np.ndarray:
        '''
        Make a list of q-points along the `main_qs` with spacing defined by `Nqs`.

        Parameters
        ----------
        main_qs: array_like, (N,3)
            List of consequtive q-points along which the path is construted.
        Nqs: array_like, (M,)
            Number of q-points in total, or list of numbers that define 
            numbers of q-points in between `main_qs`.
        return_Qinc: bool, (optional)
            If true, also return an array trailing how much distance in reciprocal space
            is made with each point.

        Returns
        -------
        Qpath: ndarray, ((N-1)*M, 3)
            Points in reciprocal space [r.l.u.] along the specified path.
        Qinc: ndarray, ((N-1)*M, 3)
            Cumultative sum of distance in reciprocal space [A-1] covered after each point in `Qpath`.
        '''

        _main_qs = np.asarray(main_qs)

        if isinstance(Nqs, int):
            _Nqs = np.array([Nqs]*(len(main_qs)-1))
        elif len(Nqs) == _main_qs.shape[0]-1:
            _Nqs = np.array(Nqs)
        else:
            raise IndexError(f'List of Nqs points must be one shorter that list of main Qs. Is {len(Nqs)}, should be {len(main_qs)-1}')

        Qpath_hkl = []
        for qstart, qend, Nq in zip(_main_qs[:-1], _main_qs[1:], _Nqs):
            Qpath_hkl.append(np.linspace(qstart, qend, Nq))

        Qpath_hkl.append([_main_qs[-1]])
        Qpath_hkl = np.vstack(Qpath_hkl)


        Qpath_xyz = self.hkl2xyz(Qpath_hkl)
        Qinc1p = np.cumsum( np.linalg.norm(Qpath_xyz[1:]-Qpath_xyz[:-1], axis=1) )
        Qinc = np.concatenate(([0], Qinc1p))

        # Prep output
        ret = Qpath_hkl
        if return_Qinc:
            ret = (ret, Qinc)

        return ret
    
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

class LatticeOriented(Lattice):
    '''
    Object representing the lattice of an oriented crystal. Facilitates transformations between Cartesian and lattice
    coordinates as well as direct and reciprocal space.
    
    Conventions:
        1. Lattice vectors are positioned in the Cartesian coordinates as:
            - a || x
            - b* || y
            - c to complete RHS
        2. All matrices, so for both direct and reciprocal space, are represented for column vectors.
        3. `B` matrix contains the 2pi factor. Consequently B.T @ A = 2pi eye(3,3). Transposition due to pt 3.
        4. The phase factors are (i) exp(i k r_xyz) and (ii) exp(2 pi i Q r_uvw). 
           So in lattice coordinates we use 2pi, in cartesian coordinates we don't.
    
    The main idea is that the orientation can be changed, but the lattice type and parameters no.
    
    Attirbutes:
        lattice_parameters: ndarray((6))
            List containing `a,b,c,alpha, beta, gamma` lattice parameters. Lengths in angstroems, angles in degrees.
        A : ndarray((3,3))
            Transforms a direct lattice point into cartesian coordinates system. Upper triangle matrix.
            [u,v,w] -> [x,y,z] (Angstroems)
        B : ndarray((3,3))
            Transforms a reciprocal lattice point into an cartesian coordinates system with 2pi factor. Upper triangle matrix.
            (h,k,l) -> [kx,ky,kz] (1/Angstroem)
        U : ndarray((3,3))
            Orientation matrix that relates the cartesian lattice coordinate system into the diffractometer/lab coordinates
            [kx,ky,kz]_{crystal} -> [kx,ky,kz]_{lab}
        UA : ndarray((3,3))
            Transforms a direct lattice point into lab coordinate system.
            (u,v,w) -> [x,y,z]_{lab} (Angstroem)
        UB : ndarray((3,3))
            Transforms a reciprocal lattice point into lab coordinate system.
            (h,k,l) -> [kx,ky,kz]_{lab} (1/Angstroem)
    '''

    def __init__(self, lattice_pars: list[float], orientation: Union[None, tuple, np.ndarray]=None):
        '''
        Object representing crystal lattice and coordinate system in cartesian space.

        Parameters:
        -----------

        lattice parameters: [a,b,c,alpha,beta,gamma]
            Lattice lengths in Angstroem, lattice angles in degrees.
        
        orientation : None
            Identity matrix
        orientation : hkl_tuple
            The chosen hkl is put perpendicular to the scattering plane, i.e. along the `z` axis.
        orientation : (hkl1_tuple, hkl2_tuple)
            hkl1 is put along the `x` axis and hkl2 in the `xy` plane.
        orientation : ndarray(3,3)
            U is given directly as an argument.
        '''
        # Initialize lattice parameters from the parent
        super().__init__(lattice_pars=lattice_pars)
        
        # Initialize the orientation, U and the UB matrix within the wrapper.
        self.updateOrientation(orientation)




    def __repr__(self):
        rr = f'{self.__class__.__name__}('
        rr += ', '.join(f'{name}={value}' for name,value in zip(['a','b','c','alpha','beta','gamma'], 
                                                                self.lattice_parameters))
        rr += f', orientation={self._current_orientation})'

        return rr
        
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
                
        # If the orientation is not None the U matrix has to be updated
        if orientation is None:
            # Initial orientation is as given by B
            U = np.eye(3,3)
        elif np.shape(orientation) == (3,):
            # Single vector
            hkl = orientation
            n = np.dot(self.B, hkl)
            U = linalg.rotate( np.cross(n, [0,0,1]), linalg.angle(n, [0,0,1]) )
        elif np.array(orientation).shape == (2,3):
            # Two vectors
            hkl1, hkl2 = orientation
            n1 = np.dot(self.B, hkl1)
            n2 = np.dot(self.B, hkl2)
            
            # This rotation puts hkl1 along `x`
            R1 = linalg.rotate( np.cross(n1, [1,0,0]), linalg.angle([1,0,0], n1) )

            # Find the angle necessary to put hkl2 in `xy` plane
            n3 = np.dot(R1, n2)
            beta2 = linalg.angle(n3, [n3[0],n3[1],0]) * np.sign(-n3[2])
            R2 = linalg.rotate( [1,0,0], beta2 )
            
            U = np.dot(R2, R1)
        elif np.array(orientation).shape == (3,3):
            U = np.array(orientation)
        else:
            raise ValueError(f'Wrong orientation argument for initializing the `{self.__class__.__name__}` object.')
            

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
    
    #################################################################################################
    # Coordinate transforms
    
    @ensure_shape(uvw=(...,3))
    def uvw2xyz(self, uvw: Sequence[float]) -> np.ndarray[float]:
        '''
        Calculate direct cartesian coordinates [x,y,z] from direct lattice coordinates [u,v,w].
        
        Parameters:
            uvw : array_like
                Crystal coordinates or list of crystal coordinates
                
        Returns: ndarray
            Vector in real space or list of vectors in real space.
        '''       
        return np.einsum('ij,...j', self.UA, uvw)
    
    @ensure_shape(xyz=(...,3))
    def xyz2uvw(self, xyz: Sequence[float]) -> np.ndarray[float]:
        '''
        Calculate the direct lattice coordinates [u,v,w] based on the direct cartesian coordinates [x,y,z].
        
        Parameters:
            xyz: array_like
                Direct space coordinates or list of thereof.
                
        Returns: ndarray
            Crystal coordinates.
        '''       
        return np.einsum('ij,...j', np.linalg.inv(self.UA), xyz)

    @ensure_shape(hkl=(...,3))
    def hkl2xyz(self, hkl: Sequence[float]) -> np.ndarray[float]:
        '''
        Calculate reciprocal cartesian coordinates (kx,ky,kz) based on the reciprocal lattice coordinates (h,k,l).
                     
        Parameters
        ----------
        hkl : array_like, (...,3)
            Reciprocal lattice coordinates.
                
        Returns
        ------- 
        Qxyz: ndarray, (...,3)
            Vector in reciprocal space or list of vectors in reciprocal space.
        '''        
        return  np.einsum('ij,...j', self.UB, hkl)
        
    @ensure_shape(xyz=(...,3))
    def xyz2hkl(self, xyz: Sequence[float]) -> np.ndarray[float]:
        '''
        Calculate the reciprocal lattice coordinates (h,k,l) from the reciprocal cartesian coordinates (kx,ky,kz).
        
        Parameters
        ----------
        xyz : array_like, (...,3)
            Reciprocal space coordinates or list of thereof.
                
        Returns
        -------
        Qhkl: ndarray, (...,3)
            Miller indices or list of Miller indices.
        '''
        return np.einsum('ij,...j', np.linalg.inv(self.UB), xyz)
        
    #################################################################################################
    # Coordinate transforms
    @ensure_shape(hkl=(...,3))
    def scattering_angle(self, hkl: Sequence[float], wavelength: float) -> np.ndarray:
        '''
        Calculate the scattering angle two-theta for the reciprocal lattice coordinates `hkl`.
        
        Parameters
        ----------
        hkl : array_like
            Reciprocal lattice coordinates.
        wavelength : float
            Wavelength of the incoming wave in Angstroems.
                
        Returns
        -------
        tth: ndarray
            Scattering angles in radians.
        '''
        
        Q_lengths = self.get_Q(hkl)        
        y = wavelength*Q_lengths/(4*np.pi)
        try:
            theta = np.arcsin(y)
        except RuntimeWarning:
            raise ValueError('Wavelength too long to be able to reach the selected hkl.')
            
        return 2*theta
        
    @ensure_shape(hkl=(...,3))
    def is_in_scattering_plane(self, hkl: Sequence[float]) -> bool:
        '''
        Test whether the given hkl is in the scattering plane i.e. `xy` plane.
        '''
        # XY is the scattering plane, to be in the scattering plane the z component must be small.
        v = self.hkl2xyz(hkl)
        # v = v/linalg.norm(v)
        
        return v[...,2] < 1e-7
    