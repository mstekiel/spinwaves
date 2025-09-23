'''Engine for the Linear Spin Wave Theory calculations of spin dynamics.

I realized how important the safety checks are for setting up the systems.
Unfortunately, some of them require additional calculations and slow down
the execution.
How about implementing a `speedrun` option for functions, which will omit 
heavy checks.
'''
# Core
# from scipy.linalg import schur, cholesky
import scipy
import numpy as np

import logging

# Typing
from typing import Callable, List, Dict, TYPE_CHECKING, Union
from numpy.typing import NDArray

from spinwaves import Crystal

if TYPE_CHECKING:
    from .crystal import Crystal
    import matplotlib.pyplot as plt


# Internal
from . import functions as funs_sw
# import functions as funs_sw


# Tobi confirmed the factor of two is missing from single-ion naisotropies.
# He also mentioned the inverted sign mistake in the phase factor somewhere
# in the spin-spin correlation function calculations.

logger = logging.getLogger('SpinW')

import inspect
from functools import wraps

def _match_shape(arr_shape: tuple[int, ...], expected_shape: tuple[int, ...]) -> bool:
    """
    Recursively match arr_shape against expected_shape with ... wildcards.
    Supports multiple ellipses anywhere.
    """
    # print(f'DEBUG: matching: {arr_shape=} with {expected_shape=}')
    if not expected_shape:
        return not arr_shape

    head, *tail = expected_shape

    if head is ...:
        # try all possible splits: let ... absorb 0,1,2,... dimensions
        for k in range(len(arr_shape) + 1):
            if _match_shape(arr_shape[k:], tuple(tail)):
                return True
        return False

    if not arr_shape or arr_shape[0] != head:
        return False

    return _match_shape(arr_shape[1:], tuple(tail))

def ensure_shape(**shapes):
    """
    Decorator to validate numpy array arguments by name and shape.

    Usage:
        @ensure_shapes(
            a=(..., 3),          # last dimension must be 3
            b=(2, ...),          # first dimension must be 2
            c=(2, 3, ..., 3, 3)  # fixed prefix and suffix, flexible middle
        )
        def func(a, b, c): ...
    """
    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for name, expected_shape in shapes.items():
                if name not in bound.arguments:
                    raise ValueError(f"Parameter '{name}' not found in function arguments")

                arr_shape = np.shape(bound.arguments[name])
                if not _match_shape(arr_shape, expected_shape):
                    raise ValueError(
                        f"Parameter '{name}' expected shape {expected_shape}, "
                        f"got {arr_shape}"
                    )

            return func(*args, **kwargs)

        return wrapper
    return decorator


def bose_occupation(energies, temperature):
    """
    Compute the Bose-Einstein occupation factor:
        n(E,T) = 1 / (exp(|E|/(kB*T)) - 1)

    Handles edge cases:
    - E = 0: returns +inf
    - T = 0: returns 0 for E>0, +inf for E=0
    - Uses expm1 for small arguments
    - Uses exp(-x) for large arguments to avoid overflow
    - Threshold for switching depends on dtype precision
    """
    energies = np.asarray(energies)
    dtype = energies.dtype if np.issubdtype(energies.dtype, np.floating) else np.float64
    energies = energies.astype(dtype, copy=False)

    # machine limits
    finfo = np.finfo(dtype)
    # Safe threshold before exp(x) overflows
    threshold = np.log(finfo.max) - 1.0

    # Handle T=0 separately
    if temperature == 0:
        out = np.zeros_like(energies, dtype=dtype)
        out[energies == 0] = np.inf
        return out

    # Normal case: finite T
    kB = 0.08617333262
    x = np.abs(energies) / (kB * temperature)
    out = np.empty_like(x, dtype=dtype)

    # Case: E=0 -> inf
    zero_mask = (energies == 0)
    out[zero_mask] = np.inf

    # Case: moderate x (safe to use expm1)
    mask = (x < threshold) & ~zero_mask
    out[mask] = 1.0 / np.expm1(x[mask])

    # Case: large x -> asymptotic exp(-x)
    big_mask = ~mask & ~zero_mask
    out[big_mask] = np.exp(-x[big_mask])

    return out


def make_exc_dtype(Etype: type=np.float64, Stype: type=np.complex128) -> 'np.dtype':
    """Create a structured dtype with:
    - E: scalar of real type `Etype`
    - Sperp: scalar of real type `Etype`
    - S: 3x3 matrix of complex type `Stype`
    - Sxx...Szz: aliases into S elements

    `E` represents the energy of the excitation, `S` the spin-spin correlation function matrix,
    `Sperp` is the perpendicular component of S to the momentum transfer Q,
    `Sij` with i,j in [x,y,z] are aliases into the S matrix.

    Parameters
    ----------
    Etype: type, optional
        Data type for the energy and Sperp scalars, default np.float64
    Stype: type, optional
        Data type for the spin-spin correlation function matrix, default np.complex128

    Notes
    -----
    ChatGPT made it. I am not sure the offset calculations are correct,
    in particular, that the S matrix is aligned properly with its components.
    Test it properly.
    """
    # Sizes in bytes
    esize = np.dtype(Etype).itemsize
    ssize = np.dtype(Stype).itemsize

    # Build names, formats, and offsets
    offset_S = 2*esize
    names = ["E", "Sperp", "S"]
    formats = [Etype, Etype, (Stype, (3, 3))]
    offsets = [0, esize, offset_S]

    # Add aliases for each element of S
    labels = ["x", "y", "z"]
    for i, row in enumerate(labels):
        for j, col in enumerate(labels):
            name = f"S{row}{col}"
            names.append(name)
            formats.append(Stype)
            offsets.append(offset_S + (i * 3 + j) * ssize)

    # Construct dtype
    return np.dtype({"names": names, "formats": formats, "offsets": offsets})

class Coupling:
    '''Coupling between atoms in the crystal.
    
    Notes
    -----
    - TODO establish the convention of the `J` matrix entries, 
      either crystal or Cartesian coordinates.
    - The fields `id1`, `id2`, and `J` are linked to the `Crystal` class.
      Their validity is not checked.
    '''
    _label: str
    _id1: int
    _id2: int
    _n_uvw: np.ndarray[int]
    _J: np.ndarray[float]
    _defining_bond: str

    # DMI_vector: np.ndarray[float] = np.zeros(3)

    # hidden_symmetry: tuple = ()

    def __init__(self, label: str, id1: int, id2: int, n_uvw: list[int],
                 J: np.ndarray[float], Jcoords: str='xyz',
                 defining_bond: str=''):
        # DEV NOTES
        # Allow using the constructor directly, thus ensure the types.
        self._label = label
        self._id1 = int(id1)
        self._id2 = int(id2)
        self._n_uvw = np.array(n_uvw, dtype=int)
        self._J = J
        self._defining_bond = defining_bond

    ############################################################################
    # Properties

    # Allow for the label setter, but no more
    @property
    def label(self) -> str:
        '''Label of the coupling'''
        return self._label
    
    @label.setter
    def label(self, new_label):
        self._label = new_label

    @property
    def id1(self) -> int:
        '''Index of the first interacting atom'''
        return self._id1
    
    @property
    def id2(self) -> int:
        '''Index of the second interacting atom'''
        return self._id2

    @property
    def n_uvw(self) -> np.ndarray[int]:
        '''Origin/index of the unit cell where the second interacting atoms resides.'''
        return self._n_uvw
    
    @property
    def J(self) -> np.ndarray[float]:
        '''Exchange interaction in matrix form.'''
        return self._J
    
    @property
    def DMI_vector(self) -> np.ndarray[float]:
        '''Ansitymmetric part of the interaction represented
        by the Dzialoshynskii-Moriya vector.'''    
        J_asym = (self.J - self.J.T) / 2
        Dx = J_asym[1,2]
        Dy = J_asym[0,2]
        Dz = J_asym[0,1]
        return np.array([Dx, Dy, Dz], dtype=float)
    
    ###########################################################################
    # Fors sorting and comparing

    # DEV
    # The main use case is when symmetrizing the couplings.
    # For that, we don't want to look into the exchange interaction matrix.

    def __hash__(self) -> int:
        return hash((self.id1, self.id2) + tuple(self.n_uvw.astype(int)))
    
    def __lt__(self, other) -> bool:
        comp_fields_left  = (self._id1, self._id2, self._n_uvw[0], self._n_uvw[1], self._n_uvw[2])
        comp_fields_right = (other._id1, other._id2, other._n_uvw[0], other._n_uvw[1], other._n_uvw[2])
        return comp_fields_left < comp_fields_right
    
    def __eq__(self, other: 'Coupling') -> bool:
        comp_fields_left  = (self._id1, self._id2, self._n_uvw[0], self._n_uvw[1], self._n_uvw[2])
        comp_fields_right = (other._id1, other._id2, other._n_uvw[0], other._n_uvw[1], other._n_uvw[2])
        return comp_fields_left == comp_fields_right
    
    ###########################################################################
    # Methods

    def revert(self) -> 'Coupling':
        '''Revert the coupling, as in exchange the coupled atoms.'''
        return Coupling(label = self.label+'_rev',
                        id1 = self.id2,
                        id2 = self.id1,
                        n_uvw = -self.n_uvw,
                        J = self.J.T,
                        defining_bond = self.label
                        )
    
    def __repr__(self) -> str:
        return f'<Coupling label={self.label}, id1={self.id1}, id2={self.id2}, n_uvw={self.n_uvw}, J={self.J.tolist()}>'
        
class Dispersion:
    pass

class SpinW:
    '''Spin wave calculator.

    Core functionalities encapsulate calculating all matrices required to derive spin wave dynamics as per linear spin wave theory.
    This includes symmetrizing the couplings.

    Fields
    ------
    crystal: 'Crystal'
        `Crystal` object holding information on position and magnetic moment of atoms as well as crystal symmetry.
    magnetic_atoms: list['Atom']
        List of magnetic atoms in the unit cell.
    magnetic_modulation: dict[str, tuple[float,float,float]]
        Dictionary holding information on propagation vector `k` and normal vector defining rotation plane `n`.
    couplings: list['Coupling']
        List of unique couplings defining interactions between magnetic atoms.
    couplings_all: list['Coupling']
        List of all couplings of atoms in the unit cell, that is a symmetrized version of `self.couplings`.



    TODO
    ----
    OK so the whole package needs restructurization into CRYSTAL = LATTICE + MSG + COUPLINGS.
    CRYSTAL will become a state machine
    - add decorator that condifrms the shape of the function parameters are ok
    - fix ground state calculation
    - I think symmetrization will fail in non 90*3 lattice. 
      I do symmetrization based on xyz, but it is implemented as uvw.
      When marrying symmetry operations with lattice, I should construct symmetry operations in cartesian representation.
    - pyLiSW is great inspiration. Split _determine_ESp into hamiltonian prep and intensity calc.
    - add functionality of magnetic field
    - seems liek we also have all components to determine the bond symmetry i nthe symbolic picture.
      Take a looka at https://doi.org/10.1103/fgc1-5blp
    - Clarify the calculation and definitions of spectral weight and intensity.
      I like the notation from gen Shirane book, I \propto kf/ki f^2(Q) exp(-W) S_perp
      I dont understand where DW-factor is coming from. Detailed balance is in S
    - add functionalities to force certain magnetic structure.
      Feels like it should be after the constructor. Maybe I can also figure out which symmetries are broken.
    - understand total moment sum rule
    '''
    def __init__(self, crystal: 'Crystal', magnetic_modulation: Dict, couplings: List[Coupling],
                 temperature: float=0):
        '''
        Parameters:
        -------
        crystal: 'Crystal'
            `Crystal` object holding information on position and magnetic moment of atoms as well as crystal symmetry.    
        magnetic_modulation: dict[str, tuple[float,float,float]]
            Dictionary holding information on propagation vector `k` and normal vector defining rotation plane `n`.
        couplings: list['Coupling']
            List of unique couplings defining interactions between magnetic atoms.
        '''
        self.crystal = crystal
        self.magnetic_atoms = crystal.atoms_magnetic
        self.magnetic_modulation = magnetic_modulation

        # Ensure input couplings are unique, otherwise symmetrization doesn't work well
        self.couplings = couplings
        self.couplings_all =  self.symmetrize_couplings(couplings)

        self.temperature = temperature

    def symmetrize_couplings(self, couplings: list['Coupling']) -> list['Coupling']:
        '''Generalize the couplings according to the symmetry of the crystal.

        Parameters
        ----------
        couplings: list[`Coupling`]
            List of unique couplings, subject to symmetrization

        Returns
        -------
        couplings_symmetrized: list[`Coupling`]
            Lsit of symmetrized couplings

        Raises
        ------
        KeyError
            When coupling a non-magnetic atom
        Warning
            When the coupling does not respect the crystal symmetry.
        
        Notes
        -----
        Algorithm
        1. Create all couplings by nesting loops
            - For each coupling in the list:
                - For each symmetry element of MSG:
                    - Symmetrize the coupling, by applying the symmetry operation.
                    - Append to list of all couplings
        2. Find only unique couplings in the list --- The way unique couplings are found might be harder than expected.
        3. Check if the symmetry constraints are obeyed by looking at equivalent couplings
        '''
        couplings_all = []
        # [1]
        for cpl in couplings:
            atom1 = self.magnetic_atoms[cpl.id1]
            atom2 = self.magnetic_atoms[cpl.id2]

            # Check that coupled atoms are both magnetic
            if not (atom1.is_mag and atom2.is_mag):
                logger.error(f'Coupling a non magnetic atom:\n\t{atom1}\n\t{atom2}')
                raise KeyError(f'Coupling a non magnetic atom: {atom1} {atom2}')

            for n,g in enumerate(self.crystal.MSG):
                r1 = g.transform_position(atom1.r)
                n_uvw1 = np.floor(r1)
                new_id1 = self.crystal.get_atom_sw_id(r1)

                r2 = g.transform_position(atom2.r+cpl.n_uvw)
                n_uvw2 = np.floor(r2)
                new_id2 = self.crystal.get_atom_sw_id(r2)

                new_n_uvw_12 = n_uvw2 - n_uvw1

                # g matrix has to be represented in xyz coordinates
                g_xyz = self.crystal.A @ g.matrix @ np.linalg.inv(self.crystal.A)
                new_J = g_xyz @ cpl.J @ np.linalg.inv(g_xyz)

                # If self interaction, anisotropy
                # we need to correct for the unnecesary double counting
                if np.allclose(r1, r2):
                    logger.info(f'Correcting for double-counting, {cpl}')
                    new_J *= 2

                cpl_new_12  = Coupling(label=f'{cpl.label}_{n}',
                                n_uvw=new_n_uvw_12,
                                id1=new_id1,
                                id2=new_id2,
                                J = new_J,
                                defining_bond=cpl.label)
                cpl_new_21  = cpl_new_12.revert()


                couplings_all.extend([cpl_new_12, cpl_new_21])

        couplings_unique, id_inverse = np.unique(couplings_all, return_inverse=True)

        # Check symmetry condidtion
        for id_unique, cpl_unique in enumerate(couplings_unique):
            id_equivalent = np.where(id_inverse==id_unique)[0]
            J_equivalent = [couplings_all[id].J for id in id_equivalent]
            J_averaged = np.average(J_equivalent, axis=0)

            # One more check is to count the equivalent couplings and compare to 
            # the order of the cupling pointgroup and MSG order
            # ---> Something doesnt work here on a non-Bavais lattice
            # r_cpl = (self.magnetic_atoms[cpl_unique.id1].r + self.magnetic_atoms[cpl_unique.id2].r + cpl_unique.n_uvw) / 2
            # cpl_point_group = self.crystal.MSG.get_point_symmetry(r_cpl)
            # print('UQ COUPLING: order=', len(cpl_point_group))
            # print('\t', cpl_unique)
            # print('\t',f'{id_equivalent=} {r_cpl=}')
            # print('\t', cpl_point_group)

            # # We need factor two as the reverse couplings are taken too
            # assert len(id_equivalent)//2 == len(cpl_point_group), 'Order of the coupling`s points group does not match the number of equivalent couplings'
            
            if not np.allclose(cpl_unique.J, J_averaged):
                warning_message = f'The following coupling does not respect the symmetry\n\t{cpl}\n'
                warning_message+= f'\tIs:        {cpl_unique.J.tolist()}\n'
                warning_message+= f'\tShould be: {J_averaged.tolist()}'
               
                logger.warning(warning_message)

        logger.info(f'Symmetrization report: generated / unique / provided = {len(couplings_all)} / {len(couplings_unique)} / {len(couplings)}')
        return sorted(couplings_unique, key=lambda cpl: cpl.label)
        
    #########################################################################################
    # Fundamental calculations

    def rot_Rn(self, n_uvw: tuple[int,int,int]) -> np.ndarray:
        '''Matrix corresponding to rotation of the spins in the `n_uvw` unit cell
        according to the magnetic modulation.

        Parameters
        ----------
        n_uvw: (3,) int
            Index of the unit cell
        
        Notes
        -----
        Wrapper around the fundamental function from `functions.rot_Rn`.
        '''

        return funs_sw.rot_Rn(n_uvw, self.magnetic_modulation['k'], self.magnetic_modulation['n'])
    
    def rot_Rprime(self, S: tuple[float, float, float]) -> np.ndarray:
        '''Matrix corresponding to rotation of the spin towards the `z` axis.

        Parameters
        ----------
        S: (3,) int
            Spi ndirection vector
        
        Notes
        -----
        Wrapper around the fundamental function from `functions.rot_RtoZ`.
        As for eq. (7) [SpinW]
        Rn' is the rotation that puts the magnetic moment along z axis.
            S'_nj = R'_n S''_nj
            S'_nj=S_0j : magnetic moment of j-th atom in the 0-th unit cell, independent on unit cell
            S''_nj : spin oriented along the ferromagnetic axis
        '''

        return funs_sw.RfromZ(S)
    
    def _construct_h(self, Qhkl):
        '''Construct the hamlitonian kernel matrix `h`, the core matrix 
        to determine magnon excitations.
        
        Parameters
        ----------
        Qhkl: np.ndarray, shape=(...,3)
            Array of momentum transfers at which to construct the matrix.

        Returns:
        h: np.ndarray, shape=(...,2*M,2*M)
            Kernel of the Hamiltonian.
        '''
        *shape, three = np.shape(Qhkl)
        shape = tuple(shape)
        assert three == 3, f'Last dimension of Qhkl must be 3, is {np.shape(Qhkl)=}'
        M = len(self.magnetic_atoms)

        # [SpinW] eq 21 and 14
        Jp0 = np.zeros((M,M, 3,3), dtype=np.float64)
        JpofK = np.zeros(shape+(M,M, 3,3), dtype=np.complex128)
        for cpl in self.couplings_all:
            # According to [SpinW] eq 21: Jp = Rm @ J @ Rn,
            # where n,m index unit cells, but m=0 in this notation so is omitted.
            Rn = self.rot_Rn(cpl.n_uvw)

            # We will be dealing with shape = (shape, 3,3)
            Jp0[cpl.id1, cpl.id2, :,:] += cpl.J @ Rn
            phase_factor = np.exp(2*np.pi*1j*np.dot(Qhkl, cpl.n_uvw)) 
            JpofK[..., cpl.id1, cpl.id2, :,:] += phase_factor[..., None, None] * (cpl.J @ Rn)    # Eq (52) [spinW]

        # [SpinW] eq 9
        u = np.zeros((M, 3), dtype=complex)
        v = np.zeros((M, 3), dtype=complex)
        S = np.asarray([atom.s for atom in self.magnetic_atoms])
        for atom_i,Sdir_i in enumerate([atom.m for atom in self.magnetic_atoms]):
            Rp_i = self.rot_Rprime(Sdir_i)
            u[atom_i, :] = Rp_i[:,0] + 1j*Rp_i[:,1]
            v[atom_i, :] = Rp_i[:,2]

        # [SpinW] eq 26
        # Indinces `ij` are magnetic atoms, `pq` are cartesian directions
        # I know that tensordot can be faster here
        JpofmK = np.conj(JpofK)
        SiSj = np.sqrt(np.einsum('i,j->ij', S, S))
        A1 = 0.5*np.einsum('ij,ip,...ijpq,jq->...ij', SiSj, u, JpofmK, np.conj(u))
        A2 = 0.5*np.conj(np.einsum('ij,ip,...ijpq,jq->...ij', SiSj, u, JpofK, np.conj(u)))
        B = 0.5*np.einsum('ij,ip,...ijpq,jq->...ij', SiSj, u, JpofmK, u)
        C = np.diag(np.einsum('l,ip,ilpq,lq->i', S, v, Jp0, v))

        # Now we assemble the h Frankenstein
        # [SpinW] eq 27
        Bdag = np.conj( B.swapaxes(-2,-1) )     # multidimensional transposition
        h = np.block([
            [A1-C, B],
            [Bdag, A2-C]
        ])

        return h
    
    def _determine_ESp(self, Qhkl: np.ndarray, includeS: bool=True) -> tuple:
        '''Determine the energies and unrotated spectral weights.

        Parameters
        ----------
        Qhkl: array_like, (..., 3)
            Momentum transfer at which the excitations are determined
        includeS: bool=True, (optional)
            If True, also determine the non-rotated spin-spin correlation function
            as in [SpinW eq. 47]
        
        Returns
        -------
        E: (...,M) ndarray
            Energies of excitations at given momentum transfer
        Sp: (...,M,3,3) ndarray, optional
            Single-mode, non-rotated spin-spin correlation function
        '''
        M = len(self.magnetic_atoms)

        h = self._construct_h(Qhkl) 

        # Ok, so I learned the hard way this numerical accuracy is important to debug properly.
        # SpinW does the block: (lines 812-840 spinwave.m)
        # Try cholesky decomposition
        # if failed: check smallest eigenvalue. 
        #   If smaller than tolerance add to diagonal and try cholesky
        #       if cholesky succeds everything is ok
        #       if cholesky fails, add tolerance to diagonal, and try again
        #           if cholesky fails, give up
        # if failed, add tole
        zeros = np.abs(np.linalg.det(h)) < 1e-10 
        h[zeros] += np.eye(2*M,2*M) * 1e-10

        # Cholesky decomposition
        try:
            K_lo = np.linalg.cholesky(h)
        except Exception as e:
            # if debug go through h in for loop and check which exactly fails?
            raise ValueError("Cholesky decomposition failed. \n"
                             "Most probably your structure is incombatible with exchange interactions.") from e

        K_up = K_lo.swapaxes(-2, -1).conj()

        def numpy_schur(A):
            '''Performr Schur decomposition of Hermitian matrix A
            using fast numpy implementation.
            
            Returns
            -------
            T, Q such that A = Q @ T @ Q^dagger
            
            Notes
            -----
            Main issue is that I want the eigenvectors to be orthonormal = 
            Q is unitary, not gauranteed by `numpy.linalg.eigh`. Also,
            `scipy.linalg.schur` is slow for large arrays.
            '''
            ### Slow scipy schur version for comparison
            # T = np.zeros(A.shape)
            # Q = np.zeros(A.shape)
            # for id, _ in np.ndenumerate(A[...,0,0]):
            #     Ai = A[id]
            #     Ti, Qi = scipy.linalg.schur(Ai)
            #     T[id] = Ti
            #     Q[id] = Qi

            #     print(f'{Ti=} {Qi=}')

            # return T, Q

            # Compute eigen-decomposition
            _, U = np.linalg.eigh(A)

            # Make eigenvectors orthonormal (U may not be unitary as-is)
            Q, R = np.linalg.qr(U)  # QR factorization gives orthonormal basis
            # print('Q bef', Q)
            # print('R bef', R)
            # print(f'{Q.real=}')
            # print(f'{R.real=}') # R should be eye
            # Deep dive trick: trust R has +-1 on diagonal, then R @ R = 1
            Q = Q @ R
            R = R @ R
            # print('Q aft', Q)
            # print('R aft', R)
            '''The main question is, when is R not identity? These are the weird
            cases, probably when the eigenvalues are degenerate, and numerics
            don't help.'''

            # Transform A into Schur form
            T = Q.conj().swapaxes(-2,-1) @ A @ Q
            # print('T', T)
            return T, Q
        
        # (30) [SpinW]
        g = np.diag( [1]*M + [-1]*M )

        # Between (31) (32) [SpinW]
        E, U = numpy_schur(K_up @ g @ K_lo)
        energies = np.diagonal(E.real, axis1=-2, axis2=-1)

        if includeS:
            # E = energies[...,:,None] * np.eye(2*M, 2*M)

            # (34) [SpinW]
            T = np.linalg.inv(K_up) @ U @ np.sqrt(E.astype(np.complex)) # shape=(...,2M,2M)

            r_i = np.asarray([atom_i.r for atom_i in self.magnetic_atoms])
            r_ij = r_i[None,:,:] - r_i[:,None,:]    # shape=(M,M,3)
            phase_factor = np.exp(2*np.pi*1j*np.tensordot(Qhkl, r_ij, axes=(-1,-1)))    # shape=(...,M,M)

            S = np.asarray([atom.s for atom in self.magnetic_atoms])
            SiSj = np.sqrt(S[None,:] * S[:,None])  # shape=(M,M,3)
            spin_phase = phase_factor[..., :,:] * SiSj  # shape=(...,M,M)

            bose_factor = bose_occupation(energies, self.temperature) \
                + (1-np.diag(g))/2  # shape=(...,2M)

            u = np.zeros((M, 3), dtype=complex)
            for atom_i,Sdir_i in enumerate([atom.m for atom in self.magnetic_atoms]):
                Rp_i = self.rot_Rprime(Sdir_i)
                u[atom_i, :] = Rp_i[:,0] + 1j*Rp_i[:,1]

            # TODO fix this final steps into numpy broadcasted arrays for speed

            # (44) [SpinW]
            # Transpositions included in the indices swap
            # Everyone here has shape=(...,3,3,M,M)
            Y = np.einsum('...ij,ia,jb->...abij', spin_phase, u,u.conjugate())
            Z = np.einsum('...ij,ia,jb->...abij', spin_phase, u,u)
            V = np.einsum('...ij,ia,jb->...abij', spin_phase, u.conjugate(),u.conjugate())
            W = np.einsum('...ij,ia,jb->...abij', spin_phase, u.conjugate(),u)

            YZVW = np.block([[Y, Z], [V, W]])   # shape=(...,3,3,2M,2M)

            # Kind of (47) [SpinW]
            Sp = np.einsum('...ji,...abjk,...ki,...i->...iab', np.conj(T), YZVW, T, bose_factor)    # shape=(...,M,3,3)
            Sp = Sp / (2*M)

            ### Surprisingly this doesn't work well.
            ### Wants to allocate a lot of data
            # # Similarity transform over the last two axes; (3,3) just broadcast as batch dims
            # print(f'{YZVW.shape=} {T.shape=}')
            # # T_YZVW_T = T.conj().swapaxes(-1, -2) @ YZVW @ T    # shape=(...,3,3,2M,2M)
            # T_YZVW_T = np.tensordot(np.tensordot(
            #         T.conj().swapaxes(-1, -2), YZVW, axes=(-1,-2)), T, axes=(-1,-2))

            # # Take the diagonal over the (2M,2M) axes (..., 3, 3, 2M)
            # T_YZVW_T_diag = np.diagonal(T_YZVW_T, axis1=-2, axis2=-1)   # view; no copy

            # # Multiply by bose_factor along the i-dimension
            # Sp = T_YZVW_T_diag * bose_factor[..., None, None, :]   # (..., 3, 3, 2M)

            # # Reorder to (..., i, a, b) = (..., 2M, 3, 3)
            # Sp = np.moveaxis(Sp, -1, -3)


        # Prepare return objects as tuple
        ret = energies
        if includeS:
            ret = (energies, Sp)

        return ret

    def determine_ES(self, 
                     Qhkl: tuple[float,float,float], 
                     includeS: bool=True) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        '''Determine energies and spin-spin correlation function of excitation with wavevector `Qhkl`.
        Note the shape of the returned arrays, as for k=0 structure we have M modes, but with k!=0
        we have 3*M modes.

        Parameters
        ----------
        Qhkl: (...,3), float
            Momentum transfer at which the excitations are determined
        includeS: bool=True, (optional)
            If True, also determine the non-rotated spin-spin correlation function

        Returns
        -------
        E: (...,3*M/M), ndarray
            Energies of excitations at given momentum transfer
        S: (...,3*M/M,3,3) ndarray, optional
            Single-mode spin-spin correlation function
        
        Notes
        -----
        As in Eq. (40) [SpinW] this is effectively a dispatcher to calculate S'(k, omega)
        for three different k. For non-modulated structures only one is calculated.

        For large `Qpath` arrays the calculations could be pooled into three processes,
        one for each modulation vector (+-0).
        '''
        results = self._determine_ESp(Qhkl, includeS=includeS)

        if includeS:
            k = np.array(self.magnetic_modulation['k'])
            R1, R2 = funs_sw.rot_Rodrigues_complex(self.magnetic_modulation['n'])
            E_0, Sp_0 = results
            if np.allclose(k % 1, [0,0,0]):
                # Special case of (40) [SpinW] with Q=0
                # S = S' * (R1 + R1.conj +R2) = S'
                E = E_0
                S = Sp_0
            else:
                # raise NotImplementedError('spin-spin correlation function for modulated structures.')
                # Here (40) and (39) [SpinW] should be properly incorporated
                # with the modulated parts
                E_plus, Sp_plus = self._determine_ESp(Qhkl + k)
                E_minus, Sp_minus = self._determine_ESp(Qhkl - k)
                # S = Sp @ R2 + Sp_plus @ R1 + Sp_minus @ R1.conj()
                S_0 = R2 @ Sp_0
                S_plus = R1 @ Sp_plus
                S_minus = R1.conj() @ Sp_minus

                E = np.concatenate((E_0, E_plus, E_minus), axis=-1)  # E arrays are (...,M)
                S = np.concatenate((S_0, S_plus, S_minus), axis=-3)  # S arrays are (...,M,3,3)

            ret = (E, S)
        else:
            ret = results

        return ret

    def _determine_ESp_old(self, q_hkl, includeS=True):
        '''Determine characteristics of the excitations of the system.

        Parameters
        ----------
        q_hkl: array_like, (3,)
            Momentum transfer at which the excitations are determined
        includeS: bool
            If True, also determine the non-rotated spin-spin correlation function
            as in [SpinW eq. 47]
        
        Returns
        -------
        E: (2*N,) ndarray
            Energies of excitations at given momentum transfer
        Sp: (2*N,3,3) ndarray, optional
            Single-mode, non-rotated spin-spin correlation function

        Notes
        -----
        I have a dream that this whole function will work fir q_hkl.shape = (N,3) or any at all (...,3).
        That requires either porting to C or writing everything in proper numpy arrays.
        '''
        # TODO
        # Phase trick for consecutive q vectors as in euphonics?
        # Refering to atom indices is unstable, if non-magnetic atoms are introduced its gonna cramble
        N = len(self.magnetic_atoms)

        # logger.info(f"Calculating pseudoHamiltonianian at q={q}")

        # [SpinW] eq 21 and 14
        Jp0 = np.zeros((N,N, 3,3), dtype=np.float64)
        JpofK = np.zeros((N,N, 3,3), dtype=np.complex128)
        for cpl in self.couplings_all:
            Rn = self.rot_Rn(cpl.n_uvw)

            # According to [SpinW] eq 21: Jp = Rm @ J @ Rn,
            # where n,m index unit cells, but m=0 in this notation so is omitted.
            Jp0[cpl.id1, cpl.id2, :,:] += cpl.J @ Rn
            JpofK[cpl.id1, cpl.id2, :,:] += np.exp(2*np.pi*1j*np.dot(q_hkl, cpl.n_uvw)) * (cpl.J @ Rn)    # Eq (52) [spinW]


        # [SpinW] eq 9
        u = np.zeros((N, 3), dtype=complex)
        v = np.zeros((N, 3), dtype=complex)
        S = np.asarray([atom.s for atom in self.magnetic_atoms])
        for atom_i,Sdir_i in enumerate([atom.m for atom in self.magnetic_atoms]):
            Rp_i = self.rot_Rprime(Sdir_i)
            u[atom_i, :] = Rp_i[:,0] + 1j*Rp_i[:,1]
            v[atom_i, :] = Rp_i[:,2]



        # [SpinW] eq 26
        JpofmK = np.conj(JpofK)
        SiSj = np.sqrt(np.einsum('i,j->ij', S, S))
        A1 = 0.5*np.einsum('ij,ip,ijpq,jq->ij', SiSj, u, JpofmK, np.conj(u))
        A2 = 0.5*np.conj(np.einsum('ij,ip,ijpq,jq->ij', SiSj, u, JpofK, np.conj(u)))
        B = 0.5*np.einsum('ij,ip,ijpq,jq->ij', SiSj, u, JpofmK, u)
        C = np.diag(np.einsum('l,ip,ilpq,lq->i', S, v, Jp0, v))

        # [SpinW] eq 27
        h = np.block([
            [A1-C, B],
            [np.conj(B.T), A2-C]
        ])

        if not np.allclose(h, h.T.conj()):
            logger.warning('Hamiltonian kernel is not hermitian.')


        # Hamlitonian kernel h should be positive definite (magnon creation/anihilation energies).
        # In case eigenvalues are negative we have to raise error.
        # In case eigenvalue is zero, we jum to numerical problems and "adding small eps" [SpinW] helps
        energies = np.linalg.eigvalsh(h)

        if np.any( np.abs(energies)<1e-10 ):
            h += np.eye(2*N,2*N) * 1e-10
            logger.warning(f'Zero energy modes at {q_hkl} addin small correction')

        # For cholesky decomposition, do we want upper or lower triangular matrix?
        # - SpinW matlab code:
        # [K, posDef]  = chol(ham(:,:,ii));     ## MS: so K is upper triangular
        # K2 = K*gComm*K';                      ## MS: prime in matlab does the hermitian conjugation
        # K2 = 1/2*(K2+K2');                    ## MS: SNEAKY!
        # % Hermitian K2 will give orthogonal eigenvectors
        # [U, D] = eig(K2);
        # D      = diag(D);
        # % sort modes accordign to the real part of the energy
        # [~, idx] = sort(real(D),'descend');
        # U = U(:,idx);
        # % omega dispersion
        # omega(:, hklIdxMEM(ii)) = D(idx);
        # % the inverse of the para-unitary transformation V
        # V(:,:,ii) = inv(K)*U*diag(sqrt(gCommd.*omega(:, hklIdxMEM(ii)))); %#ok<MINV>
        try:
            # h += np.diag( [1e-8]*2*N ) # don't fuck around like SpinW
            K = scipy.linalg.cholesky(h, lower=False)
        except np.linalg.LinAlgError:
            K = np.zeros(h.shape)
            logger.error(f'Cholesky failed for q_hkl={q_hkl}, h_eig={np.linalg.eigvals(h)}')


        g = np.diag( [1]*N + [-1]*N )       # (30) [SpinW]
        E, U, _ = scipy.linalg.schur(K @ g @ np.conj(K.T), sort='rhp')  # Between (31) (32) [SpinW]

        energies = np.diag(E.real)
        # idE = np.argsort(energies)
# 
        # why are energies not sorted as the g matrix, i.e. first N positive, last N negative?
    
        Sp = np.array([], dtype=complex)
        if includeS:
            # print('DD e', np.sqrt(E))
            try:
                T = np.linalg.inv(K) @ U @ np.sqrt(E)   # (34) [SpinW]
            except np.linalg.LinAlgError:
                logger.warning(f'Failed to invert `T` for q_hkl={q_hkl}')
                T = np.zeros(U.shape)

            # Is this orientation ok?
            phase_ij = [[2*np.pi*1j*np.dot(q_hkl, atom_i.r-atom_j.r)
                        for atom_i in self.magnetic_atoms]
                        for atom_j in self.magnetic_atoms]
            phase_factor = np.exp(phase_ij)

            spin_phase = SiSj * phase_factor


            # energy [meV] temperature [K]
            if self.temperature is None:
                temperature = 1e-5
            else:
                temperature = self.temperature

            bose_factor = 1/np.expm1(np.abs(energies)/(0.08617333262*temperature)) + (np.diag(g)+1)/2

            # (44) [SpinW]
            Y = np.einsum('ij,ia,jb->abij', spin_phase, u,u.conjugate())
            Z = np.einsum('ij,ia,jb->abij', spin_phase, u,u)
            V = np.einsum('ij,ia,jb->abij', spin_phase, u.conjugate(),u.conjugate())
            W = np.einsum('ij,ia,jb->abij', spin_phase, u.conjugate(),u)

            YZVW = np.block([[Y, Z], [V, W]])

            # Kind of (47) [SpinW]
            # I don't do the sum, to be able to extract single-mode spectral weight
            Sp = np.einsum('ij,abjk,ki,i->iab', np.conj(T.T), YZVW, T, bose_factor)


        # Prepare return objects as tuple
        ret = (energies,)

        if includeS:
            ret += (Sp,)


        if len(ret)==1:
            ret = ret[0]

        return ret
        
    def determine_ES_old(self, 
                     q_hkl: tuple[float,float,float], 
                     includeS: bool=True) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        '''Determine energies and spin-spin correlation function of excitation with wavevector `q_hkl`.
        
        Notes
        -----
        As in Eq. (40) [SpinW] this is effectively a dispatcher to calculate S'(k, omega)
        for three different k. For non-modulated structures only one is calculated.

        For large `Qpath` arrays the calculations could be pooled into three processes,
        one for each modulation vector (+-0).
        '''


        results = self._determine_ESp(q_hkl, includeS=includeS)

        if includeS:
            k = np.array(self.magnetic_modulation['k'])
            R1, R2 = funs_sw.rot_Rodrigues_complex(self.magnetic_modulation['n'])
            E_0, Sp_0 = results
            if np.allclose(k % 1, [0,0,0]):
                # Special case of (40) [SpinW] with Q=0
                # S = S' * (R1 + R1.conj +R2) = S'
                E = E_0
                S = Sp_0
            else:
                # raise NotImplementedError('spin-spin correlation function for modulated structures.')
                # Here (40) and (39) [SpinW] should be properly incorporated
                # with the modulated parts
                E_plus, Sp_plus = self._determine_ESp(q_hkl + k)
                E_minus, Sp_minus = self._determine_ESp(q_hkl - k)
                # S = Sp @ R2 + Sp_plus @ R1 + Sp_minus @ R1.conj()
                S_0 = Sp_0 @ R2
                S_plus = Sp_plus @ R1
                S_minus = Sp_minus @ R1.conj()

                E = np.concatenate((E_0, E_plus, E_minus), axis=0)  # E arrays are (M,)
                S = np.concatenate((S_0, S_plus, S_minus), axis=0)  # S arrays are (M,3,3)

            ret = (E, S)
        else:
            ret = results

        return ret

    @ensure_shape(Qhkl=(...,3))
    def calculate_ground_state(self, Qhkl: np.ndarray[float] = None) -> float:
        '''Based on SpinW paper, Eq. 20, term with no bn operators.
        
        Parameters
        ----------
        Qhkl: (...,3), array_like
            Modulation vector for which to calculate the ground state. Default is self modulation vecotr
        '''
        if Qhkl is None:
            Qhkl = self.magnetic_modulation['k']


        *shape, three = np.shape(Qhkl)
        shape = tuple(shape)
        assert three == 3, f'Last dimension of Qhkl must be 3, is {np.shape(Qhkl)=}'
        M = len(self.magnetic_atoms)

        # [SpinW] eq 21 and 14
        Jp = np.zeros(shape+(M,M, 3,3), dtype=np.complex128)
        for cpl in self.couplings_all:
            # According to [SpinW] eq 21: Jp = Rm @ J @ Rn,
            # where n,m index unit cells, but m=0 in this notation so is omitted.
            Rn = funs_sw.rot_Rn(cpl.n_uvw, Qhkl, self.magnetic_modulation['n'])   # (...,3,3)


            # We will be dealing with shape = (shape, 3,3)
            Jp[...,cpl.id1, cpl.id2, :,:] += cpl.J @ Rn


        # [SpinW] eq 9
        v = np.zeros((M, 3), dtype=complex)
        S = np.asarray([atom.s for atom in self.magnetic_atoms])
        SiSj = np.einsum('i,j->ij', S, S)   # shape = (M,M)
        for atom_i,Sdir_i in enumerate([atom.m for atom in self.magnetic_atoms]):
            Rp_i = self.rot_Rprime(Sdir_i)
            v[atom_i, :] = Rp_i[:,2]

        E = 0.5*np.einsum('pq,pi,...pqij,qj->...', SiSj, v, Jp, v)
        
        if np.any(np.abs(E.imag) > 1e-10):
            logger.warning(f'Ground state energy has imaginary part: {E.imag=}')

        return E.real
    
    ##############################################################
    # TODO
    # - Plotting should not be in here. Let's make the `SpinW` class a pure calculator.
    # - Calculations should be wrapped into a single function. It should have split into calculating
    #   spin-spin correlations or not. The simplest output should be enrgies and intensity Sperp.
    #   but in process of calculations internal fields Sab shoould be filled ,and additional acces
    #   point like, powder_Sperp, and full Sab should be available.

    def calculate_excitations(self, Qhkl: np.ndarray,
                              omit_SS: bool=False, silent: bool=True) -> np.ndarray[float]:
        '''Calculate excitation spectrum on `qPath`.
        
        Parameters
        ----------
        Qhkl: (...,3) ndarray
            Array of momentum transfer vectors along which the excitations are calculated.
        omit_SS: bool, optional
            If True, do not calculate the spin-spin correlation function.
        silent: bool, optional
            If True, do not print the progress of the calculation.
            
        Returns
        -------
        energies: (..., 3M/M) ndarray
            Array of energies of excitations along `qPath`, where N is the number of q-points
            and M is the number of magnetic atoms.
        intensities: (..., 3M/M) ndarray, optional
            Array of intensities of excitations along `qPath`, where N is the number of q-points
            and M is the number of magnetic atoms. Only returned if `omit_SS` is False.

        Notes
        -----
        About return values structure
        v1: simple arrays
        v2: structured array for S to wrap the Sperp, and allow to conveniently acces Sij

        ChatGPT helped design a dtype that will have proper ofsets and store both S, Sij and Sperp.
        It looks like an overkill now, implementation might be hard to read.
        Treat it as black box, required to store and access data conveniently.
        '''
        *shape, three = np.shape(Qhkl)
        shape = tuple(shape)
        assert three == 3, f'Last dimension of Qhkl must be 3, is {np.shape(Qhkl)=}'
        exc_dtype = make_exc_dtype()

        if omit_SS:
            E = self.determine_ES(Qhkl, includeS=False)

            excitations = np.rec.array(np.full(shape=E.shape, fill_value=np.nan, dtype=exc_dtype))
            excitations.E = E
        else:
            E, S = self.determine_ES(Qhkl, includeS=True)

            excitations = np.rec.array(np.full(shape=E.shape, fill_value=np.nan, dtype=exc_dtype))
            excitations.E = E
            excitations.S = S

            # Prepare Sperp with perpendicular projection matrix P = 1-Qi*Qj/Q^2
            Qxyz = self.crystal.hkl2xyz(Qhkl)
            Q2 = np.sum(Qxyz**2, axis=-1)
            Q2m1 = np.zeros(shape)
            Q2m1 = np.divide(1, Q2, out=Q2m1, where=(Q2!=0))
            P = np.eye(3) - np.einsum('...i,...j,...->...ij', Qxyz, Qxyz, Q2m1)
            # For stability, where Q=0 we take P from the neighbouring Q

            Sperp = P[..., None, :, :] @ excitations.S.real
            # print(f'{P.shape=} {excitations.S.real.shape=} {Sperp.shape=}')
            excitations.Sperp = np.trace(Sperp, axis1=-2, axis2=-1)

            # Intensity should have also kf/ki f^2(Q) exp(-W)
            # I = [np.trace(Si.real) for Si in S]
            # TODO check perpendicular projection operator
            # I = np.trace(funs_sw.perp_matrix(q_hkl) @ np.array([Si.real for Si in S]), axis1=-2, axis2=-1)
            # intensity = [np.trace(funs_sw.perp_matrix(q_hkl) @ Si.real) for Si in S]
            # self._intensities.append(intensity)


        self.excitations = excitations

        return excitations
    
    def calculate_spectrum(self, energy: np.ndarray, resolution: Union[float,Callable],
                           spectral_weight: np.ndarray = None) -> np.ndarray[float]:
        '''Calculate spectrum of neutron scattering on the `energy` domain and the `resolution_func`.
        The calculations path is based on the previously performed `calculate_excitations` call,
        where the path is the last two dimensions of `Qhkl`.
        
        Parameters
        ----------
        energy: (N,) ndarray
            Array of energies at which the spectrum is calculated.
        resolution: float, Callable
            Resolution of the spectrum. For `float` it is the full-width at half maximum (FWHM) 
            of the gaussian resolution shape. For `Callable` it is a resolution function 
            with signature f(E, E0) -> float, where E is the energy at which the resolution is calculated
            and E0 is the energy of the excitation. E0 is provided to simulate variable resolution with E0.
            The resolution is peaked at zero.
            
        Returns
        -------
        spectrum: (len(self._qPath), N) ndarray
            Array of intensities of the spectrum at the `energy` domain.
        '''

        if np.isscalar(resolution):
            sigma = resolution / 2.35482
            def res_func(E, E0):
                return 1/(sigma*np.sqrt(1*np.pi)) * np.exp(-E**2/(2*sigma**2))
        else:
            res_func = resolution


        # Select one block (shape (N, M)) from possibly higher-dim arrays
        # print(self.excitations.E.shape)
        ss = [0] * (self.excitations.E.ndim - 2) + [slice(None)]*2
        energy = np.asarray(energy)                           # (L,)
        E = self.excitations.E[tuple(ss)]         # (N, M)

        if spectral_weight is None:
            S = self.excitations.Sperp
        else:
            # ensure the shape is alright
            S = spectral_weight

        # Broadcast energy differences: (N, M, L)
        delta = energy[None, None, :] - E[..., None]

        # Evaluate response: (N, M, L)
        R = res_func(delta, E[..., None])

        # Weighted sum over excitations M → result shape (N, L)
        staggered_spectrum = np.sum(S[..., None] * R, axis=1)

        return staggered_spectrum.T
    
    def calculate_powder_spectrum(self, Qrange, NQ, Erange, resolution):
        '''Calculate powder-averaged spectrum over given Q and E ranges.
        
        Ideas:
        Calculate only in a single BZ and then extend it over all directions.
        '''

        ### Prepare Q grid
        Q_cube = np.random.random((NQ, 3))*2 - 1
        Q_norms = np.linalg.norm(Q_cube, axis=1)
        Q_clip = (Q_norms <=1) * (Q_norms > Qrange.min()/Qrange.max())
        Q_ball = Q_cube[Q_clip] * Qrange.max()

        Qhkl = self.crystal.xyz2hkl(Q_ball)

        # E, S = self.calculate_excitations(Q_ball)
        # print('si', self._intensities.shape)
        exc = self.calculate_excitations(Qhkl, omit_SS=False)

        ### Alternatively a uniform grid could be lines along some random directions

        if np.isscalar(resolution):
            sigma = resolution / 2.35482
            def res_func(E, E0):
                return 1/(sigma*np.sqrt(1*np.pi)) * np.exp(-E**2/(2*sigma**2))
        else:
            res_func = resolution

        Q_indices = np.digitize(np.linalg.norm(Q_ball, axis=1), bins=Qrange)

        staggered_spectrum = np.zeros((len(Qrange), len(Erange)))
        for n in range(len(Q_ball)):
            spectrum = np.zeros(len(Erange))
            for E0, I0 in zip(exc.E[n], exc.Sperp[n]):
                spectrum += I0 * res_func(Erange-E0, E0)

            staggered_spectrum[Q_indices[n]] += spectrum

        return staggered_spectrum.T * len(Qrange)/len(Q_ball)


    def plot_dispersion(self, ax: 'plt.Axes', xaxis: np.ndarray=None, 
                        plot_type: str='dispersion', plot_kwargs: dict={},
                        ret_data: bool=False) -> 'plt.Axes':
        '''
        Plot dispersions

        Parameters
        ----------
        ax: pyplot.Axes
            Axes on which to make the plot
        xaxis: numpy.ndarray, optional
            Array of x values for the plot.
        plot_type: str, optional
            Plot type from ['dispersion', 'dispersion_scaled', 'spectral_weight'].
        plot_kwargs: dict
            Additional kwargs passed to the plotting functions.
        ret_data: bool, optional
            If True, return the data used for plotting.

        Returns
        -------
        ax: pyplot.Axes
            Axes with the plot.
        ret_data: list, optional
            If `ret_data` is True, return the data used for plotting.
        '''

        plot_kwargs.setdefault('cmap', 'jet')
        plot_kwargs.setdefault('vmax', None)

        # Nice property of this array is that for any change in direction it will keep the same value
        Qinc = np.concatenate(([0], np.linalg.norm( self.qPath[:-1] - self.qPath[1:], axis=1)))

        if xaxis is not None:
            x_arg = xaxis
        else:
            x_arg = np.cumsum(Qinc)

        ax.set_xlabel('Q ((h,k,l))')
        ax.set_ylabel('E (meV)')

        # Mask to where put the xticks:
        # (1) main qpoints, (2) the last one, (3) integer positions
        it1 = (Qinc==0)

        it2 = np.zeros(len(Qinc), dtype=bool)
        it2[0] = it2[-1] = True

        it3 = (np.linalg.norm(self.qPath - self.qPath.round(), axis=1) == 0)

        xticks_it = it1 | it2 | it3
        xticks = x_arg[xticks_it]
        xtickslabels = ['\n'.join([f'{x:.2f}' for x in q]) for q in self.qPath[xticks_it]]
        ax.set_xticks(xticks, labels=xtickslabels)


        ### Plot type
        if plot_type == 'dispersion':
            Es, Is = self.excitations
            x = x_arg.repeat(2*len(self.magnetic_atoms))

            ax.scatter(x, Es, **plot_kwargs)    # 0 branch

            ret_data = [x_arg, Es]
        elif plot_type == 'dispersion_scaled':
            Es, Is = self.excitations
            Is -= Is.min()

            s = 10 + 100*Is/Is.max()
            c = np.power(Is/Is.max(), 0.1)
            plot_kwargs.pop('alpha', None)
            plot_kwargs.pop('color', None)

            # Flatten objects for plotting
            x = x_arg.repeat(2*len(self.magnetic_atoms))
            y = Es.flatten()
            z = Is.flatten()

            ax.scatter(x, y, s=s, c=c, cmap='magma_r', **plot_kwargs)    # 0 branch
            ret_data = [x, y, s]
        elif plot_type == 'spectral_weight':

            Egrid = []
            def yvals(xvals, Es, Is):
                y = np.zeros(len(xvals))
                for x0, A in zip(Es, Is):
                    sigma = 1 #+ 0.03*x0     # Imitates energy resolution
                    y += funs_sw.gauss_bkg(xvals, x0=x0, A=A/sigma, sigma=sigma, bkg=0)

                return y

            Erange = np.linspace(0, 100, 400)
            Es, Is = self.excitations
            for E, I in zip(Es, Is):
                Egrid.append(yvals(Erange, E, I))

            Egrid = np.transpose(Egrid)


            cmap = ax.pcolormesh(x_arg, Erange, Egrid, cmap=plot_kwargs['cmap'], vmax=plot_kwargs['vmax'])
            ret_data = [x_arg, Erange, Egrid]
        else:
            raise KeyError(f"Unknown plot_type {plot_type!r}")

        # Set up the return values
        ret = ax
        if ret_data:
            ret = (ax, ret_data)

        return ret
    
    def __repr__(self):
        rr = 'SpinW(\n'
        rr += self.crystal.__repr__() + '\n'
        rr += 'Couplings = '
        rr += self.couplings_all.__repr__()
        rr += '\t})'
        return rr


# TODO
# [X] Extend to multiple magnetic atoms per unit cell
# atoms:
#   - [] make spins aligned with the crystal axes
#
# modulation:
#   - [] declare conventions
#
# couplings:
#   - [X] add symmetrization option
##
#
# calculate_excitation(qPath)
# plot_dispersion at different gammas
#
# Potential traps?
# 1. [X] Solved by finding unique couplings
#    Couplings generated with ['2z','-1'] symmetry. For atom at [u,0,0] both operations will transform
#    it into [-u,0,0], but metrix representations are gonn be different. Thus there will be two couplings involved?

if __name__ == '__main__':
    print('main')

    @ensure_shape(a=(..., 3), b=(2, ...), c=(2, 3, ..., 3, 3))
    def my_function(a, b=None, c=None):
        return np.shape(a), b.shape, c.shape


    A = np.ones((5, 3))              # matches (..., 3)
    B = np.ones((2, 4, 5, 6))        # matches (2, ...)
    C = np.ones((2, 3, 4, 5, 3, 3))  # matches (2, 3, ..., 3, 3)

    # print(my_function(A, B, C))  # ✅ works

    D = np.ones((4, 3))
    my_function([1,2,3], B, C)  # ❌ raises ValueError