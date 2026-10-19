'''Engine for the Linear Spin Wave Theory calculations of spin dynamics.

I realized how important the safety checks are for setting up the systems.
Unfortunately, some of them require additional calculations and slow down
the execution.
How about implementing a `speedrun` option for functions, which will omit 
heavy checks.
'''
# Core
# from copy import deepcopy
import logging

# Typing
from typing import TYPE_CHECKING, Callable, Dict, List, Sequence, Union

import numpy as np

from .coupling import Coupling

# Internal
from .crystal import Crystal
from .utils.arrays import ensure_shape, make_exc_dtype
from . import config
from .config import NumericalToleranceError
from .utils.linalg import (RfromZ, rot_Rn, rot_Rodrigues_complex,
                           dagger, ensure_hermitian, regularized_cholesky, hermitian_eig)
from .utils.physics import bose_occupation

if TYPE_CHECKING:
    from .symmetry.magnetic_symmetry import mSymOp
    from .crystal import Crystal


logger = logging.getLogger('SpinW')

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
    - pyLiSW is great inspiration. Split _determine_ESp into hamiltonian prep and intensity calc.
    - add functionality of magnetic field
    - seems like we also have all components to determine the bond symmetry in the symbolic picture.
      Take a looka at https://doi.org/10.1103/fgc1-5blp
    - Clarify the calculation and definitions of spectral weight and intensity.
      I like the notation from gen Shirane book, I propto kf/ki f^2(Q) exp(-W) S_perp
      I dont understand where DW-factor is coming from. Detailed balance is in S
    - add functionalities to force certain magnetic structure.
      Feels like it should be after the constructor. Maybe I can also figure out which symmetries are broken.
    - understand total moment sum rule
    '''
    def __init__(self, crystal: 'Crystal', magnetic_modulation: Dict, couplings: List[Coupling],
                 temperature: float=0, Hfield: Sequence=None):
        '''Create the instance of the LSW calculator.

        Parameters
        ----------
        crystal: 'Crystal'
            `Crystal` object holding information on position and magnetic moment of atoms as well as crystal symmetry.    
        magnetic_modulation: dict[str, tuple[float,float,float]]
            Dictionary holding information on propagation vector `k` and normal vector defining rotation plane `n`.
        couplings: list['Coupling']
            List of unique couplings defining interactions between magnetic atoms.
        Hfield: float
            External magnetic field value in units of meV, such that it describes
            the term `mu_B H` and can match the `J` units.

        '''
        self.crystal = crystal
        self.magnetic_atoms = crystal.atoms_magnetic
        self.magnetic_modulation = magnetic_modulation  # check dimensionns?

        # Ensure input couplings are unique, otherwise symmetrization doesn't work well
        self.couplings = couplings
        self.couplings_all =  self.symmetrize_couplings(couplings)

        self.temperature = temperature

        # Check that all atoms have g-tensor defined
        self.Hfield = None
        if Hfield is not None:
            if not all([atom.g_tensor is not None for atom in self.magnetic_atoms]):
                raise ValueError('Applied magnetic field requires definition of g-tensor for all magnetic atoms.')

            self.Hfield = np.asarray(Hfield, dtype=float)

            


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
            # atom1 = self.magnetic_atoms[cpl.id1]
            # atom2 = self.magnetic_atoms[cpl.id2]

            # # Check that coupled atoms are both magnetic
            # if not (atom1.is_mag and atom2.is_mag):
            #     logger.error(f'Coupling a non magnetic atom:\n\t{atom1}\n\t{atom2}')
            #     raise KeyError(f'Coupling a non magnetic atom: {atom1} {atom2}')
            
            def transform_coupling(g: 'mSymOp', cpl: 'Coupling'):
                atom1 = self.magnetic_atoms[cpl.id1]
                atom2 = self.magnetic_atoms[cpl.id2]

                r1 = g.transform_position(atom1.r)
                n_uvw1 = np.floor(r1)
                new_id1 = self.crystal.get_magatom_id(r1)

                r2 = g.transform_position(atom2.r+cpl.n_uvw)
                n_uvw2 = np.floor(r2)
                new_id2 = self.crystal.get_magatom_id(r2)

                new_n_uvw_12 = n_uvw2 - n_uvw1

                # g matrix has to be represented in xyz coordinates
                g_xyz = self.crystal.A @ g.matrix @ np.linalg.inv(self.crystal.A)
                new_J = g_xyz @ cpl.J @ np.linalg.inv(g_xyz)

                # If self interaction, anisotropy
                # we need to correct for the unnecesary double counting
                if np.allclose(r1, r2):
                    logger.info(f'Correcting for double-counting, {cpl}')
                    new_J *= 2

                cpl_new  = Coupling(label=f'{cpl.label}_({g.to_string()})',
                                n_uvw=new_n_uvw_12,
                                id1=new_id1,
                                id2=new_id2,
                                J = new_J,
                                defining_bond=cpl.label)

                return cpl_new
            

            couplings_symmetrized = self.crystal.MSG.symmetrize(cpl, transform_coupling, check_attrs=['J'])
            couplings_all.extend(couplings_symmetrized)

            couplings_symmetrized = self.crystal.MSG.symmetrize(cpl.revert(), transform_coupling, check_attrs=['J'])
            couplings_all.extend(couplings_symmetrized)


        # Could it be that the bond reversal is not always working nicely as here?
        # couplings_all.extend([c.revert() for c in couplings_all]) # The bond reversal should be done already in symmetrization
        # couplings_unique = couplings_all
        couplings_unique = np.unique(couplings_all)

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

        return rot_Rn(n_uvw, self.magnetic_modulation['k'], self.magnetic_modulation['n'])
    
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

        return RfromZ(S)
    
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
        # Indices `ij` are magnetic atoms, `pq` are cartesian directions
        # I know that tensordot can be faster here
        JpofmK = np.conj(JpofK)
        SiSj = np.sqrt(np.einsum('i,j->ij', S, S))
        A1 = 0.5*np.einsum('ij,ip,...ijpq,jq->...ij', SiSj, u, JpofmK, np.conj(u))
        A2 = 0.5*np.conj(np.einsum('ij,ip,...ijpq,jq->...ij', SiSj, u, JpofK, np.conj(u)))
        B = 0.5*np.einsum('ij,ip,...ijpq,jq->...ij', SiSj, u, JpofmK, u)
        C = np.diag(np.einsum('l,ip,ilpq,lq->i', S, v, Jp0, v))

        # [SpinW] eq 28
        # Zeeman term, added to A in the [SpinW] here it is supplied to A1 and A2
        # Hfield must be along n, to avoid umklapp terms
        # as per [SpinW] text between eq. (27) and (28)
        if self.Hfield is not None:
            gi = np.asarray([atom.g_tensor for atom in self.magnetic_atoms])    # shape=(M,3,3)
            aZ = np.einsum('p,ipq,iq->i', self.Hfield, gi, v) # shape=(M)
            AZ = -0.5*np.diag(aZ)   # shape=(M,M)

            A1 += AZ
            A2 += AZ

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

        # Hermitian by construction: deviations above the numerical tolerance are bugs
        h = ensure_hermitian(self._construct_h(Qhkl), name='h')

        # SpinW (lines 812-840 spinwave.m) adds a tolerance to the diagonal when the Cholesky
        # decomposition fails on zero modes. Here definiteness of `h` is checked against
        # the global numerical tolerance, see `spinwaves.config` and `regularized_cholesky`.
        try:
            K_lo = regularized_cholesky(h, name='h')
        except NumericalToleranceError as e:
            Q_failing = [np.asarray(Qhkl)[idx].tolist() for idx in e.indices[:5]]
            raise NumericalToleranceError(
                f'{e}\nFirst failing Qhkl: {Q_failing}. Most probably the magnetic structure '
                'is not the ground state of the exchange interactions.',
                deviation=e.deviation, tolerance=e.tolerance, indices=e.indices) from e

        K_up = dagger(K_lo)

        # (30) [SpinW]
        g = np.diag( [1]*M + [-1]*M )

        # Between (31) (32) [SpinW], Hermitian by construction.
        # Eigenvalues are in ascending order: the M negative ones come first.
        energies, U = hermitian_eig(K_up @ g @ K_lo, name='K^dagger g K')

        if includeS:
            # (34) [SpinW], U @ sqrt(diag(E)) as scaling of the columns of U
            T = np.linalg.inv(K_up) @ (U * np.sqrt(energies.astype(np.complex128))[..., None, :]) # shape=(...,2M,2M)

            r_i = np.asarray([atom_i.r for atom_i in self.magnetic_atoms])  # shape=(M,3)
            r_ij = r_i[None,:,:] - r_i[:,None,:]    # shape=(M,M,3)
            phase_factor = np.exp(2*np.pi*1j*np.tensordot(Qhkl, r_ij, axes=(-1,-1)))    # shape=(...,M,M)

            S = np.asarray([atom.s for atom in self.magnetic_atoms])  # shape=(M,)
            SiSj = np.sqrt(S[None,:] * S[:,None])  # shape=(M,M)
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
            R1, R2 = rot_Rodrigues_complex(self.magnetic_modulation['n'])
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
        M = len(self.magnetic_atoms)

        # [SpinW] eq 21 and 14
        Jp = np.zeros(shape+(M,M, 3,3), dtype=np.complex128)
        for cpl in self.couplings_all:
            # According to [SpinW] eq 21: Jp = Rm @ J @ Rn,
            # where n,m index unit cells, but m=0 in this notation so is omitted.
            Rn = rot_Rn(cpl.n_uvw, Qhkl, self.magnetic_modulation['n'])   # (...,3,3)

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
        
        scale = np.max(np.abs(E), initial=0) or 1.0
        if np.any(np.abs(E.imag) > config.tolerances.numeric * scale):
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
                              omit_SS: bool=False, silent: bool=True) -> np.recarray[np.void]:
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
        excitations:
            Record array with fields corresponding to the calculated excitations.
            Contains at least the field `E` for energies, and optionally `S` for spin-spin correlation function.
            If `S` is present also `Sperp` is calculated, as well as individual components of `S`-> `Sij`
            for cartesian ij. 

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
            # excitations.Qhkl = Qhkl   # one dim has to be stack M times to match excitations number
            excitations.E = E
        else:
            E, S = self.determine_ES(Qhkl, includeS=True)

            excitations = np.rec.array(np.full(shape=E.shape, fill_value=np.nan, dtype=exc_dtype))
            # excitations.Qhkl = Qhkl   # one dim has to be stack M times to match excitations number
            excitations.E = E
            excitations.S = S

            # Prepare Sperp with perpendicular projection matrix P = 1-Qi*Qj/Q^2
            Qxyz = self.crystal.hkl2xyz(Qhkl)
            Q2 = np.sum(Qxyz**2, axis=-1)
            Q2m1 = np.zeros(shape)
            Q2m1 = np.divide(1, Q2, out=Q2m1, where=(Q2!=0))
            P = np.eye(3) - np.einsum('...i,...j,...->...ij', Qxyz, Qxyz, Q2m1)

            Sperp = P[..., None, :, :] @ excitations.S.real
            # TODO I had cases where Sperp resulted in large negative values.
            # I put abs on it then, but is it physical?
            excitations.Sperp = np.abs(np.trace(Sperp, axis1=-2, axis2=-1))


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
        spectral_weight: (len(self._qPath), M) ndarray, optional
            Spectral weight to use instead of the calculated `self.excitations.Sperp`
            
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

        Notes:
        Now take random points in a sphere and digitizes them.
        
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
    
    def __repr__(self):
        rr = 'SpinW(\n'
        rr += self.crystal.__repr__() + '\n'
        rr += 'Couplings = '
        rr += self.couplings_all.__repr__()
        rr += '\t})'
        return rr


if __name__ == '__main__':
    print('main')

    @ensure_shape(a=(..., 3), b=(2, ...), c=(2, 3, ..., 3, 3))
    def my_function(a, b=None, c=None):
        return np.shape(a), b.shape, c.shape


    A = np.ones((5, 3))              # matches (..., 3)
    B = np.ones((2, 4, 5, 6))        # matches (2, ...)
    C = np.ones((2, 3, 4, 5, 3, 3))  # matches (2, 3, ..., 3, 3)

    # print(my_function(A, B, C))  # works

    D = np.ones((4, 3))
    my_function([1,2,3], B, C)  # raises ValueError