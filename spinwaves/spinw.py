# Core
from scipy.linalg import schur, cholesky
import numpy as np

import logging
import logging.config

# Plotting


# Typing
from typing import List, Tuple, Dict, TYPE_CHECKING, Union
from numpy.typing import NDArray

from spinwaves import crystal

if TYPE_CHECKING:
    from .crystal import Crystal
    import matplotlib.pyplot as plt


# Internal
from . import functions as funs_sw
# from .lattice import Lattice
# from .plotting_vispy import SupercellPlotter

# import plotting
# from .plotting import plot_supercell, implemented_sc_plotters

# from coupling import couplings

# Tobi confirmed the factor of two is missing from single-ion naisotropies.
# He also mentioned the inverted sign mistake in the phase factor somewhere
# in the spin-spin correlation function calculations.

logger = logging.getLogger('SpinW')

# setup_logging
logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            # 'format': '%(levelname)s: %(message)s'
            'format': '[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s.%(msecs)03d: %(message)s',
            'datefmt': '%Y-%m-%dT%H:%M:%S%z'
        }
    },
    'handlers': {
        'stdout': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        }
    },
    'loggers': {
        'root': {
            'level': 'WARNING',
            'handlers': ['stdout']
        }
    }
}
logging.config.dictConfig(logging_config)

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
                 J: np.ndarray[float], defining_bond: str=''):
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
    # Functionalities

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
    '''
    Main object used to calculate spin waves.

    Fields:
    -------

    lattice:
        `Lattice` object
    unit_cell:
        Dictionary       
    magnetic_structure:
        [k, n, spins]
    couplings: 
        List[d, i, j, J]



    TODO
    ----
        - Clear convention of coordinate systems, crystal vs cartesian.
        - Clarify the calculation and definitions of spectral weight and intensity.
          I like the notation from gen Shirane book, I \propto kf/ki f^2(Q) exp(-W) S_perp
          I dont understand where DW-factor is coming from. Detailed balance is in S
    '''
    def __init__(self, crystal: 'Crystal', magnetic_modulation: Dict, couplings: List[Coupling]):
        '''
        lattice: Lattice
        unit_cell: UnitCell
        magnetic_str: [k, n, spins]
        couplings: List[Coupling]
        '''
        self.crystal = crystal
        self.magnetic_atoms = crystal.atoms_magnetic
        self.magnetic_modulation = magnetic_modulation

        # Ensure input couplings are unique, otherwise symmetrization doesn't work well
        self.couplings = couplings
        self.couplings_all =  self.symmetrize_couplings(couplings)

        # Might want to update the g_tensor of each atom now

    def symmetrize_couplings(self, couplings: list['Coupling']) -> list['Coupling']:
        '''Generalize the couplings according to the symmetry of the crystal.

        Parameters
        ----------
        couplings: list[`Couplings`]
            List of unique couplings, subject to symmetrization

        Returns
        -------
        couplings_symmetrized: list[`Coupling`]
            Lsit of symmetrized couplings

        Raises
        ------
        ValueError
            When the DMI vector of the coupling does not respect the crystal symmetry.
        
        Notes
        -----
        Algorithm
        1. Create all couplings by nesting loops
            - For each coupling in the list:
                - For each symmetry element of MSG:
                    - Symmetrize the coupling, by applying the symmetry operation.
                    - Append to list of all couplings
        2. Find only unique couplings in the list
        3. Check if the symmetry constraints are obeyed by looking at equivalent couplings
        '''
        couplings_all = []
        # [1]
        for cpl in couplings:
            atom1 = self.crystal.atoms_magnetic[cpl.id1]
            atom2 = self.crystal.atoms_magnetic[cpl.id2]

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

                new_J = g.matrix @ cpl.J @ g.inv().matrix

                cpl_new_12  = Coupling(label=f'{cpl.label}_{n}',
                                n_uvw=new_n_uvw_12,
                                id1=new_id1,
                                id2=new_id2,
                                J = new_J,
                                defining_bond=cpl.label)
                cpl_new_21  = cpl_new_12.revert()


                couplings_all.extend([cpl_new_12, cpl_new_21])

        # Ensure the identity is in the unique list -> HOW?
        couplings_unique, id_inverse = np.unique(couplings_all, return_inverse=True)

        # Check symmetry condidtion
        for id_unique, cpl_unique in enumerate(couplings_unique):
            id_equivalent = np.where(id_inverse==id_unique)[0]
            J_equivalent = [couplings_all[id].J for id in id_equivalent]
            J_averaged = np.average(J_equivalent, axis=0)
            
            if not np.allclose( cpl_unique.J, J_averaged):
                warning_message = f'The following coupling does not respect the symmetry\n\t{cpl}\n'
                warning_message+= f'\tIs:        {cpl_unique.J.tolist()}\n'
                warning_message+= f'\tShould be: {J_averaged.tolist()}'
               
                logger.warning(warning_message)

        logger.info(f'Symmetrization report: generated / unique / provided = {len(couplings_all)} / {len(couplings_unique)} / {len(couplings)}')
        return sorted(couplings_unique, key=lambda cpl: cpl.label)
        
    #########################################################################################
    # Fundamental calculations

    def rot_Rn(self, n_uvw: tuple[int,int,int]) -> np.ndarray:
        '''Matrix correspondint to rotation of the spins in the `n_uvw` unit cell
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
        '''Matrix correspondint to rotation of the spin towards the `z` axis.

        Parameters
        ----------
        S: (3,) int
            Spi ndirection vector
        
        Notes
        -----
        Wrapper around the fundamental function from `functions.rot_Rprime`.
        '''

        return funs_sw.rot_Rprime(S)

    def _determine_ESp(self, q_hkl, includeS=True):
        '''Determine characteristics of the excitations of the system.

        Parameters
        ----------
        q_hkl: array_like
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

        # Avoid numerical problems, TODO global config

        h_eigvals = np.linalg.eigvals(h)
        if np.any( h_eigvals < 0 ):
            # Take smallest negative eigenvalue and try to correct for it
            corr = -np.min(h_eigvals.real)
            if corr > 1e-10: 
                logger.error(f'h(q_hkl={q_hkl}) is negative')
            else:
                h += np.diag( [1e-10]*2*N ) # don't fuck around like SpinW
            

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
            K = cholesky(h, lower=False)
        except np.linalg.LinAlgError:
            K = np.zeros(h.shape)
            logger.error(f'Cholesky failed for q_hkl={q_hkl}')


        g = np.diag( [1]*N + [-1]*N )       # (30) [SpinW]
        E, U, _ = schur(K @ g @ np.conj(K.T), sort='rhp')  # Between (31) (32) [SpinW]

        energies = np.diag(E.real)
        # why are energies not sorted as the g matrix, i.e. first N positive, last N negative?

        Sp = np.array([], dtype=complex)
        if includeS:
            # print('DD e', np.sqrt(E))
            try:
                T = np.linalg.inv(K) @ U @ np.sqrt(np.abs(E))   # (34) [SpinW]
            except np.linalg.LinAlgError:
                T = np.zeros(U.shape)

            # Is this orientation ok?
            phase_ij = [[2*np.pi*1j*np.dot(q_hkl, atom_i.r-atom_j.r)
                        for atom_i in self.magnetic_atoms]
                        for atom_j in self.magnetic_atoms]
            phase_factor = np.exp(phase_ij)

            spin_phase = SiSj * phase_factor

            # energy [meV] temperature [K]
            temperature = 1.5
            bose_factor = 1/(np.exp(np.abs(energies)/(0.08617333262*temperature))-1) + (np.diag(g)+1)/2

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
    
        
    def determine_ES(self, 
                     q_hkl: tuple[float,float,float], 
                     includeS: bool=True) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        '''Determine energies and spin-spin correlation function of excitation with wavevector `q_hkl`.
        
        Notes
        -----
        As in Eq. (40) [SpinW] this is effectively a dispatcher to calculate S'(k, omega)
        for three different k. For non-modulated structures only one is calculated.
        '''


        results = self._determine_ESp(q_hkl, includeS=includeS)

        if includeS:
            k = np.array(self.magnetic_modulation['k'])
            R1, R2 = funs_sw.rot_Rodrigues_complex(self.magnetic_modulation['n'])
            E, Sp = results
            if np.allclose(k % 1, [0,0,0]):
                S = Sp @ R2
            else:
                raise NotImplementedError('spin-spin correlation function for modulated structures.')
                # Here (40) and (39) [SpinW] should be properly incorporated
                # with the modulated parts
                E_plus, Sp_plus = self._determine_ESp(q_hkl + k)
                E_minus, Sp_minus = self._determine_ESp(q_hkl - k)
                S = Sp @ R2 + Sp_plus @ R1 + Sp_minus @ R1.conj()

            ret = (E, S)
        else:
            ret = results

        return ret


    def calculate_ground_state(self, q_hkl=np.ndarray[float]) -> float:
        '''Based on SpinW paper, Eq. 20, term with no bn operators'''

        # Iterate through all atoms to prepare objects required in Eq (26) [spinW]
        # sqrt(S_i) is incorporated into u_i
        Jp0 = np.zeros((len(self.magnetic_atoms), len(self.magnetic_atoms), 3, 3), dtype=np.float64)
        JpofK = np.zeros((len(self.magnetic_atoms), len(self.magnetic_atoms), 3, 3), dtype=np.complex128)
        for cpl in self.couplings_all:
            Rn = self.rot_Rn(cpl.n_uvw)

            Jp0[cpl.id1, cpl.id2, :,:] += cpl.J @ Rn
            JpofK[cpl.id1, cpl.id2, :,:] += np.exp(2*np.pi*1j*np.dot(q_hkl, cpl.n_uvw)) * (cpl.J @ Rn)    # Eq (52) [spinW]

            # Just as in `determine_h()` below should be obsolete now
            # Need to fill Jij(k) and Jji(k). Jji = np.conj(Jij^T)
            # if atom_i != atom_j:
            # Jp0[cpl.at2, cpl.at1, :,:] += np.conj(cpl.J @ Rn).T
            # JpofK[cpl.at2, cpl.at1, :,:] += np.conj(np.exp(1j*np.dot(k, cpl.d_xyz)) * (cpl.J @ Rn)).T

        v = np.zeros((len(self.magnetic_atoms), 3), dtype=complex)
        S = np.asarray([atom.s for atom in self.magnetic_atoms])
        SiSj = np.einsum('i,j->ij', S, S)

        for atom_i,Sdir_i in enumerate([atom.m for atom in self.magnetic_atoms]):
            Rp_i = self.rot_Rprime(Sdir_i)
            v[atom_i, :] = Rp_i[:,2]

        E = 0.5*np.einsum('ij,ip,ijpq,jq->ij', SiSj, v, JpofK, v)

        # print(E)

        return np.real(np.sum(E))
    
    
    ##############################################################
    def calculate_excitations(self, qPath: np.ndarray, silent=True) -> np.ndarray[float]:
        Es = []
        for q_hkl in qPath:
            E = self.determine_ES(q_hkl, includeS=False)
            Es.append(E)

        self.qPath = qPath
        self.excitations = np.asarray(Es, dtype=float)

        return self.excitations
    
    def calculate_spectrum(self, qPath: np.ndarray, silent=True) -> np.ndarray[float]:
        Es, Ss, Is = [], [], []
        for q_hkl in qPath:
            E, S = self.determine_ES(q_hkl, includeS=True)
            Es.append(E)
            Ss.append(S)

            # Intensity should have also kf/ki f^2(Q) exp(-W)
            # I = [np.trace(Si.real) for Si in S]
            # TODO check perpendicular projection operator
            I = [np.trace(funs_sw.perp_matrix(q_hkl) @ Si.real) for Si in S]
            Is.append(I)

            # print(q_hkl, np.shape(I), E, I)

        self.qPath = qPath
        self.excitations = ( np.asarray(Es, dtype=float), np.asarray(Is, dtype=float) )

        return self.excitations

    
    def plot_dispersion(self, ax: 'plt.Axes', xaxis: str=None, plot_type: str='dispersion', plot_kwargs: dict={}) -> 'plt.Axes':
        '''
        Plot dispersions
        '''

        # Nice property of this array is that for any change in direction it will keep the same value
        Qinc = np.concatenate(([0], np.linalg.norm( self.qPath[:-1] - self.qPath[1:], axis=1)))

        Qs = np.cumsum(Qinc)
        x_arg = Qs

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
            Es = self.excitations
            ax.scatter(x_arg, Es, **plot_kwargs)    # 0 branch
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
        elif plot_type == 'spectral_weight':

            Egrid = []
            def yvals(xvals, Es, Is):
                y = np.zeros(len(xvals))
                for x0, A in zip(Es, Is):
                    sigma = 1 + 0.03*x0
                    y += funs_sw.gauss_bkg(xvals, x0=x0, A=A/sigma, sigma=sigma, bkg=0)

                return y

            Erange = np.linspace(0, 100, 100)
            Es, Is = self.excitations
            for E, I in zip(Es, Is):
                Egrid.append(yvals(Erange, E, I))

            cmap = ax.pcolormesh(x_arg, Erange, np.transpose(Egrid), cmap='gist_heat_r')
        else:
            raise KeyError(f"Unknown plot_type {plot_type!r}")

        return ax
    
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
#
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