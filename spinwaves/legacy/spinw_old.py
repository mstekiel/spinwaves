'''Deprecated implementation of `SpinW` methods, kept for reference.

Not imported by `spinwaves`; import explicitly, e.g.
`from spinwaves.legacy.spinw_old import determine_ES_old`.
Requires `scipy`, which is not a dependency of the core library
(`uv sync --group dev`).

Functions take the `SpinW` instance as the first argument `sws`,
in place of `self` of the original methods.
'''
import logging

from typing import TYPE_CHECKING, Union

import numpy as np

try:
    import scipy.linalg
except ImportError as e:
    raise ImportError('`spinwaves.legacy` requires `scipy`. '
                      'Install the `dev` dependencies: `uv sync --group dev`.') from e

from ..coupling import Coupling
from ..utils.linalg import rot_Rodrigues_complex

if TYPE_CHECKING:
    from ..spinw import SpinW


logger = logging.getLogger('SpinW')


def symmetrize_couplings_old(sws: 'SpinW', couplings: list['Coupling']) -> list['Coupling']:
    couplings_all = []
    # [1]
    for cpl in couplings:
        atom1 = sws.magnetic_atoms[cpl.id1]
        atom2 = sws.magnetic_atoms[cpl.id2]

        # Check that coupled atoms are both magnetic
        if not (atom1.is_mag and atom2.is_mag):
            logger.error(f'Coupling a non magnetic atom:\n\t{atom1}\n\t{atom2}')
            raise KeyError(f'Coupling a non magnetic atom: {atom1} {atom2}')
        for n,g in enumerate(sws.crystal.MSG):
            r1 = g.transform_position(atom1.r)
            n_uvw1 = np.floor(r1)
            new_id1 = sws.crystal.get_magatom_id(r1)

            r2 = g.transform_position(atom2.r+cpl.n_uvw)
            n_uvw2 = np.floor(r2)
            new_id2 = sws.crystal.get_magatom_id(r2)

            new_n_uvw_12 = n_uvw2 - n_uvw1

            # g matrix has to be represented in xyz coordinates
            g_xyz = sws.crystal.A @ g.matrix @ np.linalg.inv(sws.crystal.A)
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
        # r_cpl = (sws.magnetic_atoms[cpl_unique.id1].r + sws.magnetic_atoms[cpl_unique.id2].r + cpl_unique.n_uvw) / 2
        # cpl_point_group = sws.crystal.MSG.get_point_symmetry(r_cpl)
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


def determine_ESp_old(sws: 'SpinW', q_hkl, includeS=True):
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
    N = len(sws.magnetic_atoms)

    # logger.info(f"Calculating pseudoHamiltonianian at q={q}")

    # [SpinW] eq 21 and 14
    Jp0 = np.zeros((N,N, 3,3), dtype=np.float64)
    JpofK = np.zeros((N,N, 3,3), dtype=np.complex128)
    for cpl in sws.couplings_all:
        Rn = sws.rot_Rn(cpl.n_uvw)

        # According to [SpinW] eq 21: Jp = Rm @ J @ Rn,
        # where n,m index unit cells, but m=0 in this notation so is omitted.
        Jp0[cpl.id1, cpl.id2, :,:] += cpl.J @ Rn
        JpofK[cpl.id1, cpl.id2, :,:] += np.exp(2*np.pi*1j*np.dot(q_hkl, cpl.n_uvw)) * (cpl.J @ Rn)    # Eq (52) [spinW]


    # [SpinW] eq 9
    u = np.zeros((N, 3), dtype=complex)
    v = np.zeros((N, 3), dtype=complex)
    S = np.asarray([atom.s for atom in sws.magnetic_atoms])
    for atom_i,Sdir_i in enumerate([atom.m for atom in sws.magnetic_atoms]):
        Rp_i = sws.rot_Rprime(Sdir_i)
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
    # K2 = 1/2*(K2+K2');                    ## MS: SNEAKY! hermitization of K, while it should be hermitian already. Numerics?
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
                    for atom_i in sws.magnetic_atoms]
                    for atom_j in sws.magnetic_atoms]
        phase_factor = np.exp(phase_ij)

        spin_phase = SiSj * phase_factor


        # energy [meV] temperature [K]
        if sws.temperature is None:
            temperature = 1e-5
        else:
            temperature = sws.temperature

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


def determine_ES_old(sws: 'SpinW',
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


    results = determine_ESp_old(sws, q_hkl, includeS=includeS)

    if includeS:
        k = np.array(sws.magnetic_modulation['k'])
        R1, R2 = rot_Rodrigues_complex(sws.magnetic_modulation['n'])
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
            E_plus, Sp_plus = determine_ESp_old(sws, q_hkl + k)
            E_minus, Sp_minus = determine_ESp_old(sws, q_hkl - k)
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
