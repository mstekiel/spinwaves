# -*- coding: utf-8 -*-
r"""Tests of `spinwaves.utils.linalg`"""
import logging

import numpy as np
import pytest

import spinwaves.utils.linalg as sw_linalg
from spinwaves.config import NumericalToleranceError, override_tolerances
from spinwaves.utils.linalg import dagger, offdiagonal, cross_matrix

from ..markers import needs_scipy


ATOL = 1e-12

@pytest.fixture
def rng():
    return np.random.default_rng(20251019)

def random_hermitian(rng, n: int, batch: tuple=()) -> np.ndarray:
    X = rng.normal(size=batch+(n,n)) + 1j*rng.normal(size=batch+(n,n))
    return (X + dagger(X)) / 2

def random_unitary(rng, n: int) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n)))
    return Q

def spinw_kernel(rng, M: int) -> np.ndarray:
    '''K^dagger g K, with h = K K^dagger positive definite, as in `SpinW._determine_ESp`'''
    X = random_hermitian(rng, 2*M)
    K_lo = np.linalg.cholesky(X @ X + np.eye(2*M))
    return dagger(K_lo) @ np.diag([1]*M + [-1]*M) @ K_lo

def is_rotation(R: np.ndarray) -> bool:
    return np.allclose(R @ R.swapaxes(-2,-1), np.eye(3)) and np.allclose(np.linalg.det(R), 1)

VECTORS = [
    [0, 0, 1],
    [1, 0, 0],
    [0, -1, 0],
    [0.3, -0.2, 0.4],
    [-np.sqrt(2), 1e5, -np.pi],
    [1, 1, 1],
    [0, 0, -2],
]


#########################################################################################
# Matrix helpers

def test_dagger(rng):
    A = rng.normal(size=(2,3,4,4)) + 1j*rng.normal(size=(2,3,4,4))
    assert dagger(A).shape == A.shape
    assert np.allclose(dagger(A)[1,2], A[1,2].conj().T)
    assert np.allclose(dagger(dagger(A)), A)

def test_dagger_hermitian(rng):
    A = random_hermitian(rng, 5)
    assert np.allclose(dagger(A), A)

def test_offdiagonal(rng):
    T = rng.normal(size=(3,4,4))
    O = offdiagonal(T)
    assert np.allclose(np.diagonal(O, axis1=-2, axis2=-1), 0)
    assert np.allclose(O + np.diagonal(T, axis1=-2, axis2=-1)[...,None]*np.eye(4), T)
    assert np.allclose(offdiagonal(np.diag([1., 2, 3])), 0)

def test_cross_matrix(rng):
    n, v = rng.normal(size=(2,3))
    N = cross_matrix(n)
    assert np.allclose(N @ v, np.cross(n, v))
    assert np.allclose(N, -N.T)
    assert np.allclose(N, sw_linalg.DMI(n).T)       # DMI(d) = -[d]_x


#########################################################################################
# ensure_hermitian

def test_ensure_hermitian_exact(rng):
    A = random_hermitian(rng, 4)
    assert np.array_equal(sw_linalg.ensure_hermitian(A), A)

def test_ensure_hermitian_symmetrizes_noise(rng, caplog):
    """Below the tolerance the Hermitian part is returned and the correction logged"""
    A = random_hermitian(rng, 4)
    noisy = A + 1e-14 * rng.normal(size=(4,4))
    with caplog.at_level(logging.DEBUG, logger='Numerics'):
        result = sw_linalg.ensure_hermitian(noisy)

    assert np.array_equal(result, dagger(result))
    assert np.allclose(result, A, atol=1e-13)
    assert 'Symmetrized' in caplog.text

def test_ensure_hermitian_raises(rng):
    A = random_hermitian(rng, 4)
    A[0,1] += 1e-3
    with pytest.raises(NumericalToleranceError, match='not Hermitian') as excinfo:
        sw_linalg.ensure_hermitian(A, name='test')

    assert '`test`' in str(excinfo.value)
    assert excinfo.value.deviation > excinfo.value.tolerance
    assert excinfo.value.indices == [()]

@pytest.mark.parametrize('scale', [1e-8, 1, 1e8])
def test_ensure_hermitian_scale_invariant(rng, scale):
    """The decision depends on the deviation relative to the matrix, not on its units"""
    A = random_hermitian(rng, 4)
    asym = rng.normal(size=(4,4))
    asym = asym - asym.T                    # real antisymmetric = anti-Hermitian

    sw_linalg.ensure_hermitian(scale*(A + 1e-13*asym))
    with pytest.raises(NumericalToleranceError):
        sw_linalg.ensure_hermitian(scale*(A + 1e-6*asym))

def test_ensure_hermitian_batch_indices(rng):
    A = random_hermitian(rng, 3, (4,5))
    A[1,2,0,1] += 1
    A[3,0,2,0] += 1
    with pytest.raises(NumericalToleranceError, match=r'\(1, 2\)') as excinfo:
        sw_linalg.ensure_hermitian(A)
    assert excinfo.value.indices == [(1,2), (3,0)]

def test_ensure_hermitian_override_tolerance(rng):
    A = random_hermitian(rng, 4)
    A[0,1] += 1e-6
    with pytest.raises(NumericalToleranceError):
        sw_linalg.ensure_hermitian(A)
    with override_tolerances(numeric=1e-4):
        sw_linalg.ensure_hermitian(A)
    sw_linalg.ensure_hermitian(A, rtol=1e-4)

def test_ensure_hermitian_zero_matrix():
    assert np.allclose(sw_linalg.ensure_hermitian(np.zeros((3,3))), 0)


#########################################################################################
# regularized_cholesky

def test_regularized_cholesky_positive_definite(rng):
    X = random_hermitian(rng, 6, (5,))
    h = X @ X + np.eye(6)
    K = sw_linalg.regularized_cholesky(h)

    assert np.allclose(K @ dagger(K), h, atol=ATOL)
    assert np.allclose(np.triu(K, 1), 0)

def test_regularized_cholesky_zero_mode(rng, caplog):
    """Zero modes are shifted by `rtol*scale`, where plain Cholesky can fail"""
    V = random_unitary(rng, 4)
    h = V @ np.diag([0, 1, 2, 4.]) @ dagger(V)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(h - 1e-15*np.eye(4))

    with caplog.at_level(logging.DEBUG, logger='Numerics'):
        K = sw_linalg.regularized_cholesky(h, rtol=1e-10)

    assert np.allclose(K @ dagger(K), h + 4e-10*np.eye(4), atol=ATOL)
    assert 'zero modes' in caplog.text

def test_regularized_cholesky_zero_matrix():
    K = sw_linalg.regularized_cholesky(np.zeros((2,2)), rtol=1e-10)
    assert np.allclose(K @ dagger(K), 1e-10*np.eye(2))

def test_regularized_cholesky_not_positive_definite():
    h = np.stack([np.diag([1., 2]), np.diag([-1., 2]), np.diag([3., 4])])
    with pytest.raises(NumericalToleranceError, match='not positive definite') as excinfo:
        sw_linalg.regularized_cholesky(h)
    assert excinfo.value.indices == [(1,)]

@pytest.mark.parametrize('scale', [1e-6, 1, 1e6])
def test_regularized_cholesky_scale_invariant(scale):
    """Slightly negative eigenvalue is noise or an error relative to the largest one"""
    sw_linalg.regularized_cholesky(scale*np.diag([-1e-12, 1.]))
    with pytest.raises(NumericalToleranceError):
        sw_linalg.regularized_cholesky(scale*np.diag([-1e-8, 1.]))

def test_regularized_cholesky_not_hermitian():
    with pytest.raises(NumericalToleranceError, match='not Hermitian'):
        sw_linalg.regularized_cholesky(np.array([[2., 1], [0, 2]]))


#########################################################################################
# hermitian_eig

@pytest.mark.parametrize('n', [1, 2, 3, 6, 10])
def test_hermitian_eig_decomposition(rng, n):
    """A = U diag(E) U^dagger with unitary U and real E"""
    A = random_hermitian(rng, n)
    E, U = sw_linalg.hermitian_eig(A)

    assert E.shape == (n,) and U.shape == (n,n)
    assert np.isrealobj(E)
    assert np.allclose(dagger(U) @ U, np.eye(n), atol=ATOL)
    assert np.allclose(U @ np.diag(E) @ dagger(U), A, atol=ATOL)
    assert np.allclose(A @ U, U * E, atol=ATOL)

def test_hermitian_eig_ascending_order(rng):
    """Eigenvalues come sorted ascending, as from `np.linalg.eigh`.

    `SpinW._determine_ESp` relies on this order: its bose factor `(1-g)/2`
    assumes the negative eigenvalues of `K^dagger g K` come first.
    """
    E, _ = sw_linalg.hermitian_eig(random_hermitian(rng, 8))
    assert np.all(np.diff(E) >= 0)

@pytest.mark.parametrize('spectrum', [
    [1, 1, 1, 1],           # identity, fully degenerate
    [1, 1, -1, -1],         # g-like, two degenerate pairs
    [0, 0, 0, 3],           # rank 1
    [0, 0, 0, 0],           # zero matrix
    [1e-12, 1, 1e6, 1e6],   # wide dynamic range with degeneracy
])
def test_hermitian_eig_degenerate(rng, spectrum):
    """Degenerate spectra, where U is not unique, still give a unitary U"""
    V = random_unitary(rng, len(spectrum))
    A = V @ np.diag(spectrum) @ dagger(V)
    E, U = sw_linalg.hermitian_eig(A)
    scale = max(1, np.max(np.abs(spectrum)))

    assert np.allclose(dagger(U) @ U, np.eye(len(spectrum)), atol=ATOL)
    assert np.allclose(E, np.sort(spectrum), atol=ATOL*scale)
    assert np.allclose(offdiagonal(dagger(U) @ A @ U), 0, atol=ATOL*scale)

def test_hermitian_eig_real_symmetric(rng):
    X = rng.normal(size=(5,5))
    A = X + X.T
    E, U = sw_linalg.hermitian_eig(A)

    assert np.isrealobj(U)
    assert np.allclose(U @ np.diag(E) @ U.T, A, atol=ATOL)

@pytest.mark.parametrize('batch', [(7,), (3,4)])
def test_hermitian_eig_batched(rng, batch):
    """(...,N,N) input is decomposed matrix by matrix"""
    A = random_hermitian(rng, 4, batch)
    E, U = sw_linalg.hermitian_eig(A)

    assert E.shape == batch + (4,) and U.shape == batch + (4,4)
    for idx in np.ndindex(*batch):
        Ei, Ui = sw_linalg.hermitian_eig(A[idx])
        assert np.allclose(E[idx], Ei, atol=ATOL)
        assert np.allclose(U[idx], Ui, atol=ATOL)

def test_hermitian_eig_spinw_kernel(rng):
    """K^dagger g K has, by Sylvester's law of inertia, M positive and M negative
    eigenvalues; in ascending order the negative come first.
    """
    M = 3
    E, U = sw_linalg.hermitian_eig(spinw_kernel(rng, M))

    assert np.allclose(dagger(U) @ U, np.eye(2*M), atol=ATOL)
    assert np.all(E[:M] < 0) and np.all(E[M:] > 0)

def test_hermitian_eig_noise_and_error(rng):
    """Round-off non-hermiticity is removed, larger asymmetry raises"""
    A = random_hermitian(rng, 4)
    E, _ = sw_linalg.hermitian_eig(A + 1e-14*np.triu(np.ones((4,4)), 1))
    assert np.allclose(E, np.linalg.eigvalsh(A), atol=1e-13)

    with pytest.raises(NumericalToleranceError, match='`A`'):
        sw_linalg.hermitian_eig(A + 1e-3*np.triu(np.ones((4,4)), 1), name='A')

### Comparison with scipy.linalg.schur
@needs_scipy
@pytest.mark.parametrize('n', [2, 5, 8])
def test_hermitian_eig_vs_scipy_eigenvalues(rng, n):
    """Same spectrum as `scipy.linalg.schur`, whose triangular T is diagonal for Hermitian A"""
    import scipy.linalg

    A = random_hermitian(rng, n)
    E, _ = sw_linalg.hermitian_eig(A)
    Ts, _ = scipy.linalg.schur(A, output='complex')

    assert np.allclose(np.tril(Ts, -1), 0)
    assert np.allclose(offdiagonal(Ts), 0, atol=ATOL)
    assert np.allclose(E, np.sort(np.diagonal(Ts).real), atol=ATOL)

@needs_scipy
def test_hermitian_eig_vs_scipy_eigenvectors(rng):
    """Eigenvectors are unique only up to a phase, so compare spectral projectors |u><u|"""
    import scipy.linalg

    A = random_hermitian(rng, 6)                   # random spectrum is non-degenerate
    _, U = sw_linalg.hermitian_eig(A)
    Ts, Zs = scipy.linalg.schur(A, output='complex')
    Zs = Zs[:, np.argsort(np.diagonal(Ts).real)]

    P_numpy = np.einsum('ik,jk->kij', U, U.conj())
    P_scipy = np.einsum('ik,jk->kij', Zs, Zs.conj())
    assert np.allclose(P_numpy, P_scipy, atol=1e-10)

@needs_scipy
def test_hermitian_eig_vs_scipy_rhp_order(rng):
    """`scipy.linalg.schur(sort='rhp')`, used by the legacy implementation,
    puts the positive eigenvalues first; `hermitian_eig` sorts ascending instead.
    """
    import scipy.linalg

    M = 3
    A = spinw_kernel(rng, M)
    E, _ = sw_linalg.hermitian_eig(A)
    Ts, _, sdim = scipy.linalg.schur(A, output='complex', sort='rhp')
    E_scipy = np.diagonal(Ts).real

    assert sdim == M
    assert np.all(E_scipy[:M] > 0) and np.all(E_scipy[M:] < 0)
    assert np.all(E[:M] < 0) and np.all(E[M:] > 0)
    assert np.allclose(E, np.sort(E_scipy), atol=1e-10)

@needs_scipy
@pytest.mark.parametrize('spectrum', [[1, 1, -1, -1], [2, 2, 2, 5]])
def test_hermitian_eig_vs_scipy_degenerate_subspaces(rng, spectrum):
    """For degenerate eigenvalues compare projectors on the whole eigenspace"""
    import scipy.linalg

    V = random_unitary(rng, len(spectrum))
    A = V @ np.diag(spectrum) @ dagger(V)
    E, U = sw_linalg.hermitian_eig(A)
    Ts, Zs = scipy.linalg.schur(A, output='complex')

    for e in np.unique(spectrum):
        u = U[:, np.isclose(E, e)]
        z = Zs[:, np.isclose(np.diagonal(Ts).real, e)]
        assert u.shape == z.shape
        assert np.allclose(u @ dagger(u), z @ dagger(z), atol=1e-10)


#########################################################################################
# Exchange interaction matrices

def test_DMI_antisymmetric():
    D = sw_linalg.DMI([0.1, -0.7, 2])
    assert np.allclose(D, -D.T)
    assert np.allclose(sw_linalg.DMI([0,0,0]), 0)

def test_DMI_energy(rng):
    """S_i . DMI(d) . S_j = d . (S_i x S_j)"""
    d, Si, Sj = rng.normal(size=(3,3))
    assert np.isclose(Si @ sw_linalg.DMI(d) @ Sj, np.dot(d, np.cross(Si, Sj)))

def test_DMI_explicit():
    assert np.allclose(sw_linalg.DMI([1,2,3]), [[0, 3, -2], [-3, 0, 1], [2, -1, 0]])

def test_dipolar_symmetric(rng):
    J = sw_linalg.dipolar(rng.normal(size=3))
    assert J.shape == (3,3)
    assert np.allclose(J, J.T)

@pytest.mark.xfail(strict=True, reason='Code returns (r r^T - |r|^2 I)/|r|^3, docstring states '
                   '/|r|^5; standard dipolar tensor is (3 r r^T - |r|^2 I)/|r|^5')
def test_dipolar_standard_tensor(rng):
    r = rng.normal(size=3)
    expected = (3*np.outer(r, r) - np.eye(3)*np.dot(r, r)) / np.linalg.norm(r)**5
    assert np.allclose(sw_linalg.dipolar(r), expected)


#########################################################################################
# Rotations

def test_main_rotations():
    """Principal rotations"""
    assert np.allclose(sw_linalg.Rx(0), np.eye(3,3))
    assert np.allclose(sw_linalg.Ry(0), np.eye(3,3))
    assert np.allclose(sw_linalg.Rz(0), np.eye(3,3))

@pytest.mark.parametrize('R, axis, v, expected', [
    (sw_linalg.Rx, 0, [0,1,0], [0,0,1]),
    (sw_linalg.Ry, 1, [0,0,1], [1,0,0]),
    (sw_linalg.Rz, 2, [1,0,0], [0,1,0]),
])
def test_main_rotations_right_handed(R, axis, v, expected):
    """Rotation by pi/2 maps the axes cyclically, and keeps the rotation axis"""
    Ra = R(np.pi/2)
    assert is_rotation(Ra)
    assert np.allclose(Ra @ v, expected)
    assert np.allclose(Ra @ np.eye(3)[axis], np.eye(3)[axis])

@pytest.mark.parametrize('R', [sw_linalg.Rx, sw_linalg.Ry, sw_linalg.Rz])
def test_main_rotations_vectorized(R):
    alphas = np.linspace(-np.pi, np.pi, 12).reshape(3,4)
    Rs = R(alphas)

    assert Rs.shape == (3,4,3,3)
    for idx in np.ndindex(3,4):
        assert np.allclose(Rs[idx], R(alphas[idx]))
    assert np.allclose(R(0.3) @ R(-0.3), np.eye(3))
    assert np.allclose(R(0.3) @ R(0.4), R(0.7))

@pytest.mark.parametrize('n', VECTORS)
def test_rotate(n):
    """Rotation around `n` keeps `n`, matches Rodrigues formula, and ignores |n|"""
    alpha = 0.7
    R = sw_linalg.rotate(n, alpha)
    e = np.asarray(n) / np.linalg.norm(n)
    rodrigues = np.cos(alpha)*np.eye(3) + np.sin(alpha)*cross_matrix(e) + (1-np.cos(alpha))*np.outer(e, e)

    assert is_rotation(R)
    assert np.allclose(R @ e, e)
    assert np.allclose(R, rodrigues)
    assert np.allclose(R, sw_linalg.rotate(5*e, alpha))

def test_rotate_principal_axes():
    alpha = 0.4
    assert np.allclose(sw_linalg.rotate([1,0,0], alpha), sw_linalg.Rx(alpha))
    assert np.allclose(sw_linalg.rotate([0,1,0], alpha), sw_linalg.Ry(alpha))
    assert np.allclose(sw_linalg.rotate([0,0,1], alpha), sw_linalg.Rz(alpha))

def test_rot_Rn():
    """R_n rotates by -2 pi k.n around the global axis, eq. (6) [SpinW]"""
    axis = [0,0,1]
    assert np.allclose(sw_linalg.rot_Rn([1,2,3], [0,0,0], axis), np.eye(3))
    assert np.allclose(sw_linalg.rot_Rn([0,0,0], [0.3,0.1,0], axis), np.eye(3))
    assert np.allclose(sw_linalg.rot_Rn([1,0,0], [0.25,0,0], axis), sw_linalg.Rz(-np.pi/2))
    assert np.allclose(sw_linalg.rot_Rn([2,1,0], [0.5,0,0], [1,0,0]), np.eye(3))  # k.n integer

def test_rot_Rn_composition():
    """Phases add up: R_(n+m) = R_n @ R_m"""
    k, axis = [0.13, -0.27, 0.05], [1, 2, -1]
    n, m = np.array([1,0,2]), np.array([-3,1,1])
    assert np.allclose(sw_linalg.rot_Rn(n+m, k, axis),
                       sw_linalg.rot_Rn(n, k, axis) @ sw_linalg.rot_Rn(m, k, axis))

def test_Rprime():
    """RtoZ rotates the vector onto z"""
    assert np.allclose( sw_linalg.RtoZ([0,0,1]), np.eye(3,3))
    assert np.allclose( sw_linalg.RtoZ([1,0,0]), sw_linalg.Ry(-np.pi/2))

    v = [0.3, -0.2, 0.4]
    assert np.allclose( sw_linalg.RtoZ(v) @ v, [0, 0, sw_linalg.norm(v)])
    v = [-np.sqrt(2), 1e5, -np.pi]
    assert np.allclose( sw_linalg.RtoZ(v) @ v, [0, 0, sw_linalg.norm(v)])
    v = [1, 1, 1]
    assert np.allclose( sw_linalg.RtoZ(v) @ v, [0, 0, sw_linalg.norm(v)])

@pytest.mark.parametrize('v', VECTORS)
def test_RtoZ_RfromZ(v):
    """RfromZ is the inverse of RtoZ, both are proper rotations"""
    v = np.asarray(v, dtype=float)
    to_z, from_z = sw_linalg.RtoZ(v), sw_linalg.RfromZ(v)

    assert is_rotation(to_z) and is_rotation(from_z)
    assert np.allclose(to_z @ v, [0, 0, np.linalg.norm(v)])
    assert np.allclose(from_z @ [0, 0, np.linalg.norm(v)], v)
    assert np.allclose(from_z @ to_z, np.eye(3))

@pytest.mark.parametrize('n', [[0,0,1], [1,0,0], np.array([1,1,1])/np.sqrt(3), np.array([0.3,-0.2,0.4])/np.linalg.norm([0.3,-0.2,0.4])])
def test_Rodrigues_complex(n):
    """e^(i phi) R1 + e^(-i phi) R1* + R2 is the rotation around unit `n` by phi, (39) [SpinW]"""
    R1, R2 = sw_linalg.rot_Rodrigues_complex(n)

    assert np.allclose(R2, R2.T) and np.allclose(R2 @ R2, R2)      # projector on n
    assert np.allclose(R1 + R1.conj() + R2, np.eye(3))              # phi = 0
    for phi in [0.3, np.pi/2, 2.1, -1.0]:
        R = np.exp(1j*phi)*R1 + np.exp(-1j*phi)*R1.conj() + R2
        assert np.allclose(R.imag, 0)
        assert np.allclose(R.real, sw_linalg.rotate(n, phi))


#########################################################################################
# Vectors

@pytest.mark.parametrize('xyz, expected', [
    ([0, 0, 2],  [2, 0, 0]),
    ([0, 0, -1], [1, np.pi, 0]),
    ([1, 0, 0],  [1, np.pi/2, 0]),
    ([0, 1, 0],  [1, np.pi/2, np.pi/2]),
    ([-1, -1, 0],[np.sqrt(2), np.pi/2, -3*np.pi/4]),
])
def test_cartesian2spherical(xyz, expected):
    assert np.allclose(sw_linalg.cartesian2spherical(xyz), expected)

def test_cartesian2spherical_roundtrip(rng):
    xyz = rng.normal(size=(10,3))
    r, theta, phi = sw_linalg.cartesian2spherical(xyz)

    assert np.all(r > 0)
    assert np.all((0 <= theta) & (theta <= np.pi))
    assert np.all((-np.pi <= phi) & (phi <= np.pi))
    back = np.stack([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)], axis=-1)
    assert np.allclose(back, xyz)

def test_norm():
    assert np.isclose(sw_linalg.norm([3, 4]), 5)
    assert np.allclose(sw_linalg.norm([[3, 4], [6, 8]], axis=-1), [5, 10])

@pytest.mark.parametrize('v1, v2, expected', [
    ([1,0,0], [2,0,0], 0),
    ([1,0,0], [0,3,0], np.pi/2),
    ([1,0,0], [-1,0,0], np.pi),
    ([1,1,0], [1,0,0], np.pi/4),
])
def test_angle(v1, v2, expected):
    assert np.isclose(sw_linalg.angle(v1, v2), expected)

def test_angle_parallel_no_nan():
    """Clip protects against arccos(1+eps) = nan for (anti)parallel vectors"""
    v = [1e-8, 1, 1]
    assert np.isclose(sw_linalg.angle(v, v), 0, atol=1e-7)
    assert np.isclose(sw_linalg.angle(v, -np.array(v)), np.pi, atol=1e-7)

@pytest.mark.xfail(strict=True, reason='arccos loses precision near 0 and pi: error ~sqrt(eps) = 1.5e-8; '
                   'arctan2(|v1 x v2|, v1.v2) would be accurate')
def test_angle_parallel_precision():
    v = [1e-8, 1, 1]
    assert np.isclose(sw_linalg.angle(v, v), 0, atol=1e-12)

@pytest.mark.parametrize('q', VECTORS)
def test_perp_matrix(q):
    """P = 1 - q q^T / |q|^2 is a symmetric projector that annihilates q"""
    q = np.asarray(q, dtype=float)
    P = sw_linalg.perp_matrix(q)

    assert np.allclose(P, np.eye(3) - np.outer(q, q)/np.dot(q, q))
    assert np.allclose(P, P.T)
    assert np.allclose(P @ P, P)
    assert np.allclose(P @ q, 0, atol=1e-10*np.linalg.norm(q))

def test_perp_part(rng):
    m, q = rng.normal(size=(2,3))
    m_perp = sw_linalg.perp_part(m, q)

    assert np.isclose(np.dot(m_perp, q), 0)
    assert np.allclose(m_perp, m - np.dot(m, q)/np.dot(q, q)*q)
