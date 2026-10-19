r"""Tests of `spinwaves.coupling`

Objects to cover:
- `Coupling`
"""
import numpy as np
import pytest

from spinwaves import Coupling
from spinwaves.utils.linalg import DMI

from .helpers import assert_no_numpy_scalar_repr


#########################################################################################
# numpy 1/2 compatibility: default integer on Windows changed from int32 to int64,
# scalar reprs changed to `np.int64(1)`, `np.unique` sorts object arrays

@pytest.mark.parametrize('n_uvw', [
    [1, 0, -1],
    (1, 0, -1),
    np.array([1, 0, -1], dtype=np.int32),
    np.array([1, 0, -1], dtype=np.int64),
    np.array([1., 0, -1]),
])
def test_coupling_eq_hash_independent_of_input_type(n_uvw):
    """Equality and hash ignore label and J, and the type of `n_uvw`"""
    reference = Coupling(label='J', id1=0, id2=1, n_uvw=[1,0,-1], J=np.eye(3))
    c = Coupling(label='other', id1=np.int64(0), id2=np.int32(1), n_uvw=n_uvw, J=2*np.eye(3))

    assert c == reference
    assert hash(c) == hash(reference)
    assert np.issubdtype(c.n_uvw.dtype, np.integer)
    assert type(c.id1) is int and type(c.id2) is int

def test_coupling_ordering():
    """Sorted by (id1, id2, n_uvw), also when `n_uvw` entries are numpy integers"""
    a = Coupling(label='a', id1=0, id2=1, n_uvw=[0,0,1], J=np.eye(3))
    b = Coupling(label='b', id1=0, id2=1, n_uvw=np.array([0,1,0], dtype=np.int32), J=np.eye(3))
    c = Coupling(label='c', id1=1, id2=0, n_uvw=[-1,0,0], J=np.eye(3))

    assert [x.label for x in sorted([c, b, a])] == ['a', 'b', 'c']

def test_coupling_unique_object_array():
    """`np.unique` on a list of couplings, as in `SpinW.symmetrize_couplings`"""
    couplings = [Coupling(label=f'c{i}', id1=i%2, id2=0, n_uvw=[i%3,0,0], J=np.eye(3)) for i in range(12)]
    unique = np.unique(couplings + couplings[::-1])

    assert len(unique) == 6
    assert list(unique) == sorted(set(couplings))

def test_coupling_revert():
    c = Coupling(label='J', id1=0, id2=1, n_uvw=[1,0,0], J=np.eye(3) + DMI([0,0,0.3]))
    r = c.revert()

    assert (r.id1, r.id2, r.n_uvw.tolist()) == (1, 0, [-1, 0, 0])
    assert np.allclose(r.J, c.J.T)
    assert r.label == 'J_rev' and r.revert() == c

def test_coupling_repr():
    c = Coupling(label='J', id1=np.int64(0), id2=1, n_uvw=np.array([1,0,0]), J=np.eye(3))
    assert_no_numpy_scalar_repr(repr(c))
