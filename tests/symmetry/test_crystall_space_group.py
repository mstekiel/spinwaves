r"""Tests of `spinwaves.symmetry.crystall_space_group`

Objects to cover:
- `cSymOp`
- `SG`
"""
import numpy as np
import pytest

from spinwaves.symmetry.crystall_space_group import cSymOp, SG

from ..helpers import assert_no_numpy_scalar_repr


#########################################################################################
# numpy 1/2 compatibility: default integer on Windows changed from int32 to int64,
# scalar reprs changed to `np.int64(1)`

@pytest.mark.parametrize('dtype', [np.int8, np.int32, np.int64])
def test_cSymOp_eq_hash_independent_of_matrix_dtype(dtype):
    g = cSymOp.from_string('-y, x-y, z+1/3')
    g_dtype = cSymOp(g._matrix.astype(dtype), g._translation)

    assert g_dtype == g
    assert hash(g_dtype) == hash(g)

def test_SG_strings():
    """Strings feed `cSymOp.__hash__`, so they must not depend on the numpy version"""
    G = SG.from_xyz_strings(['x-y,x,z', '-x,-y,-z'])

    assert sorted(g.to_string() for g in G) == [
        '-x+y, -x, -z', '-x+y, -x, z', '-x, -y, -z', '-x, -y, z', '-y, x-y, -z', '-y, x-y, z',
        'x, y, -z', 'x, y, z', 'x-y, x, -z', 'x-y, x, z', 'y, -x+y, -z', 'y, -x+y, z']
    for g in G:
        assert_no_numpy_scalar_repr(repr(g))
    assert_no_numpy_scalar_repr(str(G))
