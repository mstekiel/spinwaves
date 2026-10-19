# -*- coding: utf-8 -*-
r"""Tests lattice math

"""
import numpy as np
from fractions import Fraction
import pytest

from spinwaves.symmetry.magnetic_symmetry import mSymOp, MSG

from ..helpers import assert_no_numpy_scalar_repr

g_1 = mSymOp.from_string('x, y, z, +1') # 1
g_m1 = mSymOp.from_string('-x, -y, -z, +1')  # -1
g_1p = mSymOp.from_string('x, y, z, -1') # 1'
g_3_111 = mSymOp.from_string('z, x, y, +1') # 3_111
g_3_001 = mSymOp.from_string('-y, x-y, z, +1')   # 3_001
g_2_110 = mSymOp.from_string('-x, -y, z, +1')   # 2_110
g_m_100 = mSymOp.from_string('-x, y, z, +1')   # 2_110

def test_mSymOp_constructor():
    """Test construction of the lattice"""

    # Identity
    assert np.all(g_1.matrix == np.array([[1,0,0] ,[0,1,0], [0,0,1]], dtype=int))
    assert np.all(g_1.translation == np.array([0,0,0], dtype=Fraction))
    assert g_1.time_reversal == 1
    assert g_1.str == 'x, y, z, +1'
    
def test_mSymOp_multiplication():
    '''Test multiplication of symmetry elements'''

    assert g_m1*g_m1 == g_1
    assert g_3_001*g_3_001*g_3_001 == g_1
    assert g_m_100*g_m_100 == g_1

def test_MSG_construction():
    P4pom = MSG.from_xyz_strings(generators=[
        '-y,x,z,-1',
        '-x,-y,-z,+1'
    ])

    P4pom_bilbao = [
        'x,y,z,+1',
        '-x,-y,z,+1',
        '-x,-y,-z,+1',
        'x,y,-z,+1',
        '-y,x,z,-1',
        'y,-x,z,-1',
        'y,-x,-z,-1',
        '-y,x,-z,-1'
    ]
    assert set(mSymOp.from_string(s) for s in P4pom_bilbao) == set(g for g in P4pom)

    # This is non-standard setting
    I_P4_1p2p2 = MSG.from_xyz_strings(generators=[
        'x+1/2, -y, -z+3/4, +1',
        '-y+1/2, x, z+3/4, +1',
        'x+1/2, y+1/2, z+1/2, -1'
    ])

    I_P4_1p2p2_bilbao = [
        'x,y,z,+1',
        'x+1/2,-y,-z+3/4,+1',
        '-x,y+1/2,-z+1/4,+1',
        '-x+1/2,-y+1/2,z+1/2,+1',
        '-y,-x,-z,+1',
        '-y+1/2,x,z+3/4,+1',
        'y,-x+1/2,z+1/4,+1',
        'y+1/2,x+1/2,-z+1/2,+1',
        'x+1/2,y+1/2,z+1/2,-1',
        'x,-y+1/2,-z+1/4,-1',
        '-x+1/2,y,-z+3/4,-1',
        '-x,-y,z,-1',
        '-y+1/2,-x+1/2,-z+1/2,-1',
        '-y,x+1/2,z+1/4,-1',
        'y+1/2,-x,z+3/4,-1',
        'y,x,-z,-1'
    ]
    assert set(mSymOp.from_string(s) for s in I_P4_1p2p2_bilbao) == set(g for g in I_P4_1p2p2)

    # This constructor take around 70 seconds
    # P_223p205 = MSG.from_xyz_strings(generators=[
    #     'x+1/2, y+1/2, z+1/2, +1', # I centering
    #     'z, x, y, +1',    # 3_111
    #     '-x, -y, z, +1',    # 2_110
    #     '-x, -y, -z, +1',   # -1
    #     'x, y, z, -1',   # 1'
    #     'x+1/4, -z+1/4, y+3/4, -1'
    #     ])

@pytest.mark.skip(reason="Not implemented yet")
def test_subgroups():
    Imm3 = MSG.from_xyz_strings(generators=[
        'z, x, y, +1',    # 3_111
        'x+1/2, y+1/2, z+1/2, +1', # I centering
        '-x, -y, z, +1',    # 2_110
        '-x, -y, -z, +1',   # -1
        ])
    # This groups has order 48, check only orders of subgroups

    subgroups_order = [2,4,4,4,6,8,8,8,12,16,24]
    subgroups = Imm3.get_subgroups()
    assert sorted(subgroups_order) == sorted([Gsub.order for Gsub in subgroups])

def test_orbits():
    P4_32_12 = MSG.from_xyz_strings(generators=[
        '-y+1/2, x+1/2, z+3/4, +1',
        'x+1/2, -y+1/2, -z+1/4, +1',
    ])
    positions = P4_32_12.get_orbit(position=[1/4, 1/4, 0])
    positions_bilbao = [
        [1/4, 1/4, 0],
        [3/4, 1/4, 1/4],
        [1/4, 3/4, 3/4],
        [3/4, 3/4, 1/2],
    ]
    
    for pos in positions:
        positions_bilbao.remove(list(pos))

    assert len(positions_bilbao) == 0


    ### with generators
    I = MSG.from_xyz_strings(generators=['x+1/2, y+1/2, z+1/2, +1'])
    rs, gs = I.get_orbit(position=[0,0,0], return_generators=True)

    for r,g in zip(rs, gs):
        if np.allclose(r, [0,0,0]):
            assert g == mSymOp.from_string('x, y, z, +1')
        if np.allclose(r, [1/2, 1/2, 1/2]):
            assert g == mSymOp.from_string('x+1/2, y+1/2, z+1/2, +1')

    ### graphite, to check floating point accuracy with 1/3
    P_194 = MSG.from_xyz_strings(generators=[
        '-y,x-y,z, +1',
        '-x,-y,z+1/2, +1',
        'y,x,-z, +1',
        '-x,-y,-z, +1',
    ])

    positions = P_194.get_orbit(position=[1/3, 2/3, 3/4], return_generators=False)
    positions_bilbao = [
        [1/3, 2/3, 3/4],
        [2/3, 1/3, 1/4]
    ]

    if positions[0][2] == 1/4:
        positions.reverse()
    
    assert np.allclose(positions, positions_bilbao)
    
#########################################################################################
# numpy 1/2 compatibility: default integer on Windows changed from int32 to int64,
# scalar reprs changed to `np.int64(1)`, numpy 2 wheels ship different BLAS/LAPACK builds

OH_GENERATORS = ['z, x, y, +1', '-y, x, z, +1', '-x, -y, -z, +1']    # m-3m, order 48

def test_mSymOp_matrix_is_integer():
    g = mSymOp.from_string('-y, x-y, z, +1')
    assert np.issubdtype(g.matrix.dtype, np.integer)
    assert np.issubdtype((g*g).matrix.dtype, np.integer)
    assert np.issubdtype(g.inv().matrix.dtype, np.integer)

@pytest.mark.xfail(strict=True, reason='`mSymOp.__hash__` uses `matrix.tobytes()` and `__eq__` compares hashes: '
                   'equal operations with int32 and int64 matrices, the numpy 1 and 2 defaults on Windows, are not equal')
def test_mSymOp_eq_hash_independent_of_matrix_dtype():
    g = mSymOp.from_string('-y, x, z+1/2, -1')
    g32 = mSymOp(g.matrix.astype(np.int32), g._translation, g.time_reversal)
    g64 = mSymOp(g.matrix.astype(np.int64), g._translation, g.time_reversal)

    assert g32 == g64
    assert hash(g32) == hash(g64)

@pytest.mark.parametrize('xyz', ['x, y, z, +1', '-y, x-y, z+1/3, +1', 'x+1/2, -z+1/4, y+3/4, -1', '-x+y, -x, -z+5/6, -1'])
def test_mSymOp_string_roundtrip(xyz):
    g = mSymOp.from_string(xyz)
    assert g.to_string() == xyz
    assert_no_numpy_scalar_repr(repr(g))

def test_MSG_strings():
    G = MSG.from_xyz_strings(generators=OH_GENERATORS)
    for g in G:
        assert_no_numpy_scalar_repr(g.to_string())
    assert_no_numpy_scalar_repr(str(G))

def test_inversion_and_character_Oh():
    """`inversion` truncates det(matrix) with `astype(int)`: LAPACK round-off of the
    determinant below 1 would give 0, it must be exactly +-1 for every operation.
    """
    ops = list(MSG.from_xyz_strings(generators=OH_GENERATORS))

    assert len(ops) == 48
    for g in ops:
        assert g.inversion in (-1, 1)
        assert g.inversion == np.rint(np.linalg.det(g.matrix))
        assert g.character == np.trace(g.matrix)
    assert sum(g.inversion == -1 for g in ops) == 24

def test_orbit_Oh():
    """`get_orbit` uses `np.unique(..., axis=0, return_index=True)` on rounded positions"""
    G = MSG.from_xyz_strings(generators=OH_GENERATORS)

    assert len(G.get_orbit([0.1, 0.2, 0.3])) == 48
    assert len(G.get_orbit([0, 0, 0.5])) == 3
    assert len(G.get_orbit([0, 0, 0])) == 1


if __name__ == "__main__":
    # pytest.main()

    ### Quick tests
    import time
    t_start = time.time()
    P = MSG.from_xyz_strings(generators=[
        '-y, x, z, +1', # 4_001
        # 'z, x, y, +1',    # 3_111
        # 'x+1/2, y+1/2, z+1/2, +1', # I centering
        # '-x, -y, z, +1',    # 2_110
        # '-x, -y, -z, +1',   # -1
        # 'x, y, z, -1',   # 1'
        'y, x, z, +1'   # m_110
        # '-x, y, z, +1'   # m_100
        ])
    t_end = time.time()
    print()
    for g in P.operations:
        print(g)
    print(t_end-t_start, "for MSG generation")

    x = 1
    atom = np.array([x, x, 0])
    print(P.get_orbit(atom))
    print(P.get_point_symmetry(atom))
    print(P.get_point_symmetry(atom+[1,1,0.5]))

    
