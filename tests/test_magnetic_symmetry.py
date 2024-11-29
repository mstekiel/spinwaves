# -*- coding: utf-8 -*-
r"""Tests lattice math

"""
import numpy as np
from fractions import Fraction
import pytest

from spinwaves.magnetic_symmetry import mSymOp, MSG

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

def test_subgroups():
    pass 
    # Imm3 = MSG.from_xyz_strings(generators=[
    #     'z, x, y, +1',    # 3_111
    #     'x+1/2, y+1/2, z+1/2, +1', # I centering
    #     '-x, -y, z, +1',    # 2_110
    #     '-x, -y, -z, +1',   # -1
    #     ])
    # # This groups has order 48, check only orders of subgroups

    # subgroups_order = [2,4,4,4,6,8,8,8,12,16,24]
    # subgroups = Imm3.get_subgroups()
    # assert subgroups_order==[Gsub.order for Gsub in subgroups]

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

    
