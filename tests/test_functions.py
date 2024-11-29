# -*- coding: utf-8 -*-
r"""Tests lattice math

"""
import numpy as np
from fractions import Fraction
import pytest

import spinwaves.functions as funs_sw

def test_main_rotations():
    """Test principal rotations"""
    assert np.allclose(funs_sw.Rx(0), np.eye(3,3))
    assert np.allclose(funs_sw.Ry(0), np.eye(3,3))
    assert np.allclose(funs_sw.Rz(0), np.eye(3,3))

def test_Rprime():
    """Test rotation matrix"""

    assert np.allclose( funs_sw.rot_Rprime([0,0,1]), np.eye(3,3))
    assert np.allclose( funs_sw.rot_Rprime([1,0,0]), funs_sw.Ry(-np.pi/2))

    v = [0.3, -0.2, 0.4]
    assert np.allclose( funs_sw.rot_Rprime(v) @ v, [0, 0, funs_sw.norm(v)])
    v = [-np.sqrt(2), 1e5, -np.pi]
    assert np.allclose( funs_sw.rot_Rprime(v) @ v, [0, 0, funs_sw.norm(v)])
    v = [1, 1, 1]
    assert np.allclose( funs_sw.rot_Rprime(v) @ v, [0, 0, funs_sw.norm(v)])

def test_Rodrigues_complex():
    R1, R2 = funs_sw.rot_Rodrigues_complex([0,0,1])

    pass


if __name__ == "__main__":
    # pytest.main()
    test_Rodrigues_complex()

    ### Quick tests
    # import time
    # t_start = time.time()
    # P = MSG.from_xyz_strings(generators=[
    #     # '-y, x, z, +1', # 4_001
    #     # 'z, x, y, +1',    # 3_111
    #     # 'x+1/2, y+1/2, z+1/2, +1', # I centering
    #     '-x, -y, z, +1',    # 2_110
    #     # '-x, -y, -z, +1',   # -1
    #     # 'x, y, z, -1',   # 1'
    #     'y, x, z, +1'   # m_110
    #     ])
    # t_end = time.time()
    # for g in P:
    #     print(g.print())

    # print(P.order)
    # print(P.make_cayley_table())
    # print('Constructor and Cayley table', t_end-t_start)
    # print()
    # t_start = time.time()
    # P_subs = P.get_subgroups()
    # for P_sub in P_subs:
    #     print(P_sub)
    # t_end = time.time()
    # print('subgroups', t_end-t_start)
    # print('orders', [G.order for G in P_subs])

    # pos = [0, 0.5, 0.3]
    # print('point symmetry')
    # print(P.get_point_symmetry(pos))
    # print('orbit')
    # print(P.get_orbit(pos))