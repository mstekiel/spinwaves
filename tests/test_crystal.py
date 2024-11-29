# -*- coding: utf-8 -*-
r"""Tests lattice math

"""
import numpy as np
from fractions import Fraction
import pytest

from spinwaves import Crystal, MSG, Atom
from spinwaves.spinw import Coupling
from spinwaves.functions import DMI



def test_crystal_constructor():
    """Test construction of the lattice"""

    atoms = [ Atom(r=[0,0,1/4]), Atom(r=[1/3, 2/3, 3/4])]
    P_194 = MSG.from_xyz_strings(generators=[
        '-y,x-y,z, +1',
        '-x,-y,z+1/2, +1',
        'y,x,-z, +1',
        '-x,-y,-z, +1',
    ])

    print(P_194)
    a, c = 3.6, 12
    graphite = Crystal(lattice_parameters=[a,a,c, 90,90,120],
                      atoms=atoms,
                      MSG=P_194)

    print(graphite)


def test_get_atom_index():
    atoms = [ Atom(r=[0,0,1/4]), Atom(r=[1/3, 2/3, 3/4])]
    P_194 = MSG.from_xyz_strings(generators=[
        '-y,x-y,z, +1',
        '-x,-y,z+1/2, +1',
        'y,x,-z, +1',
        '-x,-y,-z, +1',
    ])

    graphite = Crystal(lattice_parameters=[3.6,3.6,12, 90,90,120],
                      atoms=atoms,
                      MSG=P_194)

    assert graphite.get_atom_sw_id([0, 0, 0.25]) == 0
    assert graphite.get_atom_sw_id([0, 0, 0.75]) == 1
    assert graphite.get_atom_sw_id([1/3, 2/3, 0.75]) == 2
    assert graphite.get_atom_sw_id([2/3, 1/3, 0.25]) == 3

    with pytest.raises(LookupError) as e_info:
        graphite.get_atom_sw_id([1/3, 1/3, 0.25]) == 3


def test_constructor_validators():
    # 1. Magnetic moment should obey MSG
    pass

def test_crystal_couplings():
    # Atoms in base plane, with funny moments
    atoms = [
        Atom(label='Dy', r=(0.5,   0.5, 0),   m=(0,0,5), s=2.5),
        Atom(label='Fe', r=(0.1,   0.1, 0),   m=(1,1,0), s=2.5)]
    # atoms = [Atom(label='Fe', r=(0,   0.5, 0),   m=(-1,0,Fz), s=2.5)]
    P4mm = MSG.from_xyz_strings(generators=[
        '-y, x, z, +1', # 4_001
        'y, x, z, +1'   # m_110
    ])

    crystal = Crystal(lattice_parameters=(4,4,10, 90,90,90),
                      MSG=P4mm, atoms=atoms)
    
    print(crystal.MSG.get_point_symmetry([0.1,0.1,0]))
    
    D1 = DMI(2,2,0)
    DMI_appropriate = Coupling(label='D1', id1=0, id2=2, n_uvw=[0,0,0], J=D1)
    assert crystal.is_respectful_DMI(DMI_appropriate)

    D2 = DMI(0.5,0.05,0.005)
    DMI_inappropriate = Coupling(label='D2', id1=0, id2=2, n_uvw=[0,0,0], J=D2)
    inappropriate, D2_symmetrized = crystal.is_respectful_DMI(DMI_inappropriate, return_symmetrized=True)
    
    assert not inappropriate
    assert np.allclose(D2_symmetrized, [0.275, 0.275, 0.005])
    
if __name__ == "__main__":
    # pytest.main()
    test_get_atom_index()


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