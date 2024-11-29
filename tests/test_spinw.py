# -*- coding: utf-8 -*-
r"""Tests lattice math

"""
import numpy as np
from fractions import Fraction
import pytest

from spinwaves import Crystal, MSG, Atom, Coupling, SpinW
from spinwaves.plotting import plot_structure   # this takes some serious time
from spinwaves.functions import DMI

def test_matrices():
    """Test how the characteristic matrices are calculated"""

    P4mm = MSG.from_xyz_strings(generators=[
        '-y, x, z, +1', # 4_001
        'y, x, z, +1'   # m_110
    ])
    atoms = [
        Atom(label='Fe', r=(0,   0, 0),   m=(0,0,1), s=2.5),    # 4d
        ] 
    crystal = Crystal(lattice_parameters=[4,4,10, 90,90,90], MSG=P4mm, atoms=atoms)

    print(crystal)
    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,0,1)
    }

    Jab, Jc, Kz = -2, -0.2, -0.2
    couplings = [
        Coupling(label='Jab', id1=0, id2=0, n_uvw=[1,0,0], J=Jab*np.eye(3,3)),
        Coupling(label='Jc', id1=0, id2=0, n_uvw=[0,0,1], J=Jc*np.eye(3,3)),
        Coupling(label='K', id1=0, id2=0, n_uvw=[0,0,0],J=np.diag([0,0,Kz]))
    ]

    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=couplings)
    print('E0 = ', sw.calculate_ground_state(q_hkl=[0,0,0]))
    
    S = sw.determine_matrices(q_hkl = [0.1, 0.1, 0])
    print(S)


def test_coupling_symmetrization_tetr():
    """Test symmetrization of the couplings in hexagonal system"""
    # Dice with 5, tetragonal lattice
    P4mm = MSG.from_xyz_strings(generators=[
        '-y, x, z, +1', # 4_001
        'y, x, z, +1'   # m_110
    ])
    atoms = [
        Atom(label='Dy', r=(0.5,   0.5, 0),   m=(0,0,5), s=2.5),    # 1b
        Atom(label='Fe', r=(0.1,   0.1, 0),   m=(1,1,0), s=2.5),    # 4d
        ] 
    crystal = Crystal(lattice_parameters=[4,4,10, 90,90,90], MSG=P4mm, atoms=atoms)

    print(crystal)
    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,0,1)
    }

    J1 = Coupling(label='K_Dy', id1=2, id2=2, n_uvw=[0,0,0],
                  J=np.diag([0,0,-0.5]))
    J2 = Coupling(label='K_Fe', id1=0, id2=0, n_uvw=[0,0,0],
                  J=np.array([[-0.2,0,0], [0,-0.2,0], [0,0,0]]))
    J3 = Coupling(label='D1', id1=0, id2=3, n_uvw=[0,0,0],
                  J=5*np.eye(3,3) + DMI(0,0,0.1))

    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=[J1, J2, J3])
    print(sw)

    plot_opts = dict(boundaries=([-1.1, 2.1],[-1.1,2.1],[-0.2,0.2]), coupling_colors={'D1': 'Cyan', 'K':'Pink'})
    plot_structure(sw, engine='vispy', plot_options=plot_opts)

def test_coupling_symmetrization_hex():
    """Test symmetrization of the couplings in hexagonal system"""

    ### Graphite like structure
    atoms = [
        Atom(label='Er', r=(0.5,   0, 0),   m=(0,0,5), s=2.5),    # 3f
        Atom(label='Mn', r=(1/3,   2/3, 0),   m=(1,1,0), s=2.5)]    # 2c
    # atoms = [Atom(label='Fe', r=(0,   0.5, 0),   m=(-1,0,Fz), s=2.5)]
    P6ommm = MSG.from_xyz_strings(generators=[
        '-y, x-y, z, +1', # 3_001
        '-x, -y ,z, +1', # 2_001
        'y, x, -z , +1', # 2_110
        '-x, -y, -z, +1'   # -1
    ])  # Order 24
    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,0,1)
    }

    print(P6ommm.operations)

    crystal = Crystal(lattice_parameters=(4,4,10, 90,90,120),
                      MSG=P6ommm, atoms=atoms)
    
    print(crystal)

    # a1 = crystal.atoms_all[1]
    # a2 = crystal.atoms_all[3]
    # for g in crystal.MSG:
    #     # In real symmetrization there cant be to_UC
    #     print(g, g.transform_position(a1.r, to_UC=True), g.transform_position(a2.r, to_UC=True))

    ds = 0.01
    J1 = Coupling(label='D1', id1=0, id2=1, n_uvw=[0,0,0],
                  J=3*np.eye(3,3) + DMI(0, 0, ds))
    
    print(J1)
    is_DMI, Dvec = crystal.is_respectful_DMI(J1, return_symmetrized=True)
    print(is_DMI, Dvec)

    if True:
        sw = SpinW(crystal=crystal, 
            couplings=[J1],
            magnetic_modulation=magnetic_modulation)
        
        
        # plot_opts = dict(boundaries=([-0.5, 1.5],[-0.5,1.5],[-0.5,1]), coupling_colors={'J1a2': 'Cyan'})
        # plot_opts = dict(boundaries=([-1.1, 2.1],[-1.1,2.1],[-0.2,0.2]), coupling_colors={'D1': 'Cyan'})
        # plot_structure(sw, engine='vispy', plot_options=plot_opts)
    
if __name__ == "__main__":
    # pytest.main()
    test_matrices()

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
