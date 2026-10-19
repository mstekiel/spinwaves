# -*- coding: utf-8 -*-
r"""Tests lattice math

"""
from pathlib import Path

import numpy as np
from fractions import Fraction
import pytest

from spinwaves import Crystal, MSG, Atom, Coupling, SpinW
from spinwaves.plotting import plot_structure   # this takes some serious time
from spinwaves.utils.linalg import DMI

from spinwaves.config import NumericalToleranceError, override_tolerances

from .markers import needs_vispy
from .helpers import REFERENCE_SYSTEMS, assert_no_numpy_scalar_repr, lswt_invariants

'''
# Some depracated test, maybe useful in the future
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
    print('E0 = ', sw.calculate_ground_state(Qhkl=[0,0,0]))
    
    S = sw.determine_matrices(q_hkl = [0.1, 0.1, 0])
    print(S)
'''

@needs_vispy
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
                  J=5*np.eye(3,3) + DMI([0,0,0.1]))

    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=[J1, J2, J3])
    print(sw)

    ### Show the structure for setting up the test
    # plot_opts = dict(boundaries=([-1.1, 2.1],[-1.1,2.1],[-0.2,0.2]), coupling_colors={'D1': 'Cyan', 'K':'Pink'})
    # plot_structure(sw, engine='vispy', plot_options=plot_opts)

    # TODO: Add some asserts to check the symmetrization of the couplings

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
                  J=3*np.eye(3,3) + DMI([0, 0, ds]))
    
    print(J1)
    is_DMI, Dvec = crystal.is_respectful_DMI(J1, return_symmetrized=True)
    print(is_DMI, Dvec)

    ### Show the structure for setting up the test
    if False:
        sw = SpinW(crystal=crystal, 
            couplings=[J1],
            magnetic_modulation=magnetic_modulation)
        
        
        # plot_opts = dict(boundaries=([-0.5, 1.5],[-0.5,1.5],[-0.5,1]), coupling_colors={'J1a2': 'Cyan'})
        # plot_opts = dict(boundaries=([-1.1, 2.1],[-1.1,2.1],[-0.2,0.2]), coupling_colors={'D1': 'Cyan'})
        # plot_structure(sw, engine='vispy', plot_options=plot_opts)


    # TODO: Add some asserts to check the symmetrization of the couplings
    
### Numerical stability, see `spinwaves.config`
def test_fm_chain_dispersion(fm_chain):
    """E(h) = 2|J|S(1 - cos 2 pi h) for the ferromagnetic chain, SpinW tutorial 1"""
    Q = np.linspace([0.05,0,0], [0.95,0,0], 19)
    E = fm_chain.determine_ES(Q, includeS=False)

    assert np.allclose(np.max(E, axis=-1), 2*(1-np.cos(2*np.pi*Q[:,0])))
    assert np.allclose(np.min(E, axis=-1), -np.max(E, axis=-1))

@pytest.mark.parametrize('rtol', [1e-10, 1e-6])
def test_goldstone_gap(fm_chain, rtol):
    """Zero mode at Q=0 is regularized: the gap is set by the numerical tolerance.

    For the ferromagnetic chain `h(Q=0)` is the zero matrix, shifted by `rtol`.
    """
    with override_tolerances(numeric=rtol):
        E = fm_chain.determine_ES(np.zeros((1,3)), includeS=False)

    assert np.allclose(np.abs(E), rtol, rtol=1e-3)

def test_not_ground_state_raises():
    """Antiferromagnetic coupling on a ferromagnetic structure: `h` is not positive definite"""
    P1 = MSG.from_xyz_strings(generators=['x, y, z, +1'])
    crystal = Crystal(lattice_parameters=[3,8,8, 90,90,90], MSG=P1, atoms=[Atom(label='Cu', r=(0,0,0), m=(0,1,0), s=1)])
    sw = SpinW(crystal=crystal, couplings=[Coupling(label='Ja', id1=0, id2=0, n_uvw=[1,0,0], J=np.eye(3,3))],
               magnetic_modulation={'k':(0,0,0), 'n':(1,0,0)})

    with pytest.raises(NumericalToleranceError, match='not the ground state') as excinfo:
        sw.determine_ES(np.linspace([0,0,0], [1,0,0], 5))
    assert excinfo.value.indices


#########################################################################################
# numpy 1/2 compatibility

def tetragonal_system(generators=('-y, x, z, +1', 'y, x, z, +1')) -> SpinW:
    """P4mm with on-site anisotropy and a DMI bond; not a ground state, used for symmetrization"""
    P4mm = MSG.from_xyz_strings(generators=list(generators))
    atoms = [Atom(label='Dy', r=(0.5,0.5,0), m=(0,0,5), s=2.5), Atom(label='Fe', r=(0.1,0.1,0), m=(1,1,0), s=2.5)]
    crystal = Crystal(lattice_parameters=[4,4,10, 90,90,90], MSG=P4mm, atoms=atoms)
    couplings = [Coupling(label='K_Fe', id1=0, id2=0, n_uvw=[0,0,0], J=np.array([[-0.2,0,0],[0,-0.2,0],[0,0,0]])),
                 Coupling(label='D1', id1=0, id2=3, n_uvw=[1,0,0], J=5*np.eye(3) + DMI([0,0,0.1]))]
    return SpinW(crystal=crystal, couplings=couplings, magnetic_modulation={'k':(0,0,0), 'n':(0,0,1)})

def bond(cpl: Coupling) -> tuple:
    return (cpl.id1, cpl.id2, tuple(int(n) for n in cpl.n_uvw))

def test_symmetrize_couplings_bonds():
    """Bonds and their J after symmetrization are independent of the generator order,
    and every bond comes with its reverse with transposed J
    """
    sw = tetragonal_system()
    sw_swapped = tetragonal_system(generators=('y, x, z, +1', '-y, x, z, +1'))
    J = {bond(c): c.J for c in sw.couplings_all}
    J_swapped = {bond(c): c.J for c in sw_swapped.couplings_all}

    assert len(J) == len(sw.couplings_all) == 12
    assert J.keys() == J_swapped.keys()
    for b, Jb in J.items():
        assert np.allclose(Jb, J_swapped[b])
        reverse = (b[1], b[0], tuple(-n for n in b[2]))
        assert np.allclose(J[reverse], Jb.T)

@pytest.mark.xfail(strict=True, reason='`np.unique` on the object array of couplings keeps an unspecified one of '
                   'the equal couplings; labels of the survivors differ between numpy 1.23 and 1.26+')
def test_symmetrize_couplings_keeps_first_generated(monkeypatch):
    """Each bond is generated twice (from the coupling and its reverse);
    a deterministic rule, here the first generated, makes labels reproducible
    """
    generated = {}
    unique = np.unique
    def spy(ar, *args, **kwargs):
        generated['all'] = list(ar)
        return unique(ar, *args, **kwargs)
    monkeypatch.setattr(np, 'unique', spy)
    sw = tetragonal_system()
    monkeypatch.undo()

    first = {}
    for c in generated['all']:
        first.setdefault(bond(c), c.label)
    assert {bond(c): c.label for c in sw.couplings_all} == first

def test_lswt_reference():
    """Energies and mode-summed S against stored values, see `tests/data/make_lswt_reference.py`.

    The reference was verified to agree between numpy 1.23 and 2.4 to 1e-14.
    """
    ref = np.load(Path(__file__).parent / 'data' / 'lswt_reference.npz')
    for name, make in REFERENCE_SYSTEMS.items():
        E, S = lswt_invariants(make(), ref['Qhkl'])
        assert np.allclose(E, ref[f'{name}_E'], rtol=1e-10, atol=1e-12), name
        assert np.allclose(S, ref[f'{name}_S'], rtol=1e-10, atol=1e-12), name

def test_spinw_repr(fm_chain):
    assert_no_numpy_scalar_repr(repr(fm_chain))
    for c in tetragonal_system().couplings_all:
        assert_no_numpy_scalar_repr(c.label)


if __name__ == "__main__":
    # pytest.main()
    pass
    # test_matrices()

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
