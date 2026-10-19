'''Fixtures shared across the test suite'''
import numpy as np
import pytest

from spinwaves import Crystal, MSG, Atom, Coupling, SpinW


@pytest.fixture
def fm_chain() -> SpinW:
    """Heisenberg ferromagnetic nearest-neighbor spin chain, SpinW tutorial 1"""
    P1 = MSG.from_xyz_strings(generators=['x, y, z, +1'])
    atoms = [Atom(label='Cu', r=(0,0,0), m=(0,1,0), s=1)]
    crystal = Crystal(lattice_parameters=[3,8,8, 90,90,90], MSG=P1, atoms=atoms)
    couplings = [Coupling(label='Ja', id1=0, id2=0, n_uvw=[1,0,0], J=-np.eye(3,3))]

    return SpinW(crystal=crystal, couplings=couplings,
                 magnetic_modulation={'k':(0,0,0), 'n':(1,0,0)})
