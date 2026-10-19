'''Helpers shared across the test suite: reference systems and numpy 1/2 compatibility checks.'''
import re

import numpy as np

from spinwaves import Crystal, MSG, Atom, Coupling, SpinW


### numpy 2 prints scalars as `np.float64(1.0)`, numpy 1 as `1.0`
NUMPY_SCALAR_REPR = re.compile(r'np\.(float|int|uint|complex|bool|str|bytes|void|object)\w*\(')

def assert_no_numpy_scalar_repr(text: str):
    '''Strings built from numpy scalars must look the same under numpy 1 and 2'''
    match = NUMPY_SCALAR_REPR.search(text)
    assert match is None, f'numpy scalar repr {match.group(0)!r} in {text!r}'


### Reference systems for linear spin wave calculations
P1_GENERATORS = ['x, y, z, +1']

def fm_chain(K: float=0.0) -> SpinW:
    """Heisenberg ferromagnetic nearest-neighbor chain, SpinW tutorial 1, with optional easy axis `K`"""
    crystal = Crystal(lattice_parameters=[3,8,8, 90,90,90], MSG=MSG.from_xyz_strings(generators=P1_GENERATORS),
                      atoms=[Atom(label='Cu', r=(0,0,0), m=(0,1,0), s=1)])
    couplings = [Coupling(label='Ja', id1=0, id2=0, n_uvw=[1,0,0], J=-np.eye(3,3))]
    if K:
        couplings.append(Coupling(label='K', id1=0, id2=0, n_uvw=[0,0,0], J=np.diag([0,-K,0])))

    return SpinW(crystal=crystal, couplings=couplings,
                 magnetic_modulation={'k':(0,0,0), 'n':(1,0,0)})

def afm_chain() -> SpinW:
    """Two-sublattice antiferromagnetic chain with easy-axis anisotropy, degenerate modes"""
    crystal = Crystal(lattice_parameters=[6,8,8, 90,90,90], MSG=MSG.from_xyz_strings(generators=P1_GENERATORS),
                      atoms=[Atom(label='A', r=(0,0,0), m=(0,0,1), s=1),
                             Atom(label='B', r=(0.5,0,0), m=(0,0,-1), s=1)])
    couplings = [Coupling(label='J1', id1=0, id2=1, n_uvw=[0,0,0], J=np.eye(3)),
                 Coupling(label='J2', id1=1, id2=0, n_uvw=[1,0,0], J=np.eye(3)),
                 Coupling(label='K', id1=0, id2=0, n_uvw=[0,0,0], J=np.diag([0,0,-0.05])),
                 Coupling(label='K', id1=1, id2=1, n_uvw=[0,0,0], J=np.diag([0,0,-0.05]))]

    return SpinW(crystal=crystal, couplings=couplings,
                 magnetic_modulation={'k':(0,0,0), 'n':(1,0,0)})

REFERENCE_SYSTEMS = {
    'fm_chain': fm_chain,
    'fm_chain_K': lambda: fm_chain(K=0.1),
    'afm_chain': afm_chain,
}

def reference_Qhkl() -> np.ndarray:
    '''Q points of the stored reference: Q=0, a path along h, and random points'''
    return np.concatenate([np.zeros((1,3)),
                           np.linspace([0,0,0], [1,0,0], 21),
                           np.random.default_rng(0).uniform(-1, 1, (20,3))])

def lswt_invariants(sw: SpinW, Qhkl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Energies and spin-spin correlation summed over modes.

    Single-mode `S` of degenerate modes depends on the arbitrary basis chosen
    by the eigensolver; the sum over modes does not.
    '''
    E, S = sw.determine_ES(Qhkl)
    return np.sort(E.real, axis=-1), S.sum(axis=-3)
