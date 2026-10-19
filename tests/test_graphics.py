# -*- coding: utf-8 -*-
r"""Tests of the optional `graphics` dependencies

Core computations must work without the `graphics` dependencies,
the plotting tests run only if the corresponding libraries are installed.
"""
import subprocess
import sys

import numpy as np
import pytest

from spinwaves import Crystal, MSG, Atom, Coupling, SpinW
import spinwaves.plotting
from spinwaves.plotting import plot_structure, plot_dispersion
from spinwaves.utils.graphics import require_graphics, GRAPHICS_HINT

from .markers import needs_matplotlib, needs_vispy


@pytest.fixture
def fm_chain() -> SpinW:
    """Heisenberg ferromagnetic nearest-neighbor spin chain, SpinW tutorial 1"""
    P1 = MSG.from_xyz_strings(generators=['x, y, z, +1'])
    atoms = [Atom(label='Cu', r=(0,0,0), m=(0,1,0), s=1)]
    crystal = Crystal(lattice_parameters=[3,8,8, 90,90,90], MSG=P1, atoms=atoms)
    couplings = [Coupling(label='Ja', id1=0, id2=0, n_uvw=[1,0,0], J=-np.eye(3,3))]

    return SpinW(crystal=crystal, couplings=couplings,
                 magnetic_modulation={'k':(0,0,0), 'n':(1,0,0)})


### Core without graphics
def test_core_import_without_graphics():
    """Importing `spinwaves` must not import any of the graphics libraries"""
    code = ('import sys, spinwaves; '
            'print(",".join(m for m in ("matplotlib", "vispy", "PyQt6", "pyqtgraph", "networkx") '
            'if m in sys.modules))')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == ''

def test_require_graphics():
    require_graphics('numpy', feature='test')

    with pytest.raises(ImportError, match='not_a_module') as excinfo:
        require_graphics('numpy', 'not_a_module', feature='test')
    assert GRAPHICS_HINT in str(excinfo.value)

def test_plot_structure_missing_engine(fm_chain, monkeypatch):
    monkeypatch.setitem(spinwaves.plotting.SC_ENGINE_REQUIREMENTS, 'vispy', ('not_a_module',))

    with pytest.raises(ImportError, match='graphics'):
        plot_structure(fm_chain, engine='vispy')


### matplotlib
@needs_matplotlib
@pytest.mark.parametrize('plot_type', [
    'dispersion',
    'dispersion_scaled',
    'spectral_weight'])
def test_plot_dispersion(fm_chain, plot_type):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    Qpath = np.linspace([0,0,0], [1,0,0], 21)
    fm_chain.calculate_excitations(Qhkl=Qpath)

    fig, ax = plt.subplots()
    plot_dispersion(fm_chain, ax, xaxis=None, qPath=Qpath, plot_type=plot_type)

    assert ax.has_data()
    plt.close(fig)

@needs_matplotlib
def test_supercell_plotter_mpl(fm_chain):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from spinwaves.plotting.supercell_plotter_mpl import MPLSupercellPlotter

    plotter = MPLSupercellPlotter(fm_chain)
    plotter.plot(dict(boundaries=([0,2],[0,1],[0,1])))
    plt.close(plotter.fig)


### vispy
@needs_vispy
def test_supercell_plotter_vispy(fm_chain):
    """Build the scene without showing it or starting the event loop"""
    from spinwaves.plotting.supercell_plotter_vispy import VispySupercellPlotter

    plotter = VispySupercellPlotter(fm_chain, show=False)
    plotter.plot(dict(boundaries=([0,2],[0,1],[0,1])))
    plotter.canvas.close()
