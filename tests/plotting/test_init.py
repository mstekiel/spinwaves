r"""Tests of `spinwaves.plotting` (package `__init__`)"""
import numpy as np
import pytest

import spinwaves.plotting
from spinwaves.plotting import plot_structure, plot_dispersion

from ..markers import needs_matplotlib


def test_plot_structure_missing_engine(fm_chain, monkeypatch):
    monkeypatch.setitem(spinwaves.plotting.SC_ENGINE_REQUIREMENTS, 'vispy', ('not_a_module',))

    with pytest.raises(ImportError, match='graphics'):
        plot_structure(fm_chain, engine='vispy')


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
