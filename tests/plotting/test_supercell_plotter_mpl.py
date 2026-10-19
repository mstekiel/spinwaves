r"""Tests of `spinwaves.plotting.supercell_plotter_mpl`"""
from ..markers import needs_matplotlib


@needs_matplotlib
def test_supercell_plotter_mpl(fm_chain):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from spinwaves.plotting.supercell_plotter_mpl import MPLSupercellPlotter

    plotter = MPLSupercellPlotter(fm_chain)
    plotter.plot(dict(boundaries=([0,2],[0,1],[0,1])))
    plt.close(plotter.fig)
