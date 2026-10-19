r"""Tests of `spinwaves.plotting.supercell_plotter_vispy`"""
from ..markers import needs_vispy


@needs_vispy
def test_supercell_plotter_vispy(fm_chain):
    """Build the scene without showing it or starting the event loop"""
    from spinwaves.plotting.supercell_plotter_vispy import VispySupercellPlotter

    plotter = VispySupercellPlotter(fm_chain, show=False)
    plotter.plot(dict(boundaries=([0,2],[0,1],[0,1])))
    plotter.canvas.close()
