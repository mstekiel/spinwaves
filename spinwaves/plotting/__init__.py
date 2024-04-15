from typing import Tuple, Union

from ..spinw import SpinW
from .supercell_plotter_vispy import VispySupercellPlotter

# class SupercellPlotter: ...
# class MPLSupercellPlotter(SupercellPlotter): ...
# class VispySupercellPlotter(SupercellPlotter): ...
# class PlotlySupercellPlotter(SupercellPlotter): ...

supercell_plotters = {
        'vispy': VispySupercellPlotter,
        # 'mpl': MPLSupercellPlotter,
        # 'plotly': PlotlySupercellPlotter
    }

implemented_sc_plotters = supercell_plotters.keys()

def plot_structure( sws: SpinW,
                    engine: str='vispy', 
                    plot_options: dict={}) -> None:
    '''Render the 3D crystal structure and couplings
    
    Parameters
    ----------
    sws: SpinW
        Spin waves hosting system.
    boundaries: ArrayLike
        Boundaries of the supercell to be plot
    engine: str
        Library used to prepare the plot
    plot_options: dict
        Additional options for the plotting

    Returns
    -------
        Library specific objects handling the plotting widget
    '''
    # Run chosen application to render the structure
    plotter = supercell_plotters[engine](sws)
    # plotter.__init__(sws=sws)
    
    return plotter.plot(plot_options)