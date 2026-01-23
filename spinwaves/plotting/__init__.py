'''Plotting functionalites of the `spinwaves` package'''


from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from ..spinw import SpinW
    



IMPLEMENTED_SC_ENGINES = ['vispy', 'vispy+', 'mpl', 'qtgraph']

def plot_structure( sws: 'SpinW',
                    engine: str='vispy', 
                    plot_options: dict={}) -> Any:
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
    if engine not in IMPLEMENTED_SC_ENGINES:
        raise NotImplementedError(f'{engine}')
    
    if engine=='vispy':
        from .supercell_plotter_vispy import VispySupercellPlotter as SCPlotter
    if engine=='vispy+':
        from .supercell_plotter_vispy_advanced import AdvancedVispySupercellPlotter as SCPlotter
    if engine=='mpl':
        from .supercell_plotter_mpl import MPLSupercellPlotter as SCPlotter
    if engine=='qtgraph':
        from .supercell_plotter_qtgraph import QtgraphSupercellPlotter as SCPlotter

    # Run chosen application to render the structure
    plotter = SCPlotter(sws, plot_options=plot_options)
    
    return plotter.deploy()