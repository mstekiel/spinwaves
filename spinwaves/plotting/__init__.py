'''Plotting functionalites of the `spinwaves` package

Requires the optional `graphics` dependencies (`uv sync --group graphics`,
or `spinwaves[graphics]`). Graphics libraries must not be imported at module
level, missing libraries are reported when a plotting function is called.
'''


from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from ..spinw import SpinW

import numpy as np

from ..utils.functions import gauss_bkg
from ..utils.graphics import require_graphics



IMPLEMENTED_SC_ENGINES = ['vispy', 'vispy+', 'mpl', 'qtgraph']
SC_ENGINE_REQUIREMENTS = {
    'vispy':    ('vispy', 'PyQt6'),
    'vispy+':   ('vispy', 'PyQt6', 'OpenGL'),
    'mpl':      ('matplotlib',),
    'qtgraph':  ('pyqtgraph', 'PyQt6', 'OpenGL'),
}

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
    engine: str ['vispy', 'vispy+', 'mpl', 'qtgraph']
        Library used to prepare the plot
    plot_options: dict
        Additional options for the plotting

    Returns
    -------
        Library specific objects handling the plotting widget
    '''
    if engine not in IMPLEMENTED_SC_ENGINES:
        raise NotImplementedError(f'{engine}')

    require_graphics(*SC_ENGINE_REQUIREMENTS[engine], feature=f'plot_structure(engine={engine!r})')

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


def plot_dispersion(sws: 'SpinW', ax: 'plt.Axes', xaxis: np.ndarray, qPath: np.ndarray,
                    plot_type: str='dispersion', plot_kwargs: dict={},
                    ret_data: bool=False) -> tuple['plt.Axes', list]:
    '''
    Plot dispersions

    Parameters
    ----------
    sws: SpinW
        Spin waves hosting system with calculated `excitations`.
    ax: pyplot.Axes
        Axes on which to make the plot
    xaxis: numpy.ndarray, optional
        Array of x values for the plot.
    plot_type: str, optional
        Plot type from ['dispersion', 'dispersion_scaled', 'spectral_weight'].
    plot_kwargs: dict
        Additional kwargs passed to the plotting functions.
    ret_data: bool, optional
        If True, return the data used for plotting.

    Returns
    -------
    ax: pyplot.Axes
        Axes with the plot.
    ret_data: list, optional
        If `ret_data` is True, return the data used for plotting.
    '''

    plot_kwargs = dict(plot_kwargs)     # Don't modify the caller's (or the default) dict
    plot_kwargs.setdefault('cmap', 'afmhot_r')
    plot_kwargs.setdefault('vmax', None)

    # Nice property of this array is that for any change in direction it will keep the same value
    Qinc = np.concatenate(([0], np.linalg.norm( qPath[:-1] - qPath[1:], axis=1)))

    if xaxis is not None:
        x_arg = xaxis
    else:
        x_arg = np.cumsum(Qinc)

    ax.set_xlabel('Q ((h,k,l))')
    ax.set_ylabel('E (meV)')

    # Mask to where put the xticks:
    # (1) main qpoints, (2) the last one, (3) integer positions
    it1 = (Qinc==0)

    it2 = np.zeros(len(Qinc), dtype=bool)
    it2[0] = it2[-1] = True

    it3 = (np.linalg.norm(qPath - qPath.round(), axis=1) == 0)

    xticks_it = it1 | it2 | it3
    xticks = x_arg[xticks_it]
    xtickslabels = ['\n'.join([f'{x:.2f}' for x in q]) for q in qPath[xticks_it]]
    ax.set_xticks(xticks, labels=xtickslabels)


    ### Plot type
    data = None
    if plot_type == 'dispersion':
        Es = sws.excitations.E
        x = x_arg.repeat(2*len(sws.magnetic_atoms))

        ax.scatter(x, Es, **plot_kwargs)    # 0 branch

        data = [x_arg, Es]
    elif plot_type == 'dispersion_scaled':
        Es = sws.excitations.E
        Is = sws.excitations.Sperp
        Is -= Is.min()

        s = 10 + 100*Is/Is.max()
        c = np.power(Is/Is.max(), 0.1)
        plot_kwargs.pop('alpha', None)
        plot_kwargs.pop('color', None)

        # Flatten objects for plotting
        x = x_arg.repeat(2*len(sws.magnetic_atoms))
        y = Es.flatten()
        # z = Is.flatten()

        ax.scatter(x, y, s=s, c=c, **plot_kwargs)    # 0 branch
        data = [x, y, s]
    elif plot_type == 'spectral_weight':

        Egrid = []
        def yvals(xvals, Es, Is):
            y = np.zeros(len(xvals))
            for x0, A in zip(Es, Is):
                sigma = 1 #+ 0.03*x0     # Imitates energy resolution
                y += gauss_bkg(xvals, x0=x0, A=A/sigma, sigma=sigma, bkg=0)

            return y

        Erange = np.linspace(0, 100, 400)
        Es = sws.excitations.E
        Is = sws.excitations.Sperp
        for En, In in zip(Es, Is):
            Egrid.append(yvals(Erange, En, In))

        Egrid = np.transpose(Egrid)

        cmap = plot_kwargs.pop('cmap', 'RdBu')

        ax.pcolormesh(x_arg, Erange, Egrid, cmap=cmap, vmax=plot_kwargs['vmax'])
        data = [x_arg, Erange, Egrid]
    else:
        raise KeyError(f"Unknown plot_type {plot_type!r}")

    # Set up the return values
    ret = ax
    if ret_data:
        ret = (ax, data)

    return ret