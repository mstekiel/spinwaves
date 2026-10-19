'''Graphical user interfaces of the `spinwaves` package.

Requires the optional `graphics` dependencies (`uv sync --group graphics`,
or `spinwaves[graphics]`).
'''
IMPLEMENTED_GUI_ENGINES = ['vispy', 'plotly', 'qtgraph']
GUI_ENGINE_REQUIREMENTS = {
    'vispy':    ('vispy', 'PyQt6'),
    'plotly':   ('plotly', 'PyQt5'),    # Not part of the `graphics` dependencies
    'qtgraph':  ('pyqtgraph', 'PyQt6', 'OpenGL'),
}

import logging
logger = logging.getLogger('GUI')

from ..utils.graphics import require_graphics, warn_missing_graphics

warn_missing_graphics('PyQt6', feature='spinwaves.gui')


def make_window(engine: str='vispy'):
    if engine not in IMPLEMENTED_GUI_ENGINES:
        raise NotImplementedError(f'{engine}')

    require_graphics(*GUI_ENGINE_REQUIREMENTS[engine], feature=f'make_window(engine={engine!r})')

    logger.info('Loading libraries')
    if engine=='vispy':
        from .gui_vispy import WindowVispy as MainWindow
    if engine=='plotly':
        from .gui_plotly import WindowPlotly as MainWindow
    if engine=='qtgraph':
        from .gui_qtgraph import WindowQtGraph as MainWindow

    logger.info('Starting GUI')

    return MainWindow()
