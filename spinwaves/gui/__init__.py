IMPLEMENTED_GUI_ENGINES = ['vispy', 'plotly', 'qtgraph']

import logging
logger = logging.getLogger('GUI')


def make_window(engine: str='vispy'):
    if engine not in IMPLEMENTED_GUI_ENGINES:
        raise NotImplementedError(f'{engine}')
    
    logger.info('Loading libraries')
    if engine=='vispy':
        from .gui_vispy import WindowVispy as MainWindow
    if engine=='plotly':
        from .gui_plotly import WindowPlotly as MainWindow
    if engine=='qtgraph':
        from .gui_qtgraph import WindowQtGraph as MainWindow
        
    logger.info('Starting GUI')

    return MainWindow()