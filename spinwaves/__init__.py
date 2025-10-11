from .atom import Atom
from .crystal import Crystal
from .symmetry.magnetic_symmetry import MSG, mSymOp

from .spinw import SpinW, Coupling
# from plottin
# from . import plotting_plotly
#from spinwaves.plotting_pyspinw import SuperCell


# setup_logging
import logging
import logging.config
logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            # 'format': '%(levelname)s: %(message)s'
            'format': '[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s.%(msecs)03d: %(message)s',
            'datefmt': '%Y-%m-%dT%H:%M:%S%z'
        }
    },
    'handlers': {
        'stdout': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        }
    },
    'loggers': {
        'root': {
            'level': 'WARNING',
            'handlers': ['stdout']
        }
    }
}
logging.config.dictConfig(logging_config)