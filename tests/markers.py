'''Skip markers for tests requiring optional dependencies.'''
import pytest

from spinwaves.utils.graphics import missing_modules


def requires(*modules: str, group: str='graphics') -> pytest.MarkDecorator:
    missing = missing_modules(*modules)
    return pytest.mark.skipif(bool(missing),
                              reason=f'requires {", ".join(missing)} (`{group}` dependencies)')


needs_matplotlib = requires('matplotlib')
needs_vispy = requires('vispy', 'PyQt6')
needs_networkx = requires('networkx', 'matplotlib')
needs_scipy = requires('scipy', group='dev')
