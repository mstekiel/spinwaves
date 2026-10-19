'''Skip markers for tests requiring the optional `graphics` dependencies.'''
import pytest

from spinwaves.utils.graphics import missing_modules


def requires(*modules: str) -> pytest.MarkDecorator:
    missing = missing_modules(*modules)
    return pytest.mark.skipif(bool(missing),
                              reason=f'requires {", ".join(missing)} (`graphics` dependencies)')


needs_matplotlib = requires('matplotlib')
needs_vispy = requires('vispy', 'PyQt6')
needs_networkx = requires('networkx', 'matplotlib')
