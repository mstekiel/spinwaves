'''Checks for the optional `graphics` dependencies.

`spinwaves` is primarily a computational library: `numpy` is the only hard
dependency. Plotting and GUI functionalities require additional
packages, collected in the `graphics` extra / dependency group.
'''
import importlib.util
import warnings


GRAPHICS_HINT = ('Install the `graphics` dependencies: `uv sync --group graphics` '
                 'when working in the spinwaves repository, '
                 'or depend on `spinwaves[graphics]` in your project.')


def missing_modules(*modules: str) -> list[str]:
    '''Return names of `modules` that cannot be found, without importing them.'''
    return [module for module in modules if importlib.util.find_spec(module) is None]


def require_graphics(*modules: str, feature: str) -> None:
    '''Raise `ImportError` if any of `modules` required by `feature` is missing.'''
    missing = missing_modules(*modules)
    if missing:
        raise ImportError(f'{feature} requires {", ".join(missing)}. {GRAPHICS_HINT}')


def warn_missing_graphics(*modules: str, feature: str) -> None:
    '''Emit a warning if any of `modules` required by `feature` is missing.'''
    missing = missing_modules(*modules)
    if missing:
        warnings.warn(f'{feature} requires {", ".join(missing)}. {GRAPHICS_HINT}',
                      stacklevel=3)
