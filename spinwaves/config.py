'''Global settings of the `spinwaves` package.

Numerical tolerances
--------------------
`tolerances.numeric`
    Relative tolerance deciding whether a deviation is numerical noise,
    e.g. non-hermiticity of a matrix that is Hermitian by construction.
    Deviations below it, relative to the scale of the checked matrix,
    are corrected silently (logged at DEBUG level), deviations above it
    raise `NumericalToleranceError`.
    Default 1e-10: about 6 orders of magnitude above the round-off of the
    LSWT matrices, well below asymmetries resulting from physical errors.
`tolerances.geometric`
    Tolerance for comparing positions and directions given by the user,
    e.g. fractional coordinates `0.3333`. Not used yet.

Functions read the tolerances at call time, so changes apply immediately::

    import spinwaves.config as config

    config.tolerances.numeric = 1e-8        # global change

    with config.override_tolerances(numeric=1e-6):
        ...                                 # temporary change
'''
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace


@dataclass
class Tolerances:
    numeric: float = 1e-10
    geometric: float = 1e-5


tolerances = Tolerances()


@contextmanager
def override_tolerances(**changes: float):
    '''Temporarily change selected fields of the global `tolerances`.'''
    names = {f.name for f in fields(Tolerances)}
    unknown = set(changes) - names
    if unknown:
        raise TypeError(f'Unknown tolerances {sorted(unknown)}, available: {sorted(names)}')

    previous = replace(tolerances)
    for name, value in changes.items():
        setattr(tolerances, name, value)
    try:
        yield tolerances
    finally:
        for name in names:
            setattr(tolerances, name, getattr(previous, name))


class NumericalToleranceError(ValueError):
    '''A numerical check failed by more than the tolerance.

    Attributes
    ----------
    deviation: float
        Largest relative deviation found.
    tolerance: float
        Tolerance it was compared to.
    indices: list[tuple]
        Batch indices of the failing matrices, empty for unbatched input.
    '''
    def __init__(self, message: str, deviation: float, tolerance: float, indices: list=()):
        super().__init__(message)
        self.deviation = deviation
        self.tolerance = tolerance
        self.indices = list(indices)
