import inspect
from functools import wraps
from typing import Iterable
import numpy as np

##################################################################################################
# Data types
def make_exc_dtype(Qhkltype: type=np.float64, Etype: type=np.float64, Stype: type=np.complex128) -> 'np.dtype[np.void]':
    """Create a structured dtype with:
    - Qhkl: 3D vector of real type `Qhkltype`
    - E: scalar of real type `Etype`
    - Sperp: scalar of real type `Etype`
    - S: 3x3 matrix of complex type `Stype`
    - Sxx...Szz: aliases into S elements

    `Qhkl` is momentum transfer vector for which the excitations are calculated,
    `E` represents the energy of the excitation, `S` the spin-spin correlation matrix,
    `Sperp` is the perpendicular component of S to the momentum transfer Q,
    `Sij` with i,j in [x,y,z] are aliases into the S matrix.

    Parameters
    ----------
    Qhkltype: type, optional
        Data type for the momentum transfer vector, default np.float64
    Etype: type, optional
        Data type for the energy and Sperp scalars, default np.float64
    Stype: type, optional
        Data type for the spin-spin correlation function matrix, default np.complex128
        
    Returns
    -------
    Numpy datatype storing information about excitation's energy and spin-spin correlation matrix (S)
    with feasible shortcuts for S elements.
    """
    # Sizes in bytes
    qsize = np.dtype(Qhkltype).itemsize
    esize = np.dtype(Etype).itemsize
    ssize = np.dtype(Stype).itemsize

    # Build names, formats, and offsets
    offset_E = 3*qsize
    offset_S = offset_E + 2*esize
    names = ["Qhkl", "E", "Sperp", "S"]
    formats = [(Qhkltype, (3,)), Etype, Etype, (Stype, (3, 3))]
    offsets = [0, offset_E, offset_E + esize, offset_S]

    # Add aliases for each element of S
    labels = ["x", "y", "z"]
    for i, row in enumerate(labels):
        for j, col in enumerate(labels):
            name = f"S{row}{col}"
            names.append(name)
            formats.append(Stype)
            offsets.append(offset_S + (i * 3 + j) * ssize)

    # Construct dtype
    return np.dtype({"names": names, "formats": formats, "offsets": offsets})

##################################################################################################
# Shape matching
def match_shape(arr_shape: tuple[int, ...], expected_shape: tuple[int, ...]) -> bool:
    """
    Recursively match `arr_shape` against `expected_shape` with ... wildcards.
    Supports multiple ellipses anywhere.
    """
    # print(f'DEBUG: matching: {arr_shape=} with {expected_shape=}')
    if not expected_shape:
        return not arr_shape

    head, *tail = expected_shape

    if head is ...:
        # try all possible splits: let ... absorb 0,1,2,... dimensions
        for k in range(len(arr_shape) + 1):
            if match_shape(arr_shape[k:], tuple(tail)):
                return True
        return False

    if not arr_shape or arr_shape[0] != head:
        return False

    return match_shape(arr_shape[1:], tuple(tail))

def ensure_shape(**shapes):
    """
    Decorator to validate numpy array arguments by name and shape.

    Usage
    -----
    @ensure_shape(
        a=(..., 3),          # last dimension must be 3
        b=(2, ...),          # first dimension must be 2
        c=(2, 3, ..., 3, 3)  # fixed prefix and suffix, flexible middle
    )
    def func(a, b, c): ...
    """
    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for name, expected_shape in shapes.items():
                if name not in bound.arguments:
                    raise ValueError(f"Parameter '{name}' not found in function arguments")
                
                val = bound.arguments.get(name)
                if val is None:
                    continue

                arr_shape = np.shape(val)
                if not match_shape(arr_shape, expected_shape):
                    raise ValueError(
                        f"[{func.__name__}] Parameter '{name}': "
                        f"expected shape {expected_shape}, got {arr_shape}"
                    )

            return func(*args, **kwargs)

        return wrapper
    
    return decorator

##################################################################################################
# Array creation

def create_mesh(*vs: Iterable) -> np.ndarray:
    '''Create a coordinate mesh from N arrays or scalars.

    Each input can be a 1D array or a scalar. Returns an array of shape
    `(*sizes, N)` where `result[i, j, ..., :]` holds the coordinates
    of that grid point. Scalar inputs are broadcast and squeezed out of
    the spatial dimensions.
    
    >>> m = create_mesh([1,2], [3,4], 0)
    array([[[1, 3, 0],
            [1, 4, 0]],

           [[2, 3, 0],
            [2, 4, 0]]])
    >>> m.shape
    (2, 2, 3)
    '''

    arrays = [np.atleast_1d(v) for v in vs]
    grids = np.meshgrid(*arrays, indexing='ij')
    return np.stack(grids, axis=-1).squeeze()