import inspect
from functools import wraps
from typing import Iterable
import numpy as np

##################################################################################################
# Data types
def make_exc_dtype(Etype: type=np.float64, Stype: type=np.complex128) -> 'np.dtype[np.void]':
    """Create a structured dtype with:
    - E: scalar of real type `Etype`
    - Sperp: scalar of real type `Etype`
    - S: 3x3 matrix of complex type `Stype`
    - Sxx...Szz: aliases into S elements

    `E` represents the energy of the excitation, `S` the spin-spin correlation matrix,
    `Sperp` is the perpendicular component of S to the momentum transfer Q,
    `Sij` with i,j in [x,y,z] are aliases into the S matrix.

    Parameters
    ----------
    Etype: type, optional
        Data type for the energy and Sperp scalars, default np.float64
    Stype: type, optional
        Data type for the spin-spin correlation function matrix, default np.complex128
        
    Returns
    -------
    Numpy datatype storing information about excitation's energy and spin-spin correlation matrix (S)
    with feasible shortcuts for S elements.

    Notes
    -----
    ChatGPT made it. I am not sure the offset calculations are correct,
    in particular, that the S matrix is aligned properly with its components.
    Test it properly.
    """
    # Sizes in bytes
    esize = np.dtype(Etype).itemsize
    ssize = np.dtype(Stype).itemsize

    # Build names, formats, and offsets
    offset_S = 2*esize
    names = ["E", "Sperp", "S"]
    formats = [Etype, Etype, (Stype, (3, 3))]
    offsets = [0, esize, offset_S]

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

def create_mesh(v1: Iterable, v2: Iterable, v3: Iterable):
    '''Create an array mesh of shape=(...,3) from the input arrays.'''

    X, Y, Z = np.meshgrid(v1,v2,v3)
    return np.stack((X,Y,Z), axis=-1).squeeze()