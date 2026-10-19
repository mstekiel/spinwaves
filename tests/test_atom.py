r"""Tests of `spinwaves.atom`

Objects to cover:
- `Atom`
"""
import sys
from fractions import Fraction

import numpy as np
import pytest

from spinwaves import Atom

from .helpers import assert_no_numpy_scalar_repr

NUMPY_2 = np.lib.NumpyVersion(np.__version__) >= '2.0.0'
# Python 3.14 `Fraction` accepts any object with `as_integer_ratio`, including `np.float32`
FRACTION_ACCEPTS_FLOAT32 = sys.version_info >= (3, 14)


#########################################################################################
# numpy 1/2 compatibility: positions are stored as exact fractions, independent of the
# numpy scalar types of the input; scalar reprs changed to `np.float64(1.0)`;
# NEP 50 promotion keeps `np.float32` where numpy 1 promoted to `np.float64`

def test_atom_position_fractions():
    a = Atom(label='Fe', r=np.array([0.5, 1/3, 1.25]), m=(0,0,1), s=np.float64(2.5))

    assert all(type(x) is Fraction for x in a._r)
    assert a._r == (Fraction(1,2), Fraction(1,3), Fraction(1,4))
    assert a.r.dtype == np.float64

@pytest.mark.xfail(not FRACTION_ACCEPTS_FLOAT32, strict=True, raises=TypeError,
                   reason='`Fraction` before Python 3.14 rejects `np.float32`, which is not a subclass of `float`')
def test_atom_position_float32():
    a = Atom(label='Fe', r=np.array([0.5, 0.25, 0], dtype=np.float32), m=(0,0,1), s=2.5)
    assert a._r == (Fraction(1,2), Fraction(1,4), 0)

@pytest.mark.xfail(NUMPY_2 and not FRACTION_ACCEPTS_FLOAT32, strict=True, raises=TypeError,
                   reason='NEP 50: `np.float32(x) * 2.0` stays float32 in numpy 2 (float64 in numpy 1), '
                   'which `Fraction` before Python 3.14 rejects')
def test_atom_position_from_numpy_arithmetic():
    x = np.float32(0.25) * 2.0
    a = Atom(label='Fe', r=(x, 0, 0), m=(0,0,1), s=2.5)
    assert a._r == (Fraction(1,2), 0, 0)

def test_atom_eq_hash_independent_of_input_type():
    a = Atom(label='Fe', r=(0.5, 0, 0.25), m=(0,0,1), s=2.5)
    b = Atom(label='Fe', r=np.array([1, 0, 1], dtype=np.int64)/np.array([2, 1, 4], dtype=np.int32), m=(0,0,1), s=2.5)
    c = Atom(label='Fe', r=(np.int32(0), np.int64(0), 0), m=(0,0,1), s=2.5)

    assert a == b
    assert hash(a) == hash(b)
    assert c._r == (0, 0, 0)

def test_atom_repr():
    a = Atom(label='Fe', r=(np.float64(0.5), 0, np.float64(0.1)), m=np.array([0., 0, 1]), s=np.float64(2.5))
    assert_no_numpy_scalar_repr(repr(a))
