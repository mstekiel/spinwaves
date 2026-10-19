r"""Tests of `spinwaves.config`"""
import pytest

import spinwaves.config as config
from spinwaves.config import NumericalToleranceError, Tolerances, override_tolerances


def test_default_tolerances():
    assert Tolerances().numeric == 1e-10
    assert Tolerances().geometric == 1e-5

def test_override_tolerances_restores():
    before = (config.tolerances.numeric, config.tolerances.geometric)
    with override_tolerances(numeric=1e-6) as tol:
        assert tol is config.tolerances
        assert config.tolerances.numeric == 1e-6
        assert config.tolerances.geometric == before[1]
    assert (config.tolerances.numeric, config.tolerances.geometric) == before

def test_override_tolerances_nested():
    with override_tolerances(numeric=1e-6):
        with override_tolerances(numeric=1e-4, geometric=1e-3):
            assert config.tolerances.numeric == 1e-4
        assert config.tolerances.numeric == 1e-6
        assert config.tolerances.geometric == Tolerances().geometric

def test_override_tolerances_restores_on_error():
    with pytest.raises(RuntimeError):
        with override_tolerances(numeric=1e-6):
            raise RuntimeError
    assert config.tolerances.numeric == Tolerances().numeric

def test_override_tolerances_unknown():
    with pytest.raises(TypeError, match='numerc'):
        with override_tolerances(numerc=1e-6):
            pass

def test_numerical_tolerance_error():
    e = NumericalToleranceError('msg', deviation=1e-3, tolerance=1e-10, indices=[(1,)])
    assert isinstance(e, ValueError)
    assert (str(e), e.deviation, e.tolerance, e.indices) == ('msg', 1e-3, 1e-10, [(1,)])
