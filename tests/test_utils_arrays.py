import numpy as np
import pytest

from spinwaves.utils.arrays import create_mesh


def test_create_mesh_doctest():
    result = create_mesh([1, 2], [3, 4], 0)
    expected = np.array([[[1, 3, 0], [1, 4, 0]],
                         [[2, 3, 0], [2, 4, 0]]])
    assert np.array_equal(result, expected)


def test_create_mesh_shape():
    result = create_mesh([1, 2, 3], [4, 5], [6, 7, 8, 9])
    assert result.shape == (3, 2, 4, 3)


def test_create_mesh_values():
    result = create_mesh([10, 20], [30, 40], [50, 60])
    assert np.array_equal(result[0, 1, 0], [10, 40, 50])
    assert np.array_equal(result[1, 0, 1], [20, 30, 60])


def test_create_mesh_scalar_squeezed():
    result = create_mesh([1, 2], 5, [3, 4])
    assert result.shape == (2, 2, 3)
    assert np.array_equal(result[0, 0], [1, 5, 3])
    assert np.array_equal(result[1, 1], [2, 5, 4])


def test_create_mesh_single_arg():
    result = create_mesh([1, 2, 3])
    assert result.shape == (3,)
    assert np.array_equal(result, [1, 2, 3])



if __name__ == "__main__":
    test_create_mesh_doctest()
    test_create_mesh_shape()
    test_create_mesh_values()
    test_create_mesh_scalar_squeezed()
    test_create_mesh_single_arg()