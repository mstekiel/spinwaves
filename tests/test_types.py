r"""Tests dtypes for various dataholders"""
import numpy as np

from spinwaves.spinw import make_exc_dtype

def test_exc_dtype():
    Etype = np.float64
    Stype = np.complex128

    dt = make_exc_dtype(Etype, Stype)

    # Create small array
    arr = np.zeros(3, dtype=dt)

    # --- Test E ---
    arr["E"][0] = 42.0
    assert arr["E"][0] == 42.0
    assert np.allclose(arr['Sperp'], 0)
    assert np.allclose(arr['S'], 0)

    # --- Test Sperp ---
    arr["Sperp"][0] = 69.0
    assert arr["Sperp"][0] == 69.0
    assert arr["E"][0] == 42.0
    assert np.allclose(arr['S'], 0)

    # --- Test S full matrix ---
    mat = np.arange(9).reshape(3, 3) + 1j * np.arange(9).reshape(3, 3)
    arr["S"][0] = mat
    np.testing.assert_array_equal(arr["S"][0], mat)

    # --- Test alias fields map correctly ---
    assert arr["Sxx"][0] == mat[0, 0]
    assert arr["Sxy"][0] == mat[0, 1]
    assert arr["Sxz"][0] == mat[0, 2]
    assert arr["Syx"][0] == mat[1, 0]
    assert arr["Syy"][0] == mat[1, 1]
    assert arr["Syz"][0] == mat[1, 2]
    assert arr["Szx"][0] == mat[2, 0]
    assert arr["Szy"][0] == mat[2, 1]
    assert arr["Szz"][0] == mat[2, 2]
    assert arr["Sperp"][0] == 69.0
    assert arr["E"][0] == 42.0

    # --- Test aliasing (modifying S updates fields) ---
    arr["S"][0, 1, 2] = 123 + 456j
    assert arr["Syz"][0] == 123 + 456j

    # --- Test reverse aliasing (modifying field updates S) ---
    arr["Sxx"][0] = -99 + 1j
    assert arr["S"][0, 0, 0] == -99 + 1j

    # --- Test dtype offsets correctness ---
    ssize = np.dtype(Stype).itemsize
    base_offset = dt.fields["S"][1]
    assert dt.fields["Sxx"][1] == base_offset + 0 * ssize
    assert dt.fields["Syy"][1] == base_offset + 4 * ssize
    assert dt.fields["Szz"][1] == base_offset + 8 * ssize

def test_exc_array():
    shape = (3,4)
    exc_dtype = make_exc_dtype()
    excitations = np.rec.array(np.full(shape=shape, fill_value=0, dtype=exc_dtype))

    excitations.Sxx = 1
    excitations.Syy = 2
    excitations.Szz = 3

    assert np.allclose(excitations.E, 0)
    assert np.allclose(excitations.Sperp, 0)
    assert np.allclose(excitations.S[0,0], np.diag([1,2,3]))



if __name__ == "__main__":
    # pytest.main()
    test_exc_dtype()
    test_exc_array()