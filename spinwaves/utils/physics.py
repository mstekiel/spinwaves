'''Various functions involved in physical properties calculations'''

import numpy as np

def bose_occupation(energies, temperature):
    """
    Compute the Bose-Einstein occupation factor:
        n(E,T) = 1 / (exp(|E|/(kB*T)) - 1)

    Handles edge cases:
    - E = 0: returns +inf
    - T = 0: returns 0 for E>0, +inf for E=0
    - Uses expm1 for small arguments
    - Uses exp(-x) for large arguments to avoid overflow
    - Threshold for switching depends on dtype precision
    """
    energies = np.asarray(energies)
    dtype = energies.dtype if np.issubdtype(energies.dtype, np.floating) else np.float64
    energies = energies.astype(dtype, copy=False)

    # machine limits
    finfo = np.finfo(dtype)
    # Safe threshold before exp(x) overflows
    threshold = np.log(finfo.max) - 1.0

    # Handle T=0 separately
    if temperature == 0:
        out = np.zeros_like(energies, dtype=dtype)
        out[energies == 0] = np.inf
        return out

    # Normal case: finite T
    kB = 0.08617333262
    x = np.abs(energies) / (kB * temperature)
    out = np.empty_like(x, dtype=dtype)

    # Case: E=0 -> inf
    zero_mask = (energies == 0)
    out[zero_mask] = np.inf

    # Case: moderate x (safe to use expm1)
    mask = (x < threshold) & ~zero_mask
    out[mask] = 1.0 / np.expm1(x[mask])

    # Case: large x -> asymptotic exp(-x)
    big_mask = ~mask & ~zero_mask
    out[big_mask] = np.exp(-x[big_mask])

    return out