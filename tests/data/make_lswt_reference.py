'''Regenerate `lswt_reference.npz` used by `tests/test_spinw.py::test_lswt_reference`.

Run from the repository root, only after a deliberate change of the results:

    uv run python tests/data/make_lswt_reference.py
'''
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
logging.disable(logging.CRITICAL)

from tests.helpers import REFERENCE_SYSTEMS, reference_Qhkl, lswt_invariants


def main():
    Qhkl = reference_Qhkl()
    data = {'Qhkl': Qhkl, 'numpy_version': np.array(np.__version__)}
    for name, make in REFERENCE_SYSTEMS.items():
        data[f'{name}_E'], data[f'{name}_S'] = lswt_invariants(make(), Qhkl)

    path = Path(__file__).with_name('lswt_reference.npz')
    np.savez(path, **data)
    print(f'Saved {path} with numpy {np.__version__}')

if __name__ == '__main__':
    main()
