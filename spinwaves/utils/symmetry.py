from copy import deepcopy
import numpy as np

from ..coupling import Coupling
from ..atom import Atom

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..symmetry.magnetic_symmetry import mSymOp
    from ..crystal import Crystal

def transform_coupling(g: 'mSymOp', cpl: 'Coupling', crystal: 'Crystal'):
    '''Transform coupling `cpl` according do the symmetry operation `g`
    in a `crystal.
    
    Usage with MSG.symmetrize(...)
    >>> cs = Crystal(...)
    >>> g_cpl = lambda g, c: transform_coupling(g, c, cs)
    >>> MSG.symmetrize(coupling, g_cpl)
    '''
    atom1 = crystal.atoms_magnetic[cpl.id1]
    atom2 = crystal.atoms_magnetic[cpl.id2]

    r1 = g.transform_position(atom1.r)
    n_uvw1 = np.floor(r1)
    new_id1 = crystal.get_atom_sw_id(r1)

    r2 = g.transform_position(atom2.r+cpl.n_uvw)
    n_uvw2 = np.floor(r2)
    new_id2 = crystal.get_atom_sw_id(r2)

    new_n_uvw_12 = n_uvw2 - n_uvw1

    # g matrix has to be represented in xyz coordinates
    g_xyz = crystal.A @ g.matrix @ np.linalg.inv(crystal.A)
    new_J = g_xyz @ cpl.J @ np.linalg.inv(g_xyz)

    # If self interaction, anisotropy
    # we need to correct for the unnecesary double counting
    if np.allclose(r1, r2):
        # logger.info(f'Correcting for double-counting, {cpl}')
        new_J *= 2

    cpl_new  = Coupling(label=f'{cpl.label}_({g.to_string()})',
                    n_uvw=new_n_uvw_12,
                    id1=new_id1,
                    id2=new_id2,
                    J = new_J,
                    defining_bond=cpl.label)

    return cpl_new


def transform_atom(g: 'mSymOp', a: 'Atom', crystal: 'Crystal'):
    atom_new = deepcopy(a)
    atom_new.r = g.transform_position(a.r, to_UC=True)

    return atom_new