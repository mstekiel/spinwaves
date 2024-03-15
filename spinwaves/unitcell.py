# Core
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import warnings
from dataclasses import dataclass, field

# Plotting
import numpy as np
from vispy import scene
# from vispy.color import color_array
# from itertools import chain
# from vispy.visuals.filters import ShadingFilter, WireframeFilter
# from vispy.geometry import create_sphere
# import copy
# from scipy.spatial.transform import Rotation
# from scipy.spatial import ConvexHull
# from dataclasses import dataclass

# Typing
from matplotlib.figure import Figure
from typing import List, Tuple, Dict, Union
from numpy.typing import NDArray

# Internal
from .data_containers import atom_data, color_data


@dataclass
class Atom:
    '''
    Stores informations about the atom.

    Questions:
    - what is index?
    - moment in what units?
    - position in what units?
    '''
    r: np.ndarray
    m: np.ndarray = np.zeros((3))
    s: float = 0
    is_mag: bool = False
    index: int = -1
    label: str = 'atom'
    element_symbol: str = None
    color: str = None
    radius: float = 0
    spin_scale: float = 1
    gtensor_mat: np.ndarray = None
    aniso_mat: np.ndarray = None

    def __post_init__(self):
        '''
        Assert field formats, and fill the correlated fields
        '''

        # Position
        self.r = np.array(self.r)
        if not self.r.shape == (3,):
            raise ValueError(f'Atomic position must be a (3,) vector now is: {self.r.shape}')
        
        # Magnetic moment
        self.m = np.array(self.m)
        if self.m.size == 0:
            pass
        if not self.r.shape == (3,):
            raise ValueError(f'Atomic position must be a (3,) vector now is: {self.r.shape}')

        # Is magnetic atom
        if np.linalg.norm(self.m) > 1e-10 and (np.abs(self.s)>0):
            self.is_mag = True

        # Element symbol as input or derived from the label 
        if not self.element_symbol:
            # token = ''.join([])
            if self.label[:2] in atom_data:
                self.element_symbol = self.label[:2]
            elif self.label[:1] in atom_data:
                self.element_symbol = self.label[:1]

        # Color as input or derived from the element symbol
            if self.color:
                pass
            elif self.element_symbol:
                self.color = atom_data[self.element_symbol]['RGB']
            else:
                self.color = color_data['Black']

        # Atom radius
        if not bool(self.radius):
            if self.element_symbol:
                self.radius = atom_data[self.element_symbol]['radius']
            else:
                self.radius = 0.3
        
class UnitCell:

    def __init__(self, 
                 atoms:  Union[Dict, List[Dict], Atom, List[Atom]], 
                 symmetry: Union[str, int]=None, 
                 origin: np.ndarray=np.zeros((3))):
        '''
        Parameters:
        -----------
        symmetry:
            Name or number of the space group, or None.
        '''


        # Handle creating list of atoms
        if isinstance(atoms, Dict) or isinstance(atoms, Atom):
            atoms = [atoms]
        elif isinstance(atoms, list):
            pass
        else:
            raise ValueError(f'Unrecognised format for `Atom`: {type(atoms)}')
        
        if isinstance(atoms[0], Dict):
            formatted_atoms = [Atom(**atom) for atom in atoms]
        else:
            formatted_atoms = atoms

        self.atoms = self.add_atoms(atoms=formatted_atoms, use_symmetry=bool(symmetry))

        self.atoms_unique = []
        self.symmetry = -1
        self.SG_name = ''


    def add_atoms(self, atoms: List[Atom], use_symmetry=False) -> List[Atom]:
        '''
        Handle types here?
        Needs to give indices to atoms and symmetrize if requested
        '''
        out_atoms = []
        for idx, atom in enumerate(atoms):
            atom.index = idx
            out_atoms.append(atom)

        return out_atoms

    # def add_single_atom(self, atom: Union[Atom, Dict]):
    #     if isinstance(atom, Atom):
    #         self.atoms.append(atom)
    #     elif isinstance(atom, Dict)
    #         self.atoms.append(Atom(atom))
    #     else:
    #         raise ValueError(f'Unrecognised format of an atom: {type(atom)}')
        
    #     return


    def represent_tensor(self, atom: Atom, tensor='aniso'):
        '''
        Former: get_transform
        Transform a tensor into ??? for `aniso` it would take inverse matrix, for `gtensor` does nothing?
        '''
        if tensor=="aniso":
            mat = self.aniso
        else:
            mat = self.gtensor
        # diagonalise so can normalise eigenvalues 
        evals, evecs = np.linalg.eig(mat)
        if not evals.all():
            warnings.warn(f"Singular {tensor} matrix on atom {self.label}")
            return np.zeros(mat.shape)  # transform will be ignored
        else:
            if tensor=="aniso":
                # take inverse of eigenvals as large number should produce a small axis
                evals = 1/evals
            # scale such that max eigval is 1
            evals = evals/np.max(abs(evals))
            return evecs @ np.diag(evals) @ np.linalg.inv(evecs)
        
    def __repr__(self):
        rr = f'UnitCell(SG_name={self.SG_name},\n'
        rr += f'  atoms=['
        for atom in self.atoms:
            rr += '\n\t' + atom.__repr__()

        return rr + '\n\t])'
        

if __name__ == '__main__':
    atoms = [
         {'label':'Er', 'r':[0,0,0], 'm':[1,0,0], 's':1},
         {'label':'Er', 'r':[0.5,0.5,0], 'm':[-1,0,0], 's':1},
         {'label':'A', 'r':[0.5,0.5,0.5]}
        ]
    uc = UnitCell( atoms=atoms)
    print( uc )
    print(uc.atoms[2].m)