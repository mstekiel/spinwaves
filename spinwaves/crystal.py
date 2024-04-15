# Core
import numpy as np
import warnings
from dataclasses import dataclass

# Typing
from typing import Dict, Union

# Internal
from .data_containers import atom_data, color_data
from .lattice import Lattice

@dataclass
class Atom:
    '''
    Stores informations about the atom.

    Attributes
    ----------
        r: np.ndarray
            Position in crystal (uvw) coordinates.
        m: np.ndarray = np.zeros((3))
            Magnetic moment in cartesian (xyz) coordinates.
        s: float = 0
            Spin number, int of half-int.
            
        gtensor_mat: np.ndarray = None
            Matrix representing the magnetic g-tensor.
        aniso_mat: np.ndarray = None
            Matrix representing the anisotropic displacement parameters.

        label: str = 'atom'
            Label, suggested format is `ElementSymbol_x`, that allows automaitc detection of
            `self.element_symbol` field.
        element_symbol: str = None
            Element symbol.

        color: str = None
            Color of the atom, used for plotting.
        radius: float = 0
            Radius of the atom, determined from `self.element_symbol`, used for plotting, 
        spin_scale: float = 1
            Scale of the magnetic moment. used for plotting.

        _is_mag: bool = False
            Helper flag set by non-zero `self.m` and `self.s` values.


    Questions:
    - what is index?
    - moment in what units?
    - position in what units?
    '''
    r: np.ndarray[float]
    m: np.ndarray[float] = np.zeros((3))
    s: float = 0
    _is_mag: bool = False
    index: int = -1
    label: str = 'atom'
    element_symbol: str = None
    color: np.ndarray[int] = None
    radius: float = 0
    spin_scale: float = 1
    gtensor_mat: np.ndarray = None
    aniso_mat: np.ndarray = None

    def __post_init__(self):
        '''
        Assert field formats, and fill the correlated fields
        '''

        # Position
        self.r = np.array(self.r, dtype=float)
        if not self.r.shape == (3,):
            raise ValueError(f'Atomic position must be a (3,) vector now is: {self.r.shape}')
        
        # Magnetic moment and spin
        self.m = np.array(self.m)
        if not self.m.shape == (3,):
            raise ValueError(f'Atomic magnetic moment must be a (3,) vector now is: {self.r.shape}')
        if not np.shape(self.s)==():
            raise ValueError(f'Atomic spin must be a single `float`: s={self.s} shape={np.shape(self.s)}')

        # Is magnetic atom
        if (np.linalg.norm(self.m) < 1e-10) and (self.s == 0):    # non-magnetic atom
            self.is_mag = False
        elif np.linalg.norm(self.m) > 1e-10 and (np.abs(self.s)>0):
            self.is_mag = True
        else:
            raise ValueError(f'''Must provide (3,) shape vector for magnetic moment and non-zero spin number for magnetic atom.\n\tNow: m={self.m} s={self.s}\n\t{self}''')

        # Element symbol as input or derived from the label 
        # TODO what if the element is wrong?
        if not self.element_symbol:
            # token = ''.join([])
            if self.label[:2] in atom_data:
                self.element_symbol = self.label[:2]
            elif self.label[:1] in atom_data:
                self.element_symbol = self.label[:1]

        # Color as input, derived from the element symbol, or black
        if self.color:
            # TODO check type
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
        
class Crystal(Lattice):
    '''Crystal structure class. `Atoms` on a `Lattice`.

    Holds information about atoms within the unit cell and the shape of the unit cell.

    Attributes
    ----------
    atoms: list[Atom]
        All atoms within the unit cell. Not just one Wyckoff site, it has to be expanded by hand.
        

    For further list of attributes from `Lattice` see its docstring.

    Conventions
    -----------
    1. Atoms are within the first unit cell, i.e. have crystal coordinates in [0;1) range.
    '''
    def __init__(self, 
                 lattice_parameters: list[float],
                 atoms:  list[Atom], 
                 symmetry: Union[str, int]=None,  # lets make it a field or something
                 origin: np.ndarray=np.zeros((3))):
        '''
        Parameters:
        -----------
        atoms:
            Names and positions of atoms.
        symmetry:
            Name or number of the space group, or None.
        '''

        super().__init__(lattice_parameters=lattice_parameters, orientation=None)

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


    def add_atoms(self, atoms: list[Atom], use_symmetry=False) -> list[Atom]:
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
    lattice = Lattice([3,3,4,90,90,120])
    atoms = [
         {'label':'Er', 'r':[0,0,0], 'm':[1,0,0], 's':1},
         {'label':'Er', 'r':[0.5,0.5,0], 'm':[-1,0,0], 's':1},
         {'label':'A', 'r':[0.5,0.5,0.5]}
        ]
    crystal = Crystal(atoms=atoms, lattice=lattice)
    print( crystal )
    print(crystal.atoms[2].m)