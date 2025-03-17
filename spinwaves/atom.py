# Core
from fractions import Fraction
from typing import Sequence
import numpy as np
from dataclasses import dataclass

# Typing
# from typing import Dict, Union, TYPE_CHECKING
# if TYPE_CHECKING:
#     from .magnetic_symmetry import MSG

# Internal
from .databases import atom_data, color_data

# config constants
config = dict(MAX_DENOMINATOR = 1000)

class Atom:
    '''
    Stores informations about the atom.

    Attributes
    ----------
        r: np.ndarray
            Position in crystal (uvw) coordinates.
        m: np.ndarray = np.zeros((3))
            Direction of the magnetic moment in cartesian (xyz) coordinates.
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

    Notes
    -----
    - TODO decide on the convention whether the spin is represented in the crystal of cartesian coordinates

    '''

    # Physical properties
    _r: tuple[Fraction]
    m: np.ndarray[float]
    s: float
    gtensor_mat: np.ndarray = None
    aniso_mat: np.ndarray = None
    occupation: float = 1
    # is_mag: bool = False

    # Indexing/naming
    _sw_id: int = None
    label: str = 'atom'
    element_symbol: str = None

    # Plotting
    radius: float = 0
    spin_scale: float = 1
    color: np.ndarray[int] = None

    def __init__(self, r: Sequence, m: Sequence = [0,0,0], s: float = 0,
                 gtensor_mat = None, aniso_mat = None, occupation = 1,
                 label: str='atom', element_symbol: str='',
                 color: Sequence=[], radius: float=0):
        '''
        Assert field formats, and fill the correlated fields
        '''

        # Position
        self.r = r
        # self.r = np.array(self.r, dtype=float)
        if not len(self.r) == (3):
            raise ValueError(f'Atomic position must be a (3,) vector now is: {len(self.r)}')
        
        # Magnetic moment and spin
        self.m = np.array(m, dtype=float)
        self.s = s
        if not self.m.shape == (3,):
            raise ValueError(f'Atomic magnetic moment must be a (3,) vector now is: {self.r.shape}')
        # if not np.shape(s)==1:
        #     raise ValueError(f'Atomic spin must be a single `float`: s={self.s} shape={np.shape(self.s)}')
        # Normalize magnetic moment, just in case
        self.m /= np.linalg.norm(self.m)

        # Label
        self.label = label
        if element_symbol:
            # TODO ensure is in database
            pass
        else:
            if self.label[:2] in atom_data.entries:
                self.element_symbol = self.label[:2]
            elif self.label[:1] in atom_data.entries:
                self.element_symbol = self.label[:1]

        # Color as input, derived from the element symbol, or black
        if len(color):
            self.color = np.array(color)
        elif self.element_symbol:
            self.color = atom_data[self.element_symbol].RGB
        else:
            self.color = color_data['Black'].RGB

        # Atom radius
        if not bool(radius):
            if self.element_symbol:
                self.radius = atom_data[self.element_symbol].radius
            else:
                self.radius = 0.3

    @property
    def r(self) -> np.ndarray[float]:
        '''Position in the unit cell in crystal coordinates'''
        return np.array(self._r, dtype=float)
    
    @r.setter
    def r(self, r_new: Sequence):
        self._r = tuple([Fraction(x).limit_denominator(config['MAX_DENOMINATOR']) for x in r_new])
    
    @property
    def is_mag(self) -> bool:
        '''Is the atom magnetic? Based on the spin and magnetic moment vector.'''
        # Is magnetic atom
        if (np.linalg.norm(self.m) < 1e-10) and (self.s == 0):    # non-magnetic atom
            ret = False
        elif np.linalg.norm(self.m) > 1e-10 and (np.abs(self.s)>0):
            ret = True
        else:
            raise ValueError(f'''Must provide (3,) shape vector for magnetic moment and non-zero spin number for magnetic atom.\n\tNow: m={self.m} s={self.s}\n\t{self}''')

        return ret

    ##################################################################################
    def __lt__(self, other) -> bool:
        '''Implement comparison of atoms for sorting.
        Compares only atomic position.'''
        # compare on fractions for better precision
        return tuple(self._r) < tuple(other._r)
    
    def __eq__(self, other) -> bool:
        return np.allclose(self._r, other._r)
    
    def __repr__(self) -> str:
        r_str = [format(x, '') if x.denominator<7 else f'{float(x):.4f}'
                  for x in self._r]
        ret = f'Atom(label={self.label}, id={self._sw_id}, r=[{", ".join(r_str)}], m={self.m}), is_mag={self.is_mag}'
        return ret