# Core
from fractions import Fraction
from typing import Sequence
import numpy as np

from spinwaves.utils.arrays import ensure_shape

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
            
        g_tensor: np.ndarray = None
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
    - I see no better convention than the mixed notation for r_uvw and m_xyz.
    - Sorting is based on the position only. So th emixed occupation of the site is not implemented.

    '''

    # Physical properties
    _r: tuple[Fraction]
    _m: np.ndarray[float]
    _s: float
    g_tensor: np.ndarray = None
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
                 g_tensor: Sequence = [[-2,0,0],[0,-2,0],[0,0,-2]], aniso_mat = None, occupation = 1,
                 label: str='atom', element_symbol: str='',
                 color: Sequence=[], radius: float=0, 
                 atom_mesh: str='sphere', moment_mesh: str='arrow'):
        '''Create atom instance.

        Parameters
        ----------
        r: array_like
            Fractional coordinates of the atom's position. Will be converted to fractions.
        m: array_like
            Magnetic moment direction of the atom, in cartesian coordinates.
        s: float
            Spin number of the atom.
        '''
        # These have setters and validators within
        self.r = r
        self.m = m
        self.s = s
        self.g_tensor = g_tensor


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

        ### I now think that these properties should sit somewhere else, like in plot_options

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
        else:
            self.radius = radius

        self.atom_mesh = atom_mesh
        self.moment_mesh = moment_mesh

    ##############################################################################################################
    @property
    def r(self) -> tuple[Fraction]:
        '''Position of the ion in the unit cell in crystal coordinates'''
        # return self._r
        return np.array(self._r, dtype=float)
    
    @r.setter
    @ensure_shape(r_new=(3,))
    def r(self, r_new: Sequence):
        self._r = tuple([Fraction(x).limit_denominator(config['MAX_DENOMINATOR']) for x in r_new])

    @property
    def m(self) -> np.ndarray[float]:
        '''Magnetic moment of the ion in cartesian coordinates.'''
        return self._m
    
    @m.setter
    @ensure_shape(m_new=(3,))
    def m(self, m_new: Sequence):
        self._m = np.array(m_new, dtype=float)

    @property
    def s(self) -> np.ndarray[float]:
        '''Spin number of the ion'''
        return float(self._s)
    
    @s.setter
    def s(self, s_new: Sequence):
        self._s = Fraction(s_new).limit_denominator(config['MAX_DENOMINATOR'])

    @property
    def g_tensor(self) -> np.ndarray[float]:
        '''g-tensor of the ion, such that the Zeeman term is `H_Zeeman = mu_B * S * g_tensor * Hfield`'''
        return self._g_tensor
    
    @g_tensor.setter
    @ensure_shape(g_tensor_new=(3,3))
    def g_tensor(self, g_tensor_new: Sequence):
        self._g_tensor = np.array(g_tensor_new, dtype=float)


    @property
    def is_mag(self) -> bool:
        '''Is the atom magnetic? Based on the spin and magnetic moment vector.'''
        return (np.linalg.norm(self.m) > 1e-10) and (self._s != 0)

    ##############################################################################################################
    def __lt__(self, other) -> bool:
        '''Implement comparison of atoms for sorting.
        Compares only atomic position.'''
        # compare on fractions for better precision
        return tuple(self._r) < tuple(other._r)
    
    def __eq__(self, other) -> bool:
        # Comparing on Fractions, which should be safe
        return self._r == other._r
    
    def __repr__(self) -> str:
        r_str = [format(x, '') if x.denominator<7 else f'{float(x):.4f}'
                  for x in self._r]
        ret = f'Atom(label={self.label}, id={self._sw_id}, r=[{", ".join(r_str)}], m={self.m}, is_mag={self.is_mag})'
        return ret