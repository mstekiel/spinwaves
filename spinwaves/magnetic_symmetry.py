from copy import deepcopy
import numpy as np
from fractions import Fraction
from itertools import chain, combinations


from typing import Any, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from .crystal import Atom


class mSymOp():
    '''Magnetic space group symmetry operation.
    
    Attributes
    ----------
    str: str
        xyz string defining the symmetry operation.
        For conventions see `self.to_str()` method.

    matrix: np.ndarray[int]
        Matrix part of the symmetry operation
    
    translation: np.ndarray[float]
        Translation part of the symmetry operation
    
    time_reversal: int
        Time reversal of symmetry operation.
        1 means time reversal is not involved
        -1 means time revelrsal is involved
    
    inversion: int
        Inversion of symmetry operation
    
    character: int
        Character of the operation, trace of matrix part.
    
    multiplicity: int
        Understood in terms of rotational part. N for which g^N=1


    Notes
    -----------
    1. time_reversal, inversion, naming is bit awkward with int
    2. Handling the transformations of position with Fractions?
    '''
    _str: str
    _matrix: np.ndarray[int]
    _translation: np.ndarray[Fraction]
    _time_reversal: int

    ##############################################################################
    # Constructors
    def __init__(self,
                 matrix: np.ndarray[int],
                 translation: np.ndarray[Fraction],
                 time_reversal: int):

        # Matrix elements can only be [-1, 0, 1] for MSG
        if not set(matrix.flatten()).issubset({-1,0,1}):
            raise ValueError('`mSymOp` matrix elements larger than 1. Check if you operation has multiplicity <=6')
        self._matrix = matrix

        # Current implementation assumes denominator max 6
        if any([t.denominator>6 for t in translation]):
            raise ValueError(f'{translation!r} Translation part contains denominator >6. This is unnacounted for.')
        self._translation = translation

        self._time_reversal = time_reversal

        self._str = self.to_string()

    @classmethod
    def from_string(cls, xyz_str: str) -> 'mSymOp':
        '''Construct a magnetic symmetry operation object `mSymOp` from the xyz string.
        There are some heavy conventions on the xyz string, noted in ???.'''

        allowed_chars = set(' /,+-xyz123456')
        if not set(xyz_str).issubset(allowed_chars):
            raise ValueError(f'Improper `xyz_str` characters should be from {allowed_chars!r} only.')

        xyz_str = xyz_str.replace(' ', '')
        substrings = [s.strip() for s in xyz_str.split(',')]
        if not (len(substrings) == 4):
            raise ValueError('`xyz_string` must have three commas.')

        # time reversal part
        time_reversal = None
        if substrings[-1]=='+1':
            time_reversal = 1
        elif substrings[-1]=='-1':
            time_reversal = -1
        else:
            raise ValueError(f'{substrings[-1]} is not a valid time reversal string. Should be `+1` or `-1`.')

        translation = [0,0,0]
        matrix_rows = [None, None, None]
        for n,ri_str in enumerate(substrings[:3]):
            # translational part, if exists
            if ri_str[-1] in '123456':
                translation[n] = Fraction(ri_str[-4:])
                ri_str = ri_str[:-4]

            # rotational/mirror part
            ri_str = ri_str.replace('y','np.array([0,1,0])')\
                           .replace('x','np.array([1,0,0])')\
                           .replace('z','np.array([0,0,1])')          
            matrix_rows[n] = eval(ri_str)

        return cls(matrix = np.array(matrix_rows, dtype=int),
                   translation = np.array(translation, dtype=Fraction),
                   time_reversal = time_reversal)

    ##############################################################################
    # Properties

    @property
    def str(self) -> str:
        '''xyz string defining the symmetry operation.
        For conventions see `self.to_str()` method.'''
        # return self.to_string()
        return self._str

    @property
    def matrix(self) -> np.ndarray[int]:
        '''Matrix part of the symmetry operation'''
        return self._matrix
    
    @property
    def translation(self) -> np.ndarray[float]:
        '''Translaiton part of the symmetry operation'''
        return self._translation.astype(float)
    
    @property
    def time_reversal(self) -> int:
        '''Time reversal of symmetry operation.
        1 means time reversal is not involved
        -1 means time revelrsal is involved'''
        return self._time_reversal
    
    @property
    def inversion(self) -> int:
        '''Inversion of symmetry operation'''
        return np.linalg.det(self.matrix).astype(int)
    
    @property
    def character(self) -> int:
        '''Character of the operation, trace of matrix part.'''
        return np.trace(self.matrix)
    
    @property
    def multiplicity(self) -> int:
        '''Understood in terms of rotational part.
        N for which g^N=1'''
        phases = np.angle(np.linalg.eigvals(self.matrix))
        ph_fracts = [Fraction(phase/np.pi).limit_denominator(6) for phase in phases]

        return max([p.denominator for p in ph_fracts])
    
    ##############################################################################
    # Utility operations
    def to_string(self) -> str:
        '''Represent symmetry operation as xyz string.
        
        Conventions
        -----------
        1. String consists of four fields separated by `, ` [comma+space].
        2. Last field denotes time reversal adjoined operation as is form [`+1`,`-1`].
        3. First three fields represent matrix and translational part, each one of them
           has the same convention, described next.
            3.1. First we have matrix part represented by `xyz+-` symbols.
                 First symbol is either `-` or from 'xyz', where latter indicates
                 `+` sign in front of the `xyz` symbol.
            3.2 Second is the optional translational part, with symbols from
                '+[1235]/[2346]'. As such, it is always 4 characters long,
                and always positive.
        '''
        mat_tr = []

        # iterate over row==out coordinate==xyz_string subfields
        for n in range(3):
            # matrix part
            sub_i = ''
            for i in range(3):
                value = self._matrix[n,i]
                sign = {-1:'-', 1:'+', 0:''}[value]
                symbol = np.abs(value)*'xyz'[i]
                sub_i += sign + symbol

            # translational part
            value = self._translation[n]
            if not value==0:
                sign = {True:'+', False:'-'}[value>0]
                sub_i += sign+str(value)

            # trailing plus is obsolete
            if sub_i[0]=='+':
                sub_i = sub_i[1:]

            mat_tr.append(sub_i)

        # time reversal
        mat_tr.append(f'{self._time_reversal:+d}')

        return ', '.join(mat_tr)
    
    def __lt__(self, other: 'mSymOp') -> bool:
        '''Evaluated based on descending list of conditions
        1. Time reversal
        2. If the operation involves inversion/mirror
        3. Multiplicity of the operation
        4. If operation involves translation
        '''
        # python allows comparing tuples, where each next pair of fields has descending importance
        comp_left = (-self.time_reversal, 
                     -self.inversion,
                     -self.character*self.inversion,
                     self.multiplicity,
                     np.linalg.norm(self.translation))
        
        comp_right= (-other.time_reversal, 
                     -other.inversion,
                     -other.character*other.inversion,
                     other.multiplicity,
                     np.linalg.norm(other.translation))

        return comp_left < comp_right
        # return self.str < other.str
    
    @property
    def symbol_HM(self) -> str:
        ret = ''
        
        # Based on character:
        # +-3 is +-1
        ch = self.character
        if ch == -3:
            ret += '-1'
        elif ch == -2:
            ret += '?'
        elif ch == -1:
            ret += '?'
        elif ch == 0:
            ret += '?'
        elif ch == 1:
            ret += '?'
        elif ch == 2:
            ret += '?'
        elif ch == 3:
            ret += '1'

        if self.time_reversal:
            ret += "'"

        return ret
    
    def __eq__(self, other: 'mSymOp') -> bool:
        # If we truly trust in string, why not?
        # Because it is not significantly faster and sometimes slower
        # return self.str==other.str
        t1 = (self._matrix==other._matrix).all()
        t2 = (self._translation==other._translation).all()    # comapring on fractions should be ok?
        t3 = (self._time_reversal==other._time_reversal)

        return t1 and t2 and t3

    def __mul__(self, other: 'mSymOp') -> 'mSymOp':
        # self is LHS, other is RHS operation
        if not isinstance(other, mSymOp):
            raise TypeError(f'Multiplication supported only for {self.__class__.__name__!r} types')
        
        new_matrix = self._matrix @ other._matrix
        new_translation = (self._translation + self._matrix @ other._translation) % 1
        new_time_reversal = self._time_reversal*other._time_reversal

        return mSymOp(matrix = new_matrix, 
                      translation = new_translation, 
                      time_reversal = new_time_reversal)

    def __repr__(self):
        # matrix_str = ', '.join([str(row) for row in self.matrix])
        return f'<mSymOp {self.str!r}>'

    def print(self, fields: list[str]=[]) -> str:
        if not len(fields):
            fields = ['str', 'matrix', 'translation', 'time_reversal', 
                      'inversion', 'character', 'multiplicity',
                      'symbol_HM']

        ret = '<mSymOp\n'
        for field in fields:
            if field not in self.__dict__:
                Warning(f'{field!r} is not a valid field of the {self.__class__.__name__!r}')

            value = self.__getattribute__(field)
            if field == 'matrix':
                value = str(self.matrix).replace('\n','')

            ret += f' {field}\t= {value!r}\n'
        
        ret += '>'
        return ret

    def __hash__(self) -> str:
        '''I assume the xyz_string==self._str is a unique identifier
        of the symmetry operation.'''
        return hash(self._str)
    
    def inv(self) -> 'mSymOp':
        '''Inverse of the symmetry operation.'''
        gm_inv = np.linalg.inv(self.matrix).astype(int)
        gtr_new = np.array([Fraction(x).limit_denominator(6) % 1
                           for x in -gm_inv @ self.translation],
                           dtype=Fraction)
        return mSymOp(
            matrix = gm_inv,
            translation = gtr_new,
            time_reversal = self.time_reversal
        )

    def transform_position(self, position: np.ndarray, to_UC: bool=False) -> np.ndarray[float]:
        '''Transform the `position` in crystal coordinates according to `mSymOp`.
        
        Parameters
        ----------
        position: (3,) float
            Represents a position in 3D space
        to_UC: (optional) bool
            Whether to keep the position within the unit cell.
        '''
        r_new = self.matrix @ position + self.translation
        if to_UC:
            r_new = r_new % 1

        return r_new
    
    def transform_polar_vec(self, vector: np.ndarray) -> np.ndarray[float]:
        '''Transform the `vector` of polar character (electric dipole) according to `mSymOp`.
        
        Notes
        -----
        Polar vector is invariant to time reversal and inverts on spatial inversion.
        '''
        return self.matrix @ vector
    
    def transform_axial_vec(self, vector: np.ndarray) -> np.ndarray[float]:
        '''Transform the `vector` of axial character (magnetic dipole) according to `mSymOp`.
        
        Notes
        -----
        Axial vector is invariant to spatial inversion and inverts on time reversal.
        '''
        return self.time_reversal*np.linalg.det(self.matrix)*self.matrix @ vector

    def transform_atom(self, atom: 'Atom', to_UC: bool=False) -> 'Atom':
        '''Transform the position and magnetic moment of the atom
        according to the symmetry operation.
        
        Parameters
        ----------
        atom: Atom
            Object of the transformation
        to_UC: bool (optional)
            WHether to shift the atomic position to the firsst unit cell.
        '''
        atom_new = deepcopy(atom)
        atom_new.r = self.transform_position(atom.r, to_UC=to_UC)
        atom_new.m = self.transform_axial_vec(atom.m)
        #TODO
        # aotm_new.gtensor = self.g.matrix @ atom.gtensor @ self.g.inv().matrix
        return atom_new

###########################################################################
class MSG():
    '''Magnetic Space Group'''
    _name: str
    _generators: tuple[mSymOp]
    _operations: tuple[mSymOp]

    def __init__(self,
                 generators: list[mSymOp],
                 name: str='MSG'):
        
        # Name
        self._name = name

        # Ensure only unique elements and no identity
        # form the generators
        generators = set(generators) - {mSymOp.from_string('x, y, z, +1')}
        self._generators = tuple(sorted(generators))

        # Generate other symmetry operations
        self._operations = tuple(sorted(self._generate_all_elements(generators)))

    ##############################################################################
    # Constructors
    @classmethod
    def from_xyz_strings(cls, generators: list[str]):
        '''Construct Magnetic Space Group from a list of xyz_strings,
        that represent MSG operations.
        
        Notes
        -----
        1. The construction takes place in real time, so for groups with large
           number of expected elements it can take seconds:
           >>> 8 s for Ia-3d1' with 192 operations.
        2. MSGs with time reversal translations do not seem to work as expected
        '''
        return cls(generators = [mSymOp.from_string(xyz_string) for xyz_string in generators])

    def _generate_all_elements(self, generators: list[mSymOp]):
        '''Generate symmetry elements of magnetic space group from generators.'''
        # Algorithm
        # 1. Multiply all symmetry operators by each other
        # 2. Find unique symmetry operations in the multiplied list
        # 3. If the list of unique elements is longer than the
        #    original symmetry some new operators were created GOTO 1
        # Exit: When no new symmetry operators were created
        #
        # The algorithm is O(n*n) and becomes heavy for large n.
        # We dint have to multiply all elements each time, just the new ones.
        # Tried some improvements, but subtracted just constant time.

        # Add 1 to generators, to fulfill `while` loops and ensure its in the group
        ops_new = list(generators) + [mSymOp.from_string('x, y, z, +1')]
        ops = generators
        while len(ops_new) > len(ops):
            ops = ops_new
            ops_table = [gi*gj for gi in ops for gj in ops]
            ops_new = list(set(ops_table))

        return ops
        
    ##############################################################################
    # Properties
    @property
    def operations(self):
        '''Symmetry operators of the MSG'''
        return self._operations

    @property
    def order(self) -> int:
        return len(self._operations)

    ##############################################################################
    # Utility operations
    def __str__(self) -> str:
        ret = f'<{self._name} order={len(self._operations)}\n'
        for g in self._generators:
            ret += f'  {g}\n'
        ret += '>'
        return ret
    
    def __getitem__(self, index: int) -> 'mSymOp':
        return self._operations[index]
    
    def make_cayley_table(self):
        '''Construct Cayley table of the symmetry group.'''
        cayley_table = np.full((self.order, self.order), -1, dtype=int)
        for id_i, gi in enumerate(self._operations):
            for id_j, gj in enumerate(self._operations):
                cayley_table[id_i, id_j] = self._operations.index(gi*gj)

        return cayley_table
    
    def get_subgroups(self) -> list['MSG']:
        '''Determine possible subgroup of the symmetry group.'''
        # Construction based on the generators of the MSG
        # s = list(self._operations.index(g) for g in self._generators)
        
        # Get combinations of generators
        s = list(self._generators)
        gen_subsets = chain.from_iterable(combinations(s, r) for r in range(1, len(s)))
        # Make groups from different generators
        subgroups = [MSG(generators=gen) for gen in gen_subsets]
        return sorted(set(subgroups), key=lambda msg: len(msg._operations))

    def get_point_symmetry(self, position: np.ndarray[Any]) -> list[mSymOp]:
        '''Determine the point symmetry of the `position`.
        
        Notes
        -----
        So we just take a look at which positions leave it invariant?
        '''
        return [g for g in self._operations 
                if np.allclose( (position-g.transform_position(position))%1, [0,0,0])]
    
    
    def get_orbit(self, position: np.ndarray[Any], 
                  return_generators: bool=False,
                  return_indices: bool=False) -> list[mSymOp]:
        '''Get symmetry operations that generate the orbit of the Wyckoff position.
        https://dictionary.iucr.org/Crystallographic_orbit

        Parameters
        ----------
        position: (3,)
            Position in the lattice coordinates for which the orbit is generated.
        return_generators: bool, optional
            If True, return the symmetry operations generating the orbit.
        return_indices: bool, optional
            If True, return the indices of the symmetry operations generating 
            the orbit, as they are listed in the `MSG.operations`.

        Notes
        -----
        The method finds unique positions, by comparing arrays containing floating point numbers.
        To prevent floating point inaccuracies the comparison is done until 5th decimal place.
        '''
        positions_new = [g.transform_position(position)%1 for g in self.operations]
        positions, id_unique = np.unique( np.around(positions_new, 5), axis=0, return_index=True)

        generators = [self.operations[n] for n in id_unique]

        ret = (positions, )
        if return_generators:
            ret += (generators, )
        if return_indices:
            ret += (id_unique, )

        if len(ret) == 1:
            ret = ret[0]

        return ret