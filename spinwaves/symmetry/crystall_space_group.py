from fractions import Fraction
import numpy as np
import logging

# from copy import deepcopy
# from itertools import chain, combinations

from .group import SymOp, Group


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..crystal import Atom

logger = logging.getLogger('CrystallographicSymmetry')

class cSymOp(SymOp):
    '''Crystallographic space group symmetry operation.
    
    Attributes
    ----------
    str: str
        xyz string defining the symmetry operation.
        For conventions see `self.to_str()` method.

    matrix: np.ndarray[int]
        Matrix part of the symmetry operation
    
    translation: np.ndarray[float]
        Translation part of the symmetry operation
    
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
    def __init__(self, matrix: np.ndarray[int], translation: np.ndarray[Fraction]):

        # Matrix elements can only be [-1, 0, 1]
        if not set(matrix.flatten()).issubset({-1,0,1}):
            raise ValueError('`cSymOp` matrix elements larger than 1. Check if you operation has multiplicity <=6')
        self._matrix = matrix

        # Current implementation assumes denominator max 6
        if any([t.denominator>6 for t in translation]):
            raise ValueError(f'{translation!r} Translation part contains denominator >6. This is unnacounted for.')
        self._translation = translation

        self._str = self.__str__()

    @classmethod
    def from_string(cls, xyz_str: str) -> 'cSymOp':
        '''Construct a magnetic symmetry operation object `cSymOp` from the xyz string.
        There are some heavy conventions on the xyz string, noted in ???.'''

        allowed_chars = set(' /,+-xyz123456')
        if not set(xyz_str).issubset(allowed_chars):
            raise ValueError(f'Improper `{xyz_str=}`: characters should be from {allowed_chars!r} only.')

        xyz_str = xyz_str.replace(' ', '')
        substrings = [s.strip() for s in xyz_str.split(',')]
        if not (len(substrings) == 3):
            raise ValueError(f'`xyz_string` must have two commas: {xyz_str!r}')
        
        required_chars = set(',xyz')
        if not required_chars.issubset(set(xyz_str)):
            raise ValueError(f'Improper `{xyz_str}` characters must contain {required_chars!r} characters.')

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
                   translation = np.array(translation, dtype=Fraction))
    
    ##############################################################################
    # Implementation of abstract methods

    def __mul__(self, other) -> 'cSymOp':
        # self is LHS, other is RHS operation
        if not isinstance(other, cSymOp):
            raise TypeError(f'Multiplication supported only for {self.__class__.__name__!r} types')
        
        new_matrix = self._matrix @ other._matrix
        new_translation = (self._translation + self._matrix @ other._translation) % 1

        return cSymOp(matrix = new_matrix, 
                      translation = new_translation)
    
    def to_str(self) -> str:
        '''Represent symmetry operation as xyz string.
        I assume the xyz_string==self._str is a unique identifier
        of the symmetry operation.
        
        Conventions
        -----------
        1. String consists of three fields separated by `, ` [comma+space].
        3. Fields represent matrix and translational part, each one of them
           has the same convention, described next.
            3.1. First we have matrix part represented by `+-xyz` symbols.
                 First symbol is either `-` or from 'xyz', where latter implies
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

        return ', '.join(mat_tr)
    
    def __hash__(self) -> int:
        '''Hash of the function, based on the string representation.'''
        return hash(self.to_str())
    
    def __eq__(self, other) -> bool:
        t1 =  (self._matrix==other._matrix).all()
        t2 =  (self._translation==other._translation).all()
        return t1 and t2
    
    @classmethod
    def identity(cls) -> 'cSymOp':
        '''Identity element of the `cSymOp` class.'''
        return cls.from_string(xyz_str='x,y,z')
    
    def inv(self) ->'cSymOp':
        '''Inverse of the symmetry operation.'''
        gm_inv = np.linalg.inv(self._matrix).astype(int)
        gtr_new = np.array([Fraction(x).limit_denominator(6) % 1
                           for x in -gm_inv @ self._translation],
                           dtype=Fraction)
        return cSymOp(matrix = gm_inv,translation = gtr_new)

    
    ##############################################################################
    # Properties
    @property
    def matrix(self) -> np.ndarray[int]:
        '''Matrix part of the symmetry operation'''
        return self._matrix
    
    @property
    def translation(self) -> np.ndarray[float]:
        '''Translation part of the symmetry operation'''
        return self._translation.astype(float)

    ##############################################################################
    # Methods
    def __repr__(self) -> str:
        return f'<cSymOp `{self.to_str()}`>'
    
    def transform_atom(self, atom: 'Atom'):
        pass
    

class SG(Group):
    pass