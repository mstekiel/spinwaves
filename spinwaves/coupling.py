import numpy as np


class Coupling:
    '''Coupling between atoms in the crystal.
    
    Notes
    -----
    - TODO establish the convention of the `J` matrix entries, 
      either crystal or Cartesian coordinates.
    - The fields `id1`, `id2`, and `J` are linked to the `Crystal` class.
      Their validity is not checked.
    '''
    _label: str
    _id1: int
    _id2: int
    _n_uvw: np.ndarray[int]
    _J: np.ndarray[float]
    _defining_bond: str

    # DMI_vector: np.ndarray[float] = np.zeros(3)

    # hidden_symmetry: tuple = ()

    def __init__(self, label: str, id1: int, id2: int, n_uvw: list[int],
                 J: np.ndarray[float], Jcoords: str='xyz',
                 defining_bond: str=''):
        # DEV NOTES
        # Allow using the constructor directly, thus ensure the types.
        self._label = label
        self._id1 = int(id1)
        self._id2 = int(id2)
        self._n_uvw = np.array(n_uvw, dtype=int)
        self._J = J
        self._defining_bond = defining_bond

    ############################################################################
    # Properties

    # Allow for the label setter, but no more
    @property
    def label(self) -> str:
        '''Label of the coupling'''
        return self._label
    
    @label.setter
    def label(self, new_label):
        self._label = new_label

    @property
    def id1(self) -> int:
        '''Index of the first interacting atom'''
        return self._id1
    
    @property
    def id2(self) -> int:
        '''Index of the second interacting atom'''
        return self._id2

    @property
    def n_uvw(self) -> np.ndarray[int]:
        '''Origin/index of the unit cell where the second interacting atoms resides.'''
        return self._n_uvw
    
    @property
    def J(self) -> np.ndarray[float]:
        '''Exchange interaction in matrix form.'''
        return self._J
    
    @property
    def DMI_vector(self) -> np.ndarray[float]:
        '''Ansitymmetric part of the interaction represented
        by the Dzialoshynskii-Moriya vector.'''    
        J_asym = (self.J - self.J.T) / 2
        Dx =  J_asym[1,2]
        Dy = -J_asym[0,2]
        Dz =  J_asym[0,1]
        return np.array([Dx, Dy, Dz], dtype=float)
    
    ###########################################################################
    # For sorting and comparing

    # DEV
    # The main use case is when symmetrizing the couplings.
    # For that, we don't want to look into the exchange interaction matrix.

    def __hash__(self) -> int:
        return hash((self.id1, self.id2) + tuple(self.n_uvw.astype(int)))
    
    def __lt__(self, other) -> bool:
        comp_fields_left  = (self._id1, self._id2, self._n_uvw[0], self._n_uvw[1], self._n_uvw[2])
        comp_fields_right = (other._id1, other._id2, other._n_uvw[0], other._n_uvw[1], other._n_uvw[2])
        return comp_fields_left < comp_fields_right
    
    def __eq__(self, other: 'Coupling') -> bool:
        comp_fields_left  = (self._id1, self._id2, self._n_uvw[0], self._n_uvw[1], self._n_uvw[2])
        comp_fields_right = (other._id1, other._id2, other._n_uvw[0], other._n_uvw[1], other._n_uvw[2])
        return comp_fields_left == comp_fields_right
    
    ###########################################################################
    # Methods

    def revert(self) -> 'Coupling':
        '''Revert the coupling, as in exchange the coupled atoms.'''
        return Coupling(label = self.label+'_rev',
                        id1 = self.id2,
                        id2 = self.id1,
                        n_uvw = -self.n_uvw,
                        J = self.J.T,
                        defining_bond = self.label
                        )
    
    def __repr__(self) -> str:
        return f'<Coupling label={self.label}, id1={self.id1}, id2={self.id2}, n_uvw={self.n_uvw}, J={self.J.tolist()}>'
        