# Core
from copy import deepcopy
import numpy as np
import warnings

# Typing
from typing import TYPE_CHECKING, Union

# from spinwaves import atom

if TYPE_CHECKING:
    from .atom import Atom
    from .symmetry.magnetic_symmetry import MSG, mSymOp
    from .spinw import Coupling

# Internal

from .lattice import Lattice
        
class Crystal(Lattice):
    '''Crystal structure class. `Atoms` on a `Lattice` with `SpaceGroup`.

    Attributes
    ----------
    atoms_unique: list[Atoms]
        List of unique atoms generating the crystal structure.
    atoms_magnetic: list[Atoms]
        List of all magnetic atoms in the unit cell.
    atoms_all: list[Atoms]
        List of all atoms in the unit cell.

    For further list of attributes from `Lattice` see its docstring.

    Conventions
    -----------
    1. Atoms are within the first unit cell, i.e. have crystal coordinates in [0;1) range.
    2. Atoms are sorted, such that magnetic atoms are first, then the non-magnetic atoms follow.
       This is crucial for the proper indexation of the matrices required by LSW calculations.

    Notes
    -----
    TODO Generate atoms by symmetry and check that magnetic moments transform acoordingly.

    TODO IUCr tables A tab 1.3.1 with symmetry element symbols

    I am strongly worried about the stability of the generator. Meaning every time the atoms are 
    generated from symmetry, I need to be sure their order is always the same. Otherwise, 
    generation of coupling will change.
    Solutions: 
        1. [X] Sort atoms by coordinates (x,y,z) tuple sorting. This is stable AF, but not pretty.
        2. [ ] Ensure sorting of symmetry generators is stable. Pretty, but hard. How about sorting by their strings?
    '''
    _atoms_unique: tuple['Atom']
    _atoms_magnetic: tuple['Atom']
    _atoms_all: tuple['Atom']
    _MSG: 'MSG'

    def __init__(self, 
                 lattice_parameters: list[float],
                 MSG: 'MSG',
                 atoms:  list['Atom']):
        '''Construct the `Crystal` class representing the crystal (lattice and atoms)
        and its symmetry (agnetic space group).

        All atoms in the unit cell will be generated from the `atoms` parameter
        according to the symmetry of the crystal.

        Parameters
        ----------
        lattice_parameters: [a,b,c, alpha,beta,gamma]
            Lattice lengths in angstroem, lattice angles in degrees.
        MSG: `MSG`
            Magnetic space group of the crystal.
        atoms: list[`Atoms`]
            List of unique atoms in the crystal.
        '''
        # Lattice
        super().__init__(lattice_parameters)

        # MSG
        self._MSG = MSG

        # Should I ensure the provided list of atoms contains unique atoms? YES
        # Constructor should also check if provided magnetic moment respects the symmetry
        for atom in atoms:
            atom._gen_symop = MSG.operations[0].identity()

        self._atoms_unique = tuple(atoms)

        # TODO atom that lives in a crystal has to have its symmetry operations saved

        def transform_atom(g: 'mSymOp', a: 'Atom'):
            atom_new = deepcopy(a)

            atom_new._gen_symop = g
            atom_new.r = g.transform_position(a.r, to_UC=True)
            atom_new.m = self.uvw2xyz(g.transform_axial_vec(self.xyz2uvw(a.m)))
            # aotm_new.gtensor = self.g.matrix @ atom.gtensor @ self.g.inv().matrix

            return atom_new
        
        atoms_all = []
        for atom in self._atoms_unique:
            atoms_symmetrized = MSG.symmetrize(atom, transform_atom, check_attrs=['r', 'm'])
            atoms_all.extend(atoms_symmetrized)


        atoms_all = sorted(atoms_all)
        for id, atom in enumerate(atoms_all):
            atom.label += f'_{id}'

        self._atoms_all = tuple(atoms_all)

        ### atoms_magnetic
        self._atoms_magnetic = tuple(atom for atom in self.atoms_all if atom.is_mag)
        for id, atom in enumerate(self._atoms_magnetic):
            atom._sw_id = id

    ################################################################################
    # Properties

    @property
    def atoms_unique(self) -> tuple['Atom']:
        '''List of unique atoms generating the crystal structure.'''
        return self._atoms_unique
    
    @property
    def atoms_magnetic(self) -> tuple['Atom']:
        '''List of all magnetic atoms in the unit cell.'''
        return self._atoms_magnetic
    
    @property
    def atoms_all(self) -> tuple['Atom']:
        '''List of all atoms in the unit cell.'''
        return self._atoms_all
    
    @property
    def MSG(self) -> 'MSG':
        '''Magnetic Space Group of the crystal'''
        return self._MSG
    
    ################################################################################
    # Functionalities

    def get_atomic_distances(self, dmax: float=8, dmin: float=-1):
        '''Make a list of interatomic distances in the range `dmin < d < dmax`.
        
        Returns
        -------
        np.ndarray: bond_dtype.names = ['dd', 'd_xyz', 'd_uvw', 'i', 'j', 'n_uvw']
            List of bonds.

        '''

        nx_max, ny_max, nz_max = np.floor(dmax / np.array(self.lattice_parameters[:3]))
        Nx = np.arange(-nx_max-1, nx_max+1)
        Ny = np.arange(-ny_max-1, ny_max+1)
        Nz = np.arange(-nz_max-1, nz_max+1)

        Ngrids = np.meshgrid(Nx,Ny,Nz)                      # each of 3 has shape (len Nx, len Ny, len Nz)
        Ngrid_uvw = np.stack(Ngrids, axis=-1)               # shape (len Nx, len Ny, len Nz, 3)
        UC = Ngrid_uvw.reshape(len(Nx)*len(Ny)*len(Nz), 3)  # flatten to (prod Ni, k)
        
        cl_dtype = [('dd', 'f4'), ('d_xyz', 'f4', (3,)), ('d_uvw', 'i', (3,)), ('id1', 'i'), ('id2', 'i'), ('n_uvw', 'i', (3,))]
        clens = []
        for i, atom1 in enumerate(self.atoms_all):
            for n_uvw in UC:
                # print(n_UC)
                for j, atom2 in enumerate(self.atoms_all):
                    # if j <= i:
                    #     continue
                    d_uvw = atom2.r + n_uvw - atom1.r
                    d_xyz = self.uvw2xyz(d_uvw)
                    dd = np.linalg.norm(d_xyz)
                    if dd < dmax and dd > dmin:
                        clens.append( (dd, d_xyz, d_uvw, i,j, n_uvw) )

        # print(f'd = {dd:.5f} A, d_uvw = {d_uvw}, d_xyz = {d_xyz}')
        # print('\t', atom1)
        # print('\t', atom2)

        clens = np.array(clens, dtype=cl_dtype)
        id_sorting = np.argsort(clens, order=['dd'])

        return clens[id_sorting]
    
    def get_atom_sw_id(self, position: np.ndarray) -> int:
        '''Find the index of the potential magnetic atom at `position`.
        The integer part of the position, i.e. allocation to specific unit cell, is ignored.
        
        Parameters
        ----------
        position: array_like
            Position of atom in the unit cell in crystal coordinates.

        Raises
        ------
        LookupError
            If there is no atom, or more than one atom found at the `position`
        '''
        position = np.array(position, dtype=self.atoms_magnetic[0].r.dtype)
        candidates = [atom._sw_id for atom in self.atoms_magnetic
                      if np.allclose(position % 1, atom.r)]
        
        if not len(candidates) == 1:
            raise LookupError(f'No atom around {position!r} within the accuracy')

        return candidates[0]

    def is_respectful_DMI(self, coupling: 'Coupling', return_symmetrized: bool=False) -> Union[bool, tuple[bool,np.ndarray]]:
        '''Check if the DMI vector of the coupling respects the symmetry of the crystal.
        
        coupling: `Coupling`
            `Coupling` class representing the coupling.
        return_symmetrized: bool, optional
            If `True`, also returnthe symmetrized DMI vector, whose coefficients
            respect the crystal symmetry.

        Returns
        -------
        is_respectful_DMI: bool
            Does the DMI vector respect the crystal symmetry?
        DMI_symmetrized: ndarray, optional
            Symmetrized DMI vector.

        Notes
        -----
        This gives different results from the symmetrization by all symmetry elements.
        Talk with Piotr.
        '''

        bond_midpoint = (self.atoms_magnetic[coupling.id1].r + 
                         self.atoms_magnetic[coupling.id2].r + coupling.n_uvw) / 2
        DMI_point_group = self.MSG.get_point_symmetry(bond_midpoint)
        DMI_results = [g.matrix @ coupling.DMI_vector for g in DMI_point_group]
        print('DMI pg', DMI_point_group)
        print('DMI_res', DMI_results)
        DMI_symmetrized = np.average(DMI_results, axis=0)

        # Return values management
        ret = (np.allclose( coupling.DMI_vector, DMI_symmetrized), )
        if return_symmetrized:
            ret += (DMI_symmetrized, )

        if len(ret) == 1:
            ret = ret[0]

        return ret


    def represent_tensor(self, atom: 'Atom', tensor='aniso'):
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
        rr = f'Crystal(,\n'
        rr += f'  atoms=['
        for atom in self.atoms_all:
            rr += '\n\t' + atom.__repr__()

        return rr + '\n\t])'
        