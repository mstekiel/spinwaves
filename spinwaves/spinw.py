# Core
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib
import warnings
from dataclasses import dataclass

import logging

# Plotting
import numpy as np
from vispy import scene
import vispy
# from vispy.color import color_array
# from itertools import chain
# from vispy.visuals.filters import ShadingFilter, WireframeFilter
# from vispy.geometry import create_sphere
# import copy
# from scipy.spatial.transform import Rotation
# from scipy.spatial import ConvexHull
# from dataclasses import dataclass
# import warnings

# Typing
from matplotlib.figure import Figure
from typing import List, Tuple, Dict, Union
from numpy.typing import NDArray

# Internal
from . import functions as funs
# from .lattice import Lattice
from .crystal import Crystal
# from .plotting_vispy import SupercellPlotter

import mikibox as ms

# import plotting
# from .plotting import plot_supercell, implemented_sc_plotters

# from coupling import couplings

# Tobi confirmed the factor of two is missing from single-ion naisotropies.
# He also mentioned the inverted sign mistake in the phase factor somewhere
# in the spin-spin correlation function calculations.

class Couplings:
    pass

class Dispersion:
    pass



class SpinW:
    '''
    Main object used to calculate spin waves.

    Fields:
    -------

    lattice:
        `Lattice` object
    unit_cell:
        Dictionary       
    magnetic_structure:
        [k, n, spins]
    couplings: 
        List[d, i, j, J]



    TODO
    ----
        - Each object of the constructor should be its own well defined class, or np.array with defined fields.
            - `Atom`s in `UnitCell` object that lives independently on the lattice.
    '''
    def __init__(self, crystal: Crystal, magnetic_structure: Dict):
        '''
        lattice: Lattice
        unit_cell: UnitCell
        magnetic_str: [k, n, spins]
        couplings: List[d, i, j, J]
        '''
        self.crystal = crystal
        self.magnetic_atoms = [atom for atom in crystal.atoms if atom.is_mag]
        self.magnetic_structure = magnetic_structure
        self.couplings = None
        # self.add_couplings(couplings)

    def symmetry_operations(self, generators: List[str]) -> np.ndarray:
        '''
        Generate all symmetry operations in matrix form based on the generators list.
        '''

        sym_matrix = {
            '1':np.eye(3,3),
            '-1':-np.eye(3,3),
            '2x':funs.Rx(np.pi),
            '2y':funs.Ry(np.pi),
            '2z':funs.Rz(np.pi),
            '3z':funs.Rz(2*np.pi/3),
            '4z':funs.Rz(np.pi/2),
            '6z':funs.Rz(np.pi/3),
        }

        for gen in generators:
            if not gen in sym_matrix:
                raise KeyError(f'`{gen}` is not implemented/allowed symmetry operator\
                                Allowed pars: {list(sym_matrix.keys())} ')

        # Ensure identity is in the generators lsit
        symmetry = np.concatenate(([sym_matrix[gen] for gen in generators], [sym_matrix['1']]))

        # 1. Multiply all symmetry operators by each other and make a table with (N,N,3,3) shape
        # 2. Find unique symmetry operations in the flattened table
        # 3. If the flattened table is longer then the original symmetry some new perators were created GOTO 1
        # Exit: When no new symmetry operators were created
        flag = True
        while flag:
            sym_table = np.einsum('mij,njk->mnik', symmetry, symmetry)
            new_symmetry = np.unique( np.around(sym_table.reshape((-1,3,3)), 10), axis=0)
            
            if new_symmetry.shape[0] == symmetry.shape[0]:
                flag = False

            symmetry = new_symmetry

        return symmetry

    def add_couplings(self, couplings: Dict):
        '''
        Generate couplings bettween magnetic atoms.        

        couplings: Dict[ label: Tuple[n_uvw, atom_i, atom_j, J, symmetry] ]
            Define a coupling with label `label` that couples atom with index `atom_j`
            in the unit cell with index `n_uvw` with atom of index `atom_i` in
            the original unit cell. The coupling can be symmetrized, that is equivalent atoms can be coupled,
            based on the symmetry operations in `symmetry` list.

            Parameters
            ----------
            n_uvw: (3) array
            atom_i: int
            atom_j: int
            J: (3,3) array
            symmetry: List[string]
                Only strings with defined names are allowed.

        Examples:

        Easy-plane single-ion anisotropy.
        >>> spinwaves.add_couplings({'Kz': [0,0,0], 0, 0, np.diag([0,0,0.2]), ['1']})

        Nearest neighbour in-plane coupling on hexagonal lattice:
        >>> spinwaves.add_couplings({ 'Jx': [[1,0,0], 0, 0, -0.5*np.eye(3,3), ['6z']] })
        '''

        formatted_couplings = []

        for label,(n_uvw, atom_i, atom_j, J, sym_ops) in couplings.items():
            assert atom_i < len(self.crystal.atoms)    # coupled atom not in the `atom` list
            assert atom_j < len(self.crystal.atoms)    # coupled atom not in the `atom` list

            ri_xyz = self.crystal.uvw2xyz(self.crystal.atoms[atom_i].r)
            rj_xyz = self.crystal.uvw2xyz(self.crystal.atoms[atom_j].r)
            d_xyz = self.crystal.uvw2xyz(n_uvw) + rj_xyz - ri_xyz
            for sym in self.symmetry_operations(sym_ops):
                Jsym = sym @ J @ sym.T
                dsym_xyz = sym @ d_xyz
                nsym_uvw = np.round(self.crystal.xyz2uvw(ri_xyz + dsym_xyz - rj_xyz))
                formatted_couplings.append([dsym_xyz, nsym_uvw, atom_i, atom_j, Jsym])

        self.couplings = couplings                      # [n_uvw, atom_i, atom_j, J, sym]
        self.formatted_couplings = formatted_couplings  # [dsym_xyz, nsym_uvw, atom_i, atom_j, Jsym]

        return formatted_couplings

    def determine_Rn(self, n_uvw: Tuple[int,int,int]) -> np.ndarray:
        '''
        Rn is the rotation corresponding to the modulation of the magnetic moments.
            S_nj = R_n S_0j
            S_nj : magnetic moment of j-th atom in the n-th unit cell, 
                   where the n-th unit cell is indexed by triple-int `n_uvw`.

        Rn determined by the modulation vector, normal to modulation, and the unit cell coordinates.
        '''
        phi = 2*np.pi*np.dot(self.magnetic_structure['k'], n_uvw)
        Rn = ms.rotate(self.magnetic_structure['n'], phi)
        return Rn
    
    def determine_Rprime(self, S) -> np.ndarray:
        '''
        Rn' is the rotation that puts the magnetic moment along z axis.
            S'_nj = R'_n S''_nj
            S'_nj=S_0j : magnetic moment of j-th atom in the 0-th unit cell, independent on unit cell
            S''_nj : spin oriented along the ferromagnetic axis

        Rn is a function of modulation vector and normal to modulation.
        '''

        Rp = np.eye(3,3)

        n = np.cross([0,0,1], S)
        phi = ms.angle([0,0,1], S)
        Rp = ms.rotate(n, phi)

        return Rp
    
    def determine_h(self, q: Tuple, silent: bool=True) -> NDArray[np.complex128]:
        '''
        Determine the reduced Hamiltonian.

        Parameters:
            q: 
                Wavevector in reciprocal lattice units (h,k,l)
        '''

        # TODO
        # Phase trick for consecutive q vectors as in euphonics?

        # Iterate through all atoms to prepare objects required in Eq (26) [spinW]
        # sqrt(S_i) is incorporated into u_i
        Jp0 = np.zeros((len(self.magnetic_atoms), len(self.magnetic_atoms), 3, 3), dtype=np.float64)
        JpofK = np.zeros((len(self.magnetic_atoms), len(self.magnetic_atoms), 3, 3), dtype=np.complex128)
        for (d_xyz, n_uvw, atom_i, atom_j, J) in self.formatted_couplings:
            k = self.crystal.hkl2xyz(q)
            Rn = self.determine_Rn(n_uvw)

            Jp0[atom_i, atom_j, :,:] += J @ Rn
            JpofK[atom_i, atom_j, :,:] += np.exp(1j*np.dot(k, d_xyz)) * (J @ Rn)    # Eq (52) [spinW]

            # Need to fill Jij(k) and Jji(k). Jji = np.conj(Jij^T)
            # if atom_i != atom_j:
            Jp0[atom_j, atom_i, :,:] += np.conj(J @ Rn).T
            JpofK[atom_j, atom_i, :,:] += np.conj(np.exp(1j*np.dot(k, d_xyz)) * (J @ Rn)).T

        u = np.zeros((len(self.magnetic_atoms), 3), dtype=complex)
        v = np.zeros((len(self.magnetic_atoms), 3), dtype=complex)
        S = np.asarray([atom.s for atom in self.magnetic_atoms])
        for atom_i,Sdir_i in enumerate([atom.m for atom in self.magnetic_atoms]):
            Rp_i = self.determine_Rprime(Sdir_i)
            u[atom_i, :] = Rp_i[:,0] + 1j*Rp_i[:,1]
            v[atom_i, :] = Rp_i[:,2]

        JpofmK = np.conj(JpofK)

        SiSj = np.sqrt(np.einsum('i,j->ij', S, S))

        A1 = 0.5*np.einsum('ij,ip,ijpq,jq->ij', SiSj, u, JpofmK, np.conj(u))
        A2 = 0.5*np.conj(np.einsum('ij,ip,ijpq,jq->ij', SiSj, u, JpofK, np.conj(u)))
        B = 0.5*np.einsum('ij,ip,ijpq,jq->ij', SiSj, u, JpofmK, u)
        C = np.diag(np.einsum('l,ip,ilpq,lq->i', S, v, Jp0, v))

        h = np.block([
            [A1-C, B],
            [np.conj(B.T), A2-C]
        ])

        if not silent:
            print('S', S)
            print('Jp0', Jp0.shape)
            print(np.around(Jp0, 5))
            print('JpofK', JpofK.shape)
            print(np.around(JpofK, 5))
            print('u')
            print(np.around(u, 5))
            print('v')
            print(np.around(v, 5))
            print('A1', A1.shape)
            print(np.around(A1, 5))
            print('A2', A2.shape)
            print(np.around(A2, 5))
            print('B', B.shape)
            print(np.around(B, 5))
            print('C', C.shape)
            print(np.around(C, 5))
            print('h', h.shape)
            print(np.around(h, 5))


        return h

    # Plotting should be delegated to the factory.
    # plt_types: [3d_structure, dipsersions?, colormapped_dispersion?]

    def make_supercell(self, 
                       boundaries: Union[float, 
                                        Tuple[float, float, float], 
                                        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float,float]]
                                        ] = 1):
        '''
        PArameters
        ----------
        
        boundaries:
            Extent in all dimensions (float), or specified along each dimension (Tuple[float, float, float])
            or a bounding box for all three dimensions.
        '''

        # Handle bbox creation
        bbox = np.zeros((3,2))
        if np.shape(boundaries) == ():
            bbox[:,1] = boundaries
        elif np.shape(boundaries) == (3,):
            bbox[:,1] = boundaries
        elif np.shape(boundaries) == (3,2):
            bbox = np.array(boundaries)
        else:
            raise IndexError(f'Unexpected dimension of the boundary box: {np.shape(boundaries)} not in [(),(3,),(3,2)]')
        
        bbox = np.sort(bbox, axis=1)

        # Find all atoms, coupling and else 

        # TODO
        # Update atoms magnetic moments by magn struct

        # if self.do_plot_cell:
        #     self.plot_unit_cell_box(view.scene)  # plot gridlines for unit cell boundaries
        # if self.do_plot_mag:
        #     self.plot_magnetic_structure(view.scene, mj, pos, colors)
        # if self.do_plot_atoms:
        #     self.plot_atoms(view.scene, pos, colors, sizes, labels)
        # if self.do_plot_bonds:
        #     self.plot_bonds(view.scene)
        # if self.do_plot_axes:
        #     self.plot_cartesian_axes(view.scene)
        # if self.do_plot_plane:
        #     self.plot_rotation_plane(view.scene, pos[is_matom], colors[is_matom])
        # if self.do_plot_ion:
        #     self.plot_ion_ellipsoids(view.scene)
        # if self.polyhedra_args is not None:
        #     self.plot_polyhedra(view.scene)

        return self.crystal.atoms, self.couplings

    def plot_structure(self,
                       engine: str,
                       boundaries: Union[float, 
                                        Tuple[float, float, float], 
                                        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float,float]]
                                        ] = 1, 
                       plot_mag=True, plot_bonds=True, plot_atoms=True,
                       plot_labels=False, plot_cell=True, plot_axes=True, 
                       plot_plane=True, ion_type=None, polyhedra_args=None):
        '''
        Plot the structure.
        '''

        if engine not in plotting.implemented_sc_plotters:
            raise NameError(f'Requested plotting engine `{engine}` is not implemented. Try: {plotting.implemented_sc_plotters}')
        
        # Crystal objects
        sc_objects = self.make_supercell(boundaries)

        return
    
    def plot_structure_old(self, extent: Tuple[int,int,int]):

        styles = {
            'Cr':dict(color='red'),
            'Fe':dict(color='brown'),
            'Er':dict(color='green'),
            'B':dict(color='blue')
        }
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        # ax.set_proj_type('ortho')  # FOV = 0 deg
        # ax.set_box_aspect((1,1,1))  # aspect ratio is 1:1:1 in data space

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Unit cell
        a, b, c = self.crystal.A.T
        print(a,b,c)
        ax.plot(*np.transpose([[0,0,0], a]), 'r--')
        ax.plot(*np.transpose([[0,0,0], b]), 'g--')
        ax.plot(*np.transpose([[0,0,0], c]), 'b--')

        Nx, Ny, Nz = extent
        unit_cells = []
        for ix in range(-Nx,Nx):
            for iy in range(-Ny,Ny):
                for iz in range(-Nz,Nz):
                    unit_cells.append([ix,iy,iz])

        # Atoms and spins
        for n_uvw in unit_cells:
            for atom_i,atom in enumerate(self.magnetic_atoms):
                Sdir = self.magnetic_structure['spins'][atom_i]
                w = np.array(atom['w'])
                mu = atom['S']*np.array(Sdir)   # this will need a g
                r_n = self.crystal.uvw2xyz(n_uvw)
                r_atom = self.crystal.uvw2xyz(n_uvw+w)

                ax.scatter(*r_atom, **styles[atom['label']])

                S_start = r_atom
                mu_n = np.dot(self.determine_Rn(n_uvw), mu)
                # Sn = np.dot(self.determine_Rnprime(Sn), Sn)   # Check 
                S_end = S_start + mu_n

                # testing Rn Rn' matrices
                # Sn = np.dot(self.determine_Rn(rn), S)
                # S_end = r+np.dot(self.determine_Rnprime(Sn), Sn)
                ax.plot(*np.transpose([S_start, S_end]), color='red')

        # Couplings
        for (r_xyz, n_uvw, atom_i, atom_j, J) in self.formatted_couplings:
            r_start = self.crystal.uvw2xyz(self.magnetic_atoms[atom_i]['w'])
            r_end = r_start + r_xyz
            ax.plot(*np.transpose([r_start, r_end]), color='black', alpha=0.7)

        plt.show()
        return
    
    def calculate_excitations(self, qPath: np.ndarray, silent: bool=True):
        '''
        '''
        Es = []
        N = len(self.magnetic_atoms)
        for q in qPath:
            # print('q', q)
            k = self.crystal.hkl2xyz(q)
            # print('k', k)
            h = self.determine_h(q, silent=silent)
            # print('h', h)
            h += np.diag( [1e-10]*2*N )
            if np.any( np.diag(scipy.linalg.schur(h)[0]) < 0 ):
                Es.append([0,0]*N)
                # if not silent:
                print(f'h(q={q}) is negative')
                continue

            try:
                K = scipy.linalg.cholesky(h)
            except np.linalg.LinAlgError:
                Es.append([0,0]*N)
                # if not silent:
                print(f'Choleskz failed for q={q}')
                print(h)
                continue

        
            # print('K', K)
            g = np.diag( [1]*N + [-1]*N )
            E, U = scipy.linalg.schur(K @ g @ np.conj(K.T)) # Maybe K^t should be first


            Es.append(np.diag(np.real(E)))  # Take appropriate care of real E

        self.qPath = qPath
        self.excitations = np.asarray(Es, dtype=float)

        return Es
    
    def plot_dispersion(self, fig: Figure) -> Figure:
        '''
        Plot dispersions
        '''

        Es = self.excitations

        Qinc = np.concatenate(([0], np.linalg.norm( self.qPath[:-1] - self.qPath[1:], axis=1)))
        Qs = np.cumsum(Qinc)

        ax = fig.get_axes()[0]

        # ax.set_xlim(0, 2)
        ax.set_ylim(0, np.max(Es))

        ax.set_ylabel('E (meV)')

        ax.set_xlabel('Q ((h,k,l))')
        # Plot ticks on (1) main qpoints, (2) the last one, (3) integer positions
        it1 = (Qinc==0)
        it2 = np.concatenate((np.zeros(len(Qinc)-1, dtype=bool),[True]))
        it3 = (np.linalg.norm(self.qPath - self.qPath.round(), axis=1) == 0)
        xticks_it = it1 | it2 | it3
        # print('main', it1)
        # print('last', it2)
        # print('refl', it3)
        # print(self.qPath - self.qPath.round())
        # print(xticks_it)
        xticks = Qs[xticks_it]
        xtickslabels = ['\n'.join([f'{x:.2f}' for x in q]) for q in self.qPath[xticks_it]]
        ax.set_xticks(xticks, labels=xtickslabels)

        for branch in range(Es.shape[-1]):
            ax.plot(Qs, Es[:,branch])    # 0 branch
            # ax.plot(Qs+1/3, Es[:,branch])    # (1/3 1/3 0) branch
            # ax.plot(Qs+2/3, Es[:,branch])    # (2/3 2/3 0) branch

        return fig
    
    def __repr__(self):
        rr = 'SpinW(\n'
        rr += self.crystal.__repr__() + '\n'
        rr += 'Couplings = '
        rr += self.couplings.__repr__()
        rr += '\t})'
        return rr


# TODO
# [] Extend to multiple magnetic atoms per unit cell
# atoms:
#   - [] make spins aligned with the crystal axes
#
# modulation:
#   - [] declare conventions
#
# couplings:
#   - [X] add symmetrization option
#
#
# calculate_excitation(qPath)
# plot_dispersion at different gammas
#
# Potential traps?
# 1. Couplings generated with ['2z','-1'] symmetry. For atom at [u,0,0] both operations will transform
#    it into [-u,0,0], but metrix representations are gonn be different. Thus there will be two couplings involved?

if __name__ == '__main__':
    print('main')