from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
from copy import deepcopy

from ..spinw import SpinW
from ..data_containers import color_data

# Tools
def _prepare_bbox(boundaries: Union[int, list[int]]):
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
        
    return np.sort(bbox, axis=1)

class SupercellPlotter(ABC):
    '''Plots the 3d crystal.
    
    How about implementing draw_line() in each child, and here have plot_unitcell_edges(), plot_bonds(), etc 
    call draw_line?'''
    def __init__(self, sws: SpinW):
        self.spinw = sws
        self.crystal = sws.crystal

        # Config settings, such that each child will adapt those to its engine
        ...

    
    def plot(self, plot_options: dict):
        '''Main plotting function
        
        Return widgets responsible for plotting'''

        boundaries = plot_options['boundaries']

        bbox = _prepare_bbox(boundaries)
        
        atoms, edges = self.get_objects_in_supercell(bbox)

        # Atoms
        pos = self.crystal.uvw2xyz([atom.r for atom in atoms])
        sizes = [atom.radius for atom in atoms]
        colors = np.array([atom.color for atom in atoms])
        self.plot_balls(positions=pos, sizes=sizes, colors=colors)
        # TODO atom labels

        # Magnetic atoms
        magnetic_atoms = [atom for atom in atoms if atom.is_mag]
        ma_r = self.crystal.uvw2xyz([atom.r for atom in magnetic_atoms])
        ma_m = np.array([atom.m for atom in magnetic_atoms])
        ma_colors = np.array([atom.color for atom in magnetic_atoms])
        self.plot_arrows(positions=ma_r, directions=ma_m, colors=ma_colors)

        # Cell edges
        colors = np.array([color_data['Black'] for _ in edges])
        edges = np.array([self.crystal.uvw2xyz(np.array(edge)) 
                          for edge in edges])
        self.plot_lines(edges, colors, alpha=0.25)

        # zeroth cell has colorful edges, make a plot option
        main_edges = np.array([
            [[0,0,0],[1,0,0]],
            [[0,0,0],[0,1,0]],
            [[0,0,0],[0,0,1]]
        ])
        main_edges = np.array([self.crystal.uvw2xyz(np.array(edge)) 
                               for edge in main_edges])
        colors = np.array([
            color_data['Red'],
            color_data['Green'],
            color_data['Blue']
            ])

        self.plot_lines(main_edges, colors, width=5)

        return self.deploy_plotter()
    

    @abstractmethod
    def plot_balls(self, 
                   positions: np.ndarray, 
                   sizes: np.ndarray, 
                   colors: np.ndarray):
        '''Plot balls.
        Used in:
         - atoms.
        
        Parameters
        ----------
        positions: (n,3)
        sizes: (n,)
        colors: (n,3)
        '''
        pass

    @abstractmethod
    def plot_lines(self,
                   lines: np.ndarray,
                   colors: np.ndarray):
        '''Plot lines.
        Used in:
         - cell edges,
         - bonds.
        
        Parameters
        ----------
        lines: (n,2,3)
            [..., [i_line_start, i_line_end], ...]
        colors: (n,3)
        '''
        pass

    @abstractmethod
    def plot_arrows(self,
                    positions: np.ndarray,
                    directions: np.ndarray,
                    colors: np.ndarray):
        '''Plot arrows.
        Used in:
         - magnetic moments,
         - reference system,
         - DM interaction direction.
        
        Parameters
        ----------
        positions: (n,3)
        directions: (n,3)
        colors: (n,3)
        '''
        pass

    @abstractmethod
    def plot_labels(self,
                    positions: np.ndarray,
                    labels: np.ndarray,
                    colors: np.ndarray):
        '''Plot labels.
        
        Parameters
        ----------
        positions: (n,3)
        labels: (n,)
        colors: (n,3)
        '''
        pass



    @abstractmethod
    def deploy_plotter(self):
        '''Library specifif routins to deploy the plot'''



    def get_objects_in_supercell(self, boundaries):
        '''Find all objects that fit within the boundaries
        
        Returns
        -------
        atoms
        magnetic_atoms
        edges
        '''
        EPS = 1e-8
        # Atoms have to be taken in negative cells as well
        ext_atoms = np.floor(boundaries).astype(int)
        # Edges only for full cells
        ext_edges = np.trunc(boundaries).astype(int)

        ### EDGES
        edges = []
        it_nx = range(ext_edges[0][0], ext_edges[0][1]+1)
        it_ny = range(ext_edges[1][0], ext_edges[1][1]+1)
        it_nz = range(ext_edges[2][0], ext_edges[2][1]+1)

        for ny in it_ny:
            for nz in it_nz:
                edges.append([[ext_edges[0][0],ny,nz], [ext_edges[0][1],ny,nz]])
        for nx in it_nx:
            for nz in it_nz:
                edges.append([[nx,ext_edges[1][0],nz], [nx,ext_edges[1][1],nz]])
        for nx in it_nx:
            for ny in it_ny:
                edges.append([[nx,ny,ext_edges[2][0]], [nx,ny,ext_edges[2][1]]])

        edges = np.array(edges, dtype=float)

        ### ATOMS
        atoms = []
        it_nx = range(ext_atoms[0][0], ext_atoms[0][1]+1)
        it_ny = range(ext_atoms[1][0], ext_atoms[1][1]+1)
        it_nz = range(ext_atoms[2][0], ext_atoms[2][1]+1)
        low_bound = boundaries[:,0] - EPS
        high_bound = boundaries[:,1] + EPS
        for zcen in it_nz:
            for ycen in it_ny:
                for xcen in it_nx:
                    for atom in self.crystal.atoms:
                        atom_candidate = deepcopy(atom)
                        n_uvw = np.array([xcen, ycen, zcen])

                        atom_candidate.r += n_uvw

                        if atom.is_mag:
                            rotated_spin = np.dot(self.spinw.determine_Rn(n_uvw), atom.m.T)
                            atom_candidate.m = rotated_spin
                        
                        if all(atom_candidate.r > low_bound) and all(atom_candidate.r < high_bound):
                            atoms.append(atom_candidate)

                    # atoms_pos_supercell[icell*natoms:(icell+1)*natoms,:] = atoms_pos_unit_cell + np.array([xcen, ycen, zcen])


                    # atoms_mom_supercell[icell*natoms:(icell+1)*natoms,:] = rotated_spins.T

                    # icell += 1

        return atoms, edges