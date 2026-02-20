from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import numpy as np
from copy import deepcopy


from ..atom import Atom
from ..spinw import Coupling, SpinW
from ..databases import color_data

import logging
import logging.config
import traceback
sc_logger = logging.getLogger('SupercellPlotter')

#######################################################################################################
# Tools
def _format_bbox(boundaries: Union[int, list[int]]):
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

def find_matching_label_color(label: str, label_colors: dict[str, Any]):
    """In plotting options you can specify to color bonds according to provided prefix
    This will go through provided dict(prefix=colorname) and find matching color."""
    color_ret = color_data['Gray'].RGB
    for prefix, color_name in label_colors.items():
        if label.startswith(prefix):
            color_ret = color_data[color_name].RGB
            break

    return color_ret

#######################################################################################################

class SupercellPlotter(ABC):
    '''Plots the supercell of a crystal in 3D.
    
    All distances are in Angstroms. This includes decorators dimensions (hopefully).
    '''
    def __init__(self, sws: SpinW):
        self.spinw = sws
        self.crystal = sws.crystal

        self.logger = sc_logger

        # I am not sure how to do this best. 
        # self._objects = dict(atoms=[], couplings=[], cell_deges=[], ref_system=[], extra=[])

        # Config settings, such that each child will adapt those to its engine

        
    @abstractmethod
    def plot_balls(self, 
                   positions: np.ndarray, 
                   sizes: np.ndarray, 
                   colors: np.ndarray):
        '''Plot balls.
        Used for drawing:
         - atoms.
        
        Parameters
        ----------
        positions: (n,3)
        sizes: (n,)
        colors: (n,3)

        Returns
        -------
        objects: (n,)
            List of plot objects, specific for library used
        '''
        pass

    @abstractmethod
    def plot_lines(self,
                   lines: np.ndarray,
                   colors: np.ndarray):
        '''Plot lines.
        Used for drawing:
         - cell edges,
         - bonds.
        
        Parameters
        ----------
        lines: (n,2,3)
            [..., [i_line_start, i_line_end], ...]
        colors: (n,3)

        Returns
        -------
        objects: (n,)
            List of plot objects, specific for library used
        '''
        pass

    @abstractmethod
    def plot_arrows(self,
                    positions: np.ndarray,
                    directions: np.ndarray,
                    colors: np.ndarray):
        '''Plot arrows.
        Used for drawing:
         - magnetic moments,
         - reference system,
         - DM interaction direction.
        
        Parameters
        ----------
        positions: (n,3)
            Where to start the arrow.

        directions: (n,3)
            Where the arrows are pointing towards.

        colors: (n,3)
            Color of the arrows.

        Returns
        -------
        objects: (n,)
            List of plot objects, specific for library used
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

        Returns
        -------
        objects: (n,)
            List of plot objects, specific for library used
        '''
        pass

    @abstractmethod
    def plot_ellipsoids(self,
                        positions: np.ndarray,
                        matrices: np.ndarray,
                        colors: np.ndarray):
        '''Plot ellipsoids.

        Parameters
        ----------
        positions: (n,3)
        matrices: (n,3,3)
            Defined the principal axes, by solving the eigenproblem.
        colors: (n,3)

        Returns
        -------
        objects: (n,)
            List of plot objects, specific for library used.
        '''
        pass

    @abstractmethod
    def deploy(self):
        '''Library specific routines to deploy the plot.
        
        Returns
        -------
        widget:
            Main widget used to control the plot window.
        '''
        pass

    #######################################################################################################
    
    def plot(self, plot_options: dict={}):
        '''Main plotting function'''

        boundaries = [[0,1],[0,1],[0,1]]
        if 'boundaries' in plot_options:
            boundaries = plot_options['boundaries']

        self.atom_alpha = plot_options.pop('atom_alpha', 0.8)
        self.spin_scale = plot_options.pop('spin_scale', 1)
        self.arrow_width = plot_options.pop('arrow_width', 0.1)
        self.arrow_head_size = plot_options.pop('arrow_head_size', 3)

        bbox = _format_bbox(boundaries)
        self.logger.info(f"Plotting stuff in bbox: {bbox}")
        
        atoms, edges = self.get_objects_in_supercell(bbox)
        
        self.logger.info(f"Plotting atoms: {atoms}")

        # Atoms
        pos = self.crystal.uvw2xyz([atom.r for atom in atoms])
        sizes = np.array([atom.radius for atom in atoms])
        colors = np.array([atom.color for atom in atoms])
        try:
            self.plot_balls(positions=pos, sizes=sizes, colors=colors)
        except Exception as e:
            self.logger.error(traceback.format_exc())

        # 
        self._structure_center = np.average(pos, axis=0)
        self._largest_distance = np.abs(pos-self._structure_center).max()

        # TODO atom labels

        # Magnetic moments
        magnetic_atoms = [atom for atom in atoms if atom.is_mag]
        ma_r = self.crystal.uvw2xyz([atom.r for atom in magnetic_atoms])
        ma_m = np.array([atom.m*atom.s for atom in magnetic_atoms])
        ma_colors = np.array([atom.color for atom in magnetic_atoms])
        # ma_colors = np.array([[255,0,0] for atom in magnetic_atoms])
        try:
            self.plot_arrows(positions=ma_r, directions=ma_m, colors=ma_colors)
        except Exception as e:
            self.logger.error(traceback.format_exc())

        # Cell edges
        colors = np.array([color_data['Black'].RGB for _ in edges])
        edges = np.array([self.crystal.uvw2xyz(np.array(edge)) 
                          for edge in edges])
        
        try:
            self.plot_lines(edges, colors, alpha=0.25, width=1)
        except Exception as e:
            self.logger.error(traceback.format_exc())

        # zeroth cell has colorful edges, make a plot option
        main_edges = np.array([
            [[0,0,0],[1,0,0]],
            [[0,0,0],[0,1,0]],
            [[0,0,0],[0,0,1]]
        ])
        main_edges = np.array([self.crystal.uvw2xyz(np.array(edge)) 
                               for edge in main_edges])
        colors = np.array([
            color_data['Red'].RGB,
            color_data['Green'].RGB,
            color_data['Blue'].RGB
            ])

        try:
            print('DEBUG plotting main cell eges')
            self.plot_lines(main_edges, colors, width=4)
        except Exception as e:
            self.logger.error(traceback.format_exc())

        # Prepare couplings
        couplings_lines, couplings_colors = [], []
        scpl_ellipsoid_pos, scpl_ellipsoid_mat, scpl_ellipsoid_color = [], [], []
        dmi_positions, dmi_directions, dmi_colors = [], [], []
        for cpl in self.spinw.couplings_all:
            ### Use this snippet to remove bonds reaching outside bbox
            l2_uvw = self.spinw.magnetic_atoms[cpl.id2].r + cpl.n_uvw
            if any(l2_uvw < bbox[:,0]) or any(l2_uvw > bbox[:,1]):
                continue

            l1 = self.crystal.uvw2xyz(self.spinw.magnetic_atoms[cpl.id1].r)
            l2 = self.crystal.uvw2xyz(l2_uvw)

            # self coupling, as in single ion anisotropy
            if np.allclose(l1, l2):   # TODO, skip for now
                scpl_ellipsoid_pos.append(l1)

                atom = self.spinw.magnetic_atoms[cpl.id1]

                ### Make the shortest axis = atom radius
                # matrix = cpl.J - np.diag([1,1,1])*np.max(np.linalg.eigvals(cpl.J))  # highest eval is 0
                # matrix = -matrix + atom.radius
                ### Make the longest axis = twice atom radius
                a, b, c = np.linalg.eigh(cpl.J)[0] # evals are ordered such the a<b<c
                matrix = cpl.J - np.diag([1,1,1])*c # evals: a-c<b-c<0
                matrix = -matrix # evals c-a > c-b > 0
                matrix = matrix*5*atom.radius/(c-a) # evals: 2r > 
                
                scpl_ellipsoid_mat.append(matrix)
                scpl_ellipsoid_color.append(self.spinw.magnetic_atoms[cpl.id1].color)

                continue

            # interatomic coupling as line
            couplings_lines.append([l1,l2])
            # TODO handle colors properly
            if 'coupling_colors' in plot_options:
                # couplings_colors.append(find_matching_label_color(cpl.label, plot_options['coupling_colors']))
                color_name = plot_options['coupling_colors'].get(cpl.label.split('_')[0], 'Gray')
            else:
                color_name = 'Gray'     

            couplings_colors.append(color_data[color_name].RGB)

            # DMI arrow
            if not np.allclose(cpl.DMI_vector, np.zeros(3)):
                dmi_positions.append( (l1+l2)/2 )
                dmi_directions.append( 11*cpl.DMI_vector )
                dmi_colors.append(color_data['Black'].RGB)

        # Plot self couplings
        try:
            scpl_ellipsoid_pos = np.array(scpl_ellipsoid_pos)
            scpl_ellipsoid_mat = np.array(scpl_ellipsoid_mat)
            scpl_ellipsoid_color = np.array(scpl_ellipsoid_color)
            self.plot_ellipsoids(scpl_ellipsoid_pos, scpl_ellipsoid_mat, scpl_ellipsoid_color)
        except Exception as e:
            self.logger.error(traceback.format_exc())

        # Plot interatomic couplings
        self.logger.info(f"Plotting couplings: {couplings_lines}")
        try:
            couplings_lines = np.array(couplings_lines)
            couplings_colors = np.array(couplings_colors)
            # couplings_colors = np.array([color_data['Gray'].RGB for _ in self.spinw.couplings_all])
            self.plot_lines(couplings_lines, couplings_colors, width=5)
        except Exception as e:
            self.logger.error(traceback.format_exc())

        # Plot DM vectors
        try:
            dmi_positions = np.array(dmi_positions)
            dmi_directions = np.array(dmi_directions)
            dmi_colors = np.array(dmi_colors)
            # couplings_colors = np.array([color_data['Gray'].RGB for _ in self.spinw.couplings_all])
            self.plot_arrows(dmi_positions, dmi_directions, dmi_colors)
        except Exception as e:
            self.logger.error(traceback.format_exc())


        return

    def get_objects_in_supercell(self, boundaries) -> tuple[list[Atom], np.ndarray]:
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
                    for atom in self.crystal.atoms_all:
                        atom_candidate = deepcopy(atom)
                        n_uvw = np.array([xcen, ycen, zcen])

                        atom_candidate.r += n_uvw

                        if atom.is_mag:
                            rotated_spin = np.dot(self.spinw.rot_Rn(n_uvw), atom.m.T)
                            atom_candidate.m = rotated_spin
                        
                        if all(atom_candidate.r > low_bound) and all(atom_candidate.r < high_bound):
                            atoms.append(atom_candidate)

                    # atoms_pos_supercell[icell*natoms:(icell+1)*natoms,:] = atoms_pos_unit_cell + np.array([xcen, ycen, zcen])


                    # atoms_mom_supercell[icell*natoms:(icell+1)*natoms,:] = rotated_spins.T

                    # icell += 1

        return atoms, edges