import numpy as np
import copy
from itertools import chain
from dataclasses import dataclass
import warnings


from vispy import scene
from vispy.color import color_array
from vispy.visuals.filters import ShadingFilter, WireframeFilter
from vispy.geometry import create_sphere

from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

from . import functions as funs
from .spinw import SpinW
from .lattice import Lattice
from .data_containers import atom_data

import mikibox as ms

from typing import Tuple, Dict, List

class SpinW_GL:
    '''
    Main object used to calculate spin waves.

    Fields:
    -------

    lattice:
        `Lattice` object
    magnetic_atoms:
        List of dictionaries [ {'label':'Er', 'w':(0,0,0), 'S':8},...]               
    couplings: 
        List[d, i, j, J]
    magnetic_str:
        [k, n, spins]



    TODO
    ----
        - Each object of the constructor should be its own well defined class, or np.array with defined fields.
            - `Atom`s in `UnitCell` object that lives independently on the lattice.
    '''
    def __init__(self, lattice: Lattice, magnetic_atoms: List[Dict], magnetic_structure: Dict):
        '''
        lattice: Lattice
        atoms: List[label, Wyckoff pos, S]
        couplings: List[d, i, j, J]
        magnetic_str: [k, n, spins]
        '''
        self.lattice = lattice
        self.magnetic_atoms = magnetic_atoms # extend to handle bot magnetic and non-magnetic atoms

        assert len(magnetic_structure['spins']) == len(self.magnetic_atoms)  # Must define spin in the unit cell for each magnetic atom
        self.magnetic_structure = magnetic_structure

        self.couplings = None

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

        couplings: List[r_uvw, atom_i, atom_j, J, symmetry]
            Couple atom with index `atom_j` in the unit cell with index `r_uvw` with atom of index `atom_i` in
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

        (1) Define easy-plane single-ion anisotropy.
        >>> spinwaves.add_couplings([[0,0,0], 0, 0, np.diag([0,0,0.2]), ['1']])
        '''

        formatted_couplings = []

        for label,(n_uvw, atom_i, atom_j, J, sym_ops) in couplings.items():
            assert atom_i < len(self.magnetic_atoms)    # coupled atom not in the `atom` list
            assert atom_j < len(self.magnetic_atoms)    # coupled atom not in the `atom` list

            ri_xyz = self.lattice.uvw2xyz(self.magnetic_atoms[atom_i]['w'])
            rj_xyz = self.lattice.uvw2xyz(self.magnetic_atoms[atom_j]['w'])
            d_xyz = self.lattice.uvw2xyz(n_uvw) + rj_xyz - ri_xyz
            for sym in self.symmetry_operations(sym_ops):
                Jsym = sym @ J @ sym.T
                dsym_xyz = sym @ d_xyz
                nsym_uvw = np.round(self.lattice.xyz2uvw(ri_xyz + dsym_xyz - rj_xyz))
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
        k = self.lattice.hkl2xyz(self.magnetic_structure['k'])
        n_xyz = self.lattice.uvw2xyz(n_uvw)
        phi = np.dot(k, n_xyz)
        # phi = 2*np.pi*np.dot(self.modulation[0], uvw) # Why is the Miller notation not working?
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
    
    
class SpinWObjectParser:
    '''
    SpinW object, full ripoff from Matlab SpinW documentation.
    '''

    def __init__(self, spinw: SpinW_GL):
        self.spinw = spinw

        colors, labels = [], []
        for index,atom in enumerate(self.spinw.magnetic_atoms):
            labels.append(atom['label'])

            RGB = atom_data[ atom_data.name==atom['label'] ][['R','G','B']].to_numpy()[0]
            colors.append(RGB)

        self.unit_cell = dict(color=np.array(colors), label=np.array(labels))


        # coupling['idx', 'mat_idx', 'atom1', 'atom2', 'dl']


    def magstr(self, NExt: List[int]):
        '''
        Is the list of moments going to be oriented just like list of atoms?
        '''
        spins_list = []
        for n_u in range(NExt[0]-1):
            for n_v in range(NExt[1]-1):
                for n_w in range(NExt[2]-1):
                    for atom_i,atom in enumerate(self.spinw.magnetic_atoms):
                        n_uvw = [n_u, n_v, n_w]
                        
                        Sdir = self.spinw.magnetic_structure['spins'][atom_i]
                        w = np.array(atom['w'])
                        spin = np.array(atom['S'])
                        mu = atom['S']*np.array(Sdir)   # this will need a g
                        r_n = self.spinw.lattice.uvw2xyz(n_uvw)
                        r_atom = self.spinw.lattice.uvw2xyz(n_uvw+w)

                        mu_n = np.dot(self.spinw.determine_Rn(n_uvw), mu)
                        spins_list.append(mu_n)
        
        
        mstr = dict(S=np.array(spins_list).T, n=self.spinw.magnetic_structure['n'] )

        return mstr



    def intmatrix(self, plotmode= True, extend=False, sortDM=False, zeroC=False, nExt=[1, 1, 1]) -> Tuple:    # we just need single ion anisotropies here
        '''
        Ref: https://spinw.org/spinw_intmatrix
        For each atom in `self.atoms` return the single ion anisotropies

        If the single ion anisotropies were not defined for the atom set it to ???
        Take care of the matrices shape
        '''

        if not self.spinw.couplings:
            raise KeyError('Need to define couplings first.')

        SS = None
        RR = None

        Jprototype = np.zeros((3,3))
        SI = dict(aniso=np.asarray([Jprototype]*len(self.spinw.magnetic_atoms)))
        for label,(n_uvw, atom_i, atom_j, J, symmetry) in self.spinw.couplings.items():
            if n_uvw==[0,0,0] and atom_i==atom_j:
                SI['aniso'][atom_i] = J

        return SS, SI, RR
    
    def atom(self) -> List:
        '''
        https://spinw.org/spinw_atom
        '''

        rs, idx, mag = [], [], []
        for index, atom in enumerate(self.spinw.magnetic_atoms):
            rs.append(atom['w'])
            idx.append(index)
            mag.append(True)
            
        atomList = dict(r=np.array(rs).T, idx=np.array(idx), mag=np.array(mag))

        return atomList
    
    def matom(self) -> List:
        '''
        https://spinw.org/spinw_atom
        '''

        rs, idx, Ss = [], [], []
        for index, atom in enumerate(self.spinw.magnetic_atoms):
            rs.append(atom['w'])
            idx.append(index)
            Ss.append(atom['S'])
            
        matomList = dict(r=np.array(rs).T, idx=np.array(idx), Ss=np.array(Ss))

        return matomList



    # def genlattice(self, lattice_dict):
    #     '''
    #     Generates all necessary parameters to define a lattice including space group symmetry
    #     and store the result it in the `self.lattice` field.
    #     '''

    #     assert 'lat_const' in lattice_dict   # Input dictionary needs that field
    #     lat_const = np.asarray(lattice_dict['lat_const'])
    #     assert lat_const.shape == 3  # Need three lattice parameters

    #     assert 'angled' in lattice_dict   # Input dictionary needs that field
    #     angled = np.asarray(lattice_dict['angled'])
    #     assert angled.shape == 3     # Need three angle values
        
    #     if 'sym' in lattice_dict:
    #         sym = self.symmetry_operations(lattice_dict['sym'])
    #     else:
    #         sym = self.symmetry_operations(['1'])

    #     # Lets keep it simple for now
    #     origin = np.asarray([0,0,0])

    #     if 'label' in lattice_dict:
    #         label = lattice_dict['label']
    #     else:
    #         label = ''

    #     self.lattice = dict(
    #         lat_const = lat_const,
    #         angle = angled,
    #         sym = sym,
    #         origin = origin,
    #         label = label
    #     )

    #     return self.lattice
    
    # def symmetry_operations(self, generators: List[str]) -> np.ndarray:
    #     '''
    #     Generate all symmetry operations in matrix form based on the generators list.
    #     '''

    #     sym_matrix = {
    #         '1':np.eye(3,3),
    #         '-1':-np.eye(3,3),
    #         '2x':funs.Rx(np.pi),
    #         '2y':funs.Ry(np.pi),
    #         '2z':funs.Rz(np.pi),
    #         '3z':funs.Rz(2*np.pi/3),
    #         '4z':funs.Rz(np.pi/2),
    #         '6z':funs.Rz(np.pi/3),
    #     }

    #     for gen in generators:
    #         if not gen in sym_matrix:
    #             raise KeyError(f'`{gen}` is not implemented/allowed symmetry operator\
    #                             Allowed pars: {list(sym_matrix.keys())} ')

    #     # Ensure identity is in the generators lsit
    #     symmetry = np.concatenate(([sym_matrix[gen] for gen in generators], [sym_matrix['1']]))

    #     # 1. Multiply all symmetry operators by each other and make a table with (N,N,3,3) shape
    #     # 2. Find unique symmetry operations in the flattened table
    #     # 3. If the flattened table is longer then the original symmetry some new perators were created GOTO 1
    #     # Exit: When no new symmetry operators were created
    #     flag = True
    #     while flag:
    #         sym_table = np.einsum('mij,njk->mnik', symmetry, symmetry)
    #         new_symmetry = np.unique( np.around(sym_table.reshape((-1,3,3)), 10), axis=0)
            
    #         if new_symmetry.shape[0] == symmetry.shape[0]:
    #             flag = False

    #         symmetry = new_symmetry

    #     return symmetry

@dataclass
class PolyhedronMesh:
    vertices: np.ndarray
    faces: np.ndarray

class PolyhedraArgs:
    def __init__(self, atom1_idx, atom2_idx, color, n_nearest=6):
        self.atom1_idx = np.array(atom1_idx).reshape(-1)  # makes an array even if single number passed
        self.atom2_idx = np.array(atom2_idx).reshape(-1)  # makes an array even if single number passed
        self.n_nearest=n_nearest
        self.color = color

class SuperCell:
    def __init__(self, spinw: SpinW, extent=(1,1,1), plot_mag=True, plot_bonds=True, plot_atoms=True,
                 plot_labels=False, plot_cell=True, plot_axes=True, plot_plane=True, ion_type=None, polyhedra_args=None):
        """
        :param swobj: spinw object to plot 
        :param extent: Tuple of supercell dimensions default is (1,1,1) - a single unit cell
        :param plot_mag: If True the magneitc moments (in rotating frame representation) will be 
                         plotted if a magnetic structure has been set on swobj
        :param plot_bonds: If True the bonds in swobj.coupling will be plotted
        :param plot_atoms: If True the atoms will be plotted
        :param plot_labels: If True atom labels will be plotted on atom markers
        :param plot_cell: If True the unit cell boundaries will be plotted
        :param plot_axes: If True the arrows for the unit cell vectors will be plotted near the origin
        :param plot_plane: If True the rotation plane will be plotted
        :param ion_type: If not None ion_type can be one of 'aniso' or 'g' and the corresponding 
                         single-ion ellipsoid will be plotted
        :param polyhedra_args: If not None then instance of PolyhedraArgs that stores atom indices and 
                                nearest neighbours. These will be used to plot polyhedra.
        """
        # init with sw obj - could get NExt from object if not explicitly provide (i.e. make default None)
        self.do_plot_mag = plot_mag
        self.do_plot_bonds = plot_bonds
        self.do_plot_atoms = plot_atoms
        self.do_plot_labels = plot_labels
        self.do_plot_cell = plot_cell
        self.do_plot_axes = plot_axes
        self.do_plot_plane = plot_plane
        self.do_plot_ion = ion_type is not None
        self.ion_type = ion_type  # "aniso" or "g"
        self.polyhedra_args = polyhedra_args
        # magnetic structure
        self.mj = None
        self.n = None

        # Convert to expected swobject
        swobj = SpinWObjectParser(spinw)
        self.spinw = spinw

        # get properties from swobj
        self.unit_cell = UnitCell()
        # add atoms
        _, single_ion, _ = swobj.intmatrix(plotmode= True, extend=False, sortDM=False, zeroC=False, nExt=[1, 1, 1]) 
        aniso_mats = single_ion['aniso'].reshape(3,3,-1)  # make 3D array even if only one atom
        g_mats = single_ion['aniso'].reshape(3,3,-1)  # make 3D array even if only one atom
        imat = -1  # index of aniso and g matrices
        atoms_mag = np.array(swobj.atom()['mag']).reshape(-1)  # handles case when only 1 atom and swobj.atom()['mag'] is bool not list
        for iatom, atom_idx in enumerate(swobj.atom()['idx'].flatten().astype(int)):
            # get spin magnitude if magnetic
            if atoms_mag[iatom]:
                # get S
                imatom = np.argmax(swobj.matom()['idx'].flatten().astype(int)== atom_idx)
                # get single-ion matrices
                imat += 1
                g_mat = g_mats[:,:,imat]
                aniso_mat = aniso_mats[:,:,imat]
                mom = swobj.spinw.magnetic_structure['spins'][atom_idx]
            else:
                g_mat = None
                aniso_mat = None
                mom = np.array([0,0,0])
            color = swobj.unit_cell['color'][atom_idx]/255
            label = swobj.unit_cell['label'][atom_idx]
            size = atom_data[atom_data.name==label].radius.to_numpy()
            self.unit_cell.add_atom(Atom(atom_idx, swobj.atom()['r'][:,iatom], moment=mom ,is_mag=atoms_mag[iatom], size=size, color=color, label=label,
                                         gtensor_mat=g_mat, aniso_mat=aniso_mat))
            
        print(self.unit_cell)

        # This is weird AF, skip    
        # add bonds - only plot bonds for which there is a mat_idx
        bond_idx = np.squeeze(swobj.coupling['idx'])
        bond_matrices = swobj.matrix['mat'].reshape(3,3,-1)
        for ibond in np.unique(bond_idx[np.any(swobj.coupling['mat_idx'], axis=0)]):
            i_dl = np.squeeze(bond_idx==ibond)
            for mat_idx in swobj.coupling['mat_idx'][:, np.argmax(i_dl)]:
                if mat_idx > 0:
                    self.unit_cell.add_bond_vertices(name=f"bond{ibond}_mat{mat_idx}",
                                                     atom1_idx=np.squeeze(swobj.coupling['atom1'])[i_dl]-1,
                                                     atom2_idx=np.squeeze(swobj.coupling['atom2'])[i_dl]-1,
                                                     dl=swobj.coupling['dl'].T[i_dl],
                                                     mat=bond_matrices[:,:,mat_idx-1],
                                                     color=swobj.matrix['color'][:,mat_idx-1]/255)

        self.unit_cell.add_bond_vertices(name=f"bond{1}_mat{1}",
                                        atom1_idx=0,
                                        atom2_idx=1,
                                        dl=2,
                                        mat=np.diag([1,2,3]),
                                        color=swobj.matrix['color'][:,mat_idx-1]/255)
        
        # dimensions of supercell (pad by 1 for plotting)
        self.extent = np.asarray(extent)
        self.int_extent = np.ceil(self.extent).astype(int) + 1  # to plot additional unit cell along each dimension to get atoms on cell boundary
        self.ncells = np.prod(self.int_extent)
                    
        # get magnetic structure for all spins in supercell
        self.set_magnetic_structure(swobj)
        
        # transforms
        self.basis_vec = swobj.spinw.lattice.A  # FLAG
        self.inv_basis_vec = np.linalg.inv(self.basis_vec)
                
        # scale factors
        self.abc = np.sqrt(np.sum(self.basis_vec**2, axis=1))
        self.cell_scale_abc_to_xyz = min(self.abc)
        self.supercell_scale_abc_to_xyz = min(self.abc*self.extent)
        # visual properties
        self.bond_width = 5
        self.spin_scale = 1
        self.arrow_width = 8
        self.arrow_head_size = 6
        self.font_size = 20
        self.axes_font_size = 50
        self.atom_alpha = 0.75
        self.mesh_alpha = 0.25
        self.rotation_plane_radius = 0.3*self.cell_scale_abc_to_xyz
        self.ion_radius = 0.3*self.cell_scale_abc_to_xyz
        self.dm_arrow_scale = 0.2*self.cell_scale_abc_to_xyz

    def transform_points_abc_to_xyz(self, points):
        return points @ self.basis_vec
        
    def transform_points_xyz_to_abc(self, points):
        return points @ self.inv_basis_vec

    def set_magnetic_structure(self, swobj):
        magstr = swobj.magstr(NExt=[int(ext) for ext in self.int_extent])
        if not np.any(magstr['S']):
            warnings.warn('No magnetic structure defined')
            self.do_plot_mag = False
            self.do_plot_plane = False
            return
        self.mj = magstr['S'].T
        self.n = np.asarray(magstr['n'])  # plane of rotation of moment


    def plot(self):
        canvas = scene.SceneCanvas(bgcolor='white', show=True)
        view = canvas.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera()
        
        pos, mj, is_matom, colors, sizes, labels, iremove, iremove_mag = self.get_atomic_properties_in_supercell()

        print(pos, pos.shape)
        print(mj, mj.shape)
        
        # This makes no sense so far
        # delete spin vectors outside extent
        # if self.mj is not None:
        #     mj = np.delete(self.mj, iremove_mag, axis=0)
        
        if self.do_plot_cell:
            self.plot_unit_cell_box(view.scene)  # plot gridlines for unit cell boundaries
        if self.do_plot_mag:
            self.plot_magnetic_structure(view.scene, mj, pos, colors)
        if self.do_plot_atoms:
            self.plot_atoms(view.scene, pos, colors, sizes, labels)
        if self.do_plot_bonds:
            self.plot_bonds(view.scene)
        if self.do_plot_axes:
            self.plot_cartesian_axes(view.scene)
        if self.do_plot_plane:
            self.plot_rotation_plane(view.scene, pos[is_matom], colors[is_matom])
        if self.do_plot_ion:
            self.plot_ion_ellipsoids(view.scene)
        if self.polyhedra_args is not None:
            self.plot_polyhedra(view.scene)
        view.camera.set_range()  # centers camera on middle of data and auto-scales extent
        canvas.app.run()
        return canvas, view.scene

    def plot_cartesian_axes(self, canvas_scene):
        pos = np.array([[0., 0., 0.], [1., 0., 0.],
                        [0., 0., 0.], [0., 1., 0.],
                        [0., 0., 0.], [0., 0., 1.],
                        ])*0.5
        pos = pos - 0.5*np.ones(3)
        pos = self.transform_points_abc_to_xyz(pos)
        arrows = np.c_[pos[0::2], pos[1::2]]

        line_color = ['red', 'red', 'green', 'green', 'blue', 'blue']
        arrow_color = ['red', 'green', 'blue']

        scene.visuals.Arrow(pos=pos, parent=canvas_scene, connect='segments',
                    arrows=arrows, arrow_type='angle_60', arrow_size=3.,
                    width=3., antialias=False, arrow_color=arrow_color,
                    color=line_color)
        scene.visuals.Text(pos=self.transform_points_abc_to_xyz(0.7*np.eye(3)-0.5), parent=canvas_scene, text=["a", "b", "c"], color=arrow_color, 
                           font_size=self.axes_font_size*self.supercell_scale_abc_to_xyz)


    def plot_unit_cell_box(self, canvas_scene):
        for zcen in range(self.int_extent[2]):
            for ycen in range(self.int_extent[1]):
                    scene.visuals.Line(pos = self.transform_points_abc_to_xyz(np.array([[0, ycen, zcen], [np.ceil(self.extent[0]), ycen, zcen]])),
                                       parent=canvas_scene, color=color_array.Color(color="k", alpha=0.25)) # , method="gl")
        for xcen in range(self.int_extent[0]):
            for ycen in range(self.int_extent[1]):
                    scene.visuals.Line(pos = self.transform_points_abc_to_xyz(np.array([[xcen, ycen, 0], [xcen, ycen, np.ceil(self.extent[2])]])),
                                       parent=canvas_scene, color=color_array.Color(color="k", alpha=0.25)) # , method="gl")
        for xcen in range(self.int_extent[0]):
            for zcen in range(self.int_extent[2]):
                    scene.visuals.Line(pos = self.transform_points_abc_to_xyz(np.array([[xcen, 0, zcen], [xcen, np.ceil(self.extent[1]), zcen]])),
                                       parent=canvas_scene, color=color_array.Color(color="k", alpha=0.25)) # , method="gl")

    def get_atomic_properties_in_supercell(self):
        atoms_pos_unit_cell = np.array([atom.pos for atom in self.unit_cell.atoms])
        atoms_mom_unit_cell = np.array([atom.moment for atom in self.unit_cell.atoms])
        natoms = atoms_pos_unit_cell.shape[0]
        atoms_pos_supercell = np.zeros((self.ncells*natoms, 3))
        atoms_mom_supercell = np.zeros((self.ncells*natoms, 3))
        icell = 0
        # loop over unit cells in same order as in MATLAB
        for zcen in range(self.int_extent[2]):
            for ycen in range(self.int_extent[1]):
                for xcen in range(self.int_extent[0]):
                    atoms_pos_supercell[icell*natoms:(icell+1)*natoms,:] = atoms_pos_unit_cell + np.array([xcen, ycen, zcen])

                    n_uvw = np.array([xcen, ycen, zcen])

                    rotated_spins = np.dot(self.spinw.determine_Rn(n_uvw), atoms_mom_unit_cell.T)
                    atoms_mom_supercell[icell*natoms:(icell+1)*natoms,:] = rotated_spins.T
                    


                    icell += 1

        is_matom = np.tile([atom.is_mag for atom in self.unit_cell.atoms], self.ncells)
        sizes = np.tile([atom.size for atom in self.unit_cell.atoms], self.ncells)
        colors = np.tile(np.array([atom.color for atom in self.unit_cell.atoms]).reshape(-1,3), (self.ncells, 1))
        # remove points beyond extent of supercell
        atoms_pos_supercell, iremove = self._remove_points_outside_extent(atoms_pos_supercell)
        atoms_mom_supercell = np.delete(atoms_mom_supercell, iremove, axis=0)
        sizes = np.delete(sizes, iremove)
        colors = np.delete(colors, iremove, axis=0)
        # get indices of magnetic atoms outside extents
        iremove_mag = [np.sum(is_matom[:irem]) for irem in iremove if is_matom[irem]]
        is_matom = np.delete(is_matom, iremove)
        # transfrom to xyz
        atoms_pos_supercell = self.transform_points_abc_to_xyz(atoms_pos_supercell)
        # get atomic labels
        labels = np.tile([atom.label for atom in self.unit_cell.atoms], self.ncells)
        labels = np.delete(labels, iremove).tolist()
        return atoms_pos_supercell, atoms_mom_supercell, is_matom, colors, sizes, labels, iremove, iremove_mag
    
    def plot_magnetic_structure(self, canvas_scene, mj, pos, colors):
        verts = np.c_[pos, pos + self.spin_scale*mj]  # natom x 6
        # Maybe connect='strip', methof='agg' will work in some future versions and allow high quality arrows
        scene.visuals.Arrow(pos=verts.reshape(-1,3), parent=canvas_scene, connect='segments',
            arrows=verts, arrow_size=self.arrow_head_size, method='gl',
            width=self.arrow_width, antialias=True, 
            arrow_type='stealth',
            color=np.repeat(colors, 2, axis=0).tolist(),
            arrow_color= colors.tolist())
    
    def plot_rotation_plane(self, canvas_scene, pos, colors, npts=15):
        # generate vertices of disc with normal // [0,0,1]
        theta = np.linspace(0, 2*np.pi,npts)[:-1] # exclude 2pi
        disc_verts = np.zeros((npts, 3))
        disc_verts[1:,0] = self.rotation_plane_radius*np.cos(theta)
        disc_verts[1:,1] = self.rotation_plane_radius*np.sin(theta)
        # rotate given normal
        rot_mat = get_rotation_matrix(self.n)
        disc_verts = rot_mat.dot(disc_verts.T).T
        disc_faces = self._label_2D_mesh_faces(disc_verts)
        # for each row (atom) in pos to add to shift to verts (use np boradcasting)
        disc_verts = (disc_verts + pos[:,None]).reshape(-1,3)
        # increment faces indices to match larger verts array (use np boradcasting)
        disc_faces = (disc_faces + np.repeat(npts*np.arange(pos.shape[0]), 3).reshape(-1,1,3)).reshape(-1,3)
        # repeat colors
        face_colors = np.tile(colors, (npts-1, 1))
        face_colors = np.c_[face_colors, np.full((face_colors.shape[0], 1), self.mesh_alpha)]  # add transparency
        scene.visuals.Mesh(vertices=disc_verts, faces=disc_faces, face_colors=face_colors, parent=canvas_scene)
    
    def plot_ion_ellipsoids(self, canvas_scene, npts=7):
        matoms = [atom for atom in self.unit_cell.atoms if atom.is_mag and np.any(atom.get_transform(tensor=self.ion_type))]
        if len(matoms) > 0:
            # get mesh for a sphere
            meshdata = create_sphere(radius=self.ion_radius, rows=npts, cols=npts)
            sphere_verts = meshdata.get_vertices()
            sphere_faces = meshdata.get_faces()
            # loop over ions and get mesh verts and faces
            ion_verts = np.zeros((len(matoms) * self.ncells * sphere_verts.shape[0], 3))
            ion_faces = np.zeros((len(matoms) * self.ncells * sphere_faces.shape[0], 3))
            face_colors = np.full((ion_faces.shape[0], 4), self.mesh_alpha)
            irow_verts, irow_faces = 0, 0
            imesh = 0
            for atom in matoms:
                ellip_verts = sphere_verts @ atom.get_transform(tensor=self.ion_type)
                for zcen in range(self.int_extent[2]):
                    for ycen in range(self.int_extent[1]):
                        for xcen in range(self.int_extent[0]):
                            centre = (atom.pos + np.array([xcen, ycen, zcen])).reshape(1,-1) # i.e. make 2D array
                            centre, _ = self._remove_points_outside_extent(centre)
                            if centre.size > 0:
                                # atom in extents
                                centre = self.transform_points_abc_to_xyz(centre)
                                ion_verts[irow_verts:irow_verts+sphere_verts.shape[0]] = ellip_verts + centre
                                ion_faces[irow_faces:irow_faces+sphere_faces.shape[0]] = sphere_faces + sphere_verts.shape[0] * imesh
                                face_colors[irow_faces:irow_faces+sphere_faces.shape[0], :3] = atom.color  # np broadcasting allows this
                                irow_verts = irow_verts+sphere_verts.shape[0]
                                irow_faces = irow_faces+sphere_faces.shape[0]
                                imesh += 1
            mesh = scene.visuals.Mesh(vertices=ion_verts[:irow_verts,:], faces=ion_faces[:irow_faces,:].astype(int), face_colors=face_colors[:irow_faces,:], parent=canvas_scene)
            wireframe_filter = WireframeFilter(color=3*[0.7])
            mesh.attach(wireframe_filter)

    def plot_atoms(self, canvas_scene, pos, colors, sizes, labels):
        scene.visuals.Markers(
                    pos=pos,
                    size=sizes,
                    antialias=0,
                    face_color= colors,
                    edge_color='white',
                    edge_width=0,
                    scaling=True,
                    spherical=True,
                    alpha=self.atom_alpha,
                    parent=canvas_scene)
        # labels
        if self.do_plot_labels:
            scene.visuals.Text(pos=pos, parent=canvas_scene, text=labels, color="white", font_size=self.font_size*self.cell_scale_abc_to_xyz)

    def plot_bonds(self, canvas_scene):
        max_dm_norm = self.unit_cell.get_max_DM_vec_norm()
        for bond_name in self.unit_cell.bonds:
            color = self.unit_cell.get_bond_color(bond_name)
            verts = self._get_supercell_bond_verts(bond_name)
            if self.unit_cell.is_bond_symmetric(bond_name):
                scene.visuals.Line(pos=verts, parent=canvas_scene, connect='segments', 
                                   width=self.bond_width, color=color)
            else:
                # DM bond
                # generate verts of DM arrows at bond mid-points (note DM vector in xyz)
                mid_points = verts.reshape(-1,2,3).sum(axis=1)/2
                dm_vec = self.dm_arrow_scale*self.unit_cell.get_bond_DM_vec(bond_name)/max_dm_norm
                dm_verts = np.c_[mid_points, mid_points + dm_vec]
                arrow_verts = np.r_[np.c_[verts[::2], mid_points], dm_verts]  # draw arrow at mid-point of line as well as DM vec
                line_verts = np.r_[verts, dm_verts.reshape(-1,3)]
                scene.visuals.Arrow(pos=line_verts, parent=canvas_scene, connect='segments',
                                    arrows=arrow_verts, arrow_size=self.arrow_head_size,
                                    width=self.arrow_width, antialias=True, 
                                    arrow_type='triangle_60',
                                    color=color,
                                    arrow_color=color)
                                    
    def _get_supercell_bond_verts(self, bond_name):
        bond = self.unit_cell.bonds[bond_name]
        bond_verts_unit_cell = self.unit_cell.get_bond_vertices(bond_name)
        nverts_per_cell = bond_verts_unit_cell.shape[0]
        bond_verts_supercell = np.zeros((self.ncells*nverts_per_cell, 3))
        icell = 0
        for zcen in range(self.int_extent[2]):
            for ycen in range(self.int_extent[1]):
                for xcen in range(self.int_extent[0]):
                    lvec = np.array([xcen, ycen, zcen])
                    bond_verts_supercell[icell*nverts_per_cell:(icell+1)*nverts_per_cell,:] = bond_verts_unit_cell + lvec
                    icell += 1
        bond_verts_supercell, _ = self._remove_vertices_outside_extent(bond_verts_supercell)
        bond_verts_supercell = self.transform_points_abc_to_xyz(bond_verts_supercell)
        return bond_verts_supercell
        

    def plot_polyhedra(self, canvas_scene):
        polyhedra = self._calc_convex_polyhedra_mesh()
        # loop over all unit cells and add origin to mesh vertices
        npoly = self.ncells*len(polyhedra)
        nverts_per_poly = polyhedra[0].vertices.shape[0]
        verts = np.zeros((npoly*nverts_per_poly, 3))
        nfaces_per_poly = polyhedra[0].faces.shape[0]
        faces = np.zeros((npoly*nfaces_per_poly, 3))
        irow_verts, irow_faces = 0, 0
        ipoly = 0
        for zcen in range(self.int_extent[2]):
            for ycen in range(self.int_extent[1]):
                for xcen in range(self.int_extent[0]):
                    lvec = self.transform_points_abc_to_xyz(np.array([xcen, ycen, zcen]))
                    for poly in polyhedra:
                        this_verts = poly.vertices + lvec
                        _, irem = self._remove_points_outside_extent(self.transform_points_xyz_to_abc(this_verts))
                        if len(irem) < self.polyhedra_args.n_nearest:
                            # polyhedron has at least 1 vertex inside extent
                            verts[irow_verts:irow_verts+nverts_per_poly, : ] = this_verts
                            faces[irow_faces:irow_faces+nfaces_per_poly, : ] = poly.faces + nverts_per_poly * ipoly
                            irow_verts = irow_verts+nverts_per_poly
                            irow_faces = irow_faces+nfaces_per_poly
                            ipoly += 1
        mesh = scene.visuals.Mesh(vertices=verts[:irow_verts,:], faces=faces[:irow_faces,:].astype(int), 
                                  color=color_array.Color(color=self.polyhedra_args.color, alpha=self.mesh_alpha), parent=canvas_scene)
        wireframe_filter = WireframeFilter(color=3*[0.7])
        mesh.attach(wireframe_filter)

    def _calc_convex_polyhedra_mesh(self):
        atom2_pos_xyz = self.transform_points_abc_to_xyz(np.array([atom.pos for atom in self.unit_cell.atoms if atom.wyckoff_index in self.polyhedra_args.atom2_idx]))
        natom2 = atom2_pos_xyz.shape[0]
        polyhedra = []
        for atom1_pos_rlu in np.array([atom.pos for atom in self.unit_cell.atoms if atom.wyckoff_index in self.polyhedra_args.atom1_idx]):
            # find vector bewteen atom1 in unit cells +/- 1 in each direction to atom2 in first unit cell
            dr = np.zeros((27*natom2, 3))
            icell = 0
            for dz in range(-1,2):
                for dy in range(-1,2):
                    for dx in range(-1,2):
                        atom1_pos_xyz = self.transform_points_abc_to_xyz(atom1_pos_rlu + np.array([dx, dy, dz]))
                        dr[icell*natom2:(icell+1)*natom2,:] = -atom2_pos_xyz + atom1_pos_xyz  # ordered like this due to np broadcasting
                        icell += 1
            # keep unique within some tolerance (1e-3)
            _, unique_idx = np.unique(np.round(dr,3), axis=0, return_index=True)
            dr = dr[unique_idx]
            # sort and get n shortest
            isort = np.argsort(np.linalg.norm(dr, axis=1))
            dr = dr[isort[:self.polyhedra_args.n_nearest]]
            atom1_pos_xyz = self.transform_points_abc_to_xyz(atom1_pos_rlu)
            verts_xyz = dr + atom1_pos_xyz
            rank = np.linalg.matrix_rank(dr)
            if rank == 3:
                hull = ConvexHull(verts_xyz)
                polyhedra.append(PolyhedronMesh(vertices=verts_xyz[hull.vertices], faces=hull.simplices))
            elif rank == 2:
                # transform to basis of polygon plane
                *_, evecs_inv = np.linalg.svd(verts_xyz - verts_xyz[0])  # sorted in decreasing order of singular value
                verts_2d = (evecs_inv @ verts_xyz.T).T
                hull = ConvexHull(verts_2d[:, :-1]) # exclude last col (out of polygon plane - all have same value)
                verts_xyz = np.vstack((atom1_pos_xyz, verts_xyz[hull.vertices], ))  # include central atom1 position as vertex
                faces = self._label_2D_mesh_faces(verts_xyz)
                polyhedra.append(PolyhedronMesh(vertices=verts_xyz, faces=faces))
            else:
                warnings.warn('Polyhedron vertices must not be a line or point')
        return polyhedra
    
    def _label_2D_mesh_faces(self, verts):
        # assume centre point has index 0
        nverts = verts.shape[0]
        faces = np.zeros((nverts-1, 3), dtype=int)
        faces[:,1] = np.arange(1,nverts)
        faces[:,2] = np.arange(2,nverts + 1)
        faces[-1,2] = 1  # close the shape by returning to first non-central vertex
        return faces
    
    def _remove_vertices_outside_extent(self, verts):
        # DO THIS BEFORE CONVERTING TO XYZ
        # remove pairs of verts that correpsond to a line outside the extent
        _, iremove = self._remove_points_outside_extent(verts)
        iatom2 = (iremove % 2).astype(bool) # end point of pair of vertices
        # for atom2 type vertex we need to remove previous row (atom1 vertex)
        # for atom1 type vertex we need to remove the subsequent row (atom2 vertex)
        iremove = np.hstack((iremove, iremove[iatom2]-1, iremove[~iatom2]+1))
        return np.delete(verts, iremove, axis=0), iremove

    def _remove_points_outside_extent(self, points, tol=1e-10):
        # DO THIS BEFORE CONVERTING TO XYZ
        iremove = np.flatnonzero(np.logical_or(np.any(points < -tol, axis=1), np.any((points - self.extent)>tol, axis=1)))
        return np.delete(points, iremove, axis=0), iremove

class UnitCell:
    def __init__(self, atoms_list=None, bonds=None):
        self.atoms = atoms_list if atoms_list is not None else []
        self.bonds = bonds if bonds is not None else {}
    
    def add_atom(self, atom):
        self.atoms.append(atom)

    def add_bond_vertices(self, name, atom1_idx, atom2_idx, dl, mat, color):
        # get type of interaction from matrix
        self.bonds[name] = {'verts': np.array([(self.atoms[atom1_idx[ibond]].pos, 
                            self.atoms[atom2_idx[ibond]].pos + dl) for ibond, dl in enumerate(np.asarray(dl))]).reshape(-1,3)}
        self.bonds[name]['is_sym'] = np.allclose(mat, mat.T)
        self.bonds[name]['DM_vec'] = np.array([mat[1,2], mat[2,0], mat[0,1]]) if not self.bonds[name]['is_sym'] else None
        self.bonds[name]['color'] = color

    def get_bond_vertices(self, bond_name):
        return self.bonds[bond_name]['verts']

    def get_bond_DM_vec(self, bond_name):
        return self.bonds[bond_name]['DM_vec']
    
    def get_bond_color(self, bond_name):
        return self.bonds[bond_name]['color']
        
    def is_bond_symmetric(self, bond_name):
        return self.bonds[bond_name]['is_sym']
        
    def get_max_DM_vec_norm(self):
        max_norm = 0
        for bond_name in self.bonds:
            if self.bonds[bond_name]['DM_vec'] is not None:
                max_norm = max(max_norm, np.linalg.norm(self.bonds[bond_name]['DM_vec']))
        return max_norm
    
    def __str__(self) -> str:
        return '<' + '\n'.join([self.__class__.__name__] + ['\t'+str(atom) for atom in self.atoms]) + '>'

class Atom:
    '''
    Stores informations about the atom.

    Questions:
    - what is index?
    - moment in what units?
    - position in what units?
    '''
    def __init__(self, index, position, is_mag=False, moment=np.zeros(3), size=0.2, gtensor_mat=None, aniso_mat=None, label='atom', color="blue"):
        self.pos = np.asarray(position)
        self.is_mag = is_mag
        self.moment = moment
        self.gtensor = gtensor_mat
        self.aniso = aniso_mat
        self.size = size
        self.color = color
        self.label = label
        self.spin_scale = 0.3
        self.wyckoff_index = index
        
    def get_transform(self, tensor='aniso'):
        '''
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
        
    def __str__(self) -> str:
        return f'<{self.__class__.__name__} = Label: {self.label}, r_uvw: {self.pos}, mu: {self.moment}>'

def get_rotation_matrix(vec2, vec1=np.array([0,0,1])):
    vec1 = vec1/np.linalg.norm(vec1)  # unit vectors
    vec2 = vec2/np.linalg.norm(vec2)
    if np.arccos(np.clip(np.dot(vec1.flat, vec2.flat), -1.0, 1.0)) > 1e-5:
        r = Rotation.align_vectors(vec2.reshape(1,-1), vec1.reshape(1,-1))  # matrix to rotate vec1 onto vec2
        return r[0].as_matrix()
    else:
        # too small a difference for above algorithm, just return identity
        return np.eye(3)
    

basis_vec = np.diag([4,5,8])
def transform_points_abc_to_xyz(points):
        return points @ basis_vec


if __name__ == '__main__':
    canvas = scene.SceneCanvas(bgcolor='white', show=True)
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera()
    
    # pos, is_matom, colors, sizes, labels, iremove, iremove_mag = self.get_atomic_properties_in_supercell()


    extent = np.asarray([2,2,2])
    int_extent = np.ceil(extent).astype(int) + 1  # to plot additional unit cell along each dimension to get atoms on cell boundary
    abc = np.sqrt(np.sum(basis_vec**2, axis=1))
    cell_scale_abc_to_xyz = min(abc)
    supercell_scale_abc_to_xyz = min(abc*extent)
    

    # self.plot_cartesian_axes(view.scene)
    # def plot_cartesian_axes(self, canvas_scene):
    canvas_scene = view.scene

    pos = np.array([[0., 0., 0.], [1., 0., 0.],
                    [0., 0., 0.], [0., 1., 0.],
                    [0., 0., 0.], [0., 0., 1.],
                    ])*0.5
    pos = pos - 0.5*np.ones(3)
    pos = transform_points_abc_to_xyz(pos)
    arrows = np.c_[pos[0::2], pos[1::2]]

    line_color = ['red', 'red', 'green', 'green', 'blue', 'blue']
    arrow_color = ['red', 'green', 'blue']

    scene.visuals.Arrow(pos=pos, parent=canvas_scene, connect='segments',
                arrows=arrows, arrow_type='angle_60', arrow_size=3.,
                width=3., antialias=False, arrow_color=arrow_color,
                color=line_color)
    scene.visuals.Text(pos=transform_points_abc_to_xyz(0.7*np.eye(3)-0.5), parent=canvas_scene, text=["a", "b", "c"], color=arrow_color, 
                        font_size=50*supercell_scale_abc_to_xyz)
    
    



    
    

    for zcen in range(int_extent[2]):
        for ycen in range(int_extent[1]):
                scene.visuals.Line(pos = transform_points_abc_to_xyz(np.array([[0, ycen, zcen], [np.ceil(extent[0]), ycen, zcen]])),
                                    parent=canvas_scene, color=color_array.Color(color="k", alpha=0.25)) # , method="gl")
    for xcen in range(int_extent[0]):
        for ycen in range(int_extent[1]):
                scene.visuals.Line(pos = transform_points_abc_to_xyz(np.array([[xcen, ycen, 0], [xcen, ycen, np.ceil(extent[2])]])),
                                    parent=canvas_scene, color=color_array.Color(color="k", alpha=0.25)) # , method="gl")
    for xcen in range(int_extent[0]):
        for zcen in range(int_extent[2]):
                scene.visuals.Line(pos = transform_points_abc_to_xyz(np.array([[xcen, 0, zcen], [xcen, np.ceil(extent[1]), zcen]])),
                                    parent=canvas_scene, color=color_array.Color(color="k", alpha=0.25)) # , method="gl")
                    


        

    view.camera.set_range()  # centers camera on middle of data and auto-scales extent

    # im = canvas.render(alpha=True)
    # import vispy.io
    # vispy.io.image.write_png(r'C:\Users\Stekiel\Desktop\Offline-plots\vispy-test.png', im)

    canvas.app.run()