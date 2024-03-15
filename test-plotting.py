import numpy as np
import matplotlib.pyplot as plt

from vispy import scene
from vispy.color import color_array
from itertools import chain
from vispy.visuals.filters import ShadingFilter, WireframeFilter
from vispy.geometry import create_sphere
import copy
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from dataclasses import dataclass
import warnings

from matplotlib.figure import Figure

import spinwaves

#######################################################


def bare() -> None:
    basis_vec = np.diag([4,5,8])

    def transform_points_abc_to_xyz(points):
            return points @ basis_vec

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
    
    return

#######################################################
def spinwparser() -> None:
    # To recreate spinW spectra Dz anisotropy ocmponent has to be multiplied by two
    hex = spinwaves.Lattice([3.275, 3.275, 3.785, 90, 90, 120])
    atoms = [
        {'label':'Er', 'w':(0,0,0), 'S':8},
        {'label':'Er', 'w':(0.5,0,0), 'S':8},
    ]   # position in crystal coordinates
    magnetic_structure = {
        'k':(0, 0, 0.03),
        'n':(0,0,1),
        'spins':[
            (1,0,0),
            (-1,0,0),
        ]
    }

    sw_er = spinwaves.SpinW(lattice=hex, magnetic_atoms=atoms, magnetic_structure=magnetic_structure)

    print('Add couplings...')
    Jx = -0.0354
    Jxz = -0.004
    Jz, J2z = -0.0155, -0.002
    couplings = {
        'K':[[0,0,0], 0, 0, np.diag([0,0.002,6.7]), ['1']], # K

        'Jx':[[1, 0,0], 0, 0, Jx*np.eye(3,3), ['6z']],
        'Jxz':[[1,0,1], 0, 0, Jxz*np.eye(3,3), ['6z','-1']],
        'Jz':[[0,0,1], 0, 0, Jz*np.eye(3,3), ['-1']],
        'J2z':[[0,0,2], 0, 0, J2z*np.eye(3,3), ['-1']],
       }   # (d,i,j,J) d has to be symmetrized by hand; Indices here correspond to atoms in the `atoms` list
    sw_er.add_couplings(couplings)





    sc = spinwaves.SuperCell(spinw=sw_er, extent=(1,1,5))
    print(sc.unit_cell)
    sc.plot()

    return

#######################################################
def spinwclean() -> None:
    # To recreate spinW spectra Dz anisotropy ocmponent has to be multiplied by two
    hex = spinwaves.Lattice([3.275, 3.275, 3.785, 90, 90, 120])
    atoms = [
        {'label':'Er', 'r':(0,0,0), 'm':[1,0,0], 'S':8},
        {'label':'Er', 'r':(0.5,0,0), 'm':[0,1,0], 'S':8},
    ]   # position in crystal coordinates
    # magnetic_structure = {
    #     'k':(0, 0, 0.03),
    #     'n':(0,0,1),
    #     'spins':[
    #         (1,0,0),
    #         (-1,0,0),
    #     ]
    # }

    # sw_er = spinwaves.SpinW(lattice=hex, unit_cell=spinwaves.UnitCell(atoms), magnetic_structure=)

    # print('Add couplings...')
    # Jx = -0.0354
    # Jxz = -0.004
    # Jz, J2z = -0.0155, -0.002
    # couplings = {
    #     'K':[[0,0,0], 0, 0, np.diag([0,0.002,6.7]), ['1']], # K

    #     'Jx':[[1, 0,0], 0, 0, Jx*np.eye(3,3), ['6z']],
    #     'Jxz':[[1,0,1], 0, 0, Jxz*np.eye(3,3), ['6z','-1']],
    #     'Jz':[[0,0,1], 0, 0, Jz*np.eye(3,3), ['-1']],
    #     'J2z':[[0,0,2], 0, 0, J2z*np.eye(3,3), ['-1']],
    #    }   # (d,i,j,J) d has to be symmetrized by hand; Indices here correspond to atoms in the `atoms` list
    # sw_er.add_couplings(couplings)



    # sc = spinwaves.SuperCell(spinw=sw_er, extent=(1,1,5))
    # print(sc.unit_cell)
    # sc.plot()

    return

if __name__ == '__main__':
    lattice = spinwaves.Lattice([3.275, 3.275, 3.785, 90,90,120])
    atoms = [
         {'label':'Er', 'r':[0,0,0], 'm':[0,1,0], 's':1},
         {'label':'B', 'r':[0.5,0.5,0.5]}
        ]
    uc = spinwaves.UnitCell( atoms=atoms )
    sw = spinwaves.SpinW(lattice=lattice, unit_cell=uc, magnetic_structure={'k':[1/3,1/3,0], 'n':[0,0,1]})
    # print('finish')
    # spinwclean()
    couplings = {
        'Kz':[[0,0,0], 0, 0, np.diag([0,0,0.2]), ['1']],
        'Jx':[[1,0,0], 0, 0, -0.5*np.eye(3,3), ['6z']],
    }

    sw.add_couplings(couplings=couplings)

    print(sw)
    print(sw.formatted_couplings)

    spinwaves.plotSupercell(sw, extent=(2,2,2), plot_mag=True, plot_bonds=False, plot_atoms=True,
                 plot_labels=False, plot_cell=True, plot_axes=True, plot_plane=False, ion_type=None, polyhedra_args=None)
