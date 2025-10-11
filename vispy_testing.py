import numpy as np

from spinwaves.plotting import plot_structure
from spinwaves.plotting.supercell_plotter_vispy import VispySupercellPlotter
from spinwaves import Atom, MSG, SpinW, Crystal, Coupling

import logging
logger = logging.getLogger()
logger.setLevel('INFO')

def load_system() -> SpinW:
    atoms = [Atom(label='Fe', r=(0,   0.5, 0),   m=(-1,0,0.1), s=2.5),
             Atom(label='Er', r=(0.5, 0.5, 0.25),   m=(1,0,1), s=5),]
            #  Atom(label='O', r=(0.25, 0.25, 0),   m=(1,0,1), s=5)]
    Pbnm = MSG.from_xyz_strings(generators=[
        'x+1/2,-y+1/2,-z, -1',
        '-x,-y,-z, +1',
        '-x,-y,z+1/2, +1',
    ])

    cs = Crystal(lattice_parameters=[5.3, 5.6, 7.5, 90, 90, 90], 
                 atoms=atoms, MSG=Pbnm)
    
    print(cs)

    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,0,1)
    }


    ### Extract the model parameters
    # Negative couplings are FM, positive are AF
    Ka = -0.0
    Kc = -0.02
    Kac = 0
    J1ab = 2
    J1c  = 1

    couplings = []
    
    # Wrap up the couplings in one list
    # Single-ion anisotropies
    K = np.array([
        [ Ka, 0, Kac],
        [  0, 0, 0],
        [Kac, 0, Kc],
    ])
    couplings += [Coupling(label=f'K_Fe', n_uvw=[0,0,0], id1=0, id2=0, J=K)]
    couplings += [Coupling(label=f'J1c', n_uvw=[0,0,0], id1=0, id2=1, J=J1c*np.eye(3,3))]
    couplings += [Coupling(label=f'J1a', n_uvw=[0,0,0], id1=0, id2=2, J=J1ab*np.eye(3,3))]

    # Construct the main object that is able to determine excitation spectrum
    sw = SpinW(crystal=cs, 
               couplings=couplings,
               magnetic_modulation=magnetic_modulation)
    
    return sw

def crystal_vis():
    sw = load_system()
    print('Symemtrized couplings')
    print(sw.couplings_all)
    plot_opts = dict(boundaries=([-0.1, 1.1],[-0.1,1.1],[-0.1,1.02]), 
                             coupling_colors={'J1c': 'Orange', 'J1ab':'Gray', 'J2a':'Green', 'J2b':'Red', 'J2d':'Blue'})
        
    plot_structure(sw, engine='vispy', plot_options=plot_opts)

    return

##############################################################################
def vispy_example():
    import numpy as np
    from vispy import app, scene

    # Create canvas and view
    canvas = scene.SceneCanvas(keys='interactive', size=(600, 600), show=True)
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.ArcballCamera(fov=0)
    view.camera.scale_factor = 500

    # Prepare data
    np.random.seed(57983)
    data = np.random.normal(size=(40, 3), loc=0, scale=100)
    size = np.random.rand(40) * 100
    colors = np.random.rand(40, 3)

    data = np.concatenate([data, [[0, 0, 0]]], axis=0)
    size = np.concatenate([size, [100]], axis=0)
    colors = np.concatenate([colors, [[1, 0, 0]]], axis=0)


    # Create and show visual
    vis = scene.visuals.Markers(
        pos=data,
        size=100,
        antialias=0,
        face_color=colors,
        edge_color='white',
        edge_width=0,
        scaling=True,
        spherical=True,
    )
    vis.parent = view.scene

    lines = np.array([[data[i], data[-1]]
                    for i in range(len(data) - 1)])
    line_vis = []

    for line in lines:
        vis2 = scene.visuals.Tube(line, radius=5)
        vis2.parent = view.scene
        line_vis.append(vis2)

    print(lines)

    app.run()


##############################################################################
def vispy_tests():
    '''Test basic objects from the library.
    
    Notes
    -----
    
    1. Vispy objects always need a parent.
    2. The `edge_color` from Sphere is very nice. In fact its an overlayed duplicated
       Sphere mesh with different plot options. See code.
    '''
    from vispy import scene
    from vispy.visuals.transforms import STTransform

    canvas = scene.SceneCanvas(keys='interactive', bgcolor='white',
                            size=(800, 600), show=True)

    view = canvas.central_widget.add_view()
    view.camera = 'arcball'

    from vispy.scene.visuals import Sphere, Arrow, Line, Tube, Markers, XYZAxis

    ### SPHERE
    # Experiment with creation options. Works ok.
    sphere = Sphere(radius=1, method='latitude', parent=view.scene,
                    shading='smooth')
    sphere.transform = STTransform(translate=[-2.5, 0, 0])

    # [2] Sphere is a CompoundVisual, with wireframe as below
    # sphere._border = MeshVisual(vertices=mesh.get_vertices(),
    #                         faces=mesh.get_edges(),
    #                         color=edge_color, mode='lines')

    ### TUBE
    # Works well. Can be a line or arrow
    tube = Tube(points=[[0,0,0],[0,0,1e-5],[0,0,2],[0,0,2+1e-5],[0,0,4]], 
                radius=[0, 0.2, 0.2, 1, 0], 
                tube_points=32, shading='smooth', color='red',
                parent=view.scene)
    # tube = Tube(points=[[0,0,0],[0,0,1e-5],[0,0,2],[0,0,2+1e-5],[0,0,4]], 
    #             radius=[0, 0.2, 0.2, 1, 0], 
    #             tube_points=32, shading=None, mode='lines', color='black',
    #             parent=view.scene)


    ### XYZ
    # works only as 2D so far
    # axis = XYZAxis(parent=view)
    # s = STTransform(scale=(50, 50, 50))
    # # axis.transform = s.as_matrix()


    view.camera.set_range(x=[-3, 3])

    canvas.app.run()


def vispy_arrows():
    plotter = VispySupercellPlotter( load_system() )
    plotter.plot()
    plotter.present_arrows()
    canvas = plotter.deploy()

    # Use write_png to export your wonderful plot as png ! 
    # import vispy.io as io
    # canvas.size
    # canvas.pixel_scale = 10
    # from vispy.gloo.util import _screenshot
    # img = _screenshot()
    # img = canvas.render(size=(canvas.size[0]*10, canvas.size[1]*10))
    # img = canvas.render()
    # io.write_png("wonderful.png",img)

def vispy_xyz(filename):
    with open(filename, 'r') as ff:
        lines = ff.readlines()

    header = [line for line in lines if line.startswith('#')]
    u,v,w, Mx,My,Mz = np.loadtxt(filename).T

    atoms = [Atom(label='Mn', r=(0, 0, 0),   m=(0,0,1), s=2.5)]
    P1 = MSG.from_xyz_strings(generators=['x,y,z, +1'])

    cs = Crystal(lattice_parameters=[5.3729, 5.3729, 7.0954, 90, 90, 120], 
                 atoms=atoms, MSG=P1)
    magnetic_modulation = {'k':(0, 0, 0),'n':(0,0,1)}
    couplings = []

    # Construct the main object that is able to determine excitation spectrum
    sw = SpinW(crystal=cs, 
               couplings=couplings,
               magnetic_modulation=magnetic_modulation)
    
    positions = cs.uvw2xyz(np.transpose([u,v,w]))

    plotter = VispySupercellPlotter(sw)
    plotter.plot_balls(positions=positions, sizes=np.full((len(u),), 1), colors=np.full((len(u),3), (0,0,255)))
    plotter.plot_arrows(positions=positions, directions=np.transpose([Mx,My,Mz]), colors=np.full((len(u),3), (255,0,0)))
    # plotter.present_arrows()

    plotter.view.camera.center = [5,5,0]
    plotter.view.camera.distance = 50
    
    canvas = plotter.canvas.app.run()


if __name__ == "__main__":
    # crystal_vis()
    # vispy_tests()
    # vispy_example()
    # vispy_arrows()
    # opengl_test()

    # filename = r'C:\GDrive\PostDoc-Juelich\7_NBMP-Nikolaos\6_D23-finalized\NBMPO-600mK-0T.xyz'
    # vispy_xyz(filename)


    from spinwaves.plotting.supercell_plotter_vispy_advanced import AdvancedVispySupercellPlotter
    plotter = AdvancedVispySupercellPlotter( load_system() )
    plotter.launch()