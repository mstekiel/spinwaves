
import spinwaves


# Watchdog listener
# import sys
# import time
# import logging
# from watchdog.observers import Observer
# from watchdog.events import LoggingEventHandler

# class ReplottingHandler(LoggingEventHandler):
#      def on_modified(self, event):
#         # Do stuff required when the main file is being modified
#         pass
     
#      def on_any_event(self, event):
#           print(event)

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO,
#                         format='%(asctime)s - %(message)s',
#                         datefmt='%Y-%m-%d %H:%M:%S')
#     path = sys.argv[1] if len(sys.argv) > 1 else '.'
#     event_handler = ReplottingHandler()
#     observer = Observer()
#     observer.schedule(event_handler, path, recursive=True)
#     observer.start()
#     try:
#         while True:
#             time.sleep(1)
#             print('sleep again')
#     except KeyboardInterrupt:
#         observer.stop()
#     observer.join()

# watchfiles listener
from watchfiles import watch
import logging, traceback
import importlib

logging.getLogger('watchfiles').setLevel(logging.INFO) # handle only high priority events from watchlist
logging.basicConfig(level=logging.DEBUG)

for changes in watch(r'C:\Users\Stekiel\Documents\GitHub\spinwaves\sws_script.py'):
    try:
        logging.info('Reloading content.')
        import sws_script
        importlib.reload(sws_script)
        from sws_script import lattice, atoms

        print(lattice)
        print(atoms)


        logging.info('Replotting structure.')
        uc = spinwaves.UnitCell(atoms=atoms)
        sw = spinwaves.SpinW(lattice=lattice, atoms=uc, magnetic_structure={'k':[1/3,1/3,0], 'n':[0,0,1]})

        spinwaves.SupercellPlotter(sw, extent=(2,2,2), engine='vispy',
                    plot_mag=True, plot_bonds=False, plot_atoms=True,
                    plot_labels=False, plot_cell=True, plot_axes=True, 
                    plot_plane=False, ion_type=None, polyhedra_args=None)
    
    except Exception as e:
        logging.error(traceback.format_exc())
         

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

def old_main():
    lattice = spinwaves.Lattice([3.275, 3.275, 3.785, 90,90,120])
    atoms = [
         {'label':'Er', 'r':[0,0,0], 'm':[0,1,0], 's':1},
         {'label':'B', 'r':[0.5,0.5,0.5]}
        ]
    uc = spinwaves.UnitCell(atoms=atoms)
    sw = spinwaves.SpinW(lattice=lattice, atoms=uc, magnetic_structure={'k':[1/3,1/3,0], 'n':[0,0,1]})
    # print('finish')
    # spinwclean()
    # from coupling import couplings

    # sw.add_couplings(couplings=couplings)

    print(sw)
    print(sw.formatted_couplings)

    spinwaves.SupercellPlotter(sw, extent=(2,2,2), plot_mag=True, plot_bonds=False, plot_atoms=True,
                 plot_labels=False, plot_cell=True, plot_axes=True, plot_plane=False, ion_type=None, polyhedra_args=None)
    
    
    data = sw.calculate_excitations(...)
    print('finish')

    # spinwclean()
    # SupercellPlotter(sw, extent=(2,1,2), plot_mag=False, plot_bonds=False, plot_atoms=True,
    #              plot_labels=False, plot_cell=True, plot_axes=True, plot_plane=False, ion_type=None, polyhedra_args=None)

    # Plotting plotly
    # import plotly.graph_objects as go
    
    # fig = go.Figure()

    # fig.update_layout(scene_camera_eye=dict(x=-0.76, y=1.8, z=0.92))

    # trace_cones = fig.add_trace(go.Cone(x=[1], y=[1], z=[1], u=[1], v=[1], w=[0]))

    # atoms_r = np.asarray([atom.r for atom in uc.atoms])
    # print(uc.atoms)
    # print(atoms_r)
    # trace_atoms = fig.add_trace(go.Scatter3d(x=atoms_r[:,0], 
    #                            y=atoms_r[:,1], 
    #                            z=atoms_r[:,2], 
    #                            marker=go.scatter3d.Marker(size=3, color='black'), 
    #                            opacity=0.8, 
    #                            mode='markers'))

    # # fig_window = go.FigureWidget(fig) # still in browser
    # # fig_window.show()

    # fig.show()
