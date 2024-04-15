import numpy as np

from ..spinw import SpinW
from .supercell_plotter import SupercellPlotter
# from ..crystal import Atom

from vispy import scene
from vispy.color import color_array
# from vispy.visuals.filters import ShadingFilter, WireframeFilter
# from vispy.geometry import create_sphere

# from scipy.spatial.transform import Rotation
# from scipy.spatial import ConvexHull

class VispySupercellPlotter(SupercellPlotter):
    '''
    Vispy is a 3D, ray-tracing, lightweight renderer.

    Own attributes
    ----------
    canvas: vispy.scene.SceneCanvas
        Central object holding the plot window and objects related to visualisation.
    view: vispy.?
        Allows controlling the visualisation of the plotted objects.
    '''
    def __init__(self, sws: SpinW):
        super().__init__(sws=sws)

        # TODO move these setting to config file
        self.atom_alpha = 0.8

        # Init vispy objects
        self.canvas = scene.SceneCanvas(bgcolor='white', show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera()

        return
    
    def deploy_plotter(self):
        self.view.camera.set_range()  # centers camera on middle of data and auto-scales extent
        self.canvas.app.run()
    
        return self.canvas, self.view.scene
    
    def plot_balls(self, 
                   positions: np.ndarray, 
                   sizes: np.ndarray, 
                   colors: np.ndarray):
        colors = colors/255.

        # Straightforward
        obj = scene.visuals.Markers(
                    pos=positions,
                    size=sizes,
                    antialias=0,
                    face_color= colors,
                    edge_color='white',
                    edge_width=0,
                    scaling=True,
                    spherical=True,
                    alpha=self.atom_alpha,
                    parent=self.view.scene)
        
        return obj
        
    def plot_labels(self,
                    positions: np.ndarray,
                    labels: np.ndarray,
                    colors: np.ndarray):
        
        colors = colors/255.
        # Labels
        # fint size was nice and tricky in original vispy implementation
        obj = scene.visuals.Text(pos=positions, 
                                 parent=self.view.scene, 
                                 text=labels, 
                                 color="white", 
                                 font_size=12)

        return obj

    def plot_arrows(self,
                    positions: np.ndarray,
                    directions: np.ndarray,
                    colors: np.ndarray):
        colors = colors/255.

        # Magnetic moments from magnetic atoms only
        # def plot_magnetic_structure(self, canvas_scene, mj, pos, colors):
        self.spin_scale = 1
        self.arrow_width = 8
        self.arrow_head_size = 6

        verts = np.c_[positions, positions + self.spin_scale*directions]  # natom x 6
        # Maybe connect='strip', methof='agg' will work in some future versions and allow high quality arrows
        obj = scene.visuals.Arrow(pos=verts.reshape(-1,3), 
                                  parent=self.view.scene, 
                                  connect='segments',
                                  arrows=verts, 
                                  arrow_size=self.arrow_head_size, 
                                  method='gl',
                                  width=self.arrow_width, 
                                  antialias=True, 
                                  arrow_type='stealth',
                                  color = np.repeat(colors, 2, axis=0).tolist(),
                                  arrow_color= colors.tolist())

        return obj
    
    def plot_lines(self,
                   lines: np.ndarray,
                   colors: np.ndarray,
                   width: float=1,
                   alpha: float=1):
        '''
        width in px
        '''
        colors = colors/255.
        for line, color in zip(lines, colors):
            scene.visuals.Line(pos=line,
                               parent=self.view.scene, 
                               width=width,
                               color=color_array.Color(color=color, alpha=alpha)) # , method="gl")
