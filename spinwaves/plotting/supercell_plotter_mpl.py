import numpy as np

from ..spinw import SpinW
from .supercell_plotter import SupercellPlotter

# import matplotlib
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure

class MPLSupercellPlotter(SupercellPlotter):
    '''
    Matplotlib standard python plotting library

    Own attributes
    ----------
    fig
    ax
    '''
    ax: Axes
    fig: Figure

    def __init__(self, sws: SpinW):
        super().__init__(sws=sws)

        # TODO move these setting to config file
        self.atom_alpha = 0.8

        self._ball_sizes_scale = 100

        # Init vispy objects
        # print(matplotlib.__version__)
        # fig, ax = plt.subplots(projection='3D')
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')

        plt.axis('off')
        plt.grid(None)

        # pyplot is not implemeting equal aspect for 3d plot, so one needs to set the axes limits.
        # To do that, we need to list all objects and find the extreme coordinates for each direction.

        return
    
    def deploy_plotter(self):
        # self.fig.show()
        plt.show()
    
        return None, None
    
    def plot_balls(self, 
                   positions: np.ndarray, 
                   sizes: np.ndarray, 
                   colors: np.ndarray):
        colors = colors/255.
        sizes = sizes * self._ball_sizes_scale
        print('pp', positions.shape)
        print('ss', sizes.shape)

        # Straightforward
        obj = self.ax.scatter(
                    positions[:,0], positions[:,1], positions[:,2],
                    s=sizes, c=colors, marker='o')

                    # antialias=0,
                    # edge_color='white',
                    # edge_width=0,
                    # scaling=True,
                    # spherical=True,
                    # alpha=self.atom_alpha,
                    # parent=self.view.scene)
        
        return obj
        
    def plot_labels(self,
                    positions: np.ndarray,
                    labels: np.ndarray,
                    colors: np.ndarray):
        
        colors = colors/255.
        # Labels
        # fint size was nice and tricky in original vispy implementation
        # obj = scene.visuals.Text(pos=positions, 
        #                          parent=self.view.scene, 
        #                          text=labels, 
        #                          color="white", 
        #                          font_size=12)

        return

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

        self.ax.quiver(*positions.T, *directions.T, colors=colors)

        # verts = np.c_[positions, positions + self.spin_scale*directions]  # natom x 6
        # Maybe connect='strip', methof='agg' will work in some future versions and allow high quality arrows
        # obj = scene.visuals.Arrow(pos=verts.reshape(-1,3), 
        #                           parent=self.view.scene, 
        #                           connect='segments',
        #                           arrows=verts, 
        #                           arrow_size=self.arrow_head_size, 
        #                           method='gl',
        #                           width=self.arrow_width, 
        #                           antialias=True, 
        #                           arrow_type='stealth',
        #                           color = np.repeat(colors, 2, axis=0).tolist(),
        #                           arrow_color= colors.tolist())

        return 
        
    def plot_lines(self,
                   lines: np.ndarray,
                   colors: np.ndarray,
                   width: float=1,
                   alpha: float=1):
        colors = colors/255.

        for line, color in zip(lines, colors):
            self.ax.plot(*line.T, color=color)
        # for line, color in zip(lines, colors):
        #     scene.visuals.Line(pos=line,
        #                        parent=self.view.scene, 
        #                        width=width,
        #                        color=color_array.Color(color=color, alpha=alpha)) # , method="gl")

        return
