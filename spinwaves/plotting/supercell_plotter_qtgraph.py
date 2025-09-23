import sys
import numpy as np
import logging
qt_logger = logging.getLogger('QtSupercellPlotter')

from ..spinw import SpinW
from .supercell_plotter import SupercellPlotter
from ..functions import cartesian2spherical
# from ..crystal import Atom

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QSplitter, QPushButton, QPlainTextEdit, QTextEdit



                       
class QtgraphSupercellPlotter(SupercellPlotter):
    '''
    PyQtGraph is a Qt library for 2D and 3D plots.
    '''
    def __init__(self, sws: SpinW):
        super().__init__(sws=sws)

        print('Im gonna make a widget')
        self.app = pg.mkQApp()
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(0.7)


        self.config = dict(SPHERES_N_THETA=32, SPHERES_N_PHI=32, SPHERE_ALPHA=0.5)

        return
    
    def deploy(self):
        # self.view.camera.set_range()  # centers camera on middle of data and auto-scales extent
        self.view.show()
        self.app.exec()

        return
    
    def plot_balls(self, 
                   positions: np.ndarray, 
                   sizes: np.ndarray, 
                   colors: np.ndarray):
        colors /= 255
        for position, size, color in zip(positions, sizes, colors):
            qt_logger.debug(f'Plotting sphere: r={position}, size={size}, color={color}')
            mesh_sphere = gl.MeshData.sphere(self.config['SPHERES_N_THETA'], self.config['SPHERES_N_PHI'], 
                                             radius=size, offset=True)
            cc = np.array([0,0,0,self.config['SPHERE_ALPHA']])
            cc[:3] = color
            sphere = gl.GLMeshItem(meshdata=mesh_sphere, smooth=True, color=cc, shader='shaded')

            sphere.translate(*position)

            self.view.addItem(sphere)

        return
        
    def plot_labels(self,
                    positions: np.ndarray,
                    labels: np.ndarray,
                    colors: np.ndarray):
                return

    def plot_arrows(self,
                    positions: np.ndarray,
                    directions: np.ndarray,
                    colors: np.ndarray):
        return
    
    def plot_lines(self,
                   lines: np.ndarray,
                   colors: np.ndarray,
                   width: float=1,
                   alpha: float=1):

        for (r1, r2), color in zip(lines, colors):
            w = width*0.02
            r12 = r2-r1
            length = np.linalg.norm(r2-r1)
            qt_logger.debug(f'Plotting line: r1={r1}, r2={r2}, color={color}')

            mesh_cylinder = gl.MeshData.cylinder(self.config['SPHERES_N_THETA'], self.config['SPHERES_N_PHI'], 
                                             radius=[w, w], length=length, offset=True)
            cc = np.array([0,0,0,self.config['SPHERE_ALPHA']])
            cc[:3] = color
            cylinder = gl.GLMeshItem(meshdata=mesh_cylinder, smooth=False, color=cc, shader='shaded')

            _, theta, phi = cartesian2spherical(r12)
            cylinder.rotate(np.degrees(theta), *[0,1,0])
            cylinder.rotate(np.degrees(phi),   *[0,0,1])
            cylinder.translate(*r1)

            self.view.addItem(cylinder)

        return
    
##########################################################################
##########################################################################
##########################################################################3
# ARCHIVE SNIPPETS

##########################################################################
# For widgeting

# class MainWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("PyQt6 3D Plotter")
#         self.setGeometry(100, 100, 800, 600)

#         layout = QVBoxLayout()

#         self.plot_widget = gl.GLViewWidget()
#         self.plot_widget.setCameraPosition(distance=10)
#         self.plot_widget.grid = gl.GLGridItem()
#         self.plot_widget.addItem(self.plot_widget.grid)
#         layout.addWidget(self.plot_widget)

#         self.setLayout(layout)

                       
# class QtgraphSupercellPlotter(SupercellPlotter):
#     '''
#     PyQtGraph is a Qt library for 2D and 3D plots.
#     '''
#     def __init__(self, sws: SpinW):
#         super().__init__(sws=sws)

#         print('Im gonna make a widget')

#         self.app = QApplication(sys.argv)
#         self.window = MainWindow()

#         self.view = self.window.plot_widget

#         self.config = dict(SPHERES_N_THETA=4, SPHERES_N_PHI=16, LIGHT_DIRECTION=[0,0,-1])

#         return
    
#     def deploy_plotter(self):
#         # self.view.camera.set_range()  # centers camera on middle of data and auto-scales extent
#         self.window.show()
#         sys.exit(self.app.exec())

#         return
##########################################################################