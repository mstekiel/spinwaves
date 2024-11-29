# from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget

# from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure

# from ..crystal import Crystal

# logger = None

# class SWMain(QMainWindow):
#     def __init__(self):
#         QMainWindow.__init__(self)

#         print('running app')

#         self._main = QWidget()
#         self.setCentralWidget(self._main)
#         self.layout = QVBoxLayout(self._main)

#         fig = Figure()
#         ax = fig.add_axes([0,0,300,300])
#         ax.scatter([0,1], [2,3])
#         self.canvas = FigureCanvas(fig)

import numpy as np
import plotly.graph_objs as go
import plotly.offline



import os, sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QUrl
from PyQt5 import QtWebEngineWidgets


class PlotlyViewer(QtWebEngineWidgets.QWebEngineView):
    def __init__(self, fig, exec=True):
        # Create a QApplication instance or use the existing one if it exists
        self.app = QApplication.instance() if QApplication.instance() else QApplication(sys.argv)

        super().__init__()

        self.file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp.html"))
        plotly.offline.plot(fig, filename=self.file_path, auto_open=False)
        self.load(QUrl.fromLocalFile(self.file_path))
        self.setWindowTitle("spinwaves")
        self.show()

        if exec:
            self.app.exec_()

    def closeEvent(self, event):
        os.remove(self.file_path)


class SWMain():
    def __init__(self, engine: str='plotly'):
        fig = go.Figure()
        fig.add_scatter(x=np.random.rand(100), y=np.random.rand(100), mode='markers',
                marker={'size': 30, 'color': np.random.rand(100), 'opacity': 0.6,
                        'colorscale': 'Viridis'})

        return PlotlyViewer(fig)