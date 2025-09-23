# -*- coding: utf-8 -*-
# vispy: testskip
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
"""
Sandbox for experimenting with vispy.visuals.shaders
"""
from PyQt6 import QtCore
from PyQt6.QtWidgets import (QPlainTextEdit, QMainWindow, QWidget, 
                             QGridLayout, QSplitter, QApplication,
                             QDockWidget)
import sys
import traceback

class Editor(QPlainTextEdit):
    def __init__(self, parent=None, language=None):
        QPlainTextEdit.__init__(self, parent)

    def setText(self, text):
        self.setPlainText(text)

    def text(self):
        return str(self.toPlainText()).encode('UTF-8')

    def __getattr__(self, name):
        return lambda: None

class WindowVispy(QMainWindow):
    '''Main window of the application consists of docked widgets:
       text editor, console, and inspector.
       Inspector is a multitab widget that switches between:
       3D plotter (Vispy) for structure inspection,
       2D plotter (matplotlib) for any output plots, and
       tree inspector to look up variables values.
    '''
    def __init__(self):
        super().__init__()

        centralWidget = QWidget()
        layout = QGridLayout()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        self.editor = Editor(language='Python')

        inspector_dock = QDockWidget(self)

        from ..plotting.supercell_plotter_vispy import VispySupercellPlotter

        sws = self.load_system()
        sc_plotter = VispySupercellPlotter(sws, show=False)
        self.inspector = sc_plotter.canvas.native
        inspector_dock.setWidget(self.inspector)
        sc_plotter.plot()
        # inspector_dock.setFloating(True)

        self.console = Editor(language='CPP')
        print(sws)
        self.console.setPlainText(sws.__repr__())


        hsplit = QSplitter(QtCore.Qt.Orientation.Horizontal)
        vsplit = QSplitter(QtCore.Qt.Orientation.Vertical)

        layout.addWidget(hsplit)
        hsplit.addWidget(self.editor)
        hsplit.addWidget(vsplit)
        vsplit.addWidget(inspector_dock)
        vsplit.addWidget(self.console)

        self.showMaximized()

        self.editor.textChanged.connect(self.update)
        self.update()


    def update(self):
        code = self.editor.text()
        self.console.setText(str(code))

    def load_system(self) -> 'SpinW':
        from .. import Atom, MSG, Coupling, Crystal, SpinW
        import numpy as np
        atoms = [Atom(label='Fe', r=(0,   0.5, 0),   m=(-1,0,0.1), s=2.5)]
        Pbnm = MSG.from_xyz_strings(generators=[
            'x+1/2,-y+1/2,-z, -1',
            '-x,-y,-z, +1',
            '-x,-y,z+1/2, +1',
        ])

        cs = Crystal(lattice_parameters=[5.3, 5.6, 7.5, 90, 90, 90], 
                    atoms=atoms, MSG=Pbnm)
        
        # print(cs)

        magnetic_modulation = {
            'k':(0, 0, 0),
            'n':(0,0,1)
        }


        ### Extract the model parameters
        # Negative couplings are FM, positive are AF
        Ka = -0.1
        Kc = -0.02
        Kac = -2
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WindowVispy()
    window.show()
    sys.exit(app.exec())
