print('Importing libraries...')
import sys
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import io
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QSplitter, QPushButton, QPlainTextEdit, QTextEdit
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
print('Importing succesful!')

class CodeEditor(QPlainTextEdit):
    """Simple code editor for writing Python code."""
    def __init__(self, execute_callback, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Write your Python code for 3D plotting here...")
        self.setStyleSheet("font: 12px monospace;")
        
        self.appendPlainText('plot_ball(1,2,3)')
        
        # Bind Ctrl+Enter to execute code
        self.shortcut_run = QShortcut(QKeySequence("Ctrl+Enter"), self)
        self.shortcut_run.activated.connect(execute_callback)

class OutputConsole(QTextEdit):
    """Output console widget for displaying execution results and errors."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet("font: 12px monospace; background-color: #222; color: #0f0;")
        self.setPlaceholderText("Console Output...")

    def append_output(self, text):
        self.append(text)
        self.ensureCursorVisible()

class Plot3DWidget(gl.GLViewWidget):
    """3D Plot Widget using pyqtgraph's OpenGL module."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCameraPosition(distance=10)
        self.grid = gl.GLGridItem()
        self.addItem(self.grid)

    def plot(self, x, y, z):
        """Clear existing plot and render new 3D scatter plot."""
        self.clear()
        self.addItem(self.grid)
        scatter = gl.GLScatterPlotItem(pos=np.c_[x, y, z], color=(1, 1, 1, 0.5), size=5)
        self.addItem(scatter)
    
    def plot_ball(self, center, radius, color=(1, 1, 1, 1)):
        """Plot a sphere (ball) at a given center with a specified radius and color."""
        mesh_sphere = gl.MeshData.sphere(16, 16, radius=radius, offset=True)

        self.addItem(gl.GLMeshItem(meshdata=mesh_sphere))
    
    def plot_arrow(self, start, end, color=(1, 0, 0, 1), width=2):
        """Plot an arrow from start to end position with a cone as the head."""
        line = np.array([start, end])
        self.addItem(gl.GLLinePlotItem(pos=line, color=color, width=width))
        
        # Compute direction and normalize
        direction = np.array(end) - np.array(start)
        length = np.linalg.norm(direction)
        if length == 0:
            return  # Avoid division by zero
        direction /= length
        
        # Define cone properties
        cone_height = length * 0.2
        cone_radius = width * 0.5
        
        # Generate cone points
        theta = np.linspace(0, 2 * np.pi, 20)
        cone_base = np.array([cone_radius * np.cos(theta), cone_radius * np.sin(theta), np.zeros_like(theta)])
        cone_tip = np.array([0, 0, cone_height])
        
        # Rotate cone to align with arrow direction
        v = np.array([0, 0, 1])
        axis = np.cross(v, direction)
        angle = np.arccos(np.dot(v, direction))
        if np.linalg.norm(axis) != 0:
            axis /= np.linalg.norm(axis)
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            R = np.array([
                [cos_a + axis[0]**2 * (1 - cos_a), axis[0] * axis[1] * (1 - cos_a) - axis[2] * sin_a, axis[0] * axis[2] * (1 - cos_a) + axis[1] * sin_a],
                [axis[1] * axis[0] * (1 - cos_a) + axis[2] * sin_a, cos_a + axis[1]**2 * (1 - cos_a), axis[1] * axis[2] * (1 - cos_a) - axis[0] * sin_a],
                [axis[2] * axis[0] * (1 - cos_a) - axis[1] * sin_a, axis[2] * axis[1] * (1 - cos_a) + axis[0] * sin_a, cos_a + axis[2]**2 * (1 - cos_a)]
            ])
            cone_base = R @ cone_base
            cone_tip = R @ cone_tip
        
        # Translate cone to arrow end
        cone_base += np.array(end).reshape(3, 1)
        cone_tip += np.array(end)
        
        # Create and add cone
        cone_points = np.column_stack((cone_base, cone_tip))
        self.addItem(gl.GLLinePlotItem(pos=cone_points.T, color=color, width=width))
    
    def plot_line(self, points, color=(0, 1, 0, 1), width=2):
        """Plot a line connecting multiple points."""
        self.addItem(gl.GLLinePlotItem(pos=np.array(points), color=color, width=width))

class WindowQtGraph(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 3D Plotter")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_panel = QSplitter(Qt.Orientation.Vertical)

        self.editor = CodeEditor(self.execute_code)
        self.console = OutputConsole()
        self.plot_widget = Plot3DWidget()
        
        run_button = QPushButton("Run Code")
        run_button.clicked.connect(self.execute_code)
        
        left_panel.addWidget(self.editor)
        left_panel.addWidget(self.console)
        left_panel.setSizes([300, 150])
        
        splitter.addWidget(left_panel)
        splitter.addWidget(self.plot_widget)
        splitter.setSizes([400, 400])
        
        layout.addWidget(splitter)
        layout.addWidget(run_button)
        self.setLayout(layout)

    def execute_code(self):
        """Executes the code from the editor and updates the 3D plot."""
        code = self.editor.toPlainText()
        local_env = {"np": np, "plot": self.plot_widget.plot, "plot_ball": self.plot_widget.plot_ball, "plot_arrow": self.plot_widget.plot_arrow, "plot_line": self.plot_widget.plot_line}  # Safe execution environment
        self.console.clear()
        
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            exec(code, {}, local_env)
            output = sys.stdout.getvalue()
            self.console.append_output(output if output else "Execution Successful")
        except Exception as e:
            self.console.append_output(f"Error: {e}")
        finally:
            sys.stdout = old_stdout

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WindowQtGraph()
    window.show()
    sys.exit(app.exec())
