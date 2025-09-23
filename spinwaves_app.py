import spinwaves.gui
from PyQt6.QtWidgets import QApplication


if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName('spinwaves')

    mainwindow = spinwaves.gui.make_window()
    mainwindow.show()

    app.exec()