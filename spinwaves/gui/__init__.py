from PyQt5.QtWidgets import QMainWindow, QApplication


from .main import SWMain


def main():
    app = QApplication([])
    app.setApplicationName('spinwaves')

    mainwindow = SWMain()

    mainwindow.show()

    app.exec()
