r"""Tests of `spinwaves` (package `__init__`)"""
import subprocess
import sys


def test_core_import_without_graphics():
    """Importing `spinwaves` must not import any of the graphics libraries"""
    code = ('import sys, spinwaves; '
            'print(",".join(m for m in ("matplotlib", "vispy", "PyQt6", "pyqtgraph", "networkx") '
            'if m in sys.modules))')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == ''
