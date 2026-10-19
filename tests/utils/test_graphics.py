r"""Tests of `spinwaves.utils.graphics`"""
import pytest

from spinwaves.utils.graphics import require_graphics, GRAPHICS_HINT


def test_require_graphics():
    require_graphics('numpy', feature='test')

    with pytest.raises(ImportError, match='not_a_module') as excinfo:
        require_graphics('numpy', 'not_a_module', feature='test')
    assert GRAPHICS_HINT in str(excinfo.value)
