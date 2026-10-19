r"""Tests of `spinwaves.plotting.network_plotter`"""
import pytest

from spinwaves.symmetry.crystall_space_group import SG

from ..markers import needs_networkx


@needs_networkx
@pytest.mark.skip(reason="interactive")
def test_plot_network():
    sg = SG.from_xyz_strings([
            # 'x,y+1/2,z+1/2',
            # 'x+1/2,y,z+1/2',
            'x-y,x,z',
            '-x,-y,-z',
            # 'x,y,-z',
            # 'y,x,z',
            # 'y,z,x',
    ])
    print(sg)
    from spinwaves.plotting.network_plotter import plot_network

    plot_network(sg, layout='kamada_kawai', seed=10, node_size=200, font_size=8)
