from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..symmetry.group import Group
    
    from ..utils.graphics import require_graphics
require_graphics('networkx', 'matplotlib', feature='plot_network')

import networkx as nx
import matplotlib.pyplot as plt

def plot_network(group: 'Group', layout: str='spring', seed:int=0, node_size: float=100, font_size: int=8):
    """Export Cayley digraph as a NetworkX MultiDiGraph with string node ids."""
    G = nx.DiGraph(name=group._name)

    # nodes: use string ids; keep useful attrs
    for op in group._operations:
        nid = op.to_str()
        G.add_node(nid, index=group._index_of[op], op_str=nid, op=op)

    # edges: label may be SymOp or str; store as 'generator'
    for u, edges in group._adjacency.items():
        u_id = u.to_str()
        for label, v in edges:
            if label in group._generators:
                v_id = v.to_str()
                G.add_edge(u_id, v_id, generator=label.to_str())



    # positions
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size)
    nx.draw_networkx_labels(G, pos, font_size=font_size)

    # assign distinct colors per generator label
    gens = sorted({d["generator"] for _, _, d in G.edges(data=True)})
    import itertools
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_of = {g: c for g, c in zip(gens, itertools.cycle(palette))}

    # draw edges per generator with its color
    for g in gens:
        edgelist = [(u, v) for u, v, d in G.edges(data=True) if d["generator"] == g]
        nx.draw_networkx_edges(G, pos, edgelist=edgelist, edge_color=color_of[g], arrows=True, width=3)

    # edge labels (works for DiGraph)
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels={(u, v): d["generator"] for u, v, d in G.edges(data=True)},
        font_size=font_size, alpha=0.8
    )
    plt.axis("off")
    plt.show()

