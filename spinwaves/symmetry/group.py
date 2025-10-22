'''Implementation of basic entities from group theory.
Limited to application of groups to symmetry elements
which allows for certain simplifications and forces conventions.

For example, I call the group elements operations, more natural in crystallography.
Also, each element has its inverse.
'''

from abc import abstractmethod
import logging
import numpy as np
from collections import deque, defaultdict
from typing import Any, Callable, Dict, List, Tuple, TypeVar

T = TypeVar('T')
logger = logging.getLogger('Symmetry')


MAX_GROUP_ORDER = 1024

class SymOp(object):
    '''Abstract class of symmetry operations as members of the crystallographic (SG) and magnetic space groups (MSG).
    Enforces to contain the implementation of functions required to form a SG or MSG:
    - multiplication: `__mul__`
    - hashability: `__hash__`
    - equality: `__eq__`
    - string casting for hash: `to_str()`
    - identity element: `identity`
    - inverse element: `inv`

    Notes
    -----
    Hashing with string might be a trap. Need to ensure it is a unique representation.
    '''

    @abstractmethod
    def __mul__(self, other: 'SymOp') -> 'SymOp':
        pass

    @abstractmethod
    def to_str(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def identity(self) -> 'SymOp':
        pass

    @abstractmethod
    def inverse(self) -> 'SymOp':
        pass

class Group(object):
    '''Represent group, holder of group elemets `g` with functionalities.
    
    Attributes
    ----------
    operations: list[g]
        List of elements that make the group
    index_of: Dict[g, int])
        Gives the internal index (integer) which corresponds to the operation.
    mult_table: Dict[Tuple[g, g], g]
        Dictionary that holds the multiplication table of the group.
    adjacency: Dict[g, List[Tuple[str, g]]]
        Adjacency graph of the group. `adjacency[g1]=(str(g2), g3)`, where g2*g1=g3. Uses string representation of the `g2` operation.
    adjacency_matrix: 
    
    TODO
    ----
    [ ] The constructor assumes the generators are unique, i.e. are not repeated
    '''

    def __init__(self, generators: list[SymOp], name='Group'):

        # Name
        self._name = name

        # Ensure only unique elements and no identity
        # form the generators
        generators = set(generators) - {generators[0].identity()}
        self._generators = tuple(generators)

        # Generate few attributes of the group that can be done in single loop
        # and thus enhances the speed and clarity.
        (operations, mult_table, adjacency, index_of) = self.build_group_cayley(generators=self._generators)

        self._operations = operations
        self._mult_table = mult_table
        self._adjacency  = adjacency
        self._index_of   = index_of

        # dict[label] -> n*n numpy arrays
        mats = self.adjacency_tensor(self._generators, self._operations, self._adjacency)
        self._adjacency_tensor = mats

        # Uncolored directed adjacency (counts parallel edges)
        self._adjacency_matrix = np.add.reduce([mat for mat in mats.values()]).astype(int)

    ##########################################################################################################
    # Properties

    @property
    def operations(self) -> list['SymOp']:
        '''List of all symmetry operations of the group.'''
        return self._operations

    @property
    def order(self) -> int:
        '''Order of the group, that is number of operations in the group.'''
        return len(self._operations)

    ##########################################################################################################
    # Methods
    def index_of(self, g: 'SymOp') -> int:
        '''Index of the group element, as stored in `self.operations`.'''
        return self._index_of[g]

    def __repr__(self) -> str:
        gen_str = ", ".join([g.__repr__() for g in self._generators])
        return f"<{self.__class__.__name__}={self._name}, order={self.order}, gens=[{gen_str}]>"
    
    def symmetrize(self, object: T, transform_func: Callable[['SymOp', T], T], check_attrs: list[str]=[]) -> list[T]:
        '''Symmetrize the `object` of arbitrary type according to the `transform_func`
        within the symmetry of the `Group`.
        In other words, will apply each symmetry operation of the `Group` to the `object`,
        where the symmetry transformation rules are defined within the `transform_func`,
        and return only unique elements from the symmetrized elements.

        >>> class A: r=[0,0,0]
        >>> gen_fun = lambda g, a: A(g.matrix @ a.r)
        >>> MSG.symmetrize(A([0, 0, 0.5]), gen_fun) 
        
        Parameters
        ----------
        obj: T
            Object that will be symmetrized
        transform_obj: Callable[[T], T]
            Recipe how the object transforms under symmetry operations.
        check_attrs: list[str], optional
            If provided, it will check if the attributes are respecting the symmetry conditions of the MSG.

        Returns
        -------
        list[T]
            List of objects created by applyin symmetry operations of the `MSG`.
        '''
        objs_symmetrized = []
        for g in self.operations:
            objs_symmetrized.append(transform_func(g, object))

        objs_unique, id_inverse = np.unique(objs_symmetrized, return_inverse=True)


        # Check symmetry condidtion
        # [ ] Which symmetry elements to not produce compatible attributes?
        for id_unique, obj_unique in enumerate(objs_unique):
            id_equivalent = np.where(id_inverse==id_unique)[0]

            for check_field in check_attrs:
                field_eq = [objs_symmetrized[id].__getattribute__(check_field) for id in id_equivalent]
                field_averaged = np.average(field_eq, axis=0)

                field_unique = obj_unique.__getattribute__(check_field)
                if not np.allclose(field_unique, field_averaged):
                    warning_message = f'The following object property does not respect the symmetry\n\t{obj_unique}\n'
                    warning_message+= f'\tSet value        : {check_field}={field_unique}\n'
                    warning_message+= f'\tSymmetrized value: {check_field}={field_averaged}\n'

                    for id in id_equivalent:
                        warning_message += f'\t{self.operations[id]}\t-> {check_field}={objs_symmetrized[id].__getattribute__(check_field)}\n'

                    warning_message+=  'You better know what you are doing.'

                    logger.warning(warning_message)


        return objs_unique

    @staticmethod
    def build_group_cayley(generators: tuple[SymOp]) -> tuple:
        """
        Generate the group from generators using a Cayley-graph Breadth-First Search algorithm BFS,
        build the full multiplication table, and construct the Cayley graph.

        Parameters
        ----------
        generators: list[SymOp]
            List of group generators (identity NOT included)

        Returns
        -------
        tuple(operations, mult_table, adjacency, index_of)
        - operations : list of all operations (discovery order)
        - mult_table : dict mapping (a, b) -> a*b for all pairs a,b in operations
        - adjacency  : dict u -> list of (edge_label, v) for each generator/inverse
        - index_of   : map operation -> index to address rows/cols of the table
        """
        gens_inv = tuple([g.inv() for g in generators])
        gType = type(generators[0])
        identity = gType.identity()
        
        # use Spm as the generating set for BFS
        steps_bfs = set(generators + gens_inv) - {identity}
        steps_adj = set(generators) - {identity}


        # Degenerate case: only the identity exists
        if not steps_bfs:
            elements = [identity]
            table = {(identity, identity): identity*identity}
            adj = defaultdict(list)
            return (elements, table, adj, {identity: 0})
    

        # --- BFS discovery over the Cayley graph ---
        discovered = {identity}
        operations = [identity]
        index_of = {identity: 0}
        q = deque([identity])

        # Cayley graph (directed, edge-labeled)
        adjacency = defaultdict(list)

        # Full multiplication table, filled incrementally
        mult_table = {}
        mult_table[(identity, identity)] = identity*identity

        # Helper: when a new element y is discovered, fill its row/column vs all known elements
        def fill_table_for_new(y: Any) -> None:
            # row y,* and column *,y vs all existing elements (which exclude y)
            for b in operations:
                mult_table[(y, b)] = y*b
                mult_table[(b, y)] = b*y
            # finally y*y
            mult_table[(y, y)] = y*y

        while q:
            if len(operations) > MAX_GROUP_ORDER:
                raise ValueError("Group appears larger than max_size (possible infinite subgroup).")

            x = q.popleft()
            for s in steps_bfs:
                # should it be LH or RH multiplicatoin?
                y = x*s

                # record labeled edge x --(s)--> y in the Cayley graph
                if s in steps_adj:
                    adjacency[x].append((s, y))

                # if new vertex, register and extend BFS frontier
                if y not in discovered:
                    discovered.add(y)
                    fill_table_for_new(y)
                    index_of[y] = len(operations)
                    operations.append(y)
                    q.append(y)


        return (operations, mult_table, adjacency, index_of)
    
    @staticmethod
    def adjacency_tensor(
        generators: List["SymOp"],
        operations: List["SymOp"],
        adjacency: Dict["SymOp", List[Tuple[str, "SymOp"]]],
    ) -> Dict["SymOp", np.ndarray]:
        """
        Build a per-generator adjacency tensor for the Cayley diagram of the group.

        Returns a dict keyed by the *generator elements themselves* (not strings):
            { g -> A_g }  where A_g is an n*n numpy array (int)
        and (A_g)[i, j] counts directed edges operations[i] --(g)--> operations[j].

        Parameters
        ----------
        generators : Iterable[SymOp]
            The base generating set S (identity excluded). The tensor is indexed by these objects.
        operations : list[SymOp]
            All group elements, their order will define the row/column order of the matrices.
        adjacency : Dict[SymOp, List[Tuple[str, SymOp]]]
            Directed, labeled edges of the Cayley graph: u -> [(label, v), ...]

        Returns
        -------
        Dict[SymOp, np.ndarray]
            One n*n matrix per generator g in S. Edges labeled by g or g^{-1} are folded into A_g.
        """
        n = len(operations)
        idx = {g: i for i, g in enumerate(operations)}

        mats: Dict["SymOp", np.ndarray] = {g: np.zeros((n, n), dtype=int) 
                                           for g in generators}

        for g1, connections in adjacency.items():
            for gc,g2 in connections:
                if gc in mats:
                    mats[gc][idx[g1],idx[g2]] += 1

        return mats



def plot_network(group, layout: str='spring', seed:int=0, node_size: float=100, font_size: int=8):
    """Export Cayley digraph as a NetworkX MultiDiGraph with string node ids."""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Either `matplotlib` or `networkx` are not installed.") from e


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

