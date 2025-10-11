'''Implementation of basic entities from group theory.
Limited to application of groups to symmetry elements
which allows for certain simplifications and forces conventions.
'''

from abc import abstractmethod
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple

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
        (operations, mult_table, adjacency, index_of) = self._build_group_cayley(generators=self._generators)

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
    def order(self) -> int:
        '''Order of the group, that is number of operations in the group.'''
        return len(self._operations)

    ##########################################################################################################
    # Methods
    def __repr__(self) -> str():
        gen_str = ", ".join([g.__repr__() for g in self._generators])
        return f"<{self.__class__.__name__}={self._name}, order={self.order}, gens=[{gen_str}]>"

    @staticmethod
    def _build_group_cayley(generators: tuple[SymOp]) -> tuple:
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
        steps = set(generators + gens_inv) - {identity}


        # Degenerate case: only the identity exists
        if not steps:
            elements = [identity]
            table = {(identity, identity): identity*identity}
            adj = defaultdict(list)
            return (elements, table, adj, {identity: 0})
    

        # --- BFS discovery over the Cayley graph ---
        discovered = {identity}
        operations: List[gType] = [identity]
        index_of: Dict[gType, int] = {identity: 0}
        q = deque([identity])

        # Cayley graph (directed, edge-labeled)
        adjacency: Dict[gType, List[Tuple[str, gType]]] = defaultdict(list)

        # Full multiplication table, filled incrementally
        mult_table: Dict[Tuple[gType, gType], gType] = {}
        mult_table[(identity, identity)] = identity*identity

        # Helper: when a new element y is discovered, fill its row/column vs all known elements
        def fill_table_for_new(y: gType) -> None:
            # row y,* and column *,y vs all existing elements (which exclude y)
            for b in operations:
                mult_table[(y, b)] = y*b
                mult_table[(b, y)] = b*y
            # finally y*y
            mult_table[(y, y)] = y*y

        while q:
            # for g in discovered:
            #     print(g, hash(g))

            # print('BFS step:', q, discovered, operations)
            if len(operations) > MAX_GROUP_ORDER:
                raise ValueError("Group appears larger than max_size (possible infinite subgroup).")

            x = q.popleft()

            # expand x by every generator and inverse (right-multiplication is sufficient)
            for s in steps:
                y = x*s

                # record labeled edge x --(s)--> y in the Cayley graph
                adjacency[x].append((s.to_str(), y))

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
        Build a per-generator adjacency tensor for the Cayley digraph.

        Returns a dict keyed by the *generator elements themselves* (not strings):
            { g -> A_g }  where A_g is an n×n numpy array (int)
        and (A_g)[i, j] counts directed edges operations[i] --(g or g^{-1})--> operations[j].

        Assumptions:
          - `adjacency[u]` stores edges as (label_str, v) where label_str == s.to_str()
            for s in the step set used when building the graph (typically S ∪ S^{-1}).
          - `SymOp` implements `.to_str()` and `.inv()`, and equality/hash are canonical.

        Parameters
        ----------
        operations : list[SymOp]
            All group elements, in the row/column order of the matrices.
        adjacency : Dict[SymOp, List[Tuple[str, SymOp]]]
            Directed, labeled edges of the Cayley graph: u -> [(label, v), ...]
        generators : Iterable[SymOp]
            The base generating set S (identity excluded). The tensor is indexed by these objects.

        Returns
        -------
        Dict[SymOp, np.ndarray]
            One n×n matrix per generator g ∈ S. Edges labeled by g or g^{-1} are folded into A_g.
        """
        n = len(operations)
        idx = {g: i for i, g in enumerate(operations)}

        # Map every edge label seen in `adjacency` back to a *base* generator in S.
        # We cover both g and g^{-1} labels by mapping each to the same base generator g.
        gens = list(generators)
        label_to_base: Dict[str, "SymOp"] = {}

        for g in gens:
            lbl_g = g.to_str()
            lbl_inv = g.inv().to_str()

            # Enforce uniqueness of mapping
            if lbl_g in label_to_base and label_to_base[lbl_g] != g:
                raise ValueError(f"Ambiguous label '{lbl_g}' maps to multiple generators.")
            if lbl_inv in label_to_base and label_to_base[lbl_inv] != g:
                raise ValueError(f"Ambiguous label '{lbl_inv}' maps to multiple generators.")

            label_to_base[lbl_g] = g
            label_to_base[lbl_inv] = g

        # Allocate an n×n matrix per generator
        mats: Dict["SymOp", np.ndarray] = {g: np.zeros((n, n), dtype=int) for g in gens}

        # Populate counts
        for u in operations:
            i = idx[u]
            for lbl, v in adjacency.get(u, []):
                j = idx.get(v)
                if j is None:
                    # Edge points outside the provided vertex set; skip.
                    continue
                base = label_to_base.get(lbl)
                if base is None:
                    # The graph contains an edge labeled by a step that isn't tied to any base generator.
                    # This usually means `generators` is missing something (or labels don't match to_str()).
                    raise KeyError(
                        f"Adjacency label '{lbl}' not recognized from provided generators."
                    )
                mats[base][i, j] += 1

        return mats

