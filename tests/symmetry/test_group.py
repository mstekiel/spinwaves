import pytest

# CHANGE THIS IMPORT to your module path
from spinwaves.symmetry.crystall_space_group import cSymOp, SG
from spinwaves.symmetry.group import Group

import numpy as np

from . import make_perm_symop, s3_generators, c4_generator

SymOpPerm2 = make_perm_symop(2)
SymOpPerm3 = make_perm_symop(3)
SymOpPerm4 = make_perm_symop(4)

# --------------------------
# Tests
# --------------------------

# Group axiom tests
def test_closure_s3():
    s, t = SymOpPerm3((1,0,2)), SymOpPerm3((0,2,1))
    G = Group([s, t])
    ops = set(G._operations)
    for a in G._operations:
        for b in G._operations:
            assert a * b in ops

def test_identity_in_operations_s3():
    s, t = SymOpPerm3((1,0,2)), SymOpPerm3((0,2,1))
    G = Group([s, t])
    assert SymOpPerm3.identity() in G._operations

def test_inverses_in_operations_s3():
    s, t = SymOpPerm3((1,0,2)), SymOpPerm3((0,2,1))
    G = Group([s, t])
    ops = set(G._operations)
    for g in G._operations:
        assert g.inv() in ops

def test_trivial_group():
    e = SymOpPerm3.identity()
    G = Group([e])
    assert G.order == 1
    assert G._operations == [e]

# Data structure consistency

def test_index_of_consistent_s3():
    s, t = SymOpPerm3((1,0,2)), SymOpPerm3((0,2,1))
    G = Group([s, t])
    for g in G._operations:
        assert G._operations[G.index_of(g)] == g

def test_mult_table_complete_s3():
    s, t = SymOpPerm3((1,0,2)), SymOpPerm3((0,2,1))
    G = Group([s, t])
    assert len(G._mult_table) == G.order ** 2

@pytest.mark.parametrize("gens, expected_order", [
    ([SymOpPerm3((1,0,2))],                            2),   # C2
    ([SymOpPerm3((1,0,2)), SymOpPerm3((0,2,1))],       6),   # S3
    ([SymOpPerm4((1,2,3,0))],                          4),   # C4
])
def test_known_group_orders(gens, expected_order):
    assert Group(gens).order == expected_order


def _cyclic_generator(n: int):
    """Single rotation generator for C_N on n points."""
    Perm = make_perm_symop(n)
    return Perm(tuple(range(1, n)) + (0,))


@pytest.mark.parametrize("n", [5, 7, 12, 20])
def test_cyclic_group_order(n):
    assert Group([_cyclic_generator(n)]).order == n


@pytest.mark.parametrize("n, expected_order", [
    (4, 24),    # S4
    (5, 120),   # S5
])
def test_symmetric_group_order(n, expected_order):
    Perm = make_perm_symop(n)
    swap  = Perm((1, 0) + tuple(range(2, n)))       # transposition (01)
    cycle = Perm(tuple(range(1, n)) + (0,))          # full cycle (0 1 ... n-1)
    assert Group([swap, cycle]).order == expected_order


def test_group_axioms_s4():
    """Closure, identity, inverses for S4 (order 24)."""
    Perm = make_perm_symop(4)
    swap  = Perm((1, 0, 2, 3))
    cycle = Perm((1, 2, 3, 0))
    G = Group([swap, cycle])
    ops = set(G._operations)

    assert Perm.identity() in ops
    for g in G._operations:
        assert g.inv() in ops
    for a in G._operations:
        for b in G._operations:
            assert a * b in ops



def test_group_repr_and_order_s3():
    e, s, t = s3_generators()
    # Give a list with duplicates and an explicit identity; Group should dedupe and drop identity
    G = Group([s, t, s, e], name="S3")
    assert "S3" in repr(G)
    assert G.order == 6  # S3 has 6 elements

def test_generators_set_identity_removed():
    e, s, t = s3_generators()
    G = Group([s, t, e])
    # Identity should not be in generators; uniqueness enforced by set()
    assert all(g != e for g in G._generators)
    assert len(G._generators) in (1, 2)  # depending on hash order, but identity is gone

def test_mult_table_matches_operation_s3():
    _, s, t = s3_generators()
    G = Group([s, t])

    # multiplication table consistency: for all a,b in operations, table[(a,b)] == a*b
    for a in G._operations:
        for b in G._operations:
            assert G._mult_table[(a, b)] == (a * b)

def test_adjacency_has_right_labels_and_vertices_s3():
    _, s, t = s3_generators()
    G = Group([s, t])

    # Each vertex should have out-degree equal to number of steps used when building:
    # steps = set(gens u gens^{-1}) \ {identity}. Here s and t are involutions, so steps = {s,t}.
    steps = {s, t}
    for u, edges in G._adjacency.items():
        # all labels among steps, all endpoints in operations
        assert all(lbl in steps for lbl, v in edges)
        assert all(v in G._operations for _, v in edges)
        # exactly two edges per node (s and t)
        assert len(edges) == 2

def test_adjacency_tensor_keys_are_generators_and_shapes_s3():
    _, s, t = s3_generators()
    G = Group([s, t])

    mats = G._adjacency_tensor
    # Keys are the generator objects
    assert set(mats.keys()) == set(G._generators)
    # All slices have shape (n, n)
    n = G.order
    assert all(M.shape == (n, n) for M in mats.values())

    # Since s, t are involutions, each slice is a permutation matrix: one "1" per row.
    for gen, M in mats.items():
        row_sums = M.sum(axis=1)
        assert np.all(row_sums == 1), f"Row sums for {gen} should be 1 in S3"

def test_adjacency_matrix_equals_sum_of_slices_s3():
    _, s, t = s3_generators()
    G = Group([s, t])

    # By construction
    expected = np.add.reduce([M for M in G._adjacency_tensor.values()]).astype(int)
    assert np.array_equal(G._adjacency_matrix, expected)

    # In S3 with two involutory generators, out-degree per node is 2 in the uncolored digraph
    assert np.all(G._adjacency_matrix.sum(axis=1) == 2)

def test_c4_rotation_inverse_folding_in_tensor():
    e, r = c4_generator()
    # Only one generator: r (not an involution), but steps include r and r^{-1}
    G = Group([r], name="C4")

    mats = G._adjacency_tensor
    assert set(mats.keys()) == set(G._generators) == {r}

    M = mats[r]
    n = G.order
    # For a nice simple case, we have circular Cayley graph.
    assert np.all(M.sum(axis=1) == 1)
    assert np.all(M.sum(axis=0) == 1)

    # The uncolored adjacency is the same here (only one slice), still row-sum 1
    A = G._adjacency_matrix
    assert np.array_equal(A, M)
    assert np.all(A.sum(axis=1) == 1)


@pytest.mark.skip(reason="manual")
def hand_test():
    e = cSymOp.identity()
    print(e)
    g1 = cSymOp.from_string('x,y+1/2,z+1/2')
    g2 = cSymOp.from_string('x+1/2,y,z+1/2')
    g3 = cSymOp.from_string('-y,x,z')
    g4 = cSymOp.from_string('-x,-y,-z')
    g5 = cSymOp.from_string('y,x,z')
    g6 = cSymOp.from_string('y,z,x')
    SG = Group([
            cSymOp.from_string('x,y+1/2,z+1/2'),
            # cSymOp.from_string('x+1/2,y,z+1/2'),
            cSymOp.from_string('-y,x,z'),
            cSymOp.from_string('-x,-y,-z'),
            cSymOp.from_string('y,x,z'),
            cSymOp.from_string('y,z,x'),
    ])
    print(SG)
    # print(SG._adjacency_matrix)
    # print(SG._adjacency_tensor)
    print(SG.operations)

    print(g2, SG.index_of(g2))


# --------------------------
# symmetrize tests
# --------------------------
class _Pt:
    """Minimal object: equality/hash on key, checkable value attribute."""
    def __init__(self, key, value=0.0):
        self.key   = key
        self.value = value
    def __eq__(self, other):
        return isinstance(other, _Pt) and self.key == other.key
    def __hash__(self):
        return hash(self.key)
    def __repr__(self):
        return f"Pt({self.key}, {self.value})"


def test_symmetrize_general_position():
    """C4 acting on index 0 produces orbit of size 4."""
    G = Group([SymOpPerm4((1, 2, 3, 0))])
    result = G.symmetrize(0, lambda g, x: g.perm[x])
    assert set(result) == {0, 1, 2, 3}


def test_symmetrize_fixed_point():
    """Object invariant under all S3 operations → single unique result."""
    s, t = SymOpPerm3((1, 0, 2)), SymOpPerm3((0, 2, 1))
    G = Group([s, t])
    # frozenset({0,1,2}) maps to itself under any permutation of 3 elements
    transform = lambda g, obj: frozenset(g.perm[i] for i in obj)
    result = G.symmetrize(frozenset({0, 1, 2}), transform)
    assert len(result) == 1


def test_symmetrize_partial_orbit():
    """S3 acting on 0: orbit = {0,1,2}, stabilizer has order 2 so len < group order."""
    s, t = SymOpPerm3((1, 0, 2)), SymOpPerm3((0, 2, 1))
    G = Group([s, t])
    result = G.symmetrize(0, lambda g, x: g.perm[x])
    assert set(result) == {0, 1, 2}
    assert len(result) == 3          # orbit size, not group order (6)


def test_symmetrize_check_attrs_no_warning(caplog):
    """Equivalent objects with matching value attribute produce no warning."""
    import logging
    G = Group([SymOpPerm3((1, 0, 2)), SymOpPerm3((0, 2, 1))])
    # each index gets the same value → consistent
    transform = lambda g, x: _Pt(g.perm[x], value=1.0)
    with caplog.at_level(logging.WARNING, logger='Symmetry'):
        G.symmetrize(0, transform, check_attrs=['value'])
    assert not caplog.records


def test_symmetrize_check_attrs_warns(caplog):
    """Equivalent objects with conflicting value attribute trigger a warning."""
    import logging
    G = Group([SymOpPerm2((1, 0))])   # C2: identity and swap
    # both ops map to Pt('x') but with different values → conflict
    values = {G.operations[0]: 1.0, G.operations[1]: 2.0}
    transform = lambda g, _: _Pt('x', value=values[g])
    with caplog.at_level(logging.WARNING, logger='Symmetry'):
        G.symmetrize(None, transform, check_attrs=['value'])
    assert caplog.records


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
