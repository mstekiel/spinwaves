# --------------------------
# symmetrize tests
# --------------------------
import pytest

from spinwaves.symmetry import Group

from .import make_perm_symop

SymOpPerm2 = make_perm_symop(2)
SymOpPerm3 = make_perm_symop(3)
SymOpPerm4 = make_perm_symop(4)

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