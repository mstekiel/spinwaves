# spinwaves — Developer Guide

## What this library does

`spinwaves` is a Python implementation of **linear spin wave theory (LSW)** for magnetic crystals.
It is inspired by the MATLAB SpinW package. Given a crystal structure with magnetic atoms and exchange
couplings, it computes spin wave excitation energies E(Q) and spectral weights S(Q,E) across
reciprocal space.

## Module map

```
spinwaves/
├── atom.py                  # Atom: position, moment, spin, g-tensor
├── lattice.py               # Lattice, LatticeOriented: coordinate transforms
├── crystal.py               # Crystal(Lattice): atoms + MSG, generates full unit cell
├── coupling.py              # Coupling: pairwise exchange interaction J (3×3 matrix)
├── spinw.py                 # SpinW: LSW Hamiltonian, eigensolver, excitations
├── symmetry/
│   ├── group.py             # SymOp (ABC), Group[T] (generic, Cayley BFS)
│   ├── magnetic_symmetry.py # mSymOp(SymOp), MSG(Group[mSymOp])
│   └── crystall_space_group.py  # cSymOp(SymOp), SG(Group[cSymOp])
├── databases/               # Atomic data: radii, colors, masses, form factors
├── plotting/                # Visualization backends (vispy, matplotlib, qtgraph)
├── utils/
│   ├── linalg.py            # Rotation matrices, DMI, dipolar, vector ops
│   ├── arrays.py            # Structured dtypes, shape-validation decorators
│   └── physics.py           # Bose-Einstein occupation
└── data_tables/             # Raw database text files
```

## Core class hierarchy

```
SymOp (ABC)  ←── group.py
├── cSymOp   ←── crystall_space_group.py  (matrix + translation)
└── mSymOp   ←── magnetic_symmetry.py     (matrix + translation + time_reversal)

Group[T: SymOp]  ←── group.py  (generic, Cayley-BFS)
├── SG   = Group[cSymOp]
└── MSG  = Group[mSymOp]

Lattice  ←── lattice.py
└── Crystal(Lattice)  ←── crystal.py  (atoms + MSG)

SpinW(SWPlottingMixin)  ←── spinw.py
    ├── .crystal: Crystal
    ├── .couplings: list[Coupling]       # user-defined unique bonds
    └── .couplings_all: list[Coupling]   # symmetry-expanded
```

## Key conventions

| Convention | Detail |
|---|---|
| Atomic positions | `r` stored as `Fraction` array in **uvw** (crystal) coordinates |
| Magnetic moments | `m` in **xyz** (Cartesian) coordinates |
| Reciprocal lattice | `B` matrix includes `2π` factor; `A.T @ B = 2πI` |
| Atom ordering | **Magnetic atoms first** in all lists — critical for matrix indexing |
| Translation normalization | Always `% 1`, kept in `[0, 1)` as `Fraction` |
| Time reversal | `+1` = no reversal, `-1` = reversal; affects axial vector transforms |
| Hashing of SymOps | Based on raw components: `(matrix.tobytes(), tuple(translation), time_reversal)` |

## Group theory layer (`symmetry/group.py`)

`Group[T]` is generic and works with any `SymOp` subclass.
Construction uses **Cayley-graph BFS**: generates all elements, full multiplication table,
and labeled adjacency graph in one pass. Key data structures built at init:

- `_operations`: ordered list of all group elements
- `_mult_table`: `dict[(a,b)] → a*b`
- `_adjacency`: Cayley graph edges
- `_adjacency_tensor`: per-generator `n×n` adjacency matrices
- `_index_of`: `dict[g] → int`

`Group.symmetrize(obj, transform_func, check_attrs=[])` applies all operations to `obj`,
deduplicates by hash/equality, and optionally warns if attributes of equivalent objects disagree.

## mSymOp design notes

- `identity()` is a `@classmethod` using `cls.from_string(...)` — subclass-safe
- `__mul__` and `inv()` use `self.__class__(...)` — subclass-safe
- `__hash__` and `__eq__` are based on the same raw components (contract guaranteed)
- `translation` property returns `float` array; internal operations use `self._translation` (Fraction)
- `from_string()` parses 4-field xyz strings: `"x+1/2, y, -z, -1"` (last field = time reversal)

## LSW calculation flow

```
1. Atom(r, m, s)  +  MSG.from_xyz_strings([...])
        ↓
2. Crystal(lattice_params, atoms, MSG)
   → generates all unit-cell atoms via MSG.symmetrize()
        ↓
3. Coupling(id1, id2, n_uvw, J)  → SpinW.symmetrize_couplings()
        ↓
4. SpinW._construct_h(Qhkl)  → Hamiltonian matrix at each Q
        ↓
5. SpinW._determine_ESp()    → Cholesky + Schur decomposition
        ↓
6. Excitations: E(Q), S_perp(Q,E)
```

## Test structure

```
tests/
├── test_crystal.py           # Crystal construction, atom indexing, DMI validation
├── test_spinw.py             # SpinW calculations, coupling symmetrization
├── test_functions.py
├── test_utils_arrays.py
├── symmetry/
│   ├── test_group.py         # Group axioms, Cayley graph, symmetrize
│   └── test_magnetic_symmetry.py
└── SpinW_comparisons/        # Regression against original MATLAB SpinW
```

Tests use a `make_perm_symop(n)` factory (in `test_group.py`) that generates a `SymOp` class
for permutations on `n` points — used as lightweight stubs for group tests.

## Active refactoring notes

- `Group` was made generic (`Group[T]`) — ongoing work to consolidate `MSG` as `Group[mSymOp]`
- `mSymOp` now correctly inherits from `SymOp` ABC
- `SymOp` ABC still has a naming issue: abstract method declared as `to_string()` but `mSymOp` implements `to_str()` — needs reconciliation
- `MSG` in `magnetic_symmetry.py` still has its own group-generation algorithm (older, slower pairwise closure) and duplicates some `Group` logic — planned to be replaced by `Group[mSymOp]`

## Python version

Requires Python `>=3.10, <3.11`. Package managed with `uv`.
