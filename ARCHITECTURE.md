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
├── config.py                # Global tolerances, override_tolerances(), NumericalToleranceError
├── symmetry/
│   ├── group.py             # SymOp (ABC), Group[T] (generic, Cayley BFS)
│   ├── magnetic_symmetry.py # mSymOp(SymOp), MSG(Group[mSymOp])
│   └── crystall_space_group.py  # cSymOp(SymOp), SG(Group[cSymOp])
├── databases/               # Atomic data: radii, colors, masses, form factors
├── plotting/                # plot_structure, plot_dispersion; backends (vispy, matplotlib, qtgraph)
├── legacy/                  # Deprecated implementations kept for reference, not imported by `spinwaves`
│   └── spinw_old.py         # Old SpinW methods as functions of `sws`; scipy eigensolver (needs `dev` group)
├── utils/
│   ├── linalg.py            # Tolerance-checked hermitian_eig/regularized_cholesky, rotations, DMI, vector ops
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

SpinW  ←── spinw.py
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

## Numerical tolerances (`config.py`)

One rule for deviations that should be numerical noise: compare **relative** to the scale of the
checked object against `config.tolerances.numeric` (default `1e-10`); below it correct silently
(logged at DEBUG on logger `Numerics`), above it raise `NumericalToleranceError` (a `ValueError`
carrying `deviation`, `tolerance` and failing batch `indices`). Read the tolerance at call time,
never as a default argument, so `config.tolerances.numeric = ...` and `override_tolerances(...)` apply.

| Check | Helper (`utils/linalg.py`) | Below tolerance | Above tolerance |
|---|---|---|---|
| Hermiticity, `‖A−A†‖/‖A‖` | `ensure_hermitian` | return `(A+A†)/2` | raise (construction bug) |
| Definiteness of `h`, `λ_min/λ_max` | `regularized_cholesky` | shift zero modes by `tol·λ_max` | raise (not the ground state) |
| Eigen-decomposition of `K†gK` | `hermitian_eig` | `ensure_hermitian` + `eigh`, ascending order | as `ensure_hermitian` |

The zero-mode shift opens a gap at Goldstone modes of `tol·scale` (quadratic dispersion) up to
`√tol·scale` (linear dispersion). `config.tolerances.geometric` (`1e-5`) is reserved for comparing
user-given positions and directions; existing `np.allclose` calls do not use it yet.

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

**Rule: `tests/` mirrors `spinwaves/`.** Every module `spinwaves/<pkg>/<module>.py` has its tests in
`tests/<pkg>/test_<module>.py`; a package `__init__.py` with its own logic is tested in
`tests/<pkg>/test_init.py`. Modules that only re-export names (`databases/__init__.py`,
`symmetry/__init__.py`, `legacy/__init__.py`) and `plotting/archive/` have no test file.
When adding, moving or renaming a module, do the same with its test file. Modules without tests
yet have a stub file: a docstring listing the objects to cover, no test functions.

```
tests/
├── __init__.py, conftest.py   # Shared fixtures (e.g. `fm_chain`)
├── markers.py                 # Skip markers for optional dependencies: needs_matplotlib, needs_vispy, needs_scipy, ...
├── helpers.py                 # Reference LSWT systems, `assert_no_numpy_scalar_repr`
├── test_init.py               # spinwaves/__init__.py: core import loads no graphics libraries
├── test_atom.py, test_config.py, test_coupling.py, test_crystal.py, test_lattice.py, test_spinw.py
├── databases/                 # test_database.py, test_implement_txt.py, test_implement_spglib.py
├── gui/                       # test_init.py, test_gui_vispy.py, test_gui_qtgraph.py, test_gui_plotly.py
├── legacy/                    # test_spinw_old.py
├── plotting/                  # test_init.py (plot_structure, plot_dispersion), test_network_plotter.py,
│                              # test_supercell_plotter[_mpl|_qtgraph|_vispy|_vispy_advanced].py
├── symmetry/
│   ├── __init__.py            # `make_perm_symop(n)` factory and generator fixtures
│   ├── test_group.py          # Group axioms, Cayley graph, symmetrize
│   ├── test_magnetic_symmetry.py
│   └── test_crystall_space_group.py
├── utils/                     # test_init.py, test_arrays.py, test_functions.py, test_graphics.py,
│                              # test_linalg.py, test_physics.py, test_symmetry.py
├── data/                      # Input data for tests, not tests themselves
│                              # lswt_reference.npz + make_lswt_reference.py (numpy-version-independent E, S)
└── comparisons/               # Not collected by pytest (no `test_` prefix)
    ├── SpinW/                 # Regression scripts against original MATLAB SpinW tutorials
    └── various/               # Comparisons with published results
```

Every test directory is a package (has `__init__.py`), so equal basenames such as `test_init.py`
do not clash. Tests needing the optional `graphics` dependencies use the markers from
`tests/markers.py` and are skipped when those libraries are not installed.

**numpy 1/2 compatibility.** Test files of `atom`, `coupling`, `crystal`, `spinw`, `magnetic_symmetry` and
`crystall_space_group` have a section `numpy 1/2 compatibility` guarding behaviour numpy 2 changed: default
integer on Windows (int32 to int64), scalar reprs (`np.int64(1)`), NEP 50 promotion (`float32` kept), sorting of
object arrays in `np.unique`, and the BLAS/LAPACK builds (`test_lswt_reference`). Known breaks are strict xfails,
conditioned on the numpy/Python version where the behaviour differs.

`make_perm_symop(n)` (in `tests/symmetry/__init__.py`) generates a `SymOp` class
for permutations on `n` points — used as lightweight stubs for group tests.

## Active refactoring notes

- `Group` was made generic (`Group[T]`) — ongoing work to consolidate `MSG` as `Group[mSymOp]`
- `mSymOp` now correctly inherits from `SymOp` ABC
- `SymOp` ABC still has a naming issue: abstract method declared as `to_string()` but `mSymOp` implements `to_str()` — needs reconciliation
- `MSG` in `magnetic_symmetry.py` still has its own group-generation algorithm (older, slower pairwise closure) and duplicates some `Group` logic — planned to be replaced by `Group[mSymOp]`

## Supported versions

Package managed with `uv`. Core dependency is `numpy` only.

| | Supported | Tested |
|---|---|---|
| Python | `>=3.9` | 3.9 - 3.14 |
| numpy | `>=1.23.3` | 1.23.3 - 2.5, every minor with wheels for the Python version |

numpy 1.23.0-1.23.2 cannot evaluate the `np.ndarray[float]` annotations used across the code base.
The `graphics` extra is still pinned (matplotlib 3.5.2, vispy 0.12.1, built against numpy 1) and
carries `numpy<2`. Since `uv.lock` holds a single numpy for all groups and extras, the lock is
restricted to Python `<3.13` (`[tool.uv] environments`) until the graphics pins are relaxed.
