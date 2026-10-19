# Developer guide

Workflow, dependency checks, versioning rules and the release procedure of `spinwaves`.
The code structure and conventions are in [ARCHITECTURE.md](ARCHITECTURE.md).

All commands run from the repository root, in Git Bash on Windows. The project is managed with
[uv](https://docs.astral.sh/uv/) only: no `pip`, and `python` is not on the PATH, so use `uv run`.

## Environment

| Command | Purpose |
|---|---|
| `uv sync` | Install the `dev` group (pytest, scipy, lmfit, pandas, sphinx) into `.venv` |
| `uv sync --group graphics` | Also install the plotting and GUI dependencies |
| `uv run pytest` | Run the test suite in the locked environment |
| `uv run --python 3.12 pytest` | Same, on another Python covered by the lock (`<3.13`) |
| `uv lock` | Update `uv.lock` after editing dependencies in `pyproject.toml` |
| `uv lock --upgrade-package numpy` | Move one locked package to its newest allowed version |

With `graphics` installed, `test_coupling_symmetrization_tetr` opens a vispy window and blocks
until it is closed; add `--deselect tests/test_spinw.py::test_coupling_symmetrization_tetr`.

## Dependency groups

| Where | Contents | Rule |
|---|---|---|
| `[project] dependencies` | `numpy` | The core is computational: nothing else is required |
| `[project.optional-dependencies] graphics` | matplotlib, vispy, PyQt6, pyqtgraph, PyOpenGL, networkx | Imported lazily inside functions; missing packages raise via `spinwaves.utils.graphics.require_graphics` |
| `[dependency-groups] graphics` | `spinwaves[graphics]` | Mirrors the extra for `uv sync --group graphics` |
| `[dependency-groups] dev` | pytest, scipy, lmfit, pandas, sphinx | Development only; scipy is used only by `spinwaves.legacy` and comparison tests |

Pins: floors (`>=`) on tested versions, no upper caps. The exception is `numpy<2` in the `graphics`
extra, because matplotlib 3.5.2 and vispy 0.12.1 are built against numpy 1. As `uv.lock` holds a single
numpy for all groups, it forces numpy 1.26 on the lock, and `[tool.uv] environments` restricts the lock
to Python `<3.13`. Remove both once the graphics pins are relaxed.

## Compatibility checks

Two kinds of checks, both described in the [Compatibility](README.md#compatibility) section of the README.

**Locked environment**, what `uv.lock` pins:

```bash
uv run pytest
```

**Any Python × numpy**: an isolated environment, independent of `uv.lock`, with wheels only.
Graphics tests are skipped as the `graphics` extra is not installed:

```bash
uv run --no-project --isolated --python 3.13 --no-build-package numpy --with "numpy==2.2.6" --with pytest --with scipy --with-editable . pytest -q
```

**Full matrix**: every numpy minor from the floor in `pyproject.toml` (latest patch, plus the floor
itself) on every Python with a numpy wheel. About 5 s per cell, 4 min in total:

```bash
uv run --no-project --python 3.12 python scripts/compat_matrix.py --list
uv run --no-project --python 3.12 python scripts/compat_matrix.py --python 3.12 3.13 --numpy 2.0 2.5
uv run --no-project --python 3.12 python scripts/compat_matrix.py --update-readme
```

`--update-readme` rewrites the table between the `compat-matrix` markers in `README.md`. The script
needs Python `>=3.11` (`tomllib`), hence `--python 3.12`; the cells use their own Pythons. When a new
Python is released, add it to `PYTHONS` in `scripts/compat_matrix.py` and to `.github/workflows/tests.yml`.

**CI**: `.github/workflows/tests.yml` runs on every push, pull request and weekly: the locked
environment on Python 3.9 and 3.12, and the oldest and newest numpy for each Python on Ubuntu,
plus two Windows cells (Windows numpy 1 defaults to int32, numpy 2 to int64).

### Relaxing or raising a dependency floor

1. Run the full matrix, or the cells around the new floor, with the exact versions.
2. Test the floor itself, not only the latest patch of its minor: numpy 1.23.0-1.23.2 fail where 1.23.3 passes.
3. Use wheels only (`--no-build-package`); a missing wheel means unsupported, not a build to fix.
4. Run cells one after another: parallel runs race in the uv cache on Windows ("Failed to move data files").
5. Keep environment paths short on Windows: deep paths break loading numpy DLLs.
6. Update the floor in `pyproject.toml`, `uv lock`, then `uv lock --upgrade-package <name>` for packages
   whose old locked version has no wheels for newer Pythons (uv keeps locked versions otherwise).
7. Run the matrix with `--update-readme` and update the numbers in the README and ARCHITECTURE.md texts.

A behaviour that differs between versions gets a test in the `numpy 1/2 compatibility` section of the
test file mirroring the module. A known break that is not fixed yet is a strict xfail with the reason,
conditioned on the versions where it applies; an unexpected pass (`XPASS(strict)`) is a signal that
behaviour changed, e.g. Python 3.14 `Fraction` started accepting `np.float32`.

## Versioning

Versions follow [Semantic Versioning](https://semver.org): `MAJOR.MINOR.PATCH`, set in `pyproject.toml`.
The public interface is everything importable without a leading underscore, plus the physics of the
results: conventions, units and normalization.

| Bump | When |
|---|---|
| **MAJOR** | Incompatible change of the public interface: removed or renamed classes, functions, arguments or attributes, changed argument order or meaning, changed return types or shapes. Changed results beyond numerical noise from a change of convention: sign conventions, units, normalization of S(Q,E), coordinate systems. Removal of deprecated features. |
| **MINOR** | New backward-compatible functionality: new classes, functions, keyword arguments with defaults, new extras or optional dependencies. Dropping support of a Python or numpy version (raising a floor). Deprecations, which warn but keep working. Physics corrections of wrong results (bug fixes that change values), always noted in the release message. |
| **PATCH** | Backward-compatible bug fixes without changes of correct results, numerical stability fixes changing results only within tolerance, performance, documentation, tests, widened dependency support (lowering a floor, supporting a new Python). |

While the version is `0.y.z` the public interface is not stable: breaking changes bump **MINOR**,
and everything else bumps **PATCH**. Every release message lists breaking changes first.
`1.0.0` marks the first stable interface.

Pre-releases for testing a release candidate: `0.4.0b1`, `0.4.0rc1` (`--bump minor --bump beta`).

## Release procedure

Example for a minor release; replace `minor` and the versions as needed.

1. Start from an up-to-date `main` with a clean working tree:

   ```bash
   git switch main
   git pull
   git status
   ```

2. Run the tests, including graphics:

   ```bash
   uv run pytest
   uv run --group graphics pytest
   ```

3. Choose the bump according to [Versioning](#versioning) and check the result without writing:

   ```bash
   uv version --bump minor --dry-run
   ```

4. Bump the version. This updates `pyproject.toml` and `uv.lock` together; do not edit the version by
   hand, it leaves `uv.lock` behind and `uv run --locked` fails:

   ```bash
   uv version --bump minor
   ```

5. Update `CITATION.cff`: `version` to the output of `uv version --short`, and `date-released` to today.

6. Refresh the compatibility matrix in the README; check the table and fix any failing cell first:

   ```bash
   uv run --no-project --python 3.12 python scripts/compat_matrix.py --update-readme
   ```

7. Commit and create an annotated tag. The tag message is the release note: a one-line summary, then
   breaking changes, new features, fixes and dependency changes:

   ```bash
   git add pyproject.toml uv.lock CITATION.cff README.md
   git commit -m "Release v0.4.0"
   git tag -a v0.4.0 -m "v0.4.0: <one-line summary>" -m "Breaking: ..." -m "Added: ..." -m "Fixed: ..." -m "Dependencies: ..."
   ```

8. Push the commit with its tag, and publish the GitHub release from the tag message:

   ```bash
   git push origin main --follow-tags
   gh release create v0.4.0 --title "v0.4.0" --notes-from-tag
   ```

9. Check that the [tests workflow](https://github.com/mstekiel/spinwaves/actions/workflows/tests.yml)
   passes and that the README badges show the new version.

The README badges for version, Python and the core dependency read `pyproject.toml` on `main`, so
they update by themselves after the push.

## Rules established during development

- **Tests mirror the library**: `spinwaves/<pkg>/<module>.py` is tested in `tests/<pkg>/test_<module>.py`;
  see [Test structure](ARCHITECTURE.md#test-structure).
- **Known bugs are strict xfails** with the reason, not deleted or loosened tests.
- **Numerical checks** are relative to the scale of the checked object and use
  `spinwaves.config.tolerances`; see [Numerical tolerances](ARCHITECTURE.md#numerical-tolerances-configpy).
- **numpy 1/2**: store integer arrays with an explicit dtype when they are hashed via bytes, convert
  numpy scalars before `Fraction`, do not rely on which duplicate `np.unique` keeps, and keep numpy
  scalar reprs (`np.float64(1.0)`) out of strings, labels and hashes.
- **Regression data**: `tests/data/lswt_reference.npz` holds reference energies and spin-spin
  correlations. Regenerate it only after a deliberate change of results, with
  `uv run python tests/data/make_lswt_reference.py`, and say so in the release message.
