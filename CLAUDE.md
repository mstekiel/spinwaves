# CLAUDE.md

Rules for AI coding agents working on `spinwaves`, a linear spin wave theory library following SpinW.
Read [ARCHITECTURE.md](ARCHITECTURE.md) for the code structure and conventions, and
[DEV.md](DEV.md) for the workflow, dependency checks, versioning and releases.

## Working with the maintainer

- The maintainer commits, tags and pushes. Never commit, tag or push unless explicitly asked.
- When asked for a review or advice ("read only", "plan", "advise"), do not change files.
- Ask clarifying questions when a choice changes the outcome: physics conventions, scope, API.
- Report outcomes with numbers: test counts, deviations, versions. Say what was not verified.
- Physics conventions (signs, units, normalization, e.g. the `dipolar` tensor) are the maintainer's
  decision: record the discrepancy as a strict xfail and ask, do not pick one.

## Environment

- Windows, Git Bash. Managed with `uv` only: no `pip`; `python` is not on the PATH, use `uv run`.
- Do not change the maintainer's `.venv` for experiments: use `UV_PROJECT_ENVIRONMENT=<short temp path>`,
  or `uv run --no-project --isolated ...`. Keep temp paths short: deep paths break numpy DLL loading.
- Run `uv` installs one after another, not in parallel: they race in the uv cache on Windows.
- Many files use CRLF line endings: preserve them when editing.
- `uv.lock` is tracked. After editing dependencies run `uv lock`; change versions only with `uv version`.

## Code

- The core is computational and needs only `numpy`. Plotting and GUI packages (`graphics` extra) are
  imported lazily inside functions, after `spinwaves.utils.graphics.require_graphics(...)`.
  `spinwaves/plotting/__init__.py` and anything imported by `import spinwaves` must not import them.
- Plotting is done by functions taking the `SpinW` object first (`plot_dispersion(sws, ax, ...)`),
  not by mixins on `SpinW`.
- Deprecated implementations go to `spinwaves/legacy/` as functions, not imported by `spinwaves`;
  `scipy` is allowed there only (dev group).
- Numerical checks are relative to the scale of the checked object, against
  `spinwaves.config.tolerances.numeric` read at call time: below it correct silently (DEBUG log),
  above it raise `NumericalToleranceError`. Use `ensure_hermitian`, `regularized_cholesky`,
  `hermitian_eig` from `spinwaves.utils.linalg`; no ad hoc absolute thresholds.
- numpy 1 and 2 must both work (floor in `pyproject.toml`):
  - arrays hashed via `tobytes()` get an explicit dtype (`np.int64`); the default integer differs
    between platforms and numpy versions;
  - convert numpy scalars with `float()` before `Fraction`; NEP 50 keeps `float32` in numpy 2;
  - do not depend on which duplicate `np.unique` keeps on object arrays; deduplicate explicitly;
  - keep numpy scalar reprs (`np.float64(1.0)`) out of strings, labels and hashes;
  - no removed numpy 2 APIs (`np.float_`, `np.product`, `np.in1d`, `ndarray.ptp`, ...).

## Tests

- `tests/` mirrors `spinwaves/`: `spinwaves/<pkg>/<module>.py` is tested in `tests/<pkg>/test_<module>.py`.
  Move, add or rename test files together with modules; untested modules get a stub file.
- Shared fixtures in `tests/conftest.py`, reference systems and checks in `tests/helpers.py`, skip
  markers for optional dependencies in `tests/markers.py` (`needs_matplotlib`, `needs_scipy`, ...).
- Known bugs are strict xfails with the reason, conditioned on the versions where they apply.
  Never loosen or delete a test to make it pass; fix the code or record the bug.
- Check physics against analytic results, SpinW, or `scipy` (eigensolvers); compare spectral
  projectors or mode sums for degenerate modes, which are basis dependent.
- `tests/data/lswt_reference.npz` is the regression reference; regenerate it with
  `tests/data/make_lswt_reference.py` only after a deliberate change of results, and say so.
- `tests/comparisons/` holds scripts, not collected by pytest. With `graphics` installed,
  `tests/test_spinw.py::test_coupling_symmetrization_tetr` blocks on a vispy window: deselect it.
- Before finishing: `uv run pytest`, and for dependency-related changes the cells of
  `scripts/compat_matrix.py` around the change.

## Documentation

Keep in sync with the code: the module map and test tree in ARCHITECTURE.md, the commands and rules
in DEV.md, the compatibility matrix in README.md (`scripts/compat_matrix.py --update-readme`).
