# spinwaves - SpinW implmentation in python

## Bugs

 - [ ] There is some warning in `database` module that causes a lot of empty lines printout when warning levels are higher.

## TODO

- Core
  - [ ] Implement magnetic field
    - [x] implement
    - [ ] test -> sign convention unclear
    - [ ] SpinW -> only obscuretutorials found
  - [ ] Implement primitive/reduced cell such that the calculation is on smaller cell=matrix=faster, while the setup is in the coordinates of the main cell, which are easier to interpret
  - [ ] standard paths of lattices doi.org/10.1016/j.commatsci.2010.05.010
  - [x] MSG should inherit from Group
  - [x] crystallographic SG should also inherit from Group

- Future
  - [ ] Lot of core functionalities rely on hashing the objects for unique identifiers:
    -  `SymOps` do it from string for unique elements finding
    -  `Atom` same, for unique position finding
    Is there a sturdy way to implement hashing?

- Usage
  - [ ] Plotting of unit cell edges.
  - [x] coordinate systems.
  - [x] lighting.

- Make documentation:
  For list of modules and their descriptions see documentation at: 
  https://mstekiel.github.io/mikibox/build/html/index.html

- Make GUI (PyQt6) with inline editeor (https://qscintilla.com), 3D viewer (https://vispy.org). The GUI can be a direct copy of vspy/.examples/../sandbox
  - VISPY has a lot of beautiful examples for Ctrl+CV coding.
    - whole GUI as in vispy\examples\basics\scene\modular_shaders\sandbox.py
    - tubes vispy\examples\basics\visuals\tube.py
    - check out `Canvas.measure_fps()`
    - demo\gloo\boids.py for animatoins with particles (neutrons)
    - demo\gloo\camera.py with gestures recognition neural network to control the viewing.
  - https://mediapipe-studio.webapps.google.com/studio/demo/face_landmarker for gesture recognition and 3D navigation


## Managing project with uv

Moved the project to uv. 

Now, if you want to use `spinwaves` as library in editable mode include these in the `pyproject.toml`:

```toml
[project]
dependencies = [
    "spinwaves",
]

[tool.uv.sources]
spinwaves = { path = "path_to_spinwaves_project", editable=true }
```
### Testing

`uv run pytest`


## General notes
- I tried to follow the structure of https://github.com/pypa/sampleproject for the development of this project. Following descriptions from https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html
- Tobi confirmed the factor of two is missing from single-ion naisotropies. He also mentioned the inverted sign mistake in the phase factor somewhere in the spin-spin correlation function calculations.