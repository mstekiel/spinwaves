'''
Classes and functions related to symmetry operations and groups.

TODO:
- [ ] Make MSG and SG inherit from Group and implement the necessary methods.
- [ ] https://onlinelibrary.wiley.com/doi/10.1002/qua.20747 for implementation of:
    - [ ] Wyckoff positions
    - [ ] subgroups
    - [ ] supergroups
- [ ] https://journals.iucr.org/a/issues/2023/05/00/ib5114/ to check MSGs
'''

from .group import Group
from .crystall_space_group import cSymOp, SG
from .magnetic_symmetry import mSymOp, MSG