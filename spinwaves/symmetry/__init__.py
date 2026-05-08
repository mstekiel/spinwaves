'''
Classes and functions related to symmetry operations and groups.

TODO:
- [ ] Make MSG and SG inherit from Group and implement the necessary methods.
- [ ] 
'''

from .group import Group
from .crystall_space_group import cSymOp, SG
from .magnetic_symmetry import mSymOp, MSG