"""Subpackage containing databases of various atomic properties.
Databases have common interface as defined by `DataBase` class."""

# TODO I think this should be reworked a bit.
# The `_entry` classes are not just entries, they are property defining
# objects in themsleves, and should be so. So instead of `color_entry`
# let's just go with `Color`, inherit from `db_entry` and include additional
# funcitonalities. Then, each class should be in separate file.

from .implement_txt import atom_data, isotope_data, color_data, magion_data, xrayion_data
# from .implement_spglib import SG_data, MSG_data

