"""Subpackage containing databases of various atomic properties.
Databases have common interface as defined by `DataBase` class."""

from .implement_txt import atom_data, isotope_data, color_data, magion_data, xrayion_data
from .implement_spglib import SG_data, MSG_data