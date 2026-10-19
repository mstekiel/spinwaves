'''Load text-file databases into Database objects.'''

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .database import db_entry, Database, load_database_from_txt

SW_PATH = Path(__file__).parent.parent

##################################################################
# atom.dat

@dataclass
class atom_entry(db_entry):
    '''Atomic display properties (radius, color, mass).'''
    symbol:   str
    radius:   float
    red:      float
    green:    float
    blue:     float
    mass:     float
    longname: str

    units = dict(symbol=None, longname=None, radius='Bohr_radius',
                 mass='amu', red=None, green=None, blue=None)

    @property
    def unique_label(self) -> str:
        return self.symbol

    @property
    def RGB(self) -> np.ndarray:
        return np.array([self.red, self.green, self.blue])


atom_data: Database = load_database_from_txt(
    SW_PATH / 'data_tables/atom.dat',
    atom_entry,
    comment='#',
)

##################################################################
# isotope.dat

@dataclass
class isotope_entry(db_entry):
    '''Isotope properties for neutron scattering.'''
    Z:             int
    A:             int
    nat_abundance: float
    symbol:        str
    bc:            float
    mass:          float
    sigma_coh:     float
    sigma_inc:     float
    sigma_scatt:   float
    sigma_abs:     float

    units = dict(Z='protons', A='protons_neutrons', nat_abundance='/1',
                 symbol=None, bc='fm', mass='amu',
                 sigma_coh='barn', sigma_inc='barn',
                 sigma_scatt='barn', sigma_abs='barn')

    @property
    def unique_label(self) -> str:
        return str(self.A) + self.symbol


isotope_data: Database = load_database_from_txt(
    SW_PATH / 'data_tables/isotope.dat',
    isotope_entry,
    ln_data_start=16,
)

##################################################################
# magion.dat

@dataclass
class magion_entry(db_entry):
    '''Magnetic ion properties for neutron scattering form factors.'''
    name:   str
    spin:   float
    charge: int
    symbol: str
    a_1: float
    b_1: float
    a_2: float
    b_2: float
    a_3: float
    b_3: float
    a_4: float
    b_4: float
    c:   float

    units = dict(name=None, spin='hbar', charge='elementary_charge',
                 symbol=None, c=1,
                 **{f'a_{n}': 1 for n in range(1, 5)},
                 **{f'b_{n}': 1 for n in range(1, 5)})

    @property
    def unique_label(self) -> str:
        return self.name


magion_data: Database = load_database_from_txt(
    SW_PATH / 'data_tables/magion.dat',
    magion_entry,
    ln_data_start=16,
)

##################################################################
# color.dat

@dataclass
class color_entry(db_entry):
    '''Named color with RGB components.'''
    name:  str
    red:   int
    green: int
    blue:  int

    units = dict(name=None, red=None, green=None, blue=None)

    @property
    def unique_label(self) -> str:
        return self.name

    @property
    def RGB(self) -> np.ndarray:
        return np.array([self.red, self.green, self.blue])


color_data: Database = load_database_from_txt(
    SW_PATH / 'data_tables/color.dat',
    color_entry,
)

##################################################################
# xrayion.dat

@dataclass
class xrayion_entry(db_entry):
    '''Ionic form factors for X-ray scattering.'''
    name:   str
    Z:      int
    charge: int
    symbol: str
    a_1: float
    a_2: float
    a_3: float
    a_4: float
    a_5: float
    c:   float
    b_1: float
    b_2: float
    b_3: float
    b_4: float
    b_5: float

    units = dict(name=None, Z='protons', charge='elementary_charge',
                 symbol=None, c=1,
                 **{f'a_{n}': 1 for n in range(1, 6)},
                 **{f'b_{n}': 1 for n in range(1, 6)})

    @property
    def unique_label(self) -> str:
        return self.name


xrayion_data: Database = load_database_from_txt(
    SW_PATH / 'data_tables/xrayion.dat',
    xrayion_entry,
    ln_data_start=28,
    comment='#',
)
