'''
Load various databases into a common interface.
'''
from dataclasses import dataclass
import os
import numpy as np

from .database import db_entry, Database

##################################################################
# Path where spinwaves is installed
PATH = os.path.split(os.path.dirname(__file__))[0]

##################################################################
# atom.dat

@dataclass
class atom_entry(db_entry):
    symbol: str
    radius: float
    red: float
    green: float
    blue: float
    mass: float
    longname: str

    units = dict(symbol=None, longname=None, radius='Bohr_radius', mass='amu',
                 red=None, green=None, blue=None)

    @property
    def unique_label(self):
        return self.symbol

    @property
    def RGB(self):
        return np.array([self.red, self.green, self.blue])
    
            
class Database_atom(Database):
    '''Database of atomic properties used for plotting.'''

    source_filename = PATH+'\\..\\data_tables\\atom.dat'
    entry_type = atom_entry
    header = ''
    entries = {}
    
    def __init__(self):
        self.from_txt(comment='#')

atom_data = Database_atom()
atom_data.__doc__ = Database_atom.__doc__

##################################################################
# isotope.dat

@dataclass
class isotope_entry(db_entry):
    Z: int
    A: int
    nat_abundance: float
    symbol: str
    bc: float
    mass: float
    sigma_coh: float
    sigma_inc: float
    sigma_scatt: float
    sigma_abs: float

    units = dict(Z='protons', A='protons_neutrons', nat_abundance='/1',
                    symbol=None, bc='fm', mass='amu',
                    sigma_coh='barn', sigma_inc='barn',
                    sigma_scatt='barn', sigma_abs='barn')
    
    @property
    def unique_label(self):
        prefix = ''
        if self.A>0:
            prefix = self.A

        return str(self.A)+self.symbol
    

class Database_isotope(Database):
    '''Database of atomic isotopes properties for neutron scattering.'''

    source_filename = PATH+'\\..\\data_tables\\isotope.dat'
    entry_type = isotope_entry
    header = ''
    entries = {}

    def __init__(self):
        self.from_txt(ln_header_end=14, ln_data_start=16)

isotope_data = Database_isotope()

##################################################################
# magion.dat

@dataclass
class magion_entry(db_entry):
    name: str
    spin: float
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
    c: float

    units = dict(name=None, spin='hbar', charge='elementary_charge', symbol=None, c=1,
                    **{f'a_{n}':1 for n in range(5)}, **{f'b_{n}':1 for n in range(5)} )

    @property
    def unique_label(self):
        return self.name
    

class Database_magion(Database):
    '''Database of magnetic ions properties utilized in neutron scattering.'''

    source_filename = PATH+'\\..\\data_tables\\magion.dat'
    entry_type = magion_entry
    header = ''
    entries = {}

    def __init__(self):
        self.from_txt(ln_header_end=15, ln_data_start=16)

magion_data = Database_magion()
magion_data.__doc__ = Database_magion.__doc__

##################################################################
# color.dat

@dataclass
class color_entry(db_entry):
    name: str
    red: int
    green: int
    blue: int

    units = dict(name=None, red=None, green=None, blue=None)

    @property
    def unique_label(self):
        return self.name
    
    @property
    def RGB(self):
        return np.array([self.red, self.green, self.blue])
    

class Database_color(Database):
    '''Database of color RGB values.'''

    source_filename: str = PATH+'\\..\\data_tables\\color.dat'
    entry_type = color_entry
    header = ''
    entries = {}

    def __init__(self):
        self.from_txt(ln_header_end=0, ln_data_start=0)


color_data = Database_color()
color_data.__doc__ = Database_color.__doc__

##################################################################
# xrayion.dat

@dataclass
class xrayion_entry(db_entry):
    name: str
    Z: int
    charge: int
    symbol: str
    a_1: float
    a_2: float
    a_3: float
    a_4: float
    a_5: float
    c: float
    b_1: float
    b_2: float
    b_3: float
    b_4: float
    b_5: float

    units = dict(name=None, Z='protons', charge='elementary_charge', symbol=None, c=1,
                 **{f'a_{n}':1 for n in range(5+1)}, **{f'b_{n}':1 for n in range(5+1)} )

    @property
    def unique_label(self):
        return self.name

class Database_xrayion(Database):
    '''Database of ionic form factors for Xray scattering'''

    source_filename: str = PATH+'\\..\\data_tables\\xrayion.dat'
    entry_type = xrayion_entry
    header = ''
    entries = {}

    def __init__(self):
        self.from_txt(ln_header_end=26, ln_data_start=28, comment='#')

xrayion_data = Database_xrayion()
xrayion_data.__doc__ = Database_xrayion.__doc__