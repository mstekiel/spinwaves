'''
Load various databases into a common interface.
'''
from ctypes.wintypes import RGB
from dataclasses import dataclass
import os
import numpy as np

from .database import db_entry, Database

import spglib

PATH = os.path.split(os.path.dirname(__file__))[0]

####################################################################################################

@dataclass
class SG_entry(db_entry):
    number: int
    international_short: str
    international_full: str
    international: str
    schoenflies: str
    hall_number: int
    hall_symbol: str
    choice: str
    pointgroup_international: str
    pointgroup_schoenflies: str
    arithmetic_crystal_class_number: int
    arithmetic_crystal_class_symbol: str

    units = dict(number=1, international_short=None, international_full=None, international=None,
                 schoenflies=None, hall_number=None, hall_symbol=None, choice=None, 
                 pointgroup_international=None, pointgroup_schoenflies=None, 
                 arithmetic_crystal_class_number=None, arithmetic_crystal_class_symbol=None)

    @property
    def unique_label(self):
        return self.hall_number
    
    def get_symmetry_ops(self):
        return spglib.get_symmetry_from_database(self.hall_number)
    
            
class Database_SG(Database):
    '''Database of crystallographic space groups in all settings.
    
    Wrapper over the `spglib` library.
    '''

    source_filename = PATH+'\\..\\data_tables\\spglib_SGnames.dat'
    entry_type = SG_entry
    header = ''
    entries = {}
    
    def __init__(self):
        with open(self.source_filename, 'r') as ff:
            lines = ff.readlines()

            self.header = ''.join(lines[:1])

            for line in lines[2:]:
                fields = [s.strip() for s in line.split('  ') if s.strip()]
                entry = self.entry_type(*fields)
                self.entries[entry.unique_label] = entry


SG_data = Database_SG()
SG_data.__doc__ = Database_SG.__doc__

####################################################################################################

@dataclass
class MSG_entry(db_entry):
    # uni_number litvin_number bns_number og_number number type

    uni_number: int
    litvin_number: int
    bns_number: str
    og_number: str
    number: int
    type: int

    units = dict(uni_number=None, litvin_number=None, bns_number=None, 
                 og_number=None, number=None, type=None)

    @property
    def unique_label(self):
        return self.uni_number-1
    
    def get_symmetry_ops(self):
        '''Get symmetry operations of the Magnetis Space group'''
        return spglib.get_magnetic_symmetry_from_database(self.uni_number)
    
            
class Database_MSG(Database):
    '''Database of crystallographic space groups in all settings.
    
    Wrapper over the `spglib` library.
    '''

    source_filename = PATH+'\\..\\data_tables\\spglib_MSGnames.dat'
    entry_type = MSG_entry
    header = ''
    entries = {}
    
    def __init__(self):
        with open(self.source_filename, 'r') as ff:
            lines = ff.readlines()

            self.header = ''.join(lines[:1])

            for line in lines[2:]:
                fields = [s.strip() for s in line.split('  ') if s.strip()]
                entry = self.entry_type(*fields)
                self.entries[entry.unique_label] = entry


MSG_data = Database_MSG()
MSG_data.__doc__ = Database_MSG.__doc__
