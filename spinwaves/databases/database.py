'''
Common interface for various databases.

For quick implementation copy the examples at the bottom of this file.
'''

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import logging
logger = logging.getLogger('database')

# need this metaclass to enforce implementation of `unique_label` in children
@dataclass
class db_entry(metaclass=ABCMeta):
    '''Template for a database entry.
    
    For each declared field performs typecasting and requires unit implementation.

    Abstract fields:
    `unique_label` method for proper indexing of the entries
    `units` field to understand the units of stored values
    
    Example
    -------
    
    >>> @dataclass
    >>> class atom_entry(db_entry):
    ...     symbol: str
    ...     radius: float
    ...     red: float
    ...     green: float
    ...     blue: float
    ...     mass: float
    ...     longname: str
    ... 
    ...     units = dict(symbol=None, longname=None, radius='Bohr_radius', mass='amu',
    ...                 red=None, green=None, blue=None)
    ... 
    ...     @property
    ...     def unique_label(self):
    ...         return self.symbol
    ... 
    ...     @property
    ...     def RGB(self):
    ...         return np.array([self.red, self.green, self.blue])
    ...

    In the example `unites` were given as a field (TODO: does that work?!?),
    and an implementation of the derived field `RGB` is shown.
    '''

    @property
    @abstractmethod
    def unique_label(self) -> str:
        '''Unique label of the entry'''
        ...

    @property
    @abstractmethod
    def units(self) -> dict[str, str]:
        '''Units of entry's values'''
        ...

    def __post_init__(self):
        for field_name, field_type in self.__annotations__.items():
            # Perform typecasting
            if not isinstance(self.__getattribute__(field_name), field_type):
                logger.warning(f'')
            try:
                casted_field_value = field_type(self.__getattribute__(field_name))
            except ValueError:
                raise ValueError(f'Type of the field {field_name!r} must be {field_type!r}')

            self.__dict__[field_name] = casted_field_value

            # Ensure all fields have declared units
            if field_name not in self.units:
                raise KeyError(f'Missing unit for {field_name!r}')


class Database(metaclass=ABCMeta):
    '''Database template. Supports indexing and implements `search` method.
    
    Example
    -------
    >>>    @dataclass
    >>>    class atom_entry(db_entry):
    ...        symbol: str
    ...        radius: float
    ...        red: float
    ...        green: float
    ...        blue: float
    ...        mass: float
    ...        longname: str
    ...
    ...        units = dict(symbol=None, longname=None, radius='Bohr_radius', mass='amu',
    ...                    red=None, green=None, blue=None)
    ...
    ...        @property
    ...        def unique_label(self):
    ...            return self.symbol
    ...
    ...        @property
    ...        def RGB(self):
    ...            return np.array([self.red, self.green, self.blue])
                
    >>> class Database_atom(Database):
    ...    #Database of atomic properties used for plotting.
    ...
    ...    source_filename = PATH+'\\..\\data_tables\\atom.dat'
    ...    entry_type = atom_entry
    ...    header = ''
    ...    entries = {}
    ...
    ...    def __init__(self):
    ...        self.from_txt(comment='#')

    >>> atom_data = Database_atom()
    >>> atom_data.__doc__ = Database_atom.__doc__

    The above example can be contained in a file, e.g. `atomic_database.py` and imported as:
    >>> from atomic_database import atom_data 

    To access all functionalities of the database and its entries.
'''

    @property
    @abstractmethod
    def entries(self) -> dict[str, db_entry]:
        '''Entries of the database'''
        ...

    @property
    @abstractmethod
    def entry_type(self) -> db_entry:
        '''Datatype for Database entries'''
        ...

    #########################################################################################
    def from_txt(self, ln_header_end: int=0, ln_data_start: int=0, comment: str=None):
        '''Fills header and entries'''
        with open(self.source_filename, 'r') as ff:
            lines = ff.readlines()

            self.header = ''.join(lines[:ln_header_end])

            for line in lines[ln_data_start:]:
                if bool(comment):
                    if line.startswith(comment):
                        continue

                entry = self.entry_type(*line.split())
                self.entries[entry.unique_label] = entry


    @property
    def fields(self) -> list[str]:
        '''List fields of entries in the database'''
        return list(self.entry_type.__dataclass_fields__.keys())
    
    @property
    def units(self) -> list[str]:
        '''List units of entry properties'''
        return self.entry_type.units
    
    def __repr__(self) -> str:
        return self.__doc__

    def __getitem__(self, label: str) -> db_entry:
        return self.entries[label]

    def search(self, **kwargs) -> dict[str, db_entry]:
        '''Search the database with given conditions.
        
        Conditions are given as kwargs, where each keywords dictionary is `dict[str, Callable]`,
        where `str` corresponds to the column name in the database, and `Callable` is the test
        function on that column. The `Callable` must return bool.'''

        # Copy the main array and remove elements not fitting the conditions
        search_candidates = self.entries.copy()

        for property, condition in kwargs.items():
            if property not in self.fields:
                raise KeyError(f'{property!r} is an invalid entry name, try one of {self.fields}')
            
            for key in list(search_candidates.keys()):
                entry = search_candidates[key]
                if not condition( getattr(entry, property) ):
                    del search_candidates[key]
            
        return search_candidates
    
########################## EXAMPLE IMPLEMENTATIONS #################################
### db_entry 
'''
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
'''

## DataBase
'''
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
   #Database of atomic properties used for plotting.

   source_filename = PATH+'\\..\\data_tables\\atom.dat'
   entry_type = atom_entry
   header = ''
   entries = {}

   def __init__(self):
       self.from_txt(comment='#')

atom_data = Database_atom()
atom_data.__doc__ = Database_atom.__doc__
'''