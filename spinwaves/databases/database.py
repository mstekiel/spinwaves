'''
Design a common interface for various databases.
'''


from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

# need this metaclass to enforce implementation of `unique_label` in children
@dataclass
class db_entry(metaclass=ABCMeta):
    '''Template for a database entry.
    
    For each declared field performs typecasting and requires unit implementation.

    Abstract methods:
    `unique_label` for proper indexing of the entries
    `units` to understand the units of stored values'''

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
            try:
                casted_field_value = field_type(self.__getattribute__(field_name))
            except ValueError:
                raise ValueError(f'Type of the field {field_name!r} must be {field_type!r}')

            self.__dict__[field_name] = casted_field_value

            # Ensure all fields have declared units
            if field_name not in self.units:
                raise KeyError(f'Missing unit for {field_name!r}')


class Database(metaclass=ABCMeta):
    '''Database template. Supports indexing and implements `search` method.'''

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
    def properties(self) -> list[str]:
        '''List properties of entries in the database'''
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
            if property not in self.properties:
                raise KeyError(f'{property!r} is an invalid entry name, try one of {self.properties}')
            
            for key in list(search_candidates.keys()):
                entry = search_candidates[key]
                if not condition( getattr(entry, property) ):
                    del search_candidates[key]
            
        return search_candidates