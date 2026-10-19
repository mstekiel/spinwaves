'''
Common interface for databases of atomic properties.
'''

import dataclasses
import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path

logger = logging.getLogger('database')


@dataclasses.dataclass
class db_entry(metaclass=ABCMeta):
    '''Base class for a database entry.

    Subclasses must:
    - be a @dataclass with typed fields
    - define a class-level `units` dict mapping field name -> unit string or None
    - implement the `unique_label` property returning the key used for indexing

    __post_init__ casts all fields to their declared types and verifies
    that every field has a corresponding entry in `units`.
    '''

    @property
    @abstractmethod
    def unique_label(self) -> str: ...

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, field.type):
                logger.debug(f'Typecasting {field.name!r} to {field.type}')
                try:
                    object.__setattr__(self, field.name, field.type(value))
                except (ValueError, TypeError):
                    raise ValueError(f'Cannot cast field {field.name!r} value {value!r} to {field.type}')

            if field.name not in self.__class__.units:
                raise KeyError(f'Missing unit declaration for field {field.name!r}')


class Database:
    '''Thin wrapper around a dict of db_entry objects.

    Created by `load_from_txt`. Supports key lookup, field/unit
    inspection, and filtered search.

    Example
    -------
    >>> db = load_from_txt('atom.dat', atom_entry, comment='#')
    >>> db['Fe'].RGB
    >>> db.search(mass=lambda m: m > 50)
    '''

    def __init__(self, entries: dict[str, db_entry], entry_type: type):
        self._entries   = entries
        self._entry_type = entry_type

    def __getitem__(self, key: str) -> db_entry:
        return self._entries[key]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __repr__(self) -> str:
        return f'<Database[{self._entry_type.__name__}], {len(self)} entries>'

    @property
    def fields(self) -> list[str]:
        return [f.name for f in dataclasses.fields(self._entry_type)]

    @property
    def units(self) -> dict[str, str]:
        return self._entry_type.units

    def search(self, **kwargs) -> dict[str, db_entry]:
        '''Return entries matching all conditions.

        Each kwarg maps a field name to a callable predicate.

        >>> db.search(mass=lambda m: m > 50, symbol=lambda s: s.startswith('F'))
        '''
        for prop in kwargs:
            if prop not in self.fields:
                raise KeyError(f'{prop!r} is not a valid field, choose from {self.fields}')

        return {
            k: v for k, v in self._entries.items()
            if all(cond(getattr(v, prop)) for prop, cond in kwargs.items())
        }


def load_database_from_txt(
    filepath: Path,
    entry_type: type,
    ln_data_start: int = 0,
    comment: str = None,
) -> Database:
    '''Load a whitespace-delimited text file into a Database.

    Parameters
    ----------
    filepath:
        Path to the data file.
    entry_type:
        A db_entry subclass whose constructor accepts one value per column.
    ln_data_start:
        Line index (0-based) where data rows begin.
    comment:
        Lines starting with this string are skipped.
    '''
    entries: dict[str, db_entry] = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines[ln_data_start:]:
        line = line.strip()
        if not line:
            continue
        if comment and line.startswith(comment):
            continue
        entry = entry_type(*line.split())
        entries[entry.unique_label] = entry

    return Database(entries, entry_type)
