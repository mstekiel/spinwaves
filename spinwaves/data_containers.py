import numpy as np
import os
import pandas as pd

##################################################################
# Data tables container as from the numpy example:
# https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array

# Depracated
# class InfoArray(np.ndarray):    
#     def __new__(cls, input_filename: str, info_slice: slice, data_slice: slice, data_dtype: list):
#         with open(input_filename, 'r') as ff:
#             data_lines = ff.readlines()

#         obj = np.asarray([tuple(line.split()) for line in data_lines[data_slice]], dtype=data_dtype).view(cls)

#         obj.info = ''.join(data_lines[info_slice])[:-1]     # Last character is `newline`


#         # obj = np.array(data.view(cls))
#         # obj = data
#         return obj
    
#     def __array_finalize__(self, obj):
#         # see InfoArray.__array_finalize__ for comments
#         if obj is None: return
#         self.info = getattr(obj, 'info', None)

# atom_filename = PATH+'\\data_tables\\atom.dat'
# atom_entries = [('symbol', 'U2'), ('radius', 'f8'), ('R', 'i2'), ('G', 'i2'), ('B', 'i2'), ('mass', 'f8'), ('full_name', 'U16')]
# atom_data = InfoArray(input_filename=atom_filename, info_slice=slice(0, 1), data_slice=slice(2, None), data_dtype=atom_entries)

##################################################################
# Path where pySpinW is installed
PATH = os.path.split(os.path.dirname(__file__))[0]
# PATH, _ = os.path.split(os.path.dirname(spinwaves.__file__)) # split will remove last folder location

# atomdata

atom_filename = PATH+'\\data_tables\\atom.dat'
# atom_data = pd.read_csv(atom_filename, delim_whitespace=True, header=1, 
#                         names=['name','radius', 'R','G','B', 'mass', 'longname'], index_col=0).to_dict('index')
atom_data = {}
with open(atom_filename, 'r') as ff:
    for line in ff.readlines()[2:]:
        symbol, radius, R,G,B, mass, longname = line.split()
        atom_data[symbol] = dict(radius=float(radius), RGB=np.array([R,G,B], dtype=int), mass=float(mass), longname=longname)


isotope_filename = PATH+'\\data_tables\\isotope.dat'
isotope_data = pd.read_csv(isotope_filename, delim_whitespace=True, header=15)


magion_filename = PATH+'\\data_tables\\magion.dat'
magion_data = pd.read_csv(magion_filename, delim_whitespace=True, header=13)

color_filename = PATH+'\\data_tables\\color.dat'
color_data = {}
with open(color_filename, 'r') as ff:
    for line in ff.readlines():
        color, R,G,B = line.split()
        color_data[color] = np.array([R,G,B], dtype=int)


if __name__ == '__main__':
    pass
    # TODO
    # Move all loads from txt here, and replace the ones above 
    # with load from npy to increase speed
    # Alternativelu, convert to dict here and place hardcoded dicts above