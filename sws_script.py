import numpy as np
import spinwaves
from spinwaves import Crystal, Atom
from spinwaves.plotting import plot_structure

# Primary
# lattice = Lattice([3.25,3.25,3.78, 90,90,120])
crystal = Crystal(
    lattice_parameters=[3.25,3.25,3.78, 90,90,120],
    atoms=[Atom(label='Er', r=[0,0,0], m=[0,1,0], s=1),
           Atom(label='B1', r=[1/3,2/3,0.5]),
           Atom(label='B2', r=[2/3,1/3,0.5])]
           )

# Secondary
# Rework couplings, such that atom labels are used
Jx = 2.5
couplings = {
    'Kz':[[0,0,0], 0, 0, np.diag([0,0,0.2]), ['1']],
    'Jx':[[1,0,0], 0, 0, -Jx*np.eye(3,3), ['6z']],
    'J2x':[[2,0,0], 0, 0, -Jx/2*np.eye(3,3), ['6z']]
}

str_plot_options = dict(
    boundaries = 3
)

if __name__ == '__main__':
    # uc = spinwaves.UnitCell(atoms=atoms)
    sw = spinwaves.SpinW(crystal=crystal, 
                         magnetic_structure={'k':[1/3,1/3,0], 'n':[0,0,1]})
    
    str_widget = plot_structure(sw, plot_options=str_plot_options)
    
    # print(sw.make_supercell([[-1,1],[-1,1],[0,-2]]))
    # print(sw.make_supercell([[-1,1],[1,-1],[0,-2]]))
    
    # sw.add_couplings(couplings)

    # spinwaves.SupercellPlotter(sw, extent=(2,2,2), plot_mag=True, plot_bonds=False, plot_atoms=True,
    #             plot_labels=False, plot_cell=True, plot_axes=True, plot_plane=False, ion_type=None, polyhedra_args=None)
    