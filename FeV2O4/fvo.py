from attr import dataclass
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure

# Fitting
import lmfit
from lmfit import Parameters, fit_report

# import spinwaves
from spinwaves import Atom, Crystal, SpinW, Coupling
from spinwaves.magnetic_symmetry import MSG
from spinwaves.plotting import plot_structure

#######################################################
def load_system_cubic(parameters: Parameters, show_struct: bool=False, silent=True) -> SpinW:

    # Origin choice 2
    atoms = [Atom(label='Fe', r=(1/8, 1/8, 1/8),   m=(0.1,0,0), s=2), # 8a
             Atom(label='V',  r=(1/2, 1/2, 1/2),   m=(0.2,0,0), s=1),   # 16d 
             ]
    Fdm3m = MSG.from_xyz_strings(generators=[
        'x, y+1/2,z+1/2, +1', # t(0 1/2 1/2)
        'x+1/2, y,z+1/2, +1', # t(1/2 0 1/2)
        '-x+3/4,-y+1/4, z+1/2, +1',    # 2_001_d
        '-x+1/4, y+1/2, -z+3/4, +1',    # 2_010_d
        'z,x,y, +1',        # 3_111
        'y+3/4, x+1/4, -z+1/2 , +1', # 2_110_d
        '-x, -y, -z , +1', # -1
        ])

    cs = Crystal(lattice_parameters=[8.5,8.5,8.5, 90, 90, 90], 
                 atoms=atoms, MSG=Fdm3m)
    
    print(cs)

    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,0,1)
    }


    ### Extract the model parameters
    # Negative couplings are FM, positive are AF
    Da = parameters['Da'].value*2
    Db = parameters['Db'].value*2

    Jab = parameters['Jab'].value
    Jbb  = parameters['Jbb'].value
    Jpbb = parameters['Jpbb'].value

    couplings = []
    
    # Wrap up the couplings in one list
    # Single-ion anisotropies
    couplings += [Coupling(label=f'D_Fe', n_uvw=[0,0,0], id1=0, id2=0, J=np.diag([0, 0, Da]))]
    couplings += [Coupling(label=f'D_V' , n_uvw=[0,0,0], id1=2, id2=2, J=np.diag([Db,Db,Db]))]

    # Fe-V coupling
    couplings += [Coupling(label=f'Jab', n_uvw=[0,0,0], id1=6, id2=9, J=Jab*np.eye(3,3))]
    couplings += [Coupling(label=f'Jbb', n_uvw=[0,0,0], id1=2, id2=3, J=Jbb*np.eye(3,3))]
    # couplings += [Coupling(label=f'Jpbb',n_uvw=[0,0,0], id1=2, id2=3, J=Jpbb*np.eye(3,3))]

    if not silent:
        for cpl in couplings:
            print(cpl)

    # Construct the main object that is able to determine excitation spectrum
    sw = SpinW(crystal=cs, 
               couplings=couplings,
               magnetic_modulation=magnetic_modulation)
    
    if not silent: print(sw.couplings_all)
    
    if show_struct:
    #     plot_opts = dict(boundaries=([-0.5, 1.5],[-0.5,1.5],[-0.5,1]), coupling_colors={'J1a2': 'Cyan'})
        plot_opts = dict(boundaries=([-0.1, 1.1],[-0.1,1.1],[-0.1,1.02]), 
                             coupling_colors={'Jab': 'Red', 'Jbb':'Green', 'Jpbb':'Blue'})
        plot_structure(sw, engine='vispy', plot_options=plot_opts)
    
    return sw

def load_system_orthorhombic(parameters: Parameters, show_struct: bool=False, silent=True) -> SpinW:

    # Origin choice 2
    mV = 0.5
    atoms = [Atom(label='Fe', r=(1/8, 1/8, 1/8),   m=(0,0,2), s=2), # 8a
             Atom(label='V',  r=(1/2, 1/2, 1/2),   m=(mV,mV,mV), s=1),   # 16d 
             ]
    Fddd = MSG.from_xyz_strings(generators=[
        'x, y+1/2,z+1/2, +1', # t(0 1/2 1/2)
        'x+1/2, y,z+1/2, +1', # t(1/2 0 1/2)
        '-x+3/4, -y+3/4, z, +1',    # 2_001_d
        '-x+3/4, y, -z+3/4, -1',    # 2_010_d prime to get two in - two out arrangement
        '-x, -y, -z , +1', # -1
        ])

    cs = Crystal(lattice_parameters=[8.5,8.5,8.5, 90, 90, 90], 
                 atoms=atoms, MSG=Fddd)
    
    print(cs)

    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,0,1)
    }


    ### Extract the model parameters
    # Negative couplings are FM, positive are AF
    Da = parameters['Da'].value*2
    Db = parameters['Db'].value*2

    Jab = parameters['Jab'].value
    Jbb  = parameters['Jbb'].value
    Jpbb = parameters['Jpbb'].value

    couplings = []
    
    # Wrap up the couplings in one list
    # Single-ion anisotropies
    couplings += [Coupling(label=f'D_Fe', n_uvw=[0,0,0], id1=4, id2=4, J=np.diag([0, 0, Da]))]
    couplings += [Coupling(label=f'D_V' , n_uvw=[0,0,0], id1=0, id2=0, J=np.diag([Db,Db,Db]))]

    ### Coupling have to by symmetrized to the pseudo Fd-3m space group, except V-V
    # Fe-V coupling
    couplings += [Coupling(label=f'Jab', n_uvw=[0,0,0], id1=4, id2=0, J=Jab*np.eye(3,3))]
    couplings += [Coupling(label=f'Jab', n_uvw=[0,0,0], id1=4, id2=2, J=Jab*np.eye(3,3))]
    couplings += [Coupling(label=f'Jab', n_uvw=[0,0,0], id1=4, id2=12, J=Jab*np.eye(3,3))]

    # V-V couplings in V4 tetrahedron. It is not clear which one is JpBB and which JBB
    couplings += [Coupling(label=f'Jpbb', n_uvw=[0,0,0], id1=12, id2=13, J=Jpbb*np.eye(3,3))]
    couplings += [Coupling(label=f'Jpbb', n_uvw=[0,0,0], id1=13, id2=19, J=Jpbb*np.eye(3,3))]

    couplings += [Coupling(label=f'Jbb',n_uvw=[0,0,0], id1=12, id2=19, J=Jbb*np.eye(3,3))]

    if not silent:
        for cpl in couplings:
            print(cpl)

    # Construct the main object that is able to determine excitation spectrum
    sw = SpinW(crystal=cs, 
               couplings=couplings,
               magnetic_modulation=magnetic_modulation)
    
    if not silent: print(sw.couplings_all)
    
    if show_struct:
    #     plot_opts = dict(boundaries=([-0.5, 1.5],[-0.5,1.5],[-0.5,1]), coupling_colors={'J1a2': 'Cyan'})
        plot_opts = dict(boundaries=([-0.1, 1.1],[-0.1,1.1],[-0.1,1.02]), 
                             coupling_colors={'Jab': 'Gray', 'Jbb':'Blue', 'Jpbb':'Green'})
        plot_structure(sw, engine='vispy', plot_options=plot_opts)
    
    return sw

def plot_spectrum(sw_params, plot_type: str='dispersion') -> Figure:
    Npath = 31
    sw = load_system_orthorhombic(sw_params, show_struct=False)

    mosaic = [ ['2h0', '2h0_in']
              ]
    layout = dict(width_ratios=[2,1])
    fig, axs = plt.subplot_mosaic(mosaic=mosaic, figsize=(7,5), tight_layout=True, gridspec_kw=layout)


    print('Calculate spectrum `2h0`...')
    axs['2h0'].set_title("(2h0) EIGER")
    qPath = sw.crystal.make_qPath(main_qs=[[2, -6, 0], [2, 4, 0]], Nqs=[Npath])    
    omega1, _ = sw.calculate_spectrum(qPath=qPath, silent=True)
    sw.plot_dispersion(ax=axs['2h0'],    xaxis=qPath[:,1], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))
    sw.plot_dispersion(ax=axs['2h0_in'], xaxis=qPath[:,1], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))


    for ax_name, ax in axs.items():
        if ax_name.endswith('_in'):
            ax.set_xlim(1.9, 2.1)
            ax.set_ylim(0, 15)
        else:
            ax.set_ylim(0, 1.05*np.max(np.concatenate((omega1))))
            # pass

    return fig

def minimize_lfo_gs(p0: Parameters):
    """Run the ground state minimization procedure with starting parameters `p0`.
    Print the result.
    
    Returns
    -------
    Fitted parameters
    """
    print('Minimizing ground state energy for LuFeO3')
    def lfo_gs(parameters: Parameters):
        lfo_sw = load_system(parameters)

        Eg = lfo_sw.calculate_ground_state(q_hkl=[0,0,0])
        print(f'E0={Eg}')
        for p in parameters:
            print('\t', parameters[p])

        return Eg

    fit_result = lmfit.minimize(lfo_gs, p0, method='nelder')

    print(fit_report(fit_result))
    return fit_result

def load_lfo_parameters(model_name: str) -> Parameters:
    '''Define various parameter sets for LuFeO3 models.
    '''

    models = dict()


    # Simple J1-J2 model no more
    # Unstable, as it scales J1 an J2 up
    lfo_params = Parameters()
    lfo_params.add(name='Da',  value=-0.01*50, vary=True)
    lfo_params.add(name='Db',  value=-9.5*50, vary=False)

    lfo_params.add(name='Jab',  value=2.9, vary=True)
    lfo_params.add(name='Jbb',  value=15, vary=True)
    lfo_params.add(name='Jpbb',  value=-0.6, vary=True)

    models['mc'] = lfo_params


    return models[model_name]


if __name__ == '__main__':
    PATH = fr'C:\Users\Stekiel\Documents\GitHub\spinwaves\FeV2O4'

    lfo_params = load_lfo_parameters('mc')
    # sw = load_system_cubic(lfo_params, show_struct=True, silent=False)
    sw = load_system_orthorhombic(lfo_params, show_struct=False, silent=False)

    fig = plot_spectrum(lfo_params, plot_type='dispersion')
    fig.savefig(PATH+'\spinwaves-fvo-Eq.png', dpi=400)

    # fig = plot_spectrum(lfo_params, plot_type='spectral_weight')
    # fig.savefig(PATH+'\spinwaves-fvo-Sqw.png', dpi=400)

