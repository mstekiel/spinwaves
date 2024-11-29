import logging
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
from spinwaves.functions import DMI

#          #QH      QK      QL      ENlim1    ENlim2     I1       EN1       sigma1         I2        EN2    sigma2
DB_0K1 = """0    1.140000   1       2.0000   50.0000   1.0000   30.000000   0.9070         0         0         0
            0    1.090000   1       2.0000   50.0000   1.0000   20.000000   0.8884         0         0         0
            0    1.062000   1       2.0000   50.0000   1.0000   15.000000   0.6809         0         0         0
            0    1.050000   1       2.0000   50.0000   1.0000   12.500000   0.6000         0         0         0
            0    1.040000   1       2.0000   50.0000   1.0000   10.000000   0.5051         0         0         0
            0    1.020000   1       2.0000   50.0000   1.0000    7.500000   0.4270         0         0         0
            0    1.000000   1       2.0000   50.0000   1.0000    5.000000   0.3410         0         0         0
            0    0.980000   1       2.0000   50.0000   1.0000    7.500000   0.4214         0         0         0
            0    0.960000   1       2.0000   50.0000   1.0000   10.000000   0.5163         0         0         0
            0    0.950000   1       2.0000   50.0000   1.0000   12.500000   0.6000         0         0         0
            0    0.938000   1       2.0000   50.0000   1.0000   15.000000   0.6958         0         0         0
            0    0.910000   1       2.0000   50.0000   1.0000   20.000000   0.8882         0         0         0
            0    0.860000   1       2.0000   50.0000   1.0000   30.000000   0.9068         0         0         0"""
data_0K1 = np.array([line.strip().split() for line in DB_0K1.split('\n')], dtype=float)

DB_01L = """0     1     1.230000   2.0000   50.0000  1.0000  30.000000   1.1889         0         0         0
            0     1     1.100000   2.0000   50.0000  1.0000  15.000000   0.6952         0         0         0
            0     1     1.080000   2.0000   50.0000  1.0000  12.500000   0.6000         0         0         0
            0     1     1.060000   2.0000   50.0000  1.0000  10.000000   0.5163         0         0         0
            0     1     1.030000   2.0000   50.0000  1.0000  7.500000    0.4270         0         0         0
            0     1     1.000000   2.0000   50.0000  1.0000  5.000000    0.4270         0         0         0
            0     1     0.970000   2.0000   50.0000  1.0000  7.500000    0.4214         0         0         0
            0     1     0.940000   2.0000   50.0000  1.0000  10.000000   0.5051         0         0         0
            0     1     0.920000   2.0000   50.0000  1.2900  12.500000   0.6000         0         0         0
            0     1     0.900000   2.0000   50.0000  1.0000  15.000000   0.7015         0         0         0
            0     1     0.770000   2.0000   50.0000  1.0000  30.000000   1.2000         0         0         0
            0     1     0.500000   0.0000   80.0000  1.0000  54.310000   4.0000         0         0         0
            0     1     2.000000   0.0000   80.0000  1.0000  64.980000   5.0000         0         0         0"""
#            0     1     0.650000   2.0000   50.0000  1.0000  37.000000   1.6000         0         0         0

data_01L = np.array([line.strip().split() for line in DB_01L.split('\n')], dtype=float)

#######################################################
def load_system(parameters: Parameters, show_struct: bool=False, silent=True) -> SpinW:
    Fz = parameters['Fz'].value
    atoms = [Atom(label='Fe', r=(0,   0.5, 0),   m=(-1,0,Fz), s=2.5)]
    Pbnm = MSG.from_xyz_strings(generators=[
        'x+1/2,-y+1/2,-z, -1',
        '-x,-y,-z, +1',
        '-x,-y,z+1/2, +1',
    ])

    cs = Crystal(lattice_parameters=[5.3, 5.6, 7.5, 90, 90, 90], 
                 atoms=atoms, MSG=Pbnm)
    
    # print(cs)

    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,0,1)
    }


    ### Extract the model parameters
    # Negative couplings are FM, positive are AF
    Ka = parameters['Ka'].value*2
    Kc = parameters['Kc'].value*2

    J1ab = parameters['J1ab'].value
    J1c  = parameters['J1c'].value
    J2a = parameters['J2a'].value
    J2b = parameters['J2b'].value
    J2d  = parameters['J2d'].value

    Dab_x = parameters['Dab_x'].value
    Dab_y = parameters['Dab_y'].value
    Dab_z = parameters['Dab_z'].value
    
    Dc_x = parameters['Dc_x'].value
    Dc_y = parameters['Dc_y'].value
    Dc_z = parameters['Dc_z'].value

    couplings = []
    
    # Wrap up the couplings in one list
    # Single-ion anisotropies
    couplings += [Coupling(label=f'K_Fe', n_uvw=[0,0,0], id1=0, id2=0, J=np.diag([Ka, 0, Kc]))]
    # Nearest neighbour along the c axis
    DMI_c = DMI([Dc_x,-Dc_y, Dc_z])
    couplings += [Coupling(label=f'J1c', n_uvw=[0,0,0], id1=0, id2=1, J=J1c*np.eye(3,3)+DMI_c)]


    # Nearest-neighbour in-plane exchanges. Thay also contain the Dzialoszynski-Moriya interaction


    # The DMI schemeis from Park et al 2018 picture
    # One issue with that picture is it seems D32 and D32' are interexchanged, based on arrow directions.
    # Maybe this can be tested
    DMI_ab = DMI([-Dab_x, Dab_y, Dab_z])
    couplings += [Coupling(label=f'J1a', n_uvw=[0,0,0], id1=0, id2=2, J=J1ab*np.eye(3,3)+DMI_ab)]

    # Exchange between atoms along main orthorhombic directions
    couplings += [Coupling(label=f'J2a', n_uvw=[1,0,0], id1=0, id2=0, J=J2a*np.eye(3,3))]
    couplings += [Coupling(label=f'J2b', n_uvw=[0,1,0], id1=0, id2=0, J=J2b*np.eye(3,3))]

    couplings += [
        Coupling(label=f'J2d1', n_uvw=[0,0,0], id1=0, id2=3, J=J2d*np.eye(3,3)),
        Coupling(label=f'J2d2', n_uvw=[-1,0,0], id1=0, id2=3, J=J2d*np.eye(3,3))
    ]

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
                             coupling_colors={'J1c': 'Orange', 'J1ab':'Gray', 'J2a':'Green', 'J2b':'Red', 'J2d':'Blue'})
        plot_structure(sw, engine='vispy', plot_options=plot_opts)
    
    return sw

def plot_dispersions(sw_params) -> Figure:
    Npath = 201
    sw = load_system(sw_params, show_struct=False)

    mosaic = [ ['0k1', '0k1_in'], ['01l', '01l_in'] ]
    layout = dict(width_ratios=[2,1])
    fig, axs = plt.subplot_mosaic(mosaic=mosaic, figsize=(8,6), tight_layout=True, gridspec_kw=layout)


    print('Calculate excitations `0k1`...')
    qPath = sw.crystal.make_qPath(main_qs=[[0, 0.5, 1], [0,2,1]], Nqs=[Npath])
    # qPath = sw.crystal.make_qPath(main_qs=[[0,0,0.5], [0,0,2], [1,0,1], [1,0,-1]], Nqs=[5,5,5])
    # qPath = sw.crystal.make_qPath(main_qs=[[1,0,1], [1,0,-1]], Nqs=[51])
    
    omega1 = sw.calculate_excitations(qPath=qPath, silent=True)
    sw.plot_dispersion(ax=axs['0k1'], xaxis='k', plot_kwargs=dict(color='gray', alpha=0.5))
    sw.plot_dispersion(ax=axs['0k1_in'], xaxis='k', plot_kwargs=dict(color='gray', alpha=0.5))

    print('Calculate excitations `01l`...')
    qPath = sw.crystal.make_qPath(main_qs=[[0,1,0.5], [0,1,2]], Nqs=[Npath])
    # qPath = sw.crystal.make_qPath(main_qs=[[0,0,0.5], [0,0,2], [1,0,1], [1,0,-1]], Nqs=[5,5,5])
    # qPath = sw.crystal.make_qPath(main_qs=[[1,0,1], [1,0,-1]], Nqs=[51])
    
    omega2 = sw.calculate_excitations(qPath=qPath, silent=True)
    sw.plot_dispersion(ax=axs['01l'], xaxis='l', plot_kwargs=dict(color='gray', alpha=0.5))
    sw.plot_dispersion(ax=axs['01l_in'], xaxis='l', plot_kwargs=dict(color='gray', alpha=0.5))

    # Plot experimental results
    marker_style = dict(fmt='s', markersize=5, capsize=4, markeredgecolor='black')
    axs['0k1'].errorbar(data_0K1[:,1], data_0K1[:,6], yerr=data_0K1[:,7], color='red', **marker_style)
    axs['0k1_in'].errorbar(data_0K1[:,1], data_0K1[:,6], yerr=data_0K1[:,7], color='red', **marker_style)
    axs['01l'].errorbar(data_01L[:,2], data_01L[:,6], yerr=data_01L[:,7], color='orange', **marker_style)
    axs['01l_in'].errorbar(data_01L[:,2], data_01L[:,6], yerr=data_01L[:,7], color='orange', **marker_style)

    for ax_name, ax in axs.items():
        if ax_name.endswith('_in'):
            ax.set_xlim(0.92, 1.08)
            ax.set_ylim(0, 20)
        else:
            ax.set_ylim(0, 1.05*np.max(np.concatenate((omega1, omega2))))

    # print('Calculate ground state energy')
    # qz = np.linspace([0,0,-1], [0,0,1], 51)
    # qx = np.linspace([-1,0,0], [1,0,0], 51)
    # E0z = [sw.calculate_ground_state(q) for q in qz]
    # E0x = [sw.calculate_ground_state(q) for q in qx]
    # axs[1].set_ylabel('E')
    # axs[1].set_xlabel('k')
    # axs[1].scatter(qz[:,2], E0z, label='kz')
    # axs[1].scatter(qx[:,0], E0x, label='kx')
    # axs[1].legend()

    return fig
def plot_spectrum(sw_params, plot_type='dispersion_scaled') -> Figure:
    Npath = 121
    sw = load_system(sw_params, show_struct=False)

    mosaic = [ ['0k1', '0k1_in'], ['01l', '01l_in'], ['h11', 'h11_in'] ]
    layout = dict(width_ratios=[2,1])
    fig, axs = plt.subplot_mosaic(mosaic=mosaic, figsize=(8,9), tight_layout=True, gridspec_kw=layout)


    print('Calculate spectrum `0k1`...')
    axs['0k1'].set_title("0k1 scan")
    qPath = sw.crystal.make_qPath(main_qs=[[0, 0, 1], [0,2,1]], Nqs=[Npath])
    # qPath = sw.crystal.make_qPath(main_qs=[[0,0,0.5], [0,0,2], [1,0,1], [1,0,-1]], Nqs=[5,5,5])
    # qPath = sw.crystal.make_qPath(main_qs=[[1,0,1], [1,0,-1]], Nqs=[51])
    
    omega1, _ = sw.calculate_spectrum(qPath=qPath, silent=True)
    sw.plot_dispersion(ax=axs['0k1'], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))
    sw.plot_dispersion(ax=axs['0k1_in'], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))

    print('Calculate spectrum `01l`...')
    axs['01l'].set_title("01L scan")
    qPath = sw.crystal.make_qPath(main_qs=[[0,1,0], [0,1,2]], Nqs=[Npath])
    # qPath = sw.crystal.make_qPath(main_qs=[[0,0,0.5], [0,0,2], [1,0,1], [1,0,-1]], Nqs=[5,5,5])
    # qPath = sw.crystal.make_qPath(main_qs=[[1,0,1], [1,0,-1]], Nqs=[51])
    
    omega2, _ = sw.calculate_spectrum(qPath=qPath, silent=True)
    sw.plot_dispersion(ax=axs['01l'], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))
    sw.plot_dispersion(ax=axs['01l_in'], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))

    print('Calculate spectrum `h11`...')
    axs['h11'].set_title("H10 scan")
    # qPath = sw.crystal.make_qPath(main_qs=[[-0.5, 1, 0], [1,1,0]], Nqs=[Npath])
    qPath = sw.crystal.make_qPath(main_qs=[[0,1, 0], [2,1,0]], Nqs=[Npath])
    # qPath = sw.crystal.make_qPath(main_qs=[[0,0,0.5], [0,4,4.5]], Nqs=[Npath])
    
    omega3, _ = sw.calculate_spectrum(qPath=qPath, silent=True)
    sw.plot_dispersion(ax=axs['h11'], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))
    sw.plot_dispersion(ax=axs['h11_in'], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))

  

    # Plot experimental results
    marker_style = dict(fmt='s', markersize=5, capsize=4, markeredgecolor='black')
    axs['0k1'].errorbar(data_0K1[:,1], data_0K1[:,6], yerr=data_0K1[:,7], color='red', **marker_style)
    axs['0k1_in'].errorbar(data_0K1[:,1], data_0K1[:,6], yerr=data_0K1[:,7], color='red', **marker_style)
    axs['01l'].errorbar(data_01L[:,2], data_01L[:,6], yerr=data_01L[:,7], color='orange', **marker_style)
    axs['01l_in'].errorbar(data_01L[:,2], data_01L[:,6], yerr=data_01L[:,7], color='orange', **marker_style)

    for ax_name, ax in axs.items():
        if ax_name.endswith('h11_in'):
            # ax.set_xlim(-0.08, 0.08)
            ax.set_xlim(0.92, 1.08)
            ax.set_ylim(0, 20)
        elif ax_name.endswith('_in'):
            ax.set_xlim(0.92, 1.08)
            ax.set_ylim(0, 20)
        else:
            ax.set_ylim(0, 1.05*np.max(np.concatenate((omega1, omega2, omega3))))
            # pass


    # print('Calculate ground state energy')
    # qz = np.linspace([0,0,-1], [0,0,1], 51)
    # qx = np.linspace([-1,0,0], [1,0,0], 51)
    # E0z = [sw.calculate_ground_state(q) for q in qz]
    # E0x = [sw.calculate_ground_state(q) for q in qx]
    # axs[1].set_ylabel('E')
    # axs[1].set_xlabel('k')
    # axs[1].scatter(qz[:,2], E0z, label='kz')
    # axs[1].scatter(qx[:,0], E0x, label='kx')
    # axs[1].legend()

    return fig

def lfo_residuals(parameters: Parameters):
    """Calculate the residuals between the model and data of LuFeO3 measured at EIGER.
    
    Returns
    -------
    residuals
    """
    residuals = []
    weights = []

    lfo_sw = load_system(parameters)

    
    for dataline in data_0K1:
        k = dataline[1]
        E_exp = dataline[6]
        E_err = dataline[7]

        E_theo = lfo_sw.calculate_excitations([[0,k,1]])[0]

        it = np.argmin( np.abs(E_theo-E_exp) )
        residuals.append( (E_theo-E_exp)[it] )
        weights.append(1/E_err)

    for dataline in data_01L:
        l = dataline[2]
        E_exp = dataline[6]
        E_err = dataline[7]

        E_theo = lfo_sw.calculate_excitations([[0,1,l]])[0]

        # There is a lot of energies in E_theo
        # SpinW limits to certain range, but I will try to take the closes lying point
        it = np.argmin( np.abs(E_theo-E_exp) )
        residuals.append( (E_theo-E_exp)[it] )
        weights.append(1/E_err)



    residuals = np.array(residuals, dtype=float)

    weights = np.array(weights, dtype=float)
    residuals_w = residuals*weights

    chi2 = np.sum(np.square(residuals_w))
    print(f'chi2={chi2}')
    for p in parameters:
        print('\t', parameters[p])

    return residuals_w

def fit_lfo(p0: Parameters):
    """Run the fitting procedure with starting parameters `p0`.
    Print the result.
    
    Returns
    -------
    Fitted parameters
    """
    print('Fitting LuFeO3 data')
    fit_result = lmfit.minimize(lfo_residuals, p0, method='leastsq')

    print(fit_report(fit_result))
    return fit_result

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
    models = dict()
    # Simple J1-J2 model no more
    # Unstable, as it scales J1 an J2 up
    lfo_params = Parameters()
    lfo_params.add(name='Ka',  value=-0.013, vary=True)
    lfo_params.add(name='Kc',  value=0, vary=False)
    lfo_params.add(name='D1',  value=0.0, vary=False)
    lfo_params.add(name='D2',  value=0.0, vary=False)
    lfo_params.add(name='J1',  value=5.66, vary=True)
    lfo_params.add(name='J2',  value=0.21, vary=True)
    lfo_params.add(name='J1ab',  expr="J1")
    lfo_params.add(name='J1c',  expr="J1")
    lfo_params.add(name='J2ab', expr="J2")
    lfo_params.add(name='J2d',  expr="J2")
    models['J12'] = lfo_params

    # J1 J2 all split
    model2 = Parameters()
    model2.add(name='Ka',  value=-0.1, vary=True)
    model2.add(name='Kc',  value=0, vary=False)
    model2.add(name='D1',  value=0.074, vary=False)
    model2.add(name='D2',  value=0.028, vary=False)
    model2.add(name='J1ab',  value=4.8)
    model2.add(name='J1c',  value=4.2)
    model2.add(name='J2ab', value=0.2)
    model2.add(name='J2d',  value=0.1)
    models['Jall'] = lfo_params

    # MS parameters
    lfo_params = Parameters()
    lfo_params.add(name='Ka',  value=-0.036, vary=True)
    lfo_params.add(name='Kc',  value=0.0, vary=False)
    lfo_params.add(name='Dab_x',  value=0.13*0.554, vary=False)
    lfo_params.add(name='Dab_y',  value=0.13*0.553, vary=False)
    lfo_params.add(name='Dab_z',  value=0.13*0.623, vary=False)
    lfo_params.add(name='Dc_x',  value=0.158*0.191, vary=False)
    lfo_params.add(name='Dc_y',  value=0.158*0.982, vary=False)
    lfo_params.add(name='Dc_z',  value=0, vary=False)
    lfo_params.add(name='J2',  value=0.71, vary=True)
    lfo_params.add(name='J1ab',  value=7.38, vary=True)
    lfo_params.add(name='J1c',  value=6.39, vary=True)
    lfo_params.add(name='J2a', expr="J2")
    lfo_params.add(name='J2b', expr="J2")
    lfo_params.add(name='J2d',  expr="J2")
    lfo_params.add(name='Fz',  value=np.sin(np.radians(1.2)), vary=True)
    models['MS'] = lfo_params

    # PArk et al 2018
    lfo_params = Parameters()
    lfo_params.add(name='Ka',  value=-0.0124, vary=True)
    lfo_params.add(name='Kc',  value=-0.0037, vary=False)
    lfo_params.add(name='Dab_x',  value=0.13*0.554, vary=False)
    lfo_params.add(name='Dab_y',  value=0.13*0.553, vary=False)
    lfo_params.add(name='Dab_z',  value=0.13*0.623, vary=False)
    lfo_params.add(name='Dc_x',  value=0.158*0.191, vary=False)
    lfo_params.add(name='Dc_y',  value=0.158*0.982, vary=False)
    lfo_params.add(name='Dc_z',  value=0, vary=False)
    lfo_params.add(name='J1ab',  value=5.47, vary=True)
    lfo_params.add(name='J1c',  value=5.47, vary=True)
    lfo_params.add(name='Jp',  value=0.21, vary=True)
    lfo_params.add(name='J2a', expr="Jp")
    lfo_params.add(name='J2b', expr="Jp")
    lfo_params.add(name='J2d',  expr="Jp")
    lfo_params.add(name='Fz',  value=-np.sin(np.radians(0.5)), vary=True)
    models['Park'] = lfo_params

    # DB for Flipper poster
    lfo_params = Parameters()
    lfo_params.add(name='Ka',  value=-0.03, vary=True)  # Ka
    lfo_params.add(name='Kc',  value=0, vary=False)   # Kc
    lfo_params.add(name='J1c',  value=7.04, vary=True)   # J1
    lfo_params.add(name='J1ab',  value=8.92, vary=True)  # J2
    lfo_params.add(name='J2b', value=1.54, vary=True)   # J5
    lfo_params.add(name='J2a', expr='J2b')    # J3
    lfo_params.add(name='J2a', value=1.54)    # J3
    lfo_params.add(name='J2d', value=1.0, vary=True) # J4
    lfo_params.add(name='Dab_x', value=0, vary=False)
    lfo_params.add(name='Dab_y',  value=0.009, vary=False)   # Dab
    lfo_params.add(name='Dab_z',  value=0.024, vary=False)  # Dc
    lfo_params.add(name='Dc_x',  value=0, vary=False)
    lfo_params.add(name='Dc_y',  value=0, vary=False)
    lfo_params.add(name='Dc_z',  value=0, vary=False)
    lfo_params.add(name='Fz',  value=0.015, vary=False)
    models['DB'] = lfo_params

    # TAIPAN preparations
    lfo_params = Parameters()
    lfo_params.add(name='Ka',  value=-0.03, vary=True)  # Ka
    lfo_params.add(name='Kc',  value=0, vary=False)   # Kc
    lfo_params.add(name='J1c',  value=7.04, vary=True)   # J1
    lfo_params.add(name='J1ab',  value=8.92, vary=True)  # J2
    lfo_params.add(name='J2b', value=1.54, vary=True)   # J5
    lfo_params.add(name='J2a', value=1.7)    # J3
    lfo_params.add(name='J2d', value=1.0, vary=True) # J4
    lfo_params.add(name='Dab_x', value=0, vary=False)
    lfo_params.add(name='Dab_y',  value=0.009, vary=False)   # Dab
    lfo_params.add(name='Dab_z',  value=0.024, vary=False)  # Dc
    lfo_params.add(name='Dc_x',  value=0, vary=False)
    lfo_params.add(name='Dc_y',  value=0, vary=False)
    lfo_params.add(name='Dc_z',  value=0, vary=False)
    lfo_params.add(name='Fz',  value=0.015, vary=False)
    models['TAIPAN'] = lfo_params

    return models[model_name]


if __name__ == '__main__':
    # Define main parameters
    fit = False

    lfo_params = load_lfo_parameters('TAIPAN')
    sw = load_system(lfo_params, show_struct=False, silent=False)





    if fit:
        fit_result = fit_lfo(p0 = lfo_params)
        fig = plot_spectrum(fit_result.params)
    else:
        fig = plot_spectrum(lfo_params, plot_type='spectral_weight')
        # fig = plot_dispersions(lfo_params)

    ### TESTS
    def calc_exc(Q):
        print(f"Q = {Q}")
        print(sw.calculate_spectrum(qPath=[Q], silent=True))

    calc_exc([0,1,1])
    calc_exc([0,1,0])
    calc_exc([0.49,1,0])
    calc_exc([0.5,1,0])
    calc_exc([0.51,1,0])


    fig.savefig(rf'C:\Users\Stekiel\Desktop\Offline-plots\spinwaves-LuFeO3.png', dpi=400)
    # fig.savefig(rf'C:\Users\Stekiel\Desktop\Offline-plots\spinwaves-LuFeO3.pdf')

