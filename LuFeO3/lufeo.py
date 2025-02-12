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

# DB from EIGER ??.??.????
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

# DB from EIGER ??.??.????
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


# DB from TAIPAN 05.02.2025
#               QH	QK	       QL	ENlim1	ENlim2	 I1	    EN1	        sigma1	       I2    EN2	sigma2
DB_H0N = '''0.615	0.000000	1	0.0000	90.0000	1.0000	55.000000	0.000000	0	0	0
            0.65933	0.000000	1	0.0000	90.0000	1.0000	50.000000	0.000000	0	0	0
            0.728	0.000000	1	0.0000	90.0000	1.0000	45.000000	0.000000	0	0	0
            0.76	0.000000	1	0.0000	90.0000	1.0000	40.000000	0.000000	0	0	0
            0.8	    0.000000	1	0.0000	90.0000	1.0000	35.000000	0.000000	0	0	0
            0.85234	0.000000	1	0.0000	90.0000	1.0000	30.000000	0.000000	0	0	0
            0.88	0.000000	1	0.0000	90.0000	1.0000	25.000000	0.000000	0	0	0
            0.915	0.000000	1	0.0000	90.0000	1.0000	20.000000	0.000000	0	0	0
            0.938	0.000000	1	0.0000	90.0000	1.0000	15.000000	0.000000	0	0	0
            0.96844	0.000000	1	0.0000	90.0000	1.0000	10.000000	0.000000	0	0	0
            1.00383	0.000000	1	0.0000	90.0000	1.0000	5.000000	0.000000	0	0	0
            1.41221	0.000000	1	0.0000	90.0000	1.0000	55.000000	0.000000	0	0	0
            1.33037	0.000000	1	0.0000	90.0000	1.0000	50.000000	0.000000	0	0	0
            1.272	0.000000	1	0.0000	90.0000	1.0000	45.000000	0.000000	0	0	0
            1.24	0.000000	1	0.0000	90.0000	1.0000	40.000000	0.000000	0	0	0
            1.2	    0.000000	1	0.0000	90.0000	1.0000	35.000000	0.000000	0	0	0
            1.15689	0.000000	1	0.0000	90.0000	1.0000	30.000000	0.000000	0	0	0
            1.12	0.000000	1	0.0000	90.0000	1.0000	25.000000	0.000000	0	0	0
            1.085	0.000000	1	0.0000	90.0000	1.0000	20.000000	0.000000	0	0	0
            1.065	0.000000	1	0.0000	90.0000	1.0000	15.000000	0.000000	0	0	0
            1.03334	0.000000	1	0.0000	90.0000	1.0000	10.000000	0.000000	0	0	0'''

# data_H0N = np.array([line.strip().split() for line in MS_H0N.split('\n')], dtype=float)
data_H0N = np.array([line.strip().split() for line in DB_H0N.split('\n')], dtype=float)


# DB from TAIPAN 06.02.2025
# WARNING: Ql in DB_10l[-3] was replaced from 4.23 to 5.77 to fit into the q-region
#               QH	QK	        QL	ENlim1	ENlim2	I1	EN1	        sigma1	       I2	EN2	sigma2
DB_10L = '''   1	0.000000	5.79306	0.0000	90.0000	1.0000	55.000000	0.017880	0	0	0
                1	0.000000	4.58169	0.0000	90.0000	1.0000	40.000000	0.002150	0	0	0
                1	0.000000	2.67717	0.0000	90.0000	1.0000	35.000000	0.006380	0	0	0
                1	0.000000	2.74424	0.0000	90.0000	1.0000	30.000000	0.005700	0	0	0
                1	0.000000	0.88146	0.0000	90.0000	1.0000	15.000000	0.001860	0	0	0
                1	0.000000	0.91448	0.0000	90.0000	1.0000	10.000000	0.002040	0	0	0
                1	0.000000	0.97977	0.0000	90.0000	1.0000	5.000000	0.000890	0	0	0
                1	0.000000	1.03129	0.0000	90.0000	1.0000	10.000000	0.002100	0	0	0
                1	0.000000	1.07061	0.0000	90.0000	1.0000	15.000000	0.001920	0	0	0
                1	0.000000	3.21266	0.0000	90.0000	1.0000	30.000000	0.005290	0	0	0
                1	0.000000	3.24319	0.0000	90.0000	1.0000	35.000000	0.006060	0	0	0
                1	0.000000	3.31945	0.0000	90.0000	1.0000	37.500000	0.004360	0	0	0
                1	0.000000	5.45749	0.0000	90.0000	1.0000	40.000000	0.001950	0	0	0
                1	0.000000	3.46981	0.0000	90.0000	1.0000	45.000000	0.004930	0	0	0
                1	0.000000	3.54244	0.0000	90.0000	1.0000	50.000000	0.005140	0	0	0
                1	0.000000	5.77	0.0000	90.0000	1.0000	55.000000	0.000000	0	0	0
                1	0.000000	2.8622	0.0000	90.0000	1.0000	20.000000	0.004641	0	0	0
                1	0.000000	3.1028	0.0000	90.0000	1.0000	20.000000	0.004391	0	0	0'''
data_10L = np.array([line.strip().split() for line in DB_10L.split('\n')], dtype=float)


# DB from TAIPAN 06.02.2025
#                   QH	QK	        QL	ENlim1	ENlim2	I1	EN1	        sigma1	       I2	EN2	sigma2
DB_H0mH = '''   1.23593	0.000000	2.76407	0.0000	90.0000	1.0000	45.000000	0.003240	0	0	0
                1.2041	0.000000	2.7959	0.0000	90.0000	1.0000	40.000000	0.003150	0	0	0
                1.1655	0.000000	2.8345	0.0000	90.0000	1.0000	35.000000	0.003620	0	0	0
                1.13899	0.000000	2.86101	0.0000	90.0000	1.0000	30.000000	0.003370	0	0	0
                1.0992	0.000000	2.9008	0.0000	90.0000	1.0000	25.000000	0.004030	0	0	0
                1.07123	0.000000	2.92877	0.0000	90.0000	1.0000	20.000000	0.004800	0	0	0
                1.05215	0.000000	0.94785	0.0000	90.0000	1.0000	15.000000	0.001440	0	0	0
                1.03819	0.000000	0.96181	0.0000	90.0000	1.0000	10.000000	0.001090	0	0	0
                1.00835	0.000000	0.99165	0.0000	90.0000	1.0000	5.000000	0.000460	0	0	0
                0.97726	0.000000	1.02274	0.0000	90.0000	1.0000	10.000000	0.001060	0	0	0
                0.95925	0.000000	1.04075	0.0000	90.0000	1.0000	15.000000	0.001460	0	0	0
                0.93126	0.000000	3.06874	0.0000	90.0000	1.0000	20.000000	0.004490	0	0	0
                0.9	    0.000000	3.1 	0.0000	90.0000	1.0000	25.000000	0.000000	0	0	0
                0.872	0.000000	3.128	0.0000	90.0000	1.0000	30.000000	0.000000	0	0	0
                0.842	0.000000	3.158	0.0000	90.0000	1.0000	35.000000	0.000000	0	0	0
                0.8203	0.000000	3.1797	0.0000	90.0000	1.0000	40.000000	0.003050	0	0	0
                0.78013	0.000000	3.21987	0.0000	90.0000	1.0000	45.000000	0.002810	0	0	0'''
data_H0mH = np.array([line.strip().split() for line in DB_H0mH.split('\n')], dtype=float)


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
    Npath = 2001
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
def plot_spectrum(sw_params, plot_type='dispersion') -> Figure:
    Npath = 151
    sw = load_system(sw_params, show_struct=False)

    mosaic = [ ['0k1', '0k1_in'], ['01l', '01l_in'], ['10l', '10l_in'] ,['h0l', 'h0l_in'] , ['h0mh', 'h0mh_in']]
    layout = dict(width_ratios=[2,1])
    fig, axs = plt.subplot_mosaic(mosaic=mosaic, figsize=(7,12), tight_layout=True, gridspec_kw=layout)


    print('Calculate spectrum `0k1`...')
    axs['0k1'].set_title("(0K1) EIGER")
    qPath = sw.crystal.make_qPath(main_qs=[[0, 0.5, 1], [0,1.5,1]], Nqs=[Npath])    
    omega1, _ = sw.calculate_spectrum(qPath=qPath, silent=True)
    sw.plot_dispersion(ax=axs['0k1'],    xaxis=qPath[:,1], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))
    sw.plot_dispersion(ax=axs['0k1_in'], xaxis=qPath[:,1], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))

    print('Calculate spectrum `01l`...')
    axs['01l'].set_title("(01L) EIGER")
    qPath = sw.crystal.make_qPath(main_qs=[[0,1,0.5], [0,1,2]], Nqs=[Npath])
    omega2, _ = sw.calculate_spectrum(qPath=qPath, silent=True)
    sw.plot_dispersion(ax=axs['01l'],    xaxis=qPath[:,2], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))
    sw.plot_dispersion(ax=axs['01l_in'], xaxis=qPath[:,2], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))

    print('Calculate spectrum `10l`...')
    axs['10l'].set_title("(1 0 L) TAIPAN")
    qPath = sw.crystal.make_qPath(main_qs=[[1,0,0.5], [1,0,2]], Nqs=[Npath])
    omega3, _ = sw.calculate_spectrum(qPath=qPath, silent=True)
    sw.plot_dispersion(ax=axs['10l'],    xaxis=qPath[:,2], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))
    sw.plot_dispersion(ax=axs['10l_in'], xaxis=qPath[:,2], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))

    print('Calculate spectrum `h0n`...')
    axs['h0l'].set_title("(H 0 odd) TAIPAN")
    qPath = sw.crystal.make_qPath(main_qs=[[0.5,0,1], [1.5,0,1]], Nqs=[Npath])
    omega3, _ = sw.calculate_spectrum(qPath=qPath, silent=True)
    sw.plot_dispersion(ax=axs['h0l'],    xaxis=qPath[:,0], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))
    sw.plot_dispersion(ax=axs['h0l_in'], xaxis=qPath[:,0], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))

    print('Calculate spectrum `h0mh`...')
    axs['h0mh'].set_title(f"$(H 0 \\bar{{H}})$ TAIPAN")
    qPath = sw.crystal.make_qPath(main_qs=[[0.5,0,1.5], [1.5,0,0.5]], Nqs=[Npath])
    omega4, _ = sw.calculate_spectrum(qPath=qPath, silent=True)
    sw.plot_dispersion(ax=axs['h0mh'],    xaxis=qPath[:,0], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))
    sw.plot_dispersion(ax=axs['h0mh_in'], xaxis=qPath[:,0], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))
  



    # Plot experimental results
    marker_style = dict(fmt='s', markersize=5, capsize=4, markeredgecolor='black')

    axs['0k1'].errorbar(data_0K1[:,1], data_0K1[:,6], yerr=data_0K1[:,7], color='blue', **marker_style)
    axs['0k1_in'].errorbar(data_0K1[:,1], data_0K1[:,6], yerr=data_0K1[:,7], color='blue', **marker_style)

    axs['01l'].errorbar(data_01L[:,2], data_01L[:,6], yerr=data_01L[:,7], color='green', **marker_style)
    axs['01l_in'].errorbar(data_01L[:,2], data_01L[:,6], yerr=data_01L[:,7], color='green', **marker_style)

    axs['10l'].errorbar(data_10L[:,2]%2, data_10L[:,6], yerr=data_10L[:,7], color='purple', **marker_style)
    axs['10l_in'].errorbar(data_10L[:,2]%2, data_10L[:,6], yerr=data_10L[:,7], color='purple', **marker_style)

    axs['h0l'].errorbar(data_H0N[:,0], data_H0N[:,6], yerr=data_H0N[:,7], color='yellow', **marker_style)
    axs['h0l_in'].errorbar(data_H0N[:,0], data_H0N[:,6], yerr=data_H0N[:,7], color='yellow', **marker_style)

    axs['h0mh'].errorbar(data_H0mH[:,0], data_H0mH[:,6], xerr=data_H0mH[:,7], color='pink', **marker_style)
    axs['h0mh_in'].errorbar(data_H0mH[:,0], data_H0mH[:,6], xerr=data_H0mH[:,7], color='pink', **marker_style)

    for ax_name, ax in axs.items():
        if ax_name.endswith('_in'):
            ax.set_xlim(0.88, 1.12)
            ax.set_ylim(0, 33)
        else:
            ax.set_ylim(0, 1.05*np.max(np.concatenate((omega1, omega2, omega3, omega4))))
            # pass


    return fig

def lfo_residuals(parameters: Parameters):
    """Calculate the residuals between the model and data of LuFeO3.
    
    Returns
    -------
    residuals
    """
    residuals = []
    weights = []

    lfo_sw = load_system(parameters)

    def get_residual(q, E_exp, E_err):
        E_theo, I_theo = lfo_sw.calculate_spectrum([q])

        weights = np.nan_to_num(I_theo, nan=1e+30)*np.power(E_exp-E_theo, -6)
        weights /= np.sum(weights)
        # print(f'{I_theo=}')
        # print(f'{E_exp-E_theo=}')
        # print(f'{weights=}')

        return np.sum(weights*(E_exp-E_theo))
    
    for dataline in data_0K1:
        q = [0, dataline[1], 1]
        E_exp = dataline[6]
        E_err = dataline[7]

        residuals.append( get_residual(q, E_exp, E_err) )
        weights.append(1/E_err)

    for dataline in data_01L:
        q = [0, 1, dataline[2]]
        E_exp = dataline[6]
        E_err = dataline[7]

        residuals.append( get_residual(q, E_exp, E_err) )
        weights.append(1/E_err)

    for dataline in data_10L:
        q = [1, 0, dataline[2]]
        E_exp = dataline[6]
        E_err = dataline[7]

        residuals.append( get_residual(q, E_exp, E_err) )
        weights.append(1/E_err)

    for dataline in data_H0N:
        q = [dataline[0], 0, 1]
        E_exp = dataline[6]
        E_err = dataline[7]

        residuals.append( get_residual(q, E_exp, E_err) )
        weights.append(1/E_err)

    for dataline in data_H0mH:
        q = [dataline[0], 0, 2-dataline[0]]
        E_exp = dataline[6]
        E_err = dataline[7]

        residuals.append( get_residual(q, E_exp, E_err) )
        weights.append(1/E_err)



    residuals = np.array(residuals, dtype=float)
    weights = np.array(weights, dtype=float)

    # residuals_w = residuals*weights

    chi2 = np.sum(np.square(residuals))
    print(f'chi2={chi2}')
    for p in parameters:
        print('\t', parameters[p])

    return residuals

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
    lfo_params.add(name='J2d', value=1.0, vary=True) # J4
    lfo_params.add(name='Dab_x', value=0, vary=False)
    lfo_params.add(name='Dab_y',  value=0.009, vary=False)   # Dab
    lfo_params.add(name='Dab_z',  value=0.024, vary=False)  # Dc
    lfo_params.add(name='Dc_x',  value=0, vary=False)
    lfo_params.add(name='Dc_y',  value=0, vary=False)
    lfo_params.add(name='Dc_z',  value=0, vary=False)
    lfo_params.add(name='Fz',  value=0.015, vary=False)
    models['DB'] = lfo_params

    # TAIPAN with H scans
    lfo_params = Parameters()
    lfo_params.add(name='Ka',  value=-0.25, vary=True)  # Ka
    lfo_params.add(name='Kc',  value=0, vary=False)   # Kc

    lfo_params.add(name='J1c',  value=5.1, vary=True)   # J1
    lfo_params.add(name='J1ab',  value=4.15, vary=True)  # J2
    lfo_params.add(name='J2b', value=-0.2, vary=True)   # J5
    # lfo_params.add(name='J2a', value=0.224, vary=True)    # J3
    lfo_params.add(name='J2a', value=-0.2, expr='J2b')    # J3
    lfo_params.add(name='J2d', value=0.03, vary=True) # J4

    lfo_params.add(name='Dab_x',  value=0.13*0.554, vary=False)
    lfo_params.add(name='Dab_y',  value=0.13*0.553, vary=False)
    lfo_params.add(name='Dab_z',  value=0.13*0.623, vary=False)
    lfo_params.add(name='Dc_x',  value=0.158*0.191, vary=False)
    lfo_params.add(name='Dc_y',  value=0.158*0.982, vary=False)
    lfo_params.add(name='Dc_z',  value=0, vary=False)
    lfo_params.add(name='Fz',  value=0.08*0, vary=False)
    models['TAIPAN'] = lfo_params

    return models[model_name]


if __name__ == '__main__':
    PATH = fr'C:\Users\Stekiel\Documents\GitHub\spinwaves\LuFeO3'
    # Define main parameters
    fit = True

    lfo_params = load_lfo_parameters('TAIPAN')
    sw = load_system(lfo_params, show_struct=False, silent=False)


    if fit:
        fit_result = fit_lfo(p0 = lfo_params)
        lfo_params = fit_result.params

    fig = plot_spectrum(lfo_params, plot_type='spectral_weight')
    fig.savefig(PATH+'\spinwaves-LuFeO3-Sqw.png', dpi=400)
    fig = plot_spectrum(lfo_params, plot_type='dispersion')
    fig.savefig(PATH+'\spinwaves-LuFeO3-Eq.png', dpi=400)



    ### TESTS

    # fig = plot_debug_disp(lfo_params)

    # def calc_exc(Q):
    #     print(f"Q = {Q}")
    #     print(sw.calculate_spectrum(qPath=[Q], silent=True))

    # calc_exc([0,1,1])
    # calc_exc([0,1,0])
    # calc_exc([0.49,1,0])
    # calc_exc([0.5,1,0])
    # calc_exc([0.51,1,0])

    # fig.savefig(rf'C:\Users\Stekiel\Desktop\Offline-plots\spinwaves-LuFeO3.pdf')

