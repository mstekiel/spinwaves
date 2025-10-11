# -*- coding: utf-8 -*-
r"""Tests lattice math

"""

import numpy as np
import matplotlib.pyplot as plt

from spinwaves import Crystal, MSG, Atom, Coupling, SpinW
from spinwaves.plotting import plot_structure   # this takes some serious time

from pathlib import Path
PATH_PLOTS = Path(r'C:\Users\Stekiel\Documents\GitHub\spinwaves\tests\Various_comparisons')

### Altermagnets
def make_comparison():
    """Comparison picture between ferro- anti- and altermagnetic square lattices."""

    ### SETUP
    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(1,0,0)
    }


    P4p = MSG.from_xyz_strings(generators=[' -y, x, z, -1']) # just 4z'
    atoms = [Atom(label='Mn', r=(1/2,0,0),   m=(0,0,1), s=5/2)] 
    crystal_A = Crystal(lattice_parameters=[4, 4, 5, 90,90,90], MSG=P4p, atoms=atoms)

    atoms = [Atom(label='Mn', r=(0,0,0),   m=(0,0,1), s=5/2)] 
    crystal_F = Crystal(lattice_parameters=[4, 4, 5, 90,90,90], MSG=P4p, atoms=atoms)

    v1, v2 = 10, -1

    J1 = v1+2*v2
    J2 = v2
    couplings = [
        Coupling(label='J1', id1=0, id2=1, n_uvw=[0,0,0], J=J1*np.eye(3,3)),
        Coupling(label='J2', id1=0, id2=0, n_uvw=[1,0,0], J=J2*np.eye(3,3)),
        Coupling(label='J2', id1=0, id2=0, n_uvw=[0,1,0], J=J2*np.eye(3,3))
    ]
    sw_AFM = SpinW(crystal=crystal_A, magnetic_modulation=magnetic_modulation, couplings=couplings)

    J1 = v1+v2
    J2 = v2
    couplings = [
        Coupling(label='J1', id1=0, id2=1, n_uvw=[0,0,0], J=J1*np.eye(3,3)),
        Coupling(label='J2', id1=0, id2=0, n_uvw=[1,0,0], J=J2*np.eye(3,3)),
    ]
    sw_AM = SpinW(crystal=crystal_A, magnetic_modulation=magnetic_modulation, couplings=couplings)

    J1 = -v1/2
    couplings = [Coupling(label='J1', id1=0, id2=0, n_uvw=[1,0,0], J=J1*np.eye(3,3))]
    sw_FM = SpinW(crystal=crystal_F, magnetic_modulation=magnetic_modulation, couplings=couplings)
    # for cpl in sw.couplings_all:
    #     print(cpl)

    # show_struct = False
    # if show_struct:
    #     plot_opts = dict(boundaries=([-1.1, 2.1],[-1.1,2.1],[-0.1,0.1]), 
    #                      coupling_colors={'J1': 'Green', 'J2':'Red', 'Jaa':'Gray', 'J2b':'Red', 'J2d':'Blue'},
    #                      spin_scale=2)
    #     plot_structure(sw, engine='vispy', plot_options=plot_opts)




    ### PLOTTING
    margins = dict(width_ratios=[1,1,1,0.1])
    fig, axes = plt.subplots(figsize=(6,6), nrows=3, ncols=4, tight_layout=True, gridspec_kw=margins)

    def make_plot_col(sw, axes, plt_opts=dict()):
        N = 200
        # Qpath, Qinc = sw.crystal.make_qPath([[0,0,0], [1/2,0,0], [1/2,-1/2,0], [0,0,0]], N, return_Qinc=True)
        Qpath, Qinc = sw.crystal.make_qPath([[0,1/2,0], [0,0,0], [1/2,0,0]], N, return_Qinc=True)

        excitations = sw.calculate_excitations(Qhkl = Qpath)

        xticks = Qinc[::N]
        xticklabels = ['$k_x$', '0', '$k_y$']
        # xticklabels = ['$\Gamma$', 'M', 'A', '$\Gamma$']

        ylim = (0, excitations.E.max()*1.1)
        titles = ['dispersions', f'$Re(S_{{xx}}+S_{{yy}}+S_{{zz}})$', '$Im(S_{{xy}}-S_{{yx}})$']

        Erange = np.linspace(*ylim, 500)

        for n,ax in enumerate(axes):
            ax.set(xticks=xticks, xticklabels=xticklabels, 
                   ylim=ylim, yticks=[])#), title=titles[n])
        
        axes[0].plot(Qinc, excitations.E, label='branch')

        spectrum = sw.calculate_spectrum(Erange, 11, spectral_weight=(excitations.Sxx+excitations.Syy).real)
        cm_Sii = axes[1].pcolormesh(Qinc, Erange, spectrum, cmap='afmhot_r')
        # cbar = fig.colorbar(pcm, ax=axes[1], orientation='vertical', extend='max', label='intensity (a.u.)')

        spectrum = sw.calculate_spectrum(Erange, 11, spectral_weight=-(excitations.Sxy-excitations.Syx).imag)
        axes[2].plot(Qinc, excitations.E, c='black', lw=1, zorder=10)
        vv = plt_opts['vv']
        cm_Sch = axes[2].pcolormesh(Qinc, Erange, spectrum, cmap='RdBu_r', vmin=-vv, vmax=vv)

        # axes[0].legend()
        return cm_Sii, cm_Sch
    

    plt_opts = dict(vv=0.2)
    make_plot_col(sw_FM,  axes=axes[:,0], plt_opts=dict(vv=0.4))
    make_plot_col(sw_AFM, axes=axes[:,1], plt_opts=dict(vv=0.2))
    cm_Sii, cm_Sch = make_plot_col(sw_AM,  axes=axes[:,2], plt_opts=dict(vv=0.15))

    axes[0,3].set_axis_off()
    cbar = fig.colorbar(cm_Sii, cax=axes[1,3], orientation='vertical', extend='max', label='intensity (a.u.)')
    cbar = fig.colorbar(cm_Sch, cax=axes[2,3], orientation='vertical', extend='both', label='chirality',
                        ticks=[-0.15, 0.15])
    cbar.set_ticklabels(['$\circlearrowright$', '$\circlearrowleft$'])

    fig.savefig(f'{PATH_PLOTS}\compare_FM-AFM-AlterM.png', dpi=400)
    
    return


def reproduce_CrSb_Biniskos():
    """Spin wave spectrum of the CrSb altermagnet measured with RIXS"""

    ### SETUP
    P63ommc = MSG.from_xyz_strings(generators=[
        '-y, x-y,z, +1',     # 3z
        '-x,-y, z+1/2, -1',  # 2z
        ' y, x,-z, -1',      # 2_110
        '-x,-y,-z, +1',    # -1
    ])
    atoms = [
        Atom(label='Cr', r=(0,0,0),   m=(0,0,1), s=1),
        # Atom(label='Sb', r=(1/3, 2/3, 1/4)),
        ] 
    crystal = Crystal(lattice_parameters=[4.124,4.124,5.459, 90,90,120], MSG=P63ommc, atoms=atoms)

    print(crystal)
    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,0,1)
    }

    J1AB, J1AA = 50, -20
    J7AA = -3
    couplings = [
        Coupling(label='J1AB', id1=0, id2=1, n_uvw=[0,0,0], J=J1AB*np.eye(3,3)),
        Coupling(label='J1AA', id1=0, id2=0, n_uvw=[1,0,0], J=J1AA*np.eye(3,3)),
        Coupling(label='J7AA', id1=0, id2=0, n_uvw=[1,-1,1], J=J7AA*np.eye(3,3)),
    ]

    ### CALCULATIONS
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=couplings)

    print(sw.couplings_all)

    show_struct = False
    if show_struct:
    #     plot_opts = dict(boundaries=([-0.5, 1.5],[-0.5,1.5],[-0.5,1]), coupling_colors={'J1a2': 'Cyan'})
        plot_opts = dict(boundaries=([-1.1, 1.1],[-1.1,1.1],[-1.1,1.1]), 
                         coupling_colors={'J1': 'Green', 'Jdd':'Black', 'Jaa':'Gray', 'J2b':'Red', 'J2d':'Blue'},
                         spin_scale=2)
        plot_structure(sw, engine='vispy', plot_options=plot_opts)


    N = 200
    Qpath, Qinc = sw.crystal.make_qPath([[0,0,0], [1/2,0,0], [0,0,1/2], [0,0,0]], [N]*3, return_Qinc=True)

    excitations = sw.calculate_excitations(Qhkl = Qpath)


    ### PLOTTING
    fig, axes = plt.subplots(figsize=(4, 9), nrows=3, tight_layout=True)

    fig.suptitle('CrSb Biniskos et al.')

    xticks = [Qinc[id] for id in [0, N-1, 2*N-1, 3*N-1]]
    xticklabels = ['$\Gamma$', 'M', 'A', '$\Gamma$']

    ylim = (0, 300)
    titles = ['dispersions', f'$Re(S_{{xx}}+S_{{yy}}+S_{{zz}})$', '$Im(S_{{xy}}-S_{{yx}})$']

    Erange = np.linspace(*ylim, 500)

    for n,ax in enumerate(axes):
        ax.set(ylabel='energy', xticks=xticks, xticklabels=xticklabels,
               xlabel='Momentum', ylim=ylim, title=titles[n])
    
    axes[0].plot(Qinc, excitations.E)

    spectrum = sw.calculate_spectrum(Erange, 15, spectral_weight=excitations.Sxx.real)
    pcm = axes[1].pcolormesh(Qinc, Erange, spectrum, cmap='afmhot_r')
    cbar = fig.colorbar(pcm, ax=axes[1], orientation='vertical', extend='max', label='intensity (a.u.)')

    spectrum = sw.calculate_spectrum(Erange, 20, spectral_weight=(excitations.Sxy-excitations.Syx).imag)
    axes[2].plot(Qinc, excitations.E, c='black', lw=0.5, zorder=10)
    pcm = axes[2].pcolormesh(Qinc, Erange, spectrum, cmap='RdBu')
    cbar = fig.colorbar(pcm, ax=axes[2], orientation='vertical', extend='max', label='intensity (a.u.)')


    fig.savefig(f'{PATH_PLOTS}\CrSb_Biniskos2025.png')
    
    return

def reproduce_MnF2_Faure():
    """Spin wave spectrum of the CrSb altermagnet measured with RIXS"""

    ### SETUP
    msg = MSG.from_xyz_strings(generators=[
        '-x,-y,z, +1',     # 2z
        ' -y+1/2,x+1/2,z+1/2, -1',  # 4z
        ' -x+1/2,y+1/2,-z+1/2, +1',      # 2_010
        '-x,-y,-z, +1',    # -1
    ])
    atoms = [
        Atom(label='Mn', r=(0,0,0),   m=(0,0,1), s=5/2),
        # Atom(label='F', r=(1/4, 1/4, 0), m=(0,0,0.1), s=0.01),
        ] 
    crystal = Crystal(lattice_parameters=[4.87, 4.87, 3.31, 90,90,90], MSG=msg, atoms=atoms)

    print(crystal)
    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,0,1)
    }

    J1, J2, J3 = -0.075, 0.287, -0.012
    J4, J5, J6 = -0.001, 0.008,  0.001
    J7a, J7b = -0.006, -0.002
    couplings = [
        Coupling(label='J1', id1=0, id2=0, n_uvw=[0,0,1], J=J1*np.eye(3,3)),
        Coupling(label='J2', id1=0, id2=1, n_uvw=[0,0,0], J=J2*np.eye(3,3)),
        Coupling(label='J3', id1=0, id2=0, n_uvw=[1,0,0], J=J3*np.eye(3,3)),
        Coupling(label='J4', id1=0, id2=0, n_uvw=[1,0,1], J=J4*np.eye(3,3)),
        Coupling(label='J5', id1=0, id2=1, n_uvw=[0,0,1], J=J5*np.eye(3,3)),
        Coupling(label='J6', id1=0, id2=0, n_uvw=[0,0,2], J=J6*np.eye(3,3)),
        Coupling(label='J7a', id1=0, id2=0, n_uvw=[1, 1,0], J=J7a*np.eye(3,3)),
        Coupling(label='J7b', id1=0, id2=0, n_uvw=[1,-1,0], J=J7b*np.eye(3,3)),
    ]

    ### CALCULATIONS
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=couplings)

    for cpl in sw.couplings_all:
        print(cpl)

    show_struct = False
    if show_struct:
    #     plot_opts = dict(boundaries=([-0.5, 1.5],[-0.5,1.5],[-0.5,1]), coupling_colors={'J1a2': 'Cyan'})
        plot_opts = dict(boundaries=([-1.1, 1.1],[-1.1,1.1],[-1.1,1.1]), 
                         coupling_colors={'J1': 'Green', 'Jdd':'Black', 'Jaa':'Gray', 'J2b':'Red', 'J2d':'Blue'},
                         spin_scale=2)
        plot_structure(sw, engine='vispy', plot_options=plot_opts)


    main_Qs = [
        [0,0,1/2], 
        [0,0,1], 
        [1/2,1/2,1], 
        [1/2,1/2,1/2], 
        [1,1,1/2], 
        [1,1,0], 
        [1/2,1/2,0]
        ]
    N = 200
    Qpath, Qinc = sw.crystal.make_qPath(main_Qs, N, return_Qinc=True)

    excitations = sw.calculate_excitations(Qhkl = Qpath)


    ### PLOTTING
    fig, axes = plt.subplots(figsize=(4, 9), nrows=3, tight_layout=True)

    fig.suptitle('MnF2 Faure et al.')

    xticks = Qinc[::N]
    xticklabels = ['Z', '$\Gamma$', 'M', 'A', 'Z', '$\Gamma$', 'M']

    ylim = (0, 10)
    titles = ['dispersions', f'$Re(S_{{xx}}+S_{{yy}}+S_{{zz}})$', '$Im(S_{{xy}}-S_{{yx}})$']

    Erange = np.linspace(*ylim, 500)

    for n,ax in enumerate(axes):
        ax.set(ylabel='energy', xticks=xticks, xticklabels=xticklabels,
               xlabel='Momentum', ylim=ylim, title=titles[n])
    
    axes[0].plot(Qinc, excitations.E)

    spectrum = sw.calculate_spectrum(Erange, 1)
    pcm = axes[1].pcolormesh(Qinc, Erange, spectrum, vmax=10, cmap='jet')
    cbar = fig.colorbar(pcm, ax=axes[1], orientation='vertical', extend='max', label='intensity (a.u.)')

    spectrum = sw.calculate_spectrum(Erange, 1, spectral_weight=(excitations.Sxy-excitations.Syx).imag)
    axes[2].plot(Qinc, excitations.E, c='black', lw=0.5, zorder=10)
    pcm = axes[2].pcolormesh(Qinc, Erange, spectrum, cmap='RdBu')
    cbar = fig.colorbar(pcm, ax=axes[2], orientation='vertical', extend='max', label='intensity (a.u.)')


    fig.savefig(f'{PATH_PLOTS}\MnF2_Faure2025.png')
    
    return

def reproduce_MnF2_Morano():
    """Spin wave spectrum of the CrSb altermagnet measured with RIXS"""

    ### SETUP
    msg = MSG.from_xyz_strings(generators=[
        '-x,-y,z, +1',     # 2z
        ' -y+1/2,x+1/2,z+1/2, -1',  # 4z
        ' -x+1/2,y+1/2,-z+1/2, -1',      # 2_010
        '-x,-y,-z, +1',    # -1
    ])
    atoms = [
        Atom(label='Mn', r=(0,0,0),   m=(0,0,1), s=5/2),
        # Atom(label='F', r=(1/4, 1/4, 0), m=(0,0,0.1), s=0.01),
        ] 
    crystal = Crystal(lattice_parameters=[4.87, 4.87, 3.31, 90,90,90], MSG=msg, atoms=atoms)

    print(crystal)
    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,0,1)
    }

    J1, J2, J3 = -0.075, 0.287, -0.012
    J4, J5, J6 = -0.001, 0.008,  0.001
    J7a, J7b = -0.006, -0.002
    couplings = [
        Coupling(label='J1', id1=0, id2=0, n_uvw=[0,0,1], J=J1*np.eye(3,3)),
        Coupling(label='J2', id1=0, id2=1, n_uvw=[0,0,0], J=J2*np.eye(3,3)),
        Coupling(label='J3', id1=0, id2=0, n_uvw=[1,0,0], J=J3*np.eye(3,3)),
        Coupling(label='J4', id1=0, id2=0, n_uvw=[1,0,1], J=J4*np.eye(3,3)),
        Coupling(label='J5', id1=0, id2=1, n_uvw=[0,0,1], J=J5*np.eye(3,3)),
        Coupling(label='J6', id1=0, id2=0, n_uvw=[0,0,2], J=J6*np.eye(3,3)),
        Coupling(label='J7a', id1=0, id2=0, n_uvw=[1, 1,0], J=J7a*np.eye(3,3)),
        Coupling(label='J7b', id1=0, id2=0, n_uvw=[1,-1,0], J=J7b*np.eye(3,3)),
    ]

    ### CALCULATIONS
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=couplings)

    show_struct = False
    if show_struct:
    #     plot_opts = dict(boundaries=([-0.5, 1.5],[-0.5,1.5],[-0.5,1]), coupling_colors={'J1a2': 'Cyan'})
        plot_opts = dict(boundaries=([-1.1, 1.1],[-1.1,1.1],[-1.1,1.1]), 
                         coupling_colors={'J1': 'Green', 'Jdd':'Black', 'Jaa':'Gray', 'J2b':'Red', 'J2d':'Blue'},
                         spin_scale=2)
        plot_structure(sw, engine='vispy', plot_options=plot_opts)


    main_Qs = [
        [0,0,1], 
        [1/2,1/2,1], 
        [1/2,1/2,1/2], 
        [1,1,1/2], 
        [1,1,0], 
        ]
    N = 200
    Qpath, Qinc = sw.crystal.make_qPath(main_Qs, N, return_Qinc=True)

    excitations = sw.calculate_excitations(Qhkl = Qpath)


    ### PLOTTING
    fig, axes = plt.subplots(figsize=(4, 9), nrows=3, tight_layout=True)

    fig.suptitle('MnF2 Morano et al.')

    xticks = Qinc[::N]
    xticklabels = ['$\Gamma$', 'M', 'A', 'Z', '$\Gamma$']

    ylim = (0.01, 10)
    titles = ['dispersions', f'$Re(S_{{xx}}+S_{{yy}}+S_{{zz}})$', '$Im(S_{{xy}}-S_{{yx}})$']

    Erange = np.linspace(0, ylim[1], 500)

    for n,ax in enumerate(axes):
        ax.set(ylabel='energy', xticks=xticks, xticklabels=xticklabels,
               xlabel='Momentum', ylim=ylim, title=titles[n])
    
    axes[0].plot(Qinc, excitations.E, label='b')
    axes[0].legend()

    spectrum = sw.calculate_spectrum(Erange, 1)
    pcm = axes[1].pcolormesh(Qinc, Erange, spectrum, vmax=5, cmap='cubehelix_r')
    cbar = fig.colorbar(pcm, ax=axes[1], orientation='vertical', extend='max', label='intensity (a.u.)')

    spectrum = sw.calculate_spectrum(Erange, 2, spectral_weight=(excitations.Sxy-excitations.Syx).imag)
    axes[2].plot(Qinc, excitations.E, c='black', lw=0.5, zorder=10)
    pcm = axes[2].pcolormesh(Qinc, Erange, spectrum, cmap='RdBu')
    cbar = fig.colorbar(pcm, ax=axes[2], orientation='vertical', extend='max', label='intensity (a.u.)')


    fig.savefig(f'{PATH_PLOTS}\MnF2_Morano2025.png')
    
    return

def reproduce_MnTe_Liu():
    '''https://doi.org/10.1103/PhysRevLett.133.156702'''

    P63ommc = MSG.from_xyz_strings(generators=[
        '-y, x-y, z, +1',  # 3z
        '-x,  -y, z+1/2, -1',  # 2z
        ' y,   x, -z, +1',      # 2_110
        '-x,  -y, -z, +1',    # -1
    ])
    atoms = [
        Atom(label='Mn', r=(0,0,0),   m=(0,1,0), s=5/2),
        # Atom(label='Te', r=(1/3, 2/3, 1/4), m=(0,0,0.1), s=0.01, radius=0.2),
        ] 
    crystal = Crystal(lattice_parameters=[4.15, 4.15, 6.712, 90,90,120], MSG=P63ommc, atoms=atoms)

    # print(crystal)
    # print(crystal.atoms_magnetic)
    crystal.atoms_magnetic[1].m = [0,-1,0]
    print(crystal)

    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,0,1)
    }

    D = 0.0482
    J1, J2, J3 = 3.99, -0.120, 0.472
    J10, J11 = 0.0681, -0.0221
    couplings = [
        Coupling(label='K', id1=0, id2=0, n_uvw=[0,0,0], J=np.diag([0,0,D])),
        Coupling(label='J1', id1=0, id2=1, n_uvw=[0,0,0], J=J1*np.eye(3,3)),
        Coupling(label='J2', id1=0, id2=0, n_uvw=[1,0,0], J=J2*np.eye(3,3)),
        Coupling(label='J3', id1=0, id2=1, n_uvw=[1,0,0], J=J3*np.eye(3,3)),
        Coupling(label='J10', id1=0, id2=0, n_uvw=[1,-1, 1], J=J10*np.eye(3,3)),
        Coupling(label='J11', id1=0, id2=0, n_uvw=[1,-1,-1], J=J11*np.eye(3,3)),
        # Coupling(label='J11', id1=0, id2=0, n_uvw=[-1,-1,-1], J=J11*np.eye(3,3)),
    ]

    ### CALCULATIONS
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=couplings)

    # print('AFT FIX')
    # for cpl in sw.couplings_all:
    #     if cpl.label.startswith('J10') or cpl.label.startswith('J11'):
    #         print(cpl.n_uvw, cpl)

    show_struct = False
    if show_struct:
    #     plot_opts = dict(boundaries=([-0.5, 1.5],[-0.5,1.5],[-0.5,1]), coupling_colors={'J1a2': 'Cyan'})
        plot_opts = dict(boundaries=([-1.1, 2.1],[-1.1,2.1],[-1.1,2.1]), 
                         coupling_colors={'J1': 'Green', 'J2':'Black', 'J3':'Gray', 'J10':'Red', 'J11':'Blue'},
                         spin_scale=2)
        plot_structure(sw, engine='vispy', plot_options=plot_opts)


    main_Qs = [
        [-0.5, 0.5, 1], 
        [-2/3, 1/3, 1], 
        [  -1, 0, 1], 
        [-1.5, 0, 1], 
        [-1.5, 0, 1.5], 
        [  -1, 0, 1.5], 
        [  -1, 0, 1], 
        [-1/2,1/2,3/2]
        ]
    N = 200
    Qpath, Qinc = sw.crystal.make_qPath(main_Qs, N, return_Qinc=True)



    ### PLOTTING
    mosaic = [['path']*2, ['alt_S']*2, ['alt_CH']*2, ['edge_S']*2, ['edge_CH']*2, ['GL_S', 'GL_CH']]
    fig, axes = plt.subplot_mosaic(figsize=(6, 9), tight_layout=True, mosaic=mosaic)

    fig.suptitle('MnTe Liu et al. 2024')

    xticks = Qinc[::N]
    xticklabels = ['M`', 'K', '$\Gamma$', 'M', 'L', 'A', '$\Gamma$', 'H']

    ylim = (0.01, 40)
    titles = ['dispersions', f'$Re(S_{{xx}}+S_{{yy}}+S_{{zz}})$', '$Im(S_{{xy}}-S_{{yx}})$']


    # for n,ax in enumerate(axes):
    #     ax.set(ylabel='energy', xticks=xticks, xticklabels=xticklabels,
    #            xlabel='Momentum', ylim=ylim, title=titles[n])
    
    # axes['path'].plot(Qinc, excitations.E, label='b')
    # axes['path'].legend()

    Erange = np.linspace(0, ylim[1], 500)
    excitations = sw.calculate_excitations(Qhkl = Qpath)
    spectrum = sw.calculate_spectrum(Erange, 2)
    pcm = axes['path'].pcolormesh(Qinc, Erange, spectrum, vmax=2, cmap='jet')
    cbar = fig.colorbar(pcm, ax=axes['path'], orientation='vertical', extend='max', label='intensity (a.u.)')

    # spectrum = sw.calculate_spectrum(Erange, 5, spectral_weight=(excitations.Sxy-excitations.Syx).imag)
    # axes[2].plot(Qinc, excitations.E, c='black', lw=0.5, zorder=10)
    # pcm = axes[2].pcolormesh(Qinc, Erange, spectrum, cmap='RdBu')
    # cbar = fig.colorbar(pcm, ax=axes[2], orientation='vertical', extend='max', label='intensity (a.u.)')

    Erange = np.linspace(25, 40, 500)
    vmax_S = 2
    vmin_CH, vmax_CH = 5e-1*np.array([-1,1])

    for ax_name in ['alt_S', 'alt_CH', 'edge_S', 'edge_CH']:
        axes[ax_name].set(ylim=(Erange.min(), Erange.max()))

    ### Splitting path
    Qpath, Qinc = sw.crystal.make_qPath([[-4/3,0,-2], [-4/3,0,2]], 500, return_Qinc=True)
    excitations = sw.calculate_excitations(Qhkl = Qpath)

    spectrum = sw.calculate_spectrum(Erange, 2)
    pcm = axes['alt_S'].pcolormesh(Qinc, Erange, spectrum, vmax=vmax_S, cmap='jet')
    cbar = fig.colorbar(pcm, ax=axes['alt_S'], orientation='vertical', extend='max', label='intensity (a.u.)')

    spectrum = sw.calculate_spectrum(Erange, 3, spectral_weight=(excitations.Sxz-excitations.Szx).imag)
    axes['alt_CH'].plot(Qinc, excitations.E, c='black', lw=0.5, zorder=10)
    pcm = axes['alt_CH'].pcolormesh(Qinc, Erange, spectrum, cmap='RdBu', vmin=vmin_CH, vmax=vmax_CH)
    cbar = fig.colorbar(pcm, ax=axes['alt_CH'], orientation='vertical', extend='max', label='intensity (a.u.)')



    ### Non-split path
    Qpath, Qinc = sw.crystal.make_qPath([[-1.5,0,-2], [-1.5,0,2]], 500, return_Qinc=True)
    excitations = sw.calculate_excitations(Qhkl = Qpath)
    
    spectrum = sw.calculate_spectrum(Erange, 2)
    pcm = axes['edge_S'].pcolormesh(Qinc, Erange, spectrum, vmax=vmax_S, cmap='jet')
    cbar = fig.colorbar(pcm, ax=axes['edge_S'], orientation='vertical', extend='max', label='intensity (a.u.)')

    spectrum = sw.calculate_spectrum(Erange, 3, spectral_weight=(excitations.Sxz-excitations.Szx).imag)
    axes['edge_CH'].plot(Qinc, excitations.E, c='black', lw=0.5, zorder=10)
    pcm = axes['edge_CH'].pcolormesh(Qinc, Erange, spectrum, cmap='RdBu', vmin=vmin_CH, vmax=vmax_CH)
    cbar = fig.colorbar(pcm, ax=axes['edge_CH'], orientation='vertical', extend='max', label='intensity (a.u.)')

    ### Gamma-L path
    Erange = np.linspace(0, 40, 500)

    for ax_name in ['GL_S', 'GL_CH']:
        axes[ax_name].set(ylim=(Erange.min(), Erange.max()))

    Qpath, Qinc = sw.crystal.make_qPath([[-0.5,0,0.5], [-1,0,1], [-0.5,0,1.5]], 500, return_Qinc=True)
    excitations = sw.calculate_excitations(Qhkl = Qpath)
    
    spectrum = sw.calculate_spectrum(Erange, 2)
    pcm = axes['GL_S'].pcolormesh(Qinc, Erange, spectrum, vmax=vmax_S, cmap='jet')
    cbar = fig.colorbar(pcm, ax=axes['GL_S'], orientation='vertical', extend='max', label='intensity (a.u.)')

    spectrum = sw.calculate_spectrum(Erange, 3, spectral_weight=(excitations.Sxz-excitations.Szx).imag)
    axes['GL_CH'].plot(Qinc, excitations.E, c='black', lw=0.5, zorder=10)
    pcm = axes['GL_CH'].pcolormesh(Qinc, Erange, spectrum, cmap='RdBu', vmin=vmin_CH, vmax=vmax_CH)
    cbar = fig.colorbar(pcm, ax=axes['GL_CH'], orientation='vertical', extend='max', label='intensity (a.u.)')


    fig.savefig(f'{PATH_PLOTS}\MnTe_Liu2024.png')
    
    return

if __name__ == "__main__":
    # pytest.main()

    # make_comparison()

    reproduce_CrSb_Biniskos()
    reproduce_MnF2_Faure()
    reproduce_MnF2_Morano()
    reproduce_MnTe_Liu()