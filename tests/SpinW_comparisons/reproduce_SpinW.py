# -*- coding: utf-8 -*-
r"""Reproduce SpinW tutorials"""
# env: work
import numpy as np
from fractions import Fraction

import pytest

import matplotlib.pyplot as plt

from spinwaves import Crystal, MSG, Atom, Coupling, SpinW
from spinwaves.plotting import plot_structure   # this takes some serious time
from spinwaves.utils.linalg import DMI


from pathlib import Path
PATH_PLOTS = Path(__file__).parent

### BASICS
def reproduce_tutorial_1(Qsamples: float=10000):
    """Spin wave spectrum of the Heisenberg ferromagnetic nearest-neighbor spin chain"""

    ### SETUP
    P1 = MSG.from_xyz_strings(generators=[
        'x, y, z, +1', # 1
    ])
    atoms = [
        Atom(label='Cu', r=(0,0,0),   m=(0,1,0), s=1),
        ] 
    crystal = Crystal(lattice_parameters=[3,8,8, 90,90,90], MSG=P1, atoms=atoms)

    print(crystal)
    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(1,0,0)
    }

    J = -1
    couplings = [
        Coupling(label='Ja', id1=0, id2=0, n_uvw=[1,0,0], J=J*np.eye(3,3))
    ]

    ### CALCULATIONS
    sw = SpinW(crystal=crystal, couplings=couplings, 
               magnetic_modulation=magnetic_modulation)


    Egs = sw.calculate_ground_state(Qhkl=[0,0,0])
    print('E0 = ', Egs)

    # assert Egs == -1


    Qpath = np.linspace([0,0,0], [1,0,0], 300)
    exc = sw.calculate_excitations(Qhkl = Qpath)

    E_spinw = 2*(1-np.cos(2*np.pi*Qpath[:,0]))
    # assert np.allclose(exc.E[:,1], E_spinw)

    # SpinW gives S=0.5 at all Q
    # My code gives S=2 at Q=0, as the perpendicular projection operator is unstable there
    S_spinw = np.full(len(Qpath), 0.5)
    # assert np.allclose(Sperp[:,0])


    ### PLOTTING
    fig, axes = plt.subplots(nrows=3, figsize=(4, 9), tight_layout=True)

    fig.suptitle('SpinW tutorial 1\nHeisenberg ferromagnetic nearest-neighbor spin chain')

    ylabels = ['energy transfer (meV)', 'Intensity (a.u.)']
    titles = ['dispersion', 'Intensity of the spin-spin correlations functions']
    yvals = [exc.E, exc.Sperp]

    for n,ax in enumerate(axes[:2]):
        ax.set(title=titles[n], ylabel=ylabels[n],
               xlabel='L in (00L) (a.u.)')

        ax.plot(Qpath[:,0], yvals[n])

    # TODO powder spectrum
    Qrange = np.linspace(0, 2.5, 100)
    Erange = np.linspace(0, 4.5, 250)
    I = sw.calculate_powder_spectrum(Qrange, Qsamples, Erange, 0.1)
    axes[2].pcolormesh(Qrange, Erange, I)

    fig.savefig(f'{PATH_PLOTS}\spinw_tutorial_1.png')
    
    return

def reproduce_tutorial_2():
    """Antiferromagnetic nearest-neighbour spin chain"""

    ### SETUP
    P4mm = MSG.from_xyz_strings(generators=[
        'x, y, z, +1', # 1
    ])
    atoms = [
        Atom(label='Cu', r=(0,0,0),   m=(0,1,0), s=1),
        ] 
    crystal = Crystal(lattice_parameters=[3,8,8, 90,90,90], MSG=P4mm, atoms=atoms)


    print(crystal)
    magnetic_modulation = {
        'k':(1/2, 0, 0),
        'n':(1,0,0)
    }

    J = 1
    couplings = [
        Coupling(label='Ja', id1=0, id2=0, n_uvw=[1,0,0], J=J*np.eye(3,3))
    ]

    ### CALCULATIONS
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=couplings)
    Egs = sw.calculate_ground_state()
    print('E0 = ', Egs)

    # assert Egs == -1

    Qpath = np.linspace([0,0,0], [1,0,0], 523)
    excitations = sw.calculate_excitations(Qhkl = Qpath)

    # Sum over all degenerated modes
    Sperp = np.sum(excitations.Sperp, axis=1)

    # TODO The energy is 1e-8 instead of 0
    E_spinw = 2*np.abs(np.sin(2*np.pi*Qpath[:,0]))
    # print(E, E_spinw)
    # assert np.allclose(E[:,0], E_spinw)

    ### PLOTTING
    fig, axes = plt.subplots(nrows=2, figsize=(3.25, 5), tight_layout=True)

    fig.suptitle('SpinW tutorial 2\nAntiferromagnetic nearest-neighbour spin chain')

    ylabels = ['energy transfer (meV)', 'Intensity (a.u.)']
    titles = ['dispersion', 'Intensity of the spin-spin correlations functions']
    yvals = [excitations.E, np.log10(Sperp)]
    ylims = [(-2.5, 2.5), (-4, 4)]

    for n,ax in enumerate(axes):
        ax.set(title=titles[n], ylabel=ylabels[n], #ylim=ylims[n],
               xlabel='L in (00L) (a.u.)', xlim=(0,1))

        ax.plot(Qpath[:,0], yvals[n])

    fig.savefig(f'{PATH_PLOTS}\spinw_tutorial_2.png')
    
    return

def reproduce_tutorial_3():
    """Frustrated J1-J2 AFM chain"""

    ### SETUP
    P4mm = MSG.from_xyz_strings(generators=[
        'x, y, z, +1', # 1
    ])
    atoms = [
        Atom(label='Cu', r=(0,0,0),   m=(1,0,0), s=1),
        ] 
    crystal = Crystal(lattice_parameters=[3,8,10, 90,90,90], MSG=P4mm, atoms=atoms)


    J1, J2 = -1, 2
    couplings = [
        Coupling(label='J1', id1=0, id2=0, n_uvw=[1,0,0], J=J1*np.eye(3,3)),
        Coupling(label='J2', id1=0, id2=0, n_uvw=[2,0,0], J=J2*np.eye(3,3)),
    ]

    ### CALCULATIONS
    magnetic_modulation = {
        'k':(0.23005, 0, 0),
        'n':(0,0,1)
    }
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=couplings)

    # TODO starting value looks wrong
    E1 = sw.calculate_ground_state(Qhkl=[0.25,0,0])
    print('E1 = ', E1)

    E2 = sw.calculate_ground_state(Qhkl=[0.23005,0,0])
    print('E2 = ', E2)
    # assert Egs == -1

    Qpath = np.linspace([0,0,0], [1,0,0], 201)
    excitations = sw.calculate_excitations(Qhkl = Qpath)

    ### PLOTTING
    fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)

    fig.suptitle('SpinW tutorial 3\nFrustrated J1-J2 AFM chain')

    ax.set(title='dispersion', 
           ylabel='energy transfer (meV)', ylim=(0,6),
           xlabel='H in (H00) (a.u.)', xlim=(0,1))

    # TODO mode labels seem wrong.
    ax.plot(Qpath[:,0], excitations.E[:,0:2], 'o-', label='S(Q)')
    ax.plot(Qpath[:,0], excitations.E[:,2:4], 'v--', label='S(Q+k)')
    ax.plot(Qpath[:,0], excitations.E[:,4:6], '^:', label='S(Q-k)')

    ax.legend()

    fig.savefig(f'{PATH_PLOTS}\spinw_tutorial_3.png')
    
    return

def reproduce_tutorial_4():
    """Antiferromagnetic square lattice"""

    ### SETUP
    P4 = MSG.from_xyz_strings(generators=[
        '-y, x, z, +1', # 4
    ])
    print(P4)
    atoms = [
        Atom(label='Cu', r=(0,0,0),   m=(1,0,0), s=1),
        ] 
    crystal = Crystal(lattice_parameters=[3,3,6, 90,90,90], MSG=P4, atoms=atoms)


    print(crystal)
    magnetic_modulation = {
        'k':(1/2, 1/2, 0),
        'n':(0,0,1)
    }

    J1, J2 = 1, -0.1
    couplings = [
        Coupling(label='J1', id1=0, id2=0, n_uvw=[1,0,0], J=J1*np.eye(3,3)),
        Coupling(label='J2', id1=0, id2=0, n_uvw=[1,1,0], J=J2*np.eye(3,3)),
    ]

    ### CALCULATIONS
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=couplings)

    Egs = sw.calculate_ground_state(Qhkl=[0,0,0])
    print('E0 = ', Egs)
    # assert np.allclose(Egs, -2.2)

    N = 200
    Qpath, Qinc = sw.crystal.make_qPath([[0,0,0], [1/2,0,0], [1/2,1/2,0], [0,0,0]], [N]*3, return_Qinc=True)

    excitations = sw.calculate_excitations(Qhkl = Qpath)

    # TODO The energy is 1e-8 instead of 0
    # print(E, E_spinw)
    # assert np.allclose(E[:,0], E_spinw)

    # TODO my S=1 and undefined at Gamma. SpinW is 0.5 and behaves nicely at Gamma
    # Cholesky failes, maybe it needs some more stability
    # S_spinw = np.full(len(Qpath), 0.5)
    # print(Sperp)
    # assert np.allclose(Sperp[:,0])

    ### PLOTTING
    fig, axes = plt.subplots(figsize=(4, 6), nrows=2, tight_layout=True)

    fig.suptitle('SpinW tutorial 4\nAntiferromagnetic square lattice')

    xticks = [Qinc[id] for id in [0, N-1, 2*N-1, 3*N-1]]
    xticklabels = ['$\Gamma$', 'X', 'M', '$\Gamma$']

    Erange = np.linspace(0, 6.5, 500)
    spectrum = sw.calculate_spectrum(Erange, 0.4)

    for ax in axes:
        ax.set(ylabel='energy', xticks=xticks, xticklabels=xticklabels,
               xlabel='Momentum')
    
    axes[0].plot(Qinc, excitations.E)
    pcm = axes[1].pcolormesh(Qinc, Erange, spectrum, vmax=10, cmap='afmhot_r')

    cbar = fig.colorbar(pcm, ax=axes[1], orientation='vertical', extend='max', label='intensity (a.u.)')
    # cbar.ax.tick_params(axis="both", which='both', direction="out")
    # cbar.ax.text(3, 0.5, f'Intensity (a.u.)', va='center', ha='left', transform=cbar.ax.transAxes, rotation=-90)#, fontfamily='serif')


    fig.savefig(f'{PATH_PLOTS}\spinw_tutorial_4.png')
    
    return

### KAGOME
def reproduce_tutorial_5(Qsamples: float=10000):
    """Ferromagnetic first neighbor Kagome lattice"""

    ### SETUP
    Pm3 = MSG.from_xyz_strings(generators=[
        '-y, x-y, z, +1', # 3z
        '-x, -y, -z, +1', # -1
    ])
    print(Pm3)
    atoms = [
        Atom(label='Cu', r=(1/2,0,0),   m=(0,1,0), s=1),
        ] 
    crystal = Crystal(lattice_parameters=[6,6,5, 90,90,120], MSG=Pm3, atoms=atoms)


    print(crystal)
    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,1,0)
    }

    J = -1
    couplings = [
        # Coupling(label='Ka', id1=0, id2=0, n_uvw=[0,0,0], J=np.diag([0,0,1])),
        # Coupling(label='Ja', id1=0, id2=1, n_uvw=[0,0,0], J=J*np.eye(3,3)),
        Coupling(label='Ja', id1=0, id2=2, n_uvw=[0,0,0], J=J*np.eye(3,3)),
    ]

    ### CALCULATIONS
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=couplings)
    for atom in sw.crystal.atoms_magnetic:
        atom.m = [0,1,0]

    # print('COUPLINGS:')
    # for cpl in sw.couplings_all:
    #     print('\t', cpl)
    # print(sw.couplings_all)


    Egs = sw.calculate_ground_state(Qhkl=[0,0,0])
    print('E0 = ', Egs)

    # assert Egs == -1

    N = 200
    Qpath, Qinc = crystal.make_qPath([[-0.5,0,0], [0,0,0], [0.5,0.5,0]], [N,N], return_Qinc=True)
    # Qpath, Qinc = crystal.make_qPath([[-0.5,0,0], [0,0,0], [0.5,0.5,0], [-0.5,0,0]], [N,N,N], return_Qinc=True)

    exc = sw.calculate_excitations(Qhkl = Qpath)

    ### PLOTTING
    fig, axes = plt.subplots(nrows=2, figsize=(4, 6), tight_layout=True)

    fig.suptitle('SpinW tutorial 5\nFerromagnetic first neighbor Kagome lattice')

    axes[0].set(title='Dispersion relations', 
                xlabel='Qinc (A-1)',
                ylabel='energy transfer (meV)', ylim=(0,8))

    axes[0].plot(Qinc, exc.E)

    Qrange = np.linspace(0, 2.5, 100)
    Erange = np.linspace(0, 7, 250)
    print('Calculating powder spectrum...')
    I = sw.calculate_powder_spectrum(Qrange, Qsamples, Erange, 0.25)
    print('Imax for powder spectrum:', I.max())
    axes[1].set(title='Powder spectrum', 
                xlabel='Q (A-1)',
                ylabel='energy transfer (meV)', ylim=(0,7))
    axes[1].pcolormesh(Qrange, Erange, I, vmax=I.max()/3)

    fig.savefig(f'{PATH_PLOTS}\spinw_tutorial_5.png')
    
    return

def reproduce_tutorial_6(Qsamples: float=10000):
    """Ferromagnetic Kagome lattice"""

    ### SETUP
    Pm3 = MSG.from_xyz_strings(generators=[
        '-y, x-y, z, +1', # 3z
        '-x, -y, -z, +1', # -1
    ])
    print(Pm3)
    atoms = [
        Atom(label='Cu', r=(1/2,0,0),   m=(0,1,0), s=1),
        ] 
    crystal = Crystal(lattice_parameters=[6,6,8, 90,90,120], MSG=Pm3, atoms=atoms)


    print(crystal)
    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,1,0)
    }

    crystal.get_atomic_distances(dmax=6.5)

    J1, J2, J3b = -1, 0.1, 0.17
    couplings = [
        # Coupling(label='Ka', id1=0, id2=0, n_uvw=[0,0,0], J=np.diag([0,0,1])),
        # Coupling(label='Ja', id1=0, id2=1, n_uvw=[0,0,0], J=J*np.eye(3,3)),
        Coupling(label='J1', id1=0, id2=2, n_uvw=[0,0,0], J=J1*np.eye(3,3)),
        Coupling(label='J2', id1=0, id2=1, n_uvw=[0,0,0], J=J2*np.eye(3,3)),
        Coupling(label='J2', id1=0, id2=0, n_uvw=[0,1,0], J=J3b*np.eye(3,3)),
    ]

    ### CALCULATIONS
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=couplings)
    for atom in sw.crystal.atoms_magnetic:
        atom.m = [0,1,0]

    # print('COUPLINGS:')
    # for cpl in sw.couplings_all:
    #     print('\t', cpl)
    # print(sw.couplings_all)


    Egs = sw.calculate_ground_state(Qhkl=[0,0,0])
    print('E0 = ', Egs)

    # assert Egs == -1

    N = 200
    Qpath, Qinc = crystal.make_qPath([[-0.5,0,0], [0,0,0], [0.5,0.5,0]], [N,N], return_Qinc=True)
    # Qpath, Qinc = crystal.make_qPath([[-0.5,0,0], [0,0,0], [0.5,0.5,0], [-0.5,0,0]], [N,N,N], return_Qinc=True)

    exc = sw.calculate_excitations(Qhkl = Qpath)

    ### PLOTTING
    fig, axes = plt.subplots(nrows=2, figsize=(4, 6), tight_layout=True)

    fig.suptitle('SpinW tutorial 6\nFerromagnetic Kagome lattice')

    axes[0].set(title='Dispersion relations', 
                xlabel='Qinc (A-1)',
                ylabel='energy transfer (meV)', ylim=(0,8))

    axes[0].plot(Qinc, exc.E)

    Qrange = np.linspace(0, 2.5, 100)
    Erange = np.linspace(0, 7, 250)
    print('Calculating powder spectrum...')
    I = sw.calculate_powder_spectrum(Qrange, Qsamples, Erange, 0.1)
    print('Imax for powder spectrum:', I.max())
    axes[1].set(title='Powder spectrum', 
                xlabel='Q (A-1)',
                ylabel='energy transfer (meV)', ylim=(0,7))
    axes[1].pcolormesh(Qrange, Erange, I, vmax=I.max()/3)

    fig.savefig(f'{PATH_PLOTS}\spinw_tutorial_6.png')
    
    return

def reproduce_tutorial_7(Qsamples: float=10000):
    """k=0 Kagome antiferromagnet"""

    ### SETUP
    Pm3 = MSG.from_xyz_strings(generators=[
        '-y, x-y, z, +1', # 3z
        '-x, -y, -z, +1', # -1
    ])
    print(Pm3)
    atoms = [
        Atom(label='Cu', r=(1/2,0,0),   m=(0,1,0), s=1),
        ] 
    crystal = Crystal(lattice_parameters=[6,6,8, 90,90,120], MSG=Pm3, atoms=atoms)


    print(crystal)
    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,1,0)
    }

    J1, J2, J3b = 1, 0.11, 0.17
    couplings = [
        # Coupling(label='Ka', id1=0, id2=0, n_uvw=[0,0,0], J=np.diag([0,0,1])),
        # Coupling(label='Ja', id1=0, id2=1, n_uvw=[0,0,0], J=J*np.eye(3,3)),
        Coupling(label='J1', id1=0, id2=2, n_uvw=[0,0,0], J=J1*np.eye(3,3)),
        Coupling(label='J2', id1=0, id2=1, n_uvw=[0,0,0], J=J2*np.eye(3,3)),
        # Coupling(label='J2', id1=0, id2=0, n_uvw=[0,1,0], J=J3b*np.eye(3,3)),
    ]

    ### CALCULATIONS
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=couplings)

    Egs = sw.calculate_ground_state(Qhkl=[0,0,0])
    print('E0 = ', Egs)

    # assert Egs == -1

    N = 200
    Qpath, Qinc = crystal.make_qPath([[-0.5,0,0], [0,0,0], [0.5,0.5,0]], [N,N], return_Qinc=True)
    # Qpath, Qinc = crystal.make_qPath([[-0.5,0,0], [0,0,0], [0.5,0.5,0], [-0.5,0,0]], [N,N,N], return_Qinc=True)

    exc = sw.calculate_excitations(Qhkl = Qpath)

    ### PLOTTING
    fig, axes = plt.subplots(nrows=2, figsize=(4, 6), tight_layout=True)

    fig.suptitle('SpinW tutorial 7\nk=0 Kagome antiferromagnet')

    axes[0].set(title='Dispersion relations', 
                xlabel='Qinc (A-1)',
                ylabel='energy transfer (meV)', ylim=(0,3))

    axes[0].plot(Qinc, exc.E)

    Qrange = np.linspace(0, 2.5, 100)
    Erange = np.linspace(0, 3, 250)
    print('Calculating powder spectrum...')
    I = sw.calculate_powder_spectrum(Qrange, Qsamples, Erange, 0.005)
    print('Imax for powder spectrum:', I.max())
    axes[1].set(title='Powder spectrum', 
                xlabel='Q (A-1)',
                ylabel='energy transfer (meV)', ylim=(0,3))
    pcm = axes[1].pcolormesh(Qrange, Erange, I, vmax=5)
    cbar = fig.colorbar(pcm, ax=axes[1], orientation='vertical', extend='max', label='intensity (a.u.)')


    fig.savefig(f'{PATH_PLOTS}\spinw_tutorial_7.png')
    
    return

def reproduce_tutorial_8(Qsamples: float=10000):
    """sqrt(3) x sqrt(3) Kagome antiferromagnet"""
    print('Reproducing SpinW tutorial 8')

    ### SETUP
    Pm3 = MSG.from_xyz_strings(generators=[
        '-y, x-y, z, +1', # 3z
        '-x, -y, -z, +1', # -1
    ])
    print(Pm3)
    atoms = [
        Atom(label='Cu', r=(1/2,0,0),   m=(-0.5, 0.866, 0), s=1),
        ] 
    crystal = Crystal(lattice_parameters=[6,6,40, 90,90,120], MSG=Pm3, atoms=atoms)


    print(crystal)
    magnetic_modulation = {
        'k':(1/3, 1/3, 0),
        'n':(0,0,1)
    }

    J1, J2, J3b = 1, 0.11, 0.17
    couplings = [
        Coupling(label='J1', id1=0, id2=2, n_uvw=[0,0,0], J=J1*np.eye(3,3)),
    ]

    ### CALCULATIONS
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=couplings)
    forced_mm = np.array([
        [0,1,0],
        [0,1,0],
        [-1,-1, 0],
    ])
    for n,atom in enumerate(sw.crystal.atoms_magnetic):
        print(atom)
        atom.m = crystal.uvw2xyz(forced_mm[n])
        print(atom)

    show_struct = False
    if show_struct:
    #     plot_opts = dict(boundaries=([-0.5, 1.5],[-0.5,1.5],[-0.5,1]), coupling_colors={'J1a2': 'Cyan'})
        plot_opts = dict(boundaries=([-1.1, 2.1],[-1.1,2.1],[-0.1,0.1]), 
                         coupling_colors={'J1': 'Green', 'Jdd':'Black', 'Jaa':'Gray', 'J2b':'Red', 'J2d':'Blue'},
                         spin_scale=2)
        plot_structure(sw, engine='vispy', plot_options=plot_opts)


    Egs = sw.calculate_ground_state(Qhkl=[1/3,1/3,0])
    print('E0 = ', Egs)

    N = 300
    Qpath, Qinc = crystal.make_qPath([[-0.5,0,0], [0,0,0], [0.5,0.5,0]], [N,N], return_Qinc=True)
    exc = sw.calculate_excitations(Qhkl = Qpath)

    ### PLOTTING
    fig, axes = plt.subplots(nrows=3, figsize=(4, 9), tight_layout=True)

    fig.suptitle('SpinW tutorial 8\nsqrt(3) x sqrt(3) Kagome antiferromagnet')

    axes[0].set(title='Dispersion relations', 
                xlabel='Qinc (A-1)',
                ylabel='energy transfer (meV)', ylim=(0,3))

    axes[0].plot(Qinc, exc.E)

    # There is some weird mode selection process in SpinW that I don't understand.
    # 
    Erange = np.linspace(0.05, 2.5, 500)
    spectrum = sw.calculate_spectrum(Erange, 0.05)

    for ax in axes:
        ax.set(ylabel='energy', xlabel='Momentum')
    
    pcm = axes[1].pcolormesh(Qinc, Erange, spectrum,cmap='afmhot_r', vmax=5)
    cbar = fig.colorbar(pcm, ax=axes[1], orientation='vertical', extend='max', label='intensity (a.u.)')



    Qrange = np.linspace(0, 2.5, 100)
    Erange = np.linspace(0, 3, 250)
    print('Calculating powder spectrum...')
    I = sw.calculate_powder_spectrum(Qrange, Qsamples, Erange, 0.05)
    print('Imax for powder spectrum:', I.max())
    axes[2].set(title='Powder spectrum', 
                xlabel='Q (A-1)',
                ylabel='energy transfer (meV)', ylim=(0,3))
    pcm = axes[2].pcolormesh(Qrange, Erange, I, vmax=5)
    cbar = fig.colorbar(pcm, ax=axes[2], orientation='vertical', extend='max', label='intensity (a.u.)')


    fig.savefig(f'{PATH_PLOTS}\spinw_tutorial_8.png')
    
    return

def reproduce_tutorial_9(Qsamples: float=10000):
    """k=0 Kagome antiferromagnet with DM interaction"""
    print('Reproducing SpinW tutorial 8')

    ### SETUP
    Pm3 = MSG.from_xyz_strings(generators=[
        '-y, x-y, z, +1', # 3z
        '-x, -y, -z, +1', # -1
    ])
    print(Pm3)
    atoms = [
        Atom(label='Cu', r=(1/2,0,0),   m=(0,1,0), s=1),
        ] 
    crystal = Crystal(lattice_parameters=[6,6,40, 90,90,120], MSG=Pm3, atoms=atoms)


    print(crystal)
    magnetic_modulation = {
        'k':(0,0,0),
        'n':(0,0,1)
    }

    J1, DM = 1, -0.08
    couplings = [
        Coupling(label='J1', id1=0, id2=2, n_uvw=[0,0,0], J=J1*np.eye(3,3)+DMI([0,0,DM])),
    ]
    print(couplings[0])

    ### CALCULATIONS
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=couplings)

    show_struct = False
    if show_struct:
    #     plot_opts = dict(boundaries=([-0.5, 1.5],[-0.5,1.5],[-0.5,1]), coupling_colors={'J1a2': 'Cyan'})
        plot_opts = dict(boundaries=([-1.1, 2.1],[-1.1,2.1],[-0.1,0.1]), 
                         coupling_colors={'J1': 'Green', 'Jdd':'Black', 'Jaa':'Gray', 'J2b':'Red', 'J2d':'Blue'},
                         spin_scale=2)
        plot_structure(sw, engine='vispy', plot_options=plot_opts)


    Egs = sw.calculate_ground_state(Qhkl=[1/3,1/3,0])
    print('E0 = ', Egs)

    # assert Egs == -1

    N = 300
    Qpath, Qinc = crystal.make_qPath([[-0.5,0,0], [0,0,0], [0.5,0.5,0]], [N,N], return_Qinc=True)
    # Qpath, Qinc = crystal.make_qPath([[-0.5,0,0], [0,0,0], [0.5,0.5,0], [-0.5,0,0]], [N,N,N], return_Qinc=True)

    exc = sw.calculate_excitations(Qhkl = Qpath)

    ### PLOTTING
    fig, axes = plt.subplots(nrows=3, figsize=(4, 9), tight_layout=True)

    fig.suptitle('SpinW tutorial 8\nk=0 Kagome antiferromagnet with DM interaction')

    axes[0].set(title='Dispersion relations', 
                xlabel='Qinc (A-1)',
                ylabel='energy transfer (meV)', ylim=(0,3))

    axes[0].plot(Qinc, exc.E)

    Erange = np.linspace(0.05, 2.5, 500)
    spectrum = sw.calculate_spectrum(Erange, 0.05)

    for ax in axes:
        ax.set(ylabel='energy', xlabel='Momentum')
    
    pcm = axes[1].pcolormesh(Qinc, Erange, spectrum,cmap='afmhot_r', vmax=5)
    cbar = fig.colorbar(pcm, ax=axes[1], orientation='vertical', extend='max', label='intensity (a.u.)')



    Qrange = np.linspace(0, 2.5, 100)
    Erange = np.linspace(0, 3, 250)
    print('Calculating powder spectrum...')
    I = sw.calculate_powder_spectrum(Qrange, Qsamples, Erange, 0.02)
    print('Imax for powder spectrum:', I.max())
    axes[2].set(title='Powder spectrum', 
                xlabel='Q (A-1)',
                ylabel='energy transfer (meV)', ylim=(0,3))
    pcm = axes[2].pcolormesh(Qrange, Erange, I, vmax=5)
    cbar = fig.colorbar(pcm, ax=axes[2], orientation='vertical', extend='max', label='intensity (a.u.)')


    fig.savefig(f'{PATH_PLOTS}\spinw_tutorial_9.png')
    
    return

def reproduce_tutorial_11():
    """Spin wave spectrum of La2CuO4"""
    print('Reproducing SpinW tutorial 11')

    ### SETUP
    Pm4 = MSG.from_xyz_strings(generators=[
        '-y, x, z, +1', # 4z
        # '-x, -y, -z, +1', # -1
    ])
    atoms = [
        Atom(label='Cr', r=(0,0,0),   m=(1,0,0), s=0.5),
        ] 
    crystal = Crystal(lattice_parameters=[3,3,40, 90,90,90], MSG=Pm4, atoms=atoms)
    print(crystal.get_atomic_distances())


    print(crystal)
    magnetic_modulation = {
        'k':(1/2,1/2,0),
        'n':(0,0,1)
    }

    J, Jp, Jpp, Jc = 138.3, 2, 2, 38
    couplings = [
        Coupling(label='J1', id1=0, id2=0, n_uvw=[1,0,0], J=(J-Jc/2)*np.eye(3,3)),
        Coupling(label='J2', id1=0, id2=0, n_uvw=[1,1,0], J=(Jp-Jc/4)*np.eye(3,3)),
        Coupling(label='J3', id1=0, id2=0, n_uvw=[2,0,0], J=Jpp*np.eye(3,3))
    ]

    ### CALCULATIONS
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=couplings)

    show_struct = False
    if show_struct:
    #     plot_opts = dict(boundaries=([-0.5, 1.5],[-0.5,1.5],[-0.5,1]), coupling_colors={'J1a2': 'Cyan'})
        plot_opts = dict(boundaries=([-1.1, 2.1],[-1.1,2.1],[-0.1,0.1]), 
                         coupling_colors={'J1': 'Green', 'Jdd':'Black', 'Jaa':'Gray', 'J2b':'Red', 'J2d':'Blue'},
                         spin_scale=2)
        plot_structure(sw, engine='vispy', plot_options=plot_opts)


    Elims = (0,350)
    N = 100
    main_Qs = [[3/4, 1/4, 0], [1/2, 1/2, 0], [1/2, 0, 0], [3/4, 1/4, 0], [1, 0 ,0], [1/2, 0, 0]]
    main_Q_names = ['P', 'M', 'X', 'P', '\Gamma', 'X']
    Qpath, Qinc = crystal.make_qPath(main_Qs, N, return_Qinc=True)


    exc = sw.calculate_excitations(Qhkl = Qpath)

    # exc.S[exc.E<1e-3] = 0
    # exc.Sperp[exc.E<1e-3] = 0

    exc.E *= 1.18

    ### PLOTTING
    fig, axes = plt.subplots(nrows=3, figsize=(4, 9), tight_layout=True)

    fig.suptitle('SpinW tutorial 11\nSpin wave spectrum of La2CuO4')

    for ax in axes[0], axes[2]:
        ax.set(title='Dispersion relations', 
               xlabel='Qinc (A-1)', xticks=Qinc[::N], xticklabels=main_Q_names,
               ylabel='energy transfer (meV)', ylim=Elims)
        
    axes[1].set(title='Intensity of correlations Sperp', 
                xlabel='Qinc (A-1)', xticks=Qinc[::N], xticklabels=main_Q_names,
                ylabel='Intensity (a.u.)', ylim=(0, 20))

    axes[0].plot(Qinc, exc.E)
    axes[1].plot(Qinc, np.sum(exc.Sperp, axis=-1))

    Erange = np.linspace(0, Elims[1], 500)
    spectrum = sw.calculate_spectrum(Erange, 35)
    
    pcm = axes[2].pcolormesh(Qinc, Erange, spectrum, cmap='cubehelix_r', vmax=0.05)
    cbar = fig.colorbar(pcm, ax=axes[2], orientation='vertical', extend='max', label='intensity (a.u.)')


    fig.savefig(f'{PATH_PLOTS}\spinw_tutorial_11.png')
    
    return

def reproduce_tutorial_12():
    """Triangular lattice AF with easy plane anisotropy
"""
    print('Reproducing SpinW tutorial 12')

    ### SETUP
    Pm3 = MSG.from_xyz_strings(generators=[
        '-y, x-y, z, +1', # 3z
        '-x, -y, -z, +1', # -1
    ])
    print(Pm3)
    atoms = [
        Atom(label='Cr', r=(0,0,0),   m=(0,1,0), s=1.5),
        ] 
    crystal = Crystal(lattice_parameters=[3,3,4, 90,90,120], MSG=Pm3, atoms=atoms)


    print(crystal)
    magnetic_modulation = {
        'k':(-1/3,-1/3,0),
        'n':(0,0,1)
    }

    J, K = 1, 0.2
    couplings = [
        Coupling(label='K', id1=0, id2=0, n_uvw=[1,0,0], J=J*np.eye(3,3)),
        Coupling(label='J', id1=0, id2=0, n_uvw=[0,0,0], J=np.diag([0,0,K])),
    ]

    ### CALCULATIONS
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=couplings)
    for cpl in sw.couplings_all:
        print(cpl)

    show_struct = False
    if show_struct:
    #     plot_opts = dict(boundaries=([-0.5, 1.5],[-0.5,1.5],[-0.5,1]), coupling_colors={'J1a2': 'Cyan'})
        plot_opts = dict(boundaries=([-1.1, 2.1],[-1.1,2.1],[-0.1,0.1]), 
                         coupling_colors={'J1': 'Green', 'Jdd':'Black', 'Jaa':'Gray', 'J2b':'Red', 'J2d':'Blue'},
                         spin_scale=2)
        plot_structure(sw, engine='vispy', plot_options=plot_opts)


    Elims = (0,7)
    N = 300
    Qpath = crystal.make_qPath([[0,0,0], [1,1,0]], N)
    Qinc = Qpath[:,0]

    exc = sw.calculate_excitations(Qhkl = Qpath)

    ### PLOTTING
    fig, axes = plt.subplots(nrows=3, figsize=(4, 9), tight_layout=True)

    fig.suptitle('SpinW tutorial 12\nk=0 triangular lattice AF with easy plane anisotropy')

    axes[0].set(title='Dispersion relations', 
                xlabel='Qinc (A-1)',
                ylabel='energy transfer (meV)', ylim=Elims)

    axes[0].plot(Qinc, exc.E)

    Erange = np.linspace(*Elims, 500)

    for ax in axes:
        ax.set(xlabel='Momentum', xlim=(0,1),
               ylabel='energy', ylim=Elims)
    
    spectrum = sw.calculate_spectrum(Erange, 0.4)
    pcm = axes[1].pcolormesh(Qinc, Erange, spectrum, cmap='afmhot_r', vmax=4)
    cbar = fig.colorbar(pcm, ax=axes[1], orientation='vertical', extend='max', label='intensity (a.u.)')


    spectrum_xy = sw.calculate_spectrum(Erange, 0.9, spectral_weight=(exc.Sxx+exc.Syy).real)
    spectrum_zz = sw.calculate_spectrum(Erange, 0.9, spectral_weight=exc.Szz.real)
    # the trick to plot both spectra is to glue the other one on negative scale
    # and keep the scale symmetric against zero
    spectrum = spectrum_xy - 2*spectrum_zz
    v = 8
    axes[2].plot(Qinc, exc.E, c='black', lw=0.5, zorder=10)
    pcm = axes[2].pcolormesh(Qinc, Erange, spectrum, cmap='RdBu', vmin=-v, vmax=v)
    cbar = fig.colorbar(pcm, ax=axes[2], orientation='vertical', extend='max', label='intensity (a.u.)')


    fig.savefig(f'{PATH_PLOTS}\spinw_tutorial_12.png')
    
    return


def reproduce_tutorial_19():
    """Spin-spin correlation function of two different coupled magnetic ion"""

    ### SETUP
    P1 = MSG.from_xyz_strings(generators=[
        'x, y, z, +1', # 1
    ])
    atoms = [
        Atom(label='Cu', r=(0,0,0), m=(0,1,0), s=0.5),
        Atom(label='Fe', r=(0,0.5,0), m=(0,1,0), s=2),
    ]
    crystal = Crystal(lattice_parameters=[3,8,4, 90,90,90], MSG=P1, atoms=atoms)

    print(crystal)
    magnetic_modulation = {
        'k':(0.5, 0, 0),
        'n':(0,0,1)
    }

    Jcc, Jff, Jcf = 1, 1, -0.1
    couplings = [
        Coupling(label='Jcc1', id1=0, id2=0, n_uvw=[1,0,0], J=Jcc*np.eye(3,3)),
        Coupling(label='Jff2', id1=1, id2=1, n_uvw=[1,0,0], J=Jff*np.eye(3,3)),
        Coupling(label='Jcf4', id1=0, id2=1, n_uvw=[0,0,0], J=Jcf*np.eye(3,3)),
        Coupling(label='Jcf5', id1=0, id2=1, n_uvw=[0,-1,0], J=Jcf*np.eye(3,3)),
    ]

    ### CALCULATIONS
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, 
               couplings=couplings, temperature=1e-5)

    for cpl in sw.couplings_all:
        print('\t', cpl)
    
    N = 3000
    Qpath = sw.crystal.make_qPath(main_qs=[[0,0,0], [1,0,0]], Nqs=[N])
    # E, Sperp = sw.calculate_excitations(Qpath=Qpath)
    # print(E[2])
    # print(sw._SScorrelations[2])

    E, S = sw.determine_ES(Qpath)
    sw._energies = E
    sw._SScorrelations = S
    # print(E[2])
    # print(sw._SScorrelations[2])

    Erange = np.linspace(0, 4.5, 1000)
    # spectrum = sw.calculate_spectrum(Erange, 0.2)
    
    sigma = 0.2 / 2.35482
    def res_func(E, E0):
        return 1/(sigma*np.sqrt(1*np.pi)) * np.exp(-E**2/(2*sigma**2))

    staggered_spectrum = []
    for energies, SSs in zip(sw._energies, sw._SScorrelations):
        spectrum = np.zeros(len(Erange))
        for E0, SS in zip(energies, SSs):
            I = np.trace(SS.real)
            spectrum += I * res_func(Erange-E0, E0)

        staggered_spectrum.append(spectrum)

    spectrum = np.transpose(staggered_spectrum)

    ### PLOTTING
    fig, axes = plt.subplots(nrows=2, figsize=(4, 5), tight_layout=True)

    fig.suptitle('SpinW tutorial 19\nSpin-spin correlation function of two different coupled magnetic ion')

    ax = axes[0]
    ax.plot(Qpath[:,0], E)

    ax = axes[1]
    pcm = ax.pcolormesh(Qpath[:,0], Erange, spectrum, vmax=20, cmap='afmhot_r')

    fig.colorbar(pcm, ax=ax, orientation='vertical', extend='max', label='intensity (a.u.)')
    for ax in axes:
        ax.set(ylabel='energy', xlabel='Momentum', xlim=(0,1), ylim=(0,4.5))


    fig.savefig(f'{PATH_PLOTS}\spinw_tutorial_19.png')
    
    return

def reproduce_tutorial_20():
    '''Spin wave spectrum of Yb2Ti2O7 in magnetic field
    https://www.sciencedirect.com/science/article/abs/pii/002245969090182W?via%3Dihub
    '''

    ### SETUP
    Fdm3m = MSG.from_xyz_strings(generators=[
        '-z, y+3/4, x+3/4, +1', # 1
        'z+3/4, -y, x+3/4, +1', # 1
        'z+3/4, y+3/4, -x, +1', # 1
        'y+3/4, x+3/4, -z, +1', # 1
        'x+3/4, -z, y+3/4, +1', # 1
        '-z, x+3/4, y+3/4, +1', # 1
    ])
    atoms = [
        Atom(label='Yb', r=(1/2, 1/2, 1/2), m=(1,-1,0), s=1/2),
    ]
    a = 10.037
    crystal = Crystal(lattice_parameters=[a,a,a, 90,90,90], MSG=Fdm3m, atoms=atoms)

    # The magnetic structure does not respect the symmetry of the gray MSG from SG,
    # it is further complicated by field, so set moments by hand.
    for atom in crystal.atoms_magnetic:
        atom.m = np.array([1,-1,0])

    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,0,1)
    }

    J1, J2, J3, J4 = -0.09, -0.22, -0.29, 0.01
    J = np.array([
        [J1, J4, J4],
        [-J4, J2, J3],
        [-J4, J3, J2]
    ])
    couplings = [
        Coupling(label='Jmn', id1=0, id2=1, n_uvw=[0,0,0], J=J)
    ]

    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, 
               couplings=couplings, temperature=1e-5)
    
    show_struct = False
    if show_struct:
        print(crystal)
    #     plot_opts = dict(boundaries=([-0.5, 1.5],[-0.5,1.5],[-0.5,1]), coupling_colors={'J1a2': 'Cyan'})
        plot_opts = dict(boundaries=([-0.1, 1.1],[-0.1,1.1],[-0.1,1.02]), 
                             coupling_colors={'Jmn': 'Pink', 'J2':'Black', 'J3':'Gray', 'J4':'Red', 'J2d':'Blue'})
        plot_structure(sw, engine='vispy', plot_options=plot_opts)
  

    ### CALCULATIONS AND PLOTS
    mosaic = [ ['B1_Q1', 'B1_Q2'], 
               ['B2_Q1', 'B2_Q2']]
    layout = dict()
    fig, axs = plt.subplot_mosaic(mosaic=mosaic, figsize=(8, 6), layout='tight', gridspec_kw=layout)


    Npath = 101
    energy = np.linspace(0, 1, 50)


    Qs = dict(
        Q1 = [[0,0,0], [1,0,0]],
        Q2 = [[0,0,0], [1,1,0]],
    )

    pmesh_plot_kwargs = {}


    def plot_spectrum(Qchoice):
        print(f'Calculate spectrum `{Qchoice}`...')

        # axs[Qchoice+'_spec'].set_title(Qchoice)

        qPath = sw.crystal.make_qPath(main_qs=[Qs[Qchoice][0], Qs[Qchoice][1]], Nqs=101)
        
        excitations = sw.calculate_excitations(qPath)
        spectrum = sw.calculate_spectrum(energy, resolution=0.05)

        id_x = np.argmax(np.abs(qPath[-1] - qPath[0]))

        Q, E = np.meshgrid(qPath[:,id_x], energy)
        # np.save(f'{EXPORT_PATH}/{PREFIX}-{label}.npy', (Q, E, spectrum))


        axs[f'B1_{Qchoice}'].pcolormesh(Q, E, spectrum, **pmesh_plot_kwargs)

    plot_spectrum('Q1')
    plot_spectrum('Q2')

    fig.savefig(f'{PATH_PLOTS}\spinw_tutorial_20.png')

    return
    

def reproduce_tutorial_21():
    """Spin wave spectrum of YIG 
    https://doi.org/10.1038/s41535-017-0067-y
    https://doi.org/10.1103/PhysRevLett.117.217201

    Very problematic setup as the magnetic space group is trigonal:
    https://www.sciencedirect.com/science/article/abs/pii/002245969090182W?via%3Dihub
    while structural is cubic
    """

    ### SETUP
    P1 = MSG.from_xyz_strings(generators=[
        '-z+3/4, y+1/4, x+3/4, +1',
        '-z+1/2, x+1/2, y, +1',
        '-z+1/4, y+3/4, x+1/4, +1',
        '-z, x, y+1/2, +1',
        'x+1/2, y+1/2, z+1/2, +1'
    ])
    atoms = [
        Atom(label='Fe_a', r=(0,0,0), m=(0,0,1), s=5/2),
        Atom(label='Fe_d', r=(0.375,0,0.25), m=(0,0,1), s=3/2),
    ]
    crystal = Crystal(lattice_parameters=[12.3563, 12.3563, 12.3563, 90,90,90], MSG=P1, atoms=atoms)

    # print(crystal)
    # print(crystal.print_atomic_distances(dmax=6, dmin=3.8))
    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,0,1)
    }

    Jaa, Jdd, Jad = np.array([9.6, 3.24, 0.92])
    couplings = [
        Coupling(label='Jad', id1=1, id2=2, n_uvw=[0,0,0], J=Jad*np.eye(3,3)),
        Coupling(label='Jaa', id1=0, id2=11,n_uvw=[0,0,0], J=Jaa*np.eye(3,3)),
        Coupling(label='Jdd', id1=2, id2=9, n_uvw=[0,0,0], J=Jdd*np.eye(3,3)),
    ]

    ### CALCULATIONS
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, 
               couplings=couplings, temperature=1e-5)
    
    show_struct = False
    if show_struct:
    #     plot_opts = dict(boundaries=([-0.5, 1.5],[-0.5,1.5],[-0.5,1]), coupling_colors={'J1a2': 'Cyan'})
        plot_opts = dict(boundaries=([-0.1, 1.1],[-0.1,1.1],[-0.1,1.02]), 
                             coupling_colors={'Jad': 'Green', 'Jdd':'Black', 'Jaa':'Gray', 'J2b':'Red', 'J2d':'Blue'})
        plot_structure(sw, engine='vispy', plot_options=plot_opts)
    
    N = 30
    Qpath = sw.crystal.make_qPath(main_qs=[[0,0,0], [1,0,0]], Nqs=[N])
    # E, Sperp = sw.calculate_excitations(Qpath=Qpath)
    # print(E[2])
    # print(sw._SScorrelations[2])

    E, S = sw.determine_ES(Qpath)
    sw._energies = E
    sw._SScorrelations = S
    # print(E[2])
    # print(sw._SScorrelations[2])

    ### PLOTTING
    fig, axes = plt.subplots(nrows=2, figsize=(4, 5), tight_layout=True)

    fig.suptitle('SpinW tutorial 21\nSpin wave spectrum of YIG')

    ax = axes[0]
    ax.plot(Qpath[:,0], E)

    fig.savefig(f'{PATH_PLOTS}\spinw_tutorial_21.png')
    
    return

def test_new_functionalities():

    ### SETUP
    P1 = MSG.from_xyz_strings(generators=[
        'x, y, z, +1', # 1
    ])
    atoms = [
        Atom(label='Cu', r=(0,0,0), m=(0,1,0), s=0.5),
        Atom(label='Fe', r=(0,0.5,0), m=(0,1,0), s=2),
    ]
    crystal = Crystal(lattice_parameters=[3,8,4, 90,90,90], MSG=P1, atoms=atoms)

    print(crystal)
    magnetic_modulation = {
        'k':(0.5, 0, 0),
        'n':(0,0,1)
    }

    Jcc, Jff, Jcf = 1, 1, -0.1
    couplings = [
        Coupling(label='Jcc', id1=0, id2=0, n_uvw=[1,0,0], J=Jcc*np.eye(3,3)),
        Coupling(label='Jcc', id1=1, id2=1, n_uvw=[1,0,0], J=Jff*np.eye(3,3)),
        Coupling(label='Jcf', id1=0, id2=1, n_uvw=[0,0,0], J=Jcf*np.eye(3,3)),
    ]


    ### CALCULATIONS
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=couplings)

    Qhkl = np.linspace([0,0,0],[1,0,0], 3)
    # Qhkl = np.random.random((2, 3))
    new_E, new_S = sw._determine_ESp_new(Qhkl)

    full_debug = True

    for ind in np.ndindex(Qhkl.shape[:-1]):
        qhkl = Qhkl[ind]
        old_E, old_S = sw._determine_ESp(qhkl)

        print(np.allclose(old_S, new_S[ind]))
        if full_debug:
            print(ind)
            print('OLD')
            print(old_E)
            print(old_S.real)
            print('NEW')
            print(new_E[ind])
            print(new_S[ind].real)
            print()

    print('### NEW')
    # print(new_method)

    return sw
    
if __name__ == "__main__":
    # pytest.main()

    Qsamples = 3000

    ### Basics
    # reproduce_tutorial_1(Qsamples)
    # reproduce_tutorial_2()
    # reproduce_tutorial_3()
    # reproduce_tutorial_4()

    ### Kagome
    # reproduce_tutorial_5(Qsamples)
    # reproduce_tutorial_6(Qsamples)
    # reproduce_tutorial_7(Qsamples)
    # reproduce_tutorial_8(Qsamples)
    # reproduce_tutorial_9(Qsamples)

    # reproduce_tutorial_11()
    reproduce_tutorial_12()

    # reproduce_tutorial_19()
    # reproduce_tutorial_20()
    # reproduce_tutorial_21()

    # test_new_functionalities()




    # test_matrices()

    ### Quick tests
    # import time
    # t_start = time.time()
    # P = MSG.from_xyz_strings(generators=[
    #     # '-y, x, z, +1', # 4_001
    #     # 'z, x, y, +1',    # 3_111
    #     # 'x+1/2, y+1/2, z+1/2, +1', # I centering
    #     '-x, -y, z, +1',    # 2_110
    #     # '-x, -y, -z, +1',   # -1
    #     # 'x, y, z, -1',   # 1'
    #     'y, x, z, +1'   # m_110
    #     ])
    # t_end = time.time()
    # for g in P:
    #     print(g.print())

    # print(P.order)
    # print(P.make_cayley_table())
    # print('Constructor and Cayley table', t_end-t_start)
    # print()
    # t_start = time.time()
    # P_subs = P.get_subgroups()
    # for P_sub in P_subs:
    #     print(P_sub)
    # t_end = time.time()
    # print('subgroups', t_end-t_start)
    # print('orders', [G.order for G in P_subs])

    # pos = [0, 0.5, 0.3]
    # print('point symmetry')
    # print(P.get_point_symmetry(pos))
    # print('orbit')
    # print(P.get_orbit(pos))
