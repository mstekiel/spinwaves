import logging
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure

# import spinwaves
from spinwaves import Atom, Crystal, SpinW, Coupling
from spinwaves.plotting import plot_structure
from spinwaves.functions import DMI

#######################################################
def tutorial_4() -> Figure:
    # TODO
    # The energies and details of the shape are not the same as in the example.
    # Is this the bond counting, or some prefectors implementation?
    print('Define structure...')
    hex = Lattice([3,3,6, 90, 90, 90])
    atoms = [
        {'label':'Cr', 'w':(0,0,0), 'S':1},
    ]   # position in crystal coordinates
    magnetic_structure = {
        'k':(0.5, 0.5, 0),
        'n':(0,0,1),
        'spins':[
            (1,0,0)
            ]
    }

    sw = spinwaves.SpinW(lattice=hex, magnetic_atoms=atoms, magnetic_structure=magnetic_structure)

    print('Add couplings...')
    J1, J2 = 1, -0.1
    couplings = {
        'J1':[[1,0,0], 0, 0, J1*np.eye(3,3), ['4z']],
        'J2':[[1,0,0], 0, 0, J2*np.eye(3,3), ['4z']]
    }
    sw.add_couplings(couplings)

    print(sw.formatted_couplings)
    # sw.plot_structure(extent=(2,2,1))

    main_qs = [
        [0, 0, 0],
        [1/2, 0, 0],
        [1/2, 1/2, 0],
        [0, 0, 0],
    ]
    qPath = sw.lattice.make_qPath(main_qs=main_qs, Nqs=[201,201,201])
    
    print('Calculate excitations...')
    sw.calculate_excitations(qPath=qPath, silent=True)

    fig, ax = plt.subplots(tight_layout=True)
    sw.plot_dispersion(fig=fig)

    return fig

#######################################################
def tutorial_12(show_struct: bool=False) -> Figure:
    # To recreate spinW spectra Dz anisotropy ocmponent has to be multiplied by two
    print('Define structure...')
    hex = spinwaves.Lattice([3, 3, 4, 90, 90, 120])
    atoms = [
        {'label':'Cr', 'w':(0,0,0), 'S':3/2},

    ]   # position in crystal coordinates
    magnetic_structure = {
        'k':(1/3, 1/3, 0),
        'n':(0,0,1),
        'spins':[
            (0,1,0)
        ]
    }

    sw_12 = spinwaves.SpinW(lattice=hex, magnetic_atoms=atoms, magnetic_structure=magnetic_structure)

    print('Add couplings...')
    Jx = 1
    couplings = {
        'Kz':[[0,0,0], 0, 0, np.diag([0,0,0.2]), ['1']],
        'Jx':[[1,0,0], 0, 0, Jx*np.eye(3,3), ['6z']],
    }
    sw_12.add_couplings(couplings)


    # for label,(r_uvw, i,j, J, symmetry) in sw_12.couplings.items():
    #     print(label,r_uvw,i,j,symmetry)
    #     print(J)

    for d_xyz,n_uvw,i,j,J in sw_12.formatted_couplings:
        print(d_xyz,n_uvw,i,j)
        print(sw_12.lattice.xyz2uvw(d_xyz))
        print('J')
        print(J)

    # sw_12.plot_structure(extent=(3,3,1))


    print('Prepare qPath...')
    N = 301
    qs = [
        [-1,-1,0],
        [ 1,1,0]
    ]
    qPath = sw_12.lattice.make_qPath(qs, [N])
    
    print('Calculate excitations...')
    Es = sw_12.calculate_excitations(qPath, silent=True)

    fig, ax = plt.subplots(tight_layout=True)
    fig = sw_12.plot_dispersion(fig=fig)

    return fig

#######################################################
def tutorial_19(show_struct: bool=False) -> Figure:
    # PERFECT DISPERSIONS MATCH
    print('Define structure...')


    atoms = [
        Atom(label='Cu', r=(0,0,0), m=(0,1,0), s=0.5),
        Atom(label='Fe', r=(0,0.5,0), m=(0,1,0), s=2),
    ]
    cf = Crystal(lattice_parameters=[3,8,4, 90, 90, 90], atoms=atoms)
    magnetic_structure = {
        'k':(0.5, 0, 0),
        'n':(0,0,1)
    }

    sw = SpinW(crystal=cf, magnetic_modulation=magnetic_structure)

    print('Add couplings...')
    Jcc, Jff, Jcf = 1, 1, -0.1
    couplings = {
        # 'Kcr':[[0,0,0], 0, 0, np.diag([0,0,0.2]), ['1']],
        # 'Kfe':[[0,0,0], 1, 1, np.diag([0,0,0.6]), ['1']],
        'Jcc':[[1,0,0], 0, 0, Jcc*np.eye(3,3), ['2z']],
        'Jff':[[1,0,0], 1, 1, Jff*np.eye(3,3), ['2z']],
        'Jcf':[[0,0,0], 0, 1, Jcf*np.eye(3,3), ['2z']],
        # 'Jnx':[[2,1,0], 0, 0, Jnx*np.eye(3,3), ['6z']],
        # 'Jz':[[0,0,1], 0, 0, Jz*np.eye(3,3), ['-1']]
    }
    sw.add_couplings(couplings)

    if show_struct:
        sw.plot_structure(extent=(1,1,1))

    qPath = sw.crystal.make_qPath(main_qs=[[0,0,0], [1,0,0], [0,1,0]], Nqs=[501,101])
    
    print('Calculate excitations...')
    sw.calculate_excitations(qPath=qPath, silent=True)

    fig, ax = plt.subplots(tight_layout=True)
    sw.plot_dispersion(fig=fig)

    return fig

#######################################################   
def erb2(show_struct: bool=False) -> Figure:
    print('Define structure...')
    hex = spinwaves.Lattice([3.275, 3.275, 3.785, 90, 90, 120])
    atoms = [
        {'label':'Er', 'r':(0,0,0), 's':4, 'm':(1,0,0)},
    ]   # position in crystal coordinates
    magnetic_structure = {
        'k':(0, 0, 0),
        'n':(0,0,1)
    }

    sw_er = spinwaves.SpinW(lattice=hex, atoms=atoms, magnetic_structure=magnetic_structure)

    print('Add couplings...')
    Jx = -0.0354
    Jxz = -0.004
    Jz, J2z = -0.0155, 0.01
    couplings = {
        'K':[[0,0,0], 0, 0, np.diag([0,0.002,6.7]), ['1']], # K

        'Jx':[[1, 0,0], 0, 0, Jx*np.eye(3,3), ['6z']],
        'Jxz':[[1,0,1], 0, 0, Jxz*np.eye(3,3), ['6z','-1']],
        'Jz':[[0,0,1], 0, 0, Jz*np.eye(3,3), ['-1']],
        'J2z':[[0,0,2], 0, 0, J2z*np.eye(3,3), ['-1']],
       }   # (d,i,j,J) d has to be symmetrized by hand; Indices here correspond to atoms in the `atoms` list
    sw_er.add_couplings(couplings)


    # for label,(r_uvw, i,j, J, symmetry) in sw_er.couplings.items():
    #     print(label,r_uvw,i,j,symmetry)
    #     print(J)

    # for r_xyz,i,j,J in sw_er.formatted_couplings:
    #     print(r_xyz,i,j)
    #     print(J)

    if show_struct:
        sw_er.plot_structure(extent=(3,3,1))


    print('Prepare qPath...')
    N = 51
    # CAMEA path
    q1 = np.asarray([-1/3, 2/3,  -1])
    q2 = np.asarray([ 1/3,-2/3,  -1])
    q3 = np.asarray([ 1/3,-2/3,-0.5])
    q4 = np.asarray([   0,   0,-0.5])
    q5 = np.asarray([   0,   0,-1.5])
    qs = [q1, q2, q3, q4, q5]
    qPath = sw_er.lattice.make_qPath(qs, [N,N,N,N])
    
    print('Calculate excitations...')
    Es = sw_er.calculate_excitations(qPath, silent=True)

    fig, ax = plt.subplots(tight_layout=True)
    fig = sw_er.plot_dispersion(fig=fig)

    return fig

#######################################################   
def nbcp(show_struct: bool=False) -> Figure:
    print('Define structure...')
    a, c = 5.3285, 7.0081
    hex = Lattice([3*a, 3*a, c, 90, 90, 120])
    atoms = [
        {'label':'Co1', 'w':(0,0,0), 'S':1/2},
        {'label':'Co1', 'w':(1/3,0,0), 'S':1/2},
        {'label':'Co1', 'w':(2/3,0,0), 'S':1/2},
        {'label':'Co1', 'w':(0,1/3,0), 'S':1/2},
        {'label':'Co1', 'w':(1/3,1/3,0), 'S':1/2},
        {'label':'Co1', 'w':(2/3,1/3,0), 'S':1/2},
        {'label':'Co1', 'w':(0,1/3,0), 'S':1/2},
        {'label':'Co1', 'w':(1/3,1/3,0), 'S':1/2},
        {'label':'Co1', 'w':(2/3,1/3,0), 'S':1/2}
    ]   # position in crystal coordinates

    mx, mz = 0.3, 0.4
    magnetic_structure = {
        'k':(0, 0, 0),
        'n':(0,0,1),
        'spins':[
            (0,0,-1),
            (0,0,1),
            (0,0,-1)
        ]
    }

    sw_er = spinwaves.SpinW(lattice=hex, magnetic_atoms=atoms, magnetic_structure=magnetic_structure)

    print('Add couplings...')
    Jx = -0.0354
    Jxz = -0.004
    Jz, J2z = -0.0155, -0.002
    couplings = {
        'K':[[0,0,0], 0, 0, np.diag([0,0.002,6.7]), ['1']], # K

        'Jx':[[1, 0,0], 0, 0, Jx*np.eye(3,3), ['6z']],
        'Jxz':[[1,0,1], 0, 0, Jxz*np.eye(3,3), ['6z','-1']],
        'Jz':[[0,0,1], 0, 0, Jz*np.eye(3,3), ['-1']],
        'J2z':[[0,0,2], 0, 0, J2z*np.eye(3,3), ['-1']],
       }   # (d,i,j,J) d has to be symmetrized by hand; Indices here correspond to atoms in the `atoms` list
    sw_er.add_couplings(couplings)


    # for label,(r_uvw, i,j, J, symmetry) in sw_er.couplings.items():
    #     print(label,r_uvw,i,j,symmetry)
    #     print(J)

    # for r_xyz,i,j,J in sw_er.formatted_couplings:
    #     print(r_xyz,i,j)
    #     print(J)

    if show_struct:
        sw_er.plot_structure(extent=(3,3,1))


    print('Prepare qPath...')
    N = 51
    # CAMEA path
    q1 = np.asarray([-1/3, 2/3,  -1])
    q2 = np.asarray([ 1/3,-2/3,  -1])
    q3 = np.asarray([ 1/3,-2/3,-0.5])
    q4 = np.asarray([   0,   0,-0.5])
    q5 = np.asarray([   0,   0,-1.5])
    qs = [q1, q2, q3, q4, q5]
    qPath = sw_er.lattice.make_qPath(qs, [N,N,N,N])
    
    print('Calculate excitations...')
    Es = sw_er.calculate_excitations(qPath, silent=True)

    fig, ax = plt.subplots(tight_layout=True)
    fig = sw_er.plot_dispersion(fig=fig)

    return fig

#######################################################
def ceaual(show_struct: bool=False) -> Figure:
    # PERFECT DISPERSIONS MATCH
    print('Define structure...')

    atoms = [
        Atom(label='Ce_1', r=(0,0,0), m=(1,0,0), s=2.5),
        Atom(label='Ce_2', r=(0.5,0.5,0.5), m=(0,1,0), s=2.5),
    ]
    cf = Crystal(lattice_parameters=[4.3,4.3,10.65, 90, 90, 90], 
                 atoms=atoms)
    magnetic_structure = {
        'k':(0, 0, 0.5),
        'n':(0,0,1)
    }

    # Negative couplings are FM, positive are AF
    K = 0
    Ja, Jd, Jc = 0.15, -0.05, -0.05
    # Ja, Jd, Jc = -0.1, -0.1, -0.1
    couplings = [
        Coupling(label='K', n_uvw=[0,0,0], id1=0, id2=0, J=K*np.diag([0.1,0,1]), symmetry=['1']),
        Coupling(label='Ja1', n_uvw=[1,0,0], id1=0, id2=0, J=Ja*np.eye(3,3), symmetry=['4z']),
        Coupling(label='Ja2', n_uvw=[1,0,0], id1=1, id2=1, J=Ja*np.eye(3,3), symmetry=['4z']),
        Coupling(label='Jd1', n_uvw=[0,0,0], id1=0, id2=1, J=Jd*np.eye(3,3), symmetry=['4z','-1']),
        Coupling(label='Jd2', n_uvw=[0,0,0], id1=1, id2=0, J=Jd*np.eye(3,3), symmetry=['4z','-1']),
        Coupling(label='Jc1', n_uvw=[0,0,1], id1=0, id2=0, J=Jc*np.eye(3,3), symmetry=['-1']),
        Coupling(label='Jc2', n_uvw=[0,0,1], id1=1, id2=1, J=Jc*np.eye(3,3), symmetry=['-1']),
    ]

    print('Create SW system...')
    sw = SpinW(crystal=cf, 
               couplings=couplings,
               magnetic_modulation=magnetic_structure)


    if show_struct:
        plot_opts = dict(boundaries=(2,2,3))
        plot_structure(sw, engine='vispy', plot_options=plot_opts)

    print('Calculate ground state energy')
    qz = np.linspace([0,0,-1], [0,0,1], 51)
    qx = np.linspace([-1,0,0], [1,0,0], 51)
    E0z = [sw.calculate_ground_state(q) for q in qz]
    E0x = [sw.calculate_ground_state(q) for q in qx]

    qPath = sw.crystal.make_qPath(main_qs=[[0,0,0.5], [0,0,2], [1,0,1], [1,0,-1]], Nqs=[51,51,51])
    # qPath = sw.crystal.make_qPath(main_qs=[[0,0,0.5], [0,0,2], [1,0,1], [1,0,-1]], Nqs=[5,5,5])
    # qPath = sw.crystal.make_qPath(main_qs=[[1,0,1], [1,0,-1]], Nqs=[51])
    
    print('Calculate excitations...')
    sw.calculate_excitations(qPath=qPath, silent=True)

    fig, axs = plt.subplots(figsize=(6,8), nrows=2, tight_layout=True)
    sw.plot_dispersion(fig=fig)

    axs[1].set_ylabel('E')
    axs[1].set_xlabel('k')
    axs[1].scatter(qz[:,2], E0z, label='kz')
    axs[1].scatter(qx[:,0], E0x, label='kx')
    axs[1].legend()

    return fig



if __name__ == '__main__':
    # fig = tutorial_4()
    # fig.savefig(r'C:\Users\Stekiel\Desktop\Offline-plots\spinwaves-t4.png', dpi=200)

    # fig = tutorial_12()
    # fig.savefig(r'C:\Users\Stekiel\Desktop\Offline-plots\spinwaves-t12.png', dpi=200)

    # fig = tutorial_19()
    # fig.savefig(r'C:\Users\Stekiel\Desktop\Offline-plots\spinwaves-t19.png', dpi=200)

    # fig = erb2(show_struct=True)
    # fig.savefig(r'C:\Users\Stekiel\Desktop\Offline-plots\spinwaves-ErB2.png', dpi=200)

    # fig = ceaual(show_struct=True)
    # fig.savefig(r'C:\Users\Stekiel\Desktop\Offline-plots\spinwaves-CeAuAl3.png', dpi=200)