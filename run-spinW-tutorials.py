import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure

import spinwaves

#######################################################
def tutorial_4() -> Figure:
    # TODO
    # The energies and details of the shape are not the same as in the example.
    # Is this the bond counting, or some prefectors implementation?
    print('Define structure...')
    hex = spinwaves.Lattice([3,3,6, 90, 90, 90])
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

    sw = spinwaves.Structure(lattice=hex, magnetic_atoms=atoms, magnetic_structure=magnetic_structure)

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

    sw_12 = spinwaves.Structure(lattice=hex, magnetic_atoms=atoms, magnetic_structure=magnetic_structure)

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
    hex = spinwaves.Lattice([3,8,4, 90, 90, 90])
    atoms = [
        {'label':'Cu', 'w':(0,0,0), 'S':0.5},
        {'label':'Fe', 'w':(0,0.5,0), 'S':2},
    ]   # position in crystal coordinates
    magnetic_structure = {
        'k':(0.5, 0, 0),
        'n':(0,0,1),
        'spins':[
            (0,1,0),
            (0,1,0)
        ]
    }

    sw = spinwaves.Structure(lattice=hex, magnetic_atoms=atoms, magnetic_structure=magnetic_structure)

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

    qPath = sw.lattice.make_qPath(main_qs=[[0,0,0], [1,0,0]], Nqs=[501])
    
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
        {'label':'Er', 'w':(0,0,0), 'S':8},
    ]   # position in crystal coordinates
    magnetic_structure = {
        'k':(0, 0, 0),
        'n':(0,0,1),
        'spins':[
            (1,0,0)
        ]
    }

    sw_er = spinwaves.Structure(lattice=hex, magnetic_atoms=atoms, magnetic_structure=magnetic_structure)

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


if __name__ == '__main__':
    fig = tutorial_4()
    fig.savefig(r'C:\Users\Stekiel\Desktop\Offline-plots\spinwaves-t4.png', dpi=200)

    fig = tutorial_12()
    fig.savefig(r'C:\Users\Stekiel\Desktop\Offline-plots\spinwaves-t12.png', dpi=200)

    fig = tutorial_19()
    fig.savefig(r'C:\Users\Stekiel\Desktop\Offline-plots\spinwaves-t19.png', dpi=200)

    fig = erb2(show_struct=False)
    fig.savefig(r'C:\Users\Stekiel\Desktop\Offline-plots\spinwaves-ErB2.png', dpi=200)


