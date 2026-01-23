import numpy as np
import matplotlib.pyplot as plt

from spinwaves import Crystal, MSG, Atom, Coupling, SpinW
from spinwaves.plotting import plot_structure   # this takes some serious time

from pathlib import Path

from spinwaves.utils.linalg import DMI
PATH_PLOTS = Path(r'C:\Users\Stekiel\Documents\GitHub\spinwaves\tests\Various_comparisons')


def reproduce_MnTe2_Nikolaos():
    """Spin wave spectrum of MnTe2 antialtermagnet?"""

    ### SETUP
    P63ommc = MSG.from_xyz_strings(generators=[
        '-x+1/2, -y, z+1/2, +1',
        ' -x,y+1/2,-z+1/2, +1',
        ' z,x,y, +1',
        '  -x,-y,-z, +1'
    ])
    atoms = [Atom(label='Mn', r=(0,0,0),   m=(1,1,1), s=5/2)]
    crystal = Crystal(lattice_parameters=[7,7,7, 90,90,90], MSG=P63ommc, atoms=atoms)

    print(crystal)
    magnetic_modulation = {
        'k':(0, 0, 0),
        'n':(0,0,1)
    }

    K = -1
    J1, J2, J3 = 2, -1, -1
    J4, J5 = -0.5, 0.3
    J7AA = -3
    couplings = [
        Coupling(label='K', id1=0, id2=0, n_uvw=[0,0,0], J=K*np.eye(3,3)),
        Coupling(label='J1', id1=0, id2=3, n_uvw=[0,0,0], J=J1*np.eye(3,3)),
        # Coupling(label='J1', id1=0, id2=3, n_uvw=[0,0,0], J=J1*np.eye(3,3)-0.1*DMI([0,0,1])),
        Coupling(label='J2', id1=0, id2=0, n_uvw=[1,0,0], J=J2*np.eye(3,3)),
        Coupling(label='J3', id1=0, id2=0, n_uvw=[1,1,0], J=J3*np.eye(3,3)),
        Coupling(label='J4', id1=0, id2=3, n_uvw=[0,0,1], J=J4*np.eye(3,3)),
        # Coupling(label='J5', id1=0, id2=0, n_uvw=[1,1,1], J=J5*np.eye(3,3)),
        # Coupling(label='J7AA', id1=0, id2=0, n_uvw=[1,-1,1], J=J7AA*np.eye(3,3)),
    ]

    ### CALCULATIONS
    sw = SpinW(crystal=crystal, magnetic_modulation=magnetic_modulation, couplings=couplings)

    # print(sw.couplings_all)

    show_struct = False
    if show_struct:
    #     plot_opts = dict(boundaries=([-0.5, 1.5],[-0.5,1.5],[-0.5,1]), coupling_colors={'J1a2': 'Cyan'})
        plot_opts = dict(boundaries=([-0.1, 1.1],[-0.1,1.1],[-0.1,1.1]), 
                         coupling_colors={'J1': 'Green', 'J2':'Orange', 'J3':'Yellow', 'J4':'Red', 'J5':'Blue'},
                         spin_scale=2)
        plot_structure(sw, engine='vispy', plot_options=plot_opts)


    N = 100
    Qpath, Qinc = sw.crystal.make_qPath([[0,0,1], [0,0,1.5], [1,1,0], [1.5,1.5,0], [1,1,1], [1.5,1.5,1.5]], N, return_Qinc=True)
    Qpath, Qinc = sw.crystal.make_qPath([[0,0,1], [0,0,1.5], [0,0.5,1.5], [0,0,1], [0.5, 0.5, 1.5], [0,0,1.5], [0,0.5,1.5], [0.5, 0.5, 1.5]], N, return_Qinc=True)

    excitations = sw.calculate_excitations(Qhkl = Qpath)


    ### PLOTTING
    fig, axes = plt.subplots(figsize=(4, 9), nrows=3, tight_layout=True)

    fig.suptitle('MnTe2 Biniskos')

    xticks = Qinc[::N]
    xticklabels = '$\Gamma$ X M $\Gamma$ R X M R'.split()

    ylim = (0, 80)
    titles = ['dispersions', f'$Re(S_{{xx}}+S_{{yy}}+S_{{zz}})$', '$Im(S_{{xy}}-S_{{yx}})$']

    Erange = np.linspace(*ylim, 500)

    for n,ax in enumerate(axes):
        ax.set(ylabel='energy', xticks=xticks, xticklabels=xticklabels,
               xlabel='Momentum', ylim=ylim, title=titles[n])
    
    axes[0].plot(Qinc, excitations.E)

    spectrum = sw.calculate_spectrum(Erange, 2, spectral_weight=excitations.Sxx.real)
    pcm = axes[1].pcolormesh(Qinc, Erange, spectrum, cmap='afmhot_r')
    cbar = fig.colorbar(pcm, ax=axes[1], orientation='vertical', extend='max', label='intensity (a.u.)')

    spectrum = sw.calculate_spectrum(Erange, 2, spectral_weight=(excitations.Sxy-excitations.Syx).imag)
    axes[2].plot(Qinc, excitations.E, c='black', lw=0.5, zorder=10)
    pcm = axes[2].pcolormesh(Qinc, Erange, spectrum, cmap='RdBu')
    cbar = fig.colorbar(pcm, ax=axes[2], orientation='vertical', extend='max', label='intensity (a.u.)')

    print('Saving MnTe2_Nikolaos.png')
    fig.savefig(PATH_PLOTS / "MnTe2_Nikolaos.png")
    
    return


if __name__ == '__main__':
    reproduce_MnTe2_Nikolaos()