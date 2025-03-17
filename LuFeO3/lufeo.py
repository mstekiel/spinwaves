from dataclasses import dataclass
from matplotlib.axes import Axes
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


@dataclass
class DataEntry:
    QH: float
    QK: float
    QL: float
    E: float
    E_err: float
    I: float
    path_desc: str

    @property
    def Q(self) -> tuple[float,float,float]:
        '''Momentum transfer vector in rlu.'''
        return (self.QH, self.QK, self.QL)
    
    @property
    def qred(self) -> float:
        '''Return the Q wavevector reduced to the (101) BZ'''
        return (self.QH%2, self.QK%2, self.QL%2)


class Data:
    '''Class containing all loaded datasets.
    Q
    q'''
    entries: list[DataEntry]

    def __init__(self):
        self.entries = list()

    def __new__(cls):
        return super(Data, cls).__new__(cls)

    def append_file(self, filename: str, path_desc: str):
        '''Load data from file in column format.
        #QH	QK	QL	ENlim1	ENlim2	I1	EN1	 sigma1	I2	EN2	sigma2'''
        data = np.loadtxt(filename, comments='#')

        for row in data:
            QH, QK, QL, ENlim1, ENlim2, I1, EN1, sigma1, I2, EN2, sigma2 = row
            self.entries.append(DataEntry(QH=QH, QK=QK, QL=QL, E=EN1, E_err=sigma1, I=I1, path_desc=path_desc))

    def get_Qpath(self, path_desc: str) -> 'Data':
        '''Get list of measures momentum and energy transfers on the path.'''
        import copy

        NewData = copy.deepcopy(self)
        NewData.entries = [de for de in self.entries if de.path_desc==path_desc]
        return NewData
    
    @property
    def QHred(self) -> list[float]:
        return [e.qred[0] for e in self.entries]
    
    @property
    def QLred(self) -> list[float]:
        return [e.qred[2] for e in self.entries]
    
    @property
    def QH(self) -> list[float]:
        return [e.Q[0] for e in self.entries]
    
    @property
    def QK(self) -> list[float]:
        return [e.Q[1] for e in self.entries]
    
    @property
    def QL(self) -> list[float]:
        return [e.Q[2] for e in self.entries]
    
    @property
    def E(self) -> list[float]:
        return [e.E for e in self.entries]
    
    @property
    def E_err(self) -> list[float]:
        return [e.E_err for e in self.entries]
    
            

    def __repr__(self):
        return '\n'.join(['<Data']+[e.__repr__() for e in self.entries[:10]]+['>'])
        
def load_data() -> Data:
    '''Load data from files.'''

    PATH_DATA = rf'C:\Users\Stekiel\Documents\GitHub\spinwaves\LuFeO3\data'

    DATA = Data()

    DATA.append_file(f'{PATH_DATA}\HmHScan_LuFeO3_Taipan_R.txt', path_desc='H0mH')
    DATA.append_file(f'{PATH_DATA}\HScan_LuFeO3_Taipan_R.txt', path_desc='H0N')
    DATA.append_file(f'{PATH_DATA}\LScan_LuFeO3_Taipan_R.txt', path_desc='10L')
    # DATA.append_file(f'{PATH_DATA}\AG-CF Crazy Scan Q points.txt', path_desc='AG-CF')
    DATA.append_file(f'{PATH_DATA}\AG-CF-5p5.txt', path_desc='AG-CF-5p5')
    DATA.append_file(f'{PATH_DATA}\AG-CF-4p5.txt', path_desc='AG-CF-4p5')

    return DATA

def plot_spectrum(sw_params, DATA: Data, plot_type: str='dispersion') -> Figure:
    Npath = 151
    sw = load_system(sw_params, show_struct=False)

    mosaic = [ ['0k1', '0k1_in'], 
              ['01l', '01l_in'], 
              ['10l', '10l_in'] ,
              ['h0l', 'h0l_in'] , 
              ['h0mh', 'h0mh_in'], 
              ['AG-CF-5p5', 'AG-CF-5p5_in'],
              ['AG-CF-4p5', 'AG-CF-4p5_in']
              ]
    layout = dict(width_ratios=[2,1])
    fig, axs = plt.subplot_mosaic(mosaic=mosaic, figsize=(7,14), tight_layout=True, gridspec_kw=layout)


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
    qPath = sw.crystal.make_qPath(main_qs=[[1,0,0], [1,0,6]], Nqs=[Npath])
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
  

    print('Calculate spectrum `AG-CF-5p5`...')
    axs['AG-CF-5p5'].set_title(f"(H 0 5.5-H) TAIPAN")
    qPath = sw.crystal.make_qPath(main_qs=[[0,0,5.5], [2,0,3.5]], Nqs=[Npath])
    omega4, _ = sw.calculate_spectrum(qPath=qPath, silent=True)
    sw.plot_dispersion(ax=axs['AG-CF-5p5'],    xaxis=qPath[:,2], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))
    sw.plot_dispersion(ax=axs['AG-CF-5p5_in'], xaxis=qPath[:,2], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))

    print('Calculate spectrum `AG-CF-4p5`...')
    axs['AG-CF-4p5'].set_title(f"(H 0 4.5-H) TAIPAN")
    qPath = sw.crystal.make_qPath(main_qs=[[0,0,4.5], [2,0,2.5]], Nqs=[Npath])
    omega4, _ = sw.calculate_spectrum(qPath=qPath, silent=True)
    sw.plot_dispersion(ax=axs['AG-CF-4p5'],    xaxis=qPath[:,2], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))
    sw.plot_dispersion(ax=axs['AG-CF-4p5_in'], xaxis=qPath[:,2], plot_type=plot_type, plot_kwargs=dict(color='gray', alpha=0.5))



    # Plot experimental results
    marker_style = dict(fmt='s', markersize=5, capsize=4, markeredgecolor='black')

    axs['0k1'].errorbar(data_0K1[:,1], data_0K1[:,6], yerr=data_0K1[:,7], color='blue', **marker_style)
    axs['0k1_in'].errorbar(data_0K1[:,1], data_0K1[:,6], yerr=data_0K1[:,7], color='blue', **marker_style)

    axs['01l'].errorbar(data_01L[:,2], data_01L[:,6], yerr=data_01L[:,7], color='green', **marker_style)
    axs['01l_in'].errorbar(data_01L[:,2], data_01L[:,6], yerr=data_01L[:,7], color='green', **marker_style)

    data_10L = DATA.get_Qpath('10L')
    axs['10l'].errorbar(data_10L.QL, data_10L.E, yerr=data_10L.E_err, color='purple', **marker_style)
    axs['10l_in'].errorbar(data_10L.QL, data_10L.E, yerr=data_10L.E_err, color='purple', **marker_style)

    data_H0N = DATA.get_Qpath('H0N')
    axs['h0l'].errorbar(data_H0N.QH, data_H0N.E, yerr=data_H0N.E_err, color='magenta', **marker_style)
    axs['h0l_in'].errorbar(data_H0N.QH, data_H0N.E, yerr=data_H0N.E_err, color='magenta', **marker_style)

    data_H0mH = DATA.get_Qpath('H0mH')
    axs['h0mh'].errorbar(data_H0mH.QH, data_H0mH.E, xerr=data_H0mH.E_err, color='pink', **marker_style)
    axs['h0mh_in'].errorbar(data_H0mH.QH, data_H0mH.E, xerr=data_H0mH.E_err, color='pink', **marker_style)

    data_crazy = DATA.get_Qpath('AG-CF-5p5')
    axs['AG-CF-5p5'].errorbar(data_crazy.QL, data_crazy.E, xerr=data_crazy.E_err, color='cyan', **marker_style)
    axs['AG-CF-5p5_in'].errorbar(data_crazy.QL, data_crazy.E, xerr=data_crazy.E_err, color='cyan', **marker_style)

    data_crazy = DATA.get_Qpath('AG-CF-4p5')
    axs['AG-CF-4p5'].errorbar(data_crazy.QL, data_crazy.E, xerr=data_crazy.E_err, color='cyan', **marker_style)
    axs['AG-CF-4p5_in'].errorbar(data_crazy.QL, data_crazy.E, xerr=data_crazy.E_err, color='cyan', **marker_style)


    for ax_name, ax in axs.items():
        if ax_name.endswith('_in'):
            ax.set_xlim(0.88, 1.12)
            ax.set_ylim(0, 33)
        else:
            ax.set_ylim(0, 1.05*np.max(np.concatenate((omega1, omega2, omega3, omega4))))
            # pass

    axs['AG-CF-5p5'].set_ylim(40, 75)
    axs['AG-CF-4p5'].set_ylim(40, 75)

    return fig

def plot_fit(sw_system: SpinW, DATA: Data, plot_type: str='dispersion') -> Figure:

    mosaic = [['10L'] ,
              ['H0N'] , 
              ['H0mH'], 
              ['AG-CF-5p5'],
              ['AG-CF-4p5']
              ]
    layout = dict()
    fig, axs = plt.subplot_mosaic(mosaic=mosaic, figsize=(6,8), tight_layout=True, gridspec_kw=layout)


    # Plot experimental results
    marker_style = dict(fmt='s', markersize=5, capsize=4, markeredgecolor='black')
    def plot_res(label: str):
        ax = axs[label]
        data = DATA.get_Qpath(label)

        ax.set_title(label)
        ax.set_ylim(-10, 10)
        ax.axhline(0, color='tab:blue')

        for n,dd in enumerate(data.entries):
            _, E_theo, I_theo, weights = get_residuum(sw_system, dd.Q, dd.E, 0, return_calc=True)
            x = [n]*len(E_theo)

            ax.scatter(x, E_theo-dd.E, alpha=0.5, c='white', ec='black')
            ax.scatter(x, E_theo-dd.E, alpha=weights, c='black')

        return

    plot_res('10L')
    plot_res('H0N')
    plot_res('H0mH')
    plot_res('AG-CF-5p5')
    plot_res('AG-CF-4p5')

    return fig

def get_residuum(lfo_sw: SpinW, q: list[float], E_exp: float, E_err: float, return_calc: bool=False):
    '''Calculate the residual for a single data point.
    
    The main problem here is to define which point from the model corresponds to the data point.
    
    Parameters
    ----------
    lfo_sw:
        SpinW model
    q: (3,)
        Vector of momentum transfer in crystal coordinates.
    E_exp:
        Measured energy of the excitations.
    E_err:
        Error of measured energy.
    return_calc (optional)
        Flag enabling the return of calculated values and weights for diagnostics.

    Return
    ------
    residuum:
        Redisuum
    E_theo: (optional)
        Calculated excitation energies.
    I_theo: (optional)
        Calculated excitation intensity.
    weights: (optional)
        Applied excitation weights.
    '''
    E_theo, I_theo = np.squeeze(lfo_sw.calculate_spectrum([q]))

    # Intensity can be 0- due to numerics? Clip for safety
    I_theo = np.clip(I_theo, a_min=0, a_max=None)

    weights = np.nan_to_num(I_theo, nan=1e-30)*np.power(E_exp-E_theo, -6)
    weights /= np.sum(weights)

    ret = np.sum(weights*(E_exp-E_theo))

    if return_calc:
        ret = (ret, E_theo, I_theo, weights)

    return ret

Nplot = 0
def lfo_residuals(parameters: Parameters, DATA: Data, debug: bool=False):
# def lfo_residuals(parameters: Parameters):
    """Calculate the residuals between the model and data of LuFeO3.
    
    Returns
    -------
    residuals
    """
    residuals = []
    weights = []

    lfo_sw = load_system(parameters)

    for ee in DATA.entries:
        q = ee.Q
        E_exp = ee.E
        E_err = ee.E_err

        residuals.append( get_residuum(lfo_sw, q, E_exp, E_err) )
        weights.append(1/E_err)
    

    residuals = np.array(residuals, dtype=float)
    weights = np.array(weights, dtype=float)

    if debug:
        global Nplot
        Nplot += 1
        fig = plot_fit(lfo_sw, DATA, plot_type='dispersion')
        fig.savefig(PATH+f'\Fit\spinwaves-LuFeO3-residuals-{Nplot}.png')

    chi2 = np.sum(np.square(residuals))
    print(f'chi2={chi2}')
    for p in parameters:
        print('\t', parameters[p])

    return residuals

def fit_lfo(p0: Parameters, DATA: Data):
    """Run the fitting procedure with starting parameters `p0`.
    Print the result.
    
    Returns
    -------
    Fitted parameters
    """
    print('Fitting LuFeO3 data')
    fit_result = lmfit.minimize(lfo_residuals, p0, method='leastsq', kws={'DATA': DATA})
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
    '''Define various parameter sets for LuFeO3 models.

    Some interesting models:
    - J1 with fit of J1, Ka, and Fz, where J2=0 and DMI is from Park. -> J1=4.45 Fz=0.06, Ka=-0.11
    '''

    models = dict()

    # Simple J1 model no more
    # Unstable, as it scales J1 an J2 up
    lfo_params = Parameters()
    lfo_params.add(name='Ka',  value=-0.11, vary=True)
    lfo_params.add(name='Kc',  value=0, vary=False)
    lfo_params.add(name='J1',  value=5.4, vary=True)
    lfo_params.add(name='J2',  value=0, vary=False)
    lfo_params.add(name='J1ab',  expr="J1")
    lfo_params.add(name='J1c',  expr="J1")
    lfo_params.add(name='J2a', expr="J2")
    lfo_params.add(name='J2b', expr="J2")
    lfo_params.add(name='J2d',  expr="J2")
    lfo_params.add(name='Dab_x',  value=0.13*0.554, vary=False)
    lfo_params.add(name='Dab_y',  value=0.13*0.553, vary=False)
    lfo_params.add(name='Dab_z',  value=0.13*0.623, vary=False)
    lfo_params.add(name='Dc_x',  value=0.158*0.191, vary=False)
    lfo_params.add(name='Dc_y',  value=0.158*0.982, vary=False)
    lfo_params.add(name='Dc_z',  value=0, vary=False)
    lfo_params.add(name='Fz',  value=0.05, vary=True)
    models['J1'] = lfo_params


    # Simple J1-J2 model no more
    # Unstable, as it scales J1 an J2 up
    lfo_params = Parameters()
    lfo_params.add(name='Ka',  value=-0.28, vary=True)
    lfo_params.add(name='Kc',  value=0, vary=False)
    lfo_params.add(name='J1',  value=5.7, vary=True)
    lfo_params.add(name='J2',  value=0.51, vary=True)
    lfo_params.add(name='J1ab',  expr="J1")
    lfo_params.add(name='J1c',  expr="J1")
    lfo_params.add(name='J2a', expr="J2")
    lfo_params.add(name='J2b', expr="J2")
    lfo_params.add(name='J2d',  expr="J2")
    lfo_params.add(name='Dab_x',  value=0.13*0.554, vary=False)
    lfo_params.add(name='Dab_y',  value=0.13*0.553, vary=False)
    lfo_params.add(name='Dab_z',  value=0.13*0.623, vary=False)
    lfo_params.add(name='Dc_x',  value=0.158*0.191, vary=False)
    lfo_params.add(name='Dc_y',  value=0.158*0.982, vary=False)
    lfo_params.add(name='Dc_z',  value=0, vary=False)
    lfo_params.add(name='Fz',  value=0, vary=True)
    models['J12'] = lfo_params

    # J1 J2 all split
    # Simple J1-J2 model no more
    # Unstable, as it scales J1 an J2 up
    # Ka:    -0.09879743 +/- 0.05797001 (58.68%) (init = -0.28)
    # Kc:     0 (fixed)
    # J1ab:   3.79200079 +/- 3.54502045 (93.49%) (init = 5)
    # J1c:    6.08123666 +/- 0.67811506 (11.15%) (init = 4)
    # J2a:   -0.71368333 +/- 2.10736316 (295.28%) (init = 0.8)
    # J2b:   -0.71368333 +/- 2.10736316 (295.28%) == 'J2a'
    # J2d:    0.36005280 +/- 0.80274241 (222.95%) (init = 0.3)
    # Dab_x:  0.07202 (fixed)
    # Dab_y:  0.07189 (fixed)
    # Dab_z:  0.08099 (fixed)
    # Dc_x:   0.030178 (fixed)
    # Dc_y:   0.155156 (fixed)
    # Dc_z:   0 (fixed)
    # Fz:     0 (fixed)
    lfo_params = Parameters()
    lfo_params.add(name='Ka',  value=-0.1, vary=True)
    lfo_params.add(name='Kc',  value=0, vary=False)
    lfo_params.add(name='J1ab', value=3.8, vary=True)
    lfo_params.add(name='J1c',  value=6, vary=True)
    lfo_params.add(name='J2a',  value=-0.7, vary=True)
    lfo_params.add(name='J2b',  expr='J2a')
    lfo_params.add(name='J2d',  value=0.36, vary=True)
    lfo_params.add(name='Dab_x',  value=0.13*0.554, vary=False)
    lfo_params.add(name='Dab_y',  value=0.13*0.553, vary=False)
    lfo_params.add(name='Dab_z',  value=0.13*0.623, vary=False)
    lfo_params.add(name='Dc_x',  value=0.158*0.191, vary=False)
    lfo_params.add(name='Dc_y',  value=0.158*0.982, vary=False)
    lfo_params.add(name='Dc_z',  value=0, vary=False)
    lfo_params.add(name='Fz',  value=0.05, vary=True)
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
    lfo_params.add(name='J2a', value=-0.2, vary=True)   # J5
    # lfo_params.add(name='J2b', value=0.224, vary=True)    # J3
    lfo_params.add(name='J2b', value=-0.2, expr='J2a')    # J3
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

    DATA = load_data()
    print(DATA)

    # Define main parameters
    fit = True

    lfo_params = load_lfo_parameters('J1')
    sw = load_system(lfo_params, show_struct=False, silent=False)

    if fit:
        fit_result = fit_lfo(p0=lfo_params, DATA=DATA)
        lfo_params = fit_result.params

    sw = load_system(lfo_params, show_struct=False, silent=False)
    fig = plot_fit(sw, DATA, plot_type='dispersion')
    fig.savefig(PATH+'\spinwaves-LuFeO3-residuals-final.png', dpi=400)

    fig = plot_spectrum(lfo_params, DATA, plot_type='dispersion')
    fig.savefig(PATH+'\spinwaves-LuFeO3-Eq.png', dpi=400)
    # fig = plot_spectrum(lfo_params, DATA, plot_type='spectral_weight')
    # fig.savefig(PATH+'\spinwaves-LuFeO3-Sqw.png', dpi=400)



