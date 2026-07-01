path = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/schwarmax_pkg/'
import sys
sys.path.append(path)

import os 
import jax
import jax.numpy as jnp

from functools import partial
from utils import logMenc_logc_to_logM_logRs, load_data_bootstrap
from constants import EPSILON
from model import build_model, model_deltaChi2_jackknife
from potentials import NFW_potential, MiyamotoNagai_potential, MiyamotoNagai_density, T3_density, T3_potential, V4_density, V4_potential

# ---- Fixed V4 bulge parameters ----
V4_A, V4_B, V4_L, V4_GAMMA = 0.5, 0.5, 0.1, 0.0
GAMMA_BAR = 1.0

@jax.jit
def density_func(x, y, z, params):
    """
    Returns Stellar Density nu(R, z) using:
      MiyamotoNagai disc + T3 bar + V4 bulge
    """
    # MN disc density
    mn_params = {
        'logM_disc': params['logM_disc'],
        'Rs_disc': params['Rs_disc'],
        'Hs_disc': params['Hs_disc'],
        'x_origin': params['x_origin'],
        'y_origin': params['y_origin'],
        'z_origin': params['z_origin'],
        'dirx': params['dirx'],
        'diry': params['diry'],
        'dirz': params['dirz'],
    }
    rho_mn = MiyamotoNagai_density(x, y, z, mn_params)

    # T3 bar density
    M_bar = 10.0 ** params['logM_bar']
    L_bar = params['L_bar']
    a_bar = params['a_bar']
    b_bar = params['b_bar']
    rho_t3 = T3_density(x, y, z, M_bar, a_bar, b_bar, L_bar, GAMMA_BAR)

    # V4 bulge density (M = M_bar, fixed shape)
    rho_v4 = V4_density(x, y, z, M_bar, V4_A, V4_B, V4_L, V4_GAMMA)

    return rho_mn + rho_t3 + rho_v4


@jax.jit
def potential_func(x, y, z, params_baryon, params_halo):
    """Returns Phi(x,y,z) = NFW + MiyamotoNagai + T3 + V4"""
    phi_halo = NFW_potential(x, y, z, params_halo)

    mn_params = {
        'logM_disc': params_baryon['logM_disc'],
        'Rs_disc': params_baryon['Rs_disc'],
        'Hs_disc': params_baryon['Hs_disc'],
        'x_origin': params_baryon['x_origin'],
        'y_origin': params_baryon['y_origin'],
        'z_origin': params_baryon['z_origin'],
        'dirx': params_baryon['dirx'],
        'diry': params_baryon['diry'],
        'dirz': params_baryon['dirz'],
    }
    phi_mn = MiyamotoNagai_potential(x, y, z, mn_params)

    M_bar = 10.0 ** params_baryon['logM_bar']
    L_bar = params_baryon['L_bar']
    a_bar = params_baryon['a_bar']
    b_bar = params_baryon['b_bar']
    phi_t3 = T3_potential(x, y, z, M_bar, a_bar, b_bar, L_bar, GAMMA_BAR)
    phi_v4 = V4_potential(x, y, z, M_bar, V4_A, V4_B, V4_L, V4_GAMMA)

    return phi_halo + phi_mn + phi_t3 + phi_v4

def chi2_func(y_Rzphi, y_Rzphi_model, sigma_Rzphi,
              y_xy, y_xy_model, sigma_xy,
              y_h1, y_h1_model, sigma_h1,
              y_h2, y_h2_model, sigma_h2,
              y_h3, y_h3_model, sigma_h3,
              y_h4, y_h4_model, sigma_h4,):

    sig_Rzphi = sigma_Rzphi #jnp.where(sigma_Rzphi < 1e10, 0.1 * y_Rzphi, sigma_Rzphi)
    sig_xy = y_xy * 0.01 #jnp.where(sigma_xy < 1e10, 0.1 * y_xy, sigma_xy)
    sig_h1 = sigma_h1
    sig_h2 = sigma_h2
    sig_h3 = sigma_h3
    sig_h4 = sigma_h4

    res_Rzphi = jnp.nansum(((y_Rzphi - y_Rzphi_model) / sig_Rzphi + EPSILON)**2)
    res_xy = jnp.nansum(((y_xy - y_xy_model) / sig_xy + EPSILON)**2)
    res_h1 = jnp.nansum(((y_h1 - y_h1_model) / sig_h1 + EPSILON)**2)
    res_h2 = jnp.nansum(((y_h2 - y_h2_model) / sig_h2 + EPSILON)**2)
    res_h3 = jnp.nansum(((y_h3 - y_h3_model) / sig_h3 + EPSILON)**2)
    res_h4 = jnp.nansum(((y_h4 - y_h4_model) / sig_h4 + EPSILON)**2)

    return -0.5 * (res_h1 + res_h2 + res_h3 + res_h4 + res_xy)

@partial(jax.jit, static_argnames=('num_Vbin', 'Rzphi_n_tot'))
def calculate_delta_chi2(params, dict_data, num_Vbin, Rzphi_n_tot):
    """
    Same as logl_angular_input but uses model_bootstrap() which marginalizes
    logL over observational noise via bootstrap resampling:
        - 100 perturbed observation vectors y_i = y + N(0, sig)
        - Shared Cholesky, vmapped ADMM solves
        - logL_marg = log(mean(exp(logL_i)))
    """
    logM_enc = params[0]
    log_c = params[3]
    logM_halo, logRs_halo = logMenc_logc_to_logM_logRs(logM_enc, log_c, r_enc=10.0, Delta=200., rho_crit=277.54)

    # logM_halo = params[0]
    logM_disc = params[1]
    logM_bar = params[2]
    # logRs_halo = params[3]
    logRs_disk = params[4]
    logHs_disk = params[5]
    logL_bar = params[6]
    alpha = params[7]
    beta = params[8]
    gamma = params[9]
    log_mass_to_light_ratio = params[10]
    log_Omega_bar = params[11]

    alpha = alpha * 180 / jnp.pi
    beta = beta * 180 / jnp.pi
    gamma = gamma * 180 / jnp.pi

    # Derived bar parameters
    L_bar = 10.0 ** logL_bar
    a_bar = L_bar / 5.0
    Hs_disc = 10.0 ** logHs_disk
    b_bar = Hs_disc

    params_halo_pot = {
        'logM': logM_halo,
        'Rs':10 ** logRs_halo,
        'a':1.0,
        'b':1.0,
        'c':1.0,
        'x_origin':0.0,
        'y_origin':0.0,
        'z_origin':0.0,
        'dirx':0.0,
        'diry':0.0,
        'dirz':1.0
    }

    params_baryon_rho = {
        'logM_disc': logM_disc,
        'Rs_disc': 10 ** logRs_disk,
        'Hs_disc': Hs_disc,
        'logM_bar': logM_bar,
        'L_bar': L_bar,
        'a_bar': a_bar,
        'b_bar': b_bar,
        'mass_to_light_ratio': 10 ** log_mass_to_light_ratio,
        'Omega_bar': 10 ** log_Omega_bar,
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
    }

    X_minmax = dict_data['X_minmax']
    Y_minmax = dict_data['Y_minmax']
    nX, nY = dict_data['nX_nY']
    xy_lim_grid = jnp.array([X_minmax, Y_minmax])
    xy_n_grid = jnp.array([nX, nY])

    Rmin, Rmax = dict_data['R_minmax']
    zmin, zmax = dict_data['z_minmax']
    phimin, phimax = dict_data['phi_minmax']
    Rzphi_n_grid = dict_data['Rzphi_n_grid']

    delta_chi2 = model_deltaChi2_jackknife(density_func, potential_func, chi2_func,
                    params_halo_pot, params_baryon_rho, dict_data, num_Vbin,
                    Rzphi_n_tot, Rzphi_n_grid, Rzphi_lim_grid=jnp.array([[Rmin, Rmax],[zmin, zmax],[phimin, phimax]]),
                    xy_lim_grid=xy_lim_grid, xy_n_grid=xy_n_grid,
                    nnls_maxiter=200, regularization=1.0,
                    n_groups = num_Vbin, batch_size = 25)
    return delta_chi2

path = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/SchwarMAX'
filename = '/mock_data2/mock_Nbody_bar_XY_withRot_Nbins600_beta65_gamma45_D50_gal2.pkl'
# filename = 'mock_Nbody_bar_XY_withRot_gal2_Nbins1000.pkl'
dict_data = load_data_bootstrap(path, filename, n_samples = 7_500)

import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

path = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/'

N_CHAINS = 32              # number of parallel chains (must fit in GPU vmap)
N_STEPS = 1000             # total MCMC steps per chain
CHECKPOINT_EVERY = 5
BURNIN = 300
ADAPT_EVERY = 50          # re-estimate covariance every N steps
ADAPT_AFTER = 150          # delay adaptation until chains have explored


MINIMISER_RESULT = os.path.join(path, 'results/minimise_0531_Nbins600_beta65_gamma45_D50_gal2.npy')

# res = np.load(MINIMISER_RESULT)[:-1]
# res[-2] = -res[-2]
# res = jnp.array(res)
res = np.array([10.674979  , 10.717889  , 10.196065  ,  0.9875556 ,  0.97132087,
       -0.25643212,  0.74568737,  0.46487555,  1.1576498 ,  1.0315903 ,
       -0.1659575 ,  1.3822253 ])

delta_chi2 = calculate_delta_chi2(res, dict_data, dict_data['total_bins'], dict_data['Rzphi_n_tot'])
print(delta_chi2)