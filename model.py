import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial

from utils import (makeRotationMatrix, get_rotation_curve, estimate_orbital_timescale,
                   axisymmetrize_potential)
from constants import EPSILON
from integrates import _integrate_adaptive_batch_chunked_vmap, _integrate_adaptive_withtraj_batch_vmap
from nnls import solve_nnls_admm_bootstrap, solve_nnls_admm
from jeans import get_w0_new, get_w0_new_improved
from ghmoments import h_to_V_sigma, h_to_V_sigma_GH

@jax.jit
def compute_model_single(
    w,
    A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4, gamma_kin,
    v0, s,
    ):
    """
    Compute best-fit model vectors and logL for each bootstrap weight vector.

    Each weight vector is evaluated against its corresponding bootstrapped data.
    y_Rzphi is model-computed (shared across all bootstraps).

    Args:
        weights_all: (N_boot, n_orb)
        A_*: orbital library matrices — shared.
            A_h_k = sw_k / T_integrated (un-normalised vdMF moments).
        gamma_kin: per-bin base-Gaussian amplitude (n_xy_bins,) = M_b / (norm_b · 2√π · s_b).
        y_Rzphi: model-computed 3D density (n_Rzphi_bins,) — shared
        y_xy: surface density (n_xy_bins,)
        y_h1..y_h4: kinematics (n_xy_bins,)
        sig_*: error vectors — shared
        v0, s: for h_to_V_sigma conversion
        light_to_mass_ratio: conversion factor from light to mass
        mass_per_orb: mass per orbit

    Returns:
        logl_marg: scalar, log(mean(exp(logL_i)))
        density_2DXY_all: (N_boot, n_xy_bins)
        h1_all, h2_all, h3_all, h4_all: (N_boot, n_xy_bins)
        V_all, sigma_all: (N_boot, n_xy_bins)
        logl_all: (N_boot,)
    """
    eps = 1e-8
    gamma_kin_safe = jnp.where(jnp.abs(gamma_kin) > eps, gamma_kin, eps)

    density_3d = A_Rzphi @ w
    density_2DXY = A_xy @ w
    # Model GH moments: divide by gamma_kin (data-side base-Gaussian amplitude),
    # NOT by the model surface density. This is the Agama option-b convention.
    h1_model = (A_h1 @ w) / gamma_kin_safe
    h2_model = (A_h2 @ w) / gamma_kin_safe
    h3_model = (A_h3 @ w) / gamma_kin_safe
    h4_model = (A_h4 @ w) / gamma_kin_safe

    clip_val = 10.0
    h1_model = jnp.clip(h1_model, -clip_val, clip_val)
    h2_model = jnp.clip(h2_model, -clip_val, clip_val)
    h3_model = jnp.clip(h3_model, -clip_val, clip_val)
    h4_model = jnp.clip(h4_model, -clip_val, clip_val)
    V_model, sigma_model = h_to_V_sigma(h1_model, h2_model, v0, s,
                                        h3=h3_model, h4=h4_model)
    
    return density_3d, density_2DXY, h1_model, h2_model, h3_model, h4_model, V_model, sigma_model

def compute_model_single_GH(
    w,
    A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4, gamma_kin,
    v0, s,
    ):
    """
    Compute best-fit model vectors and logL for each bootstrap weight vector.

    Each weight vector is evaluated against its corresponding bootstrapped data.
    y_Rzphi is model-computed (shared across all bootstraps).

    Args:
        weights_all: (N_boot, n_orb)
        A_*: orbital library matrices — shared.
            A_h_k = sw_k / T_integrated (un-normalised vdMF moments).
        gamma_kin: per-bin base-Gaussian amplitude (n_xy_bins,) = M_b / (norm_b · 2√π · s_b).
        y_Rzphi: model-computed 3D density (n_Rzphi_bins,) — shared
        y_xy: surface density (n_xy_bins,)
        y_h1..y_h4: kinematics (n_xy_bins,)
        sig_*: error vectors — shared
        v0, s: for h_to_V_sigma conversion
        light_to_mass_ratio: conversion factor from light to mass
        mass_per_orb: mass per orbit

    Returns:
        logl_marg: scalar, log(mean(exp(logL_i)))
        density_2DXY_all: (N_boot, n_xy_bins)
        h1_all, h2_all, h3_all, h4_all: (N_boot, n_xy_bins)
        V_all, sigma_all: (N_boot, n_xy_bins)
        logl_all: (N_boot,)
    """
    eps = 1e-8
    gamma_kin_safe = jnp.where(jnp.abs(gamma_kin) > eps, gamma_kin, eps)

    density_3d = A_Rzphi @ w
    density_2DXY = A_xy @ w
    # Model GH moments: divide by gamma_kin (data-side base-Gaussian amplitude),
    # NOT by the model surface density. This is the Agama option-b convention.
    h1_model = (A_h1 @ w) / gamma_kin_safe
    h2_model = (A_h2 @ w) / gamma_kin_safe
    h3_model = (A_h3 @ w) / gamma_kin_safe
    h4_model = (A_h4 @ w) / gamma_kin_safe

    clip_val = 10.0
    h1_model = jnp.clip(h1_model, -clip_val, clip_val)
    h2_model = jnp.clip(h2_model, -clip_val, clip_val)
    h3_model = jnp.clip(h3_model, -clip_val, clip_val)
    h4_model = jnp.clip(h4_model, -clip_val, clip_val)
    # pPXF / paper convention: report the GH reference-Gaussian (h1=h2=0), matching how
    # the data's v0, s are defined.  The classical-moment h_to_V_sigma() adds a
    # sqrt(3)*s*h3 skewness term the data does not have.
    V_model, sigma_model = h_to_V_sigma_GH(h1_model, h2_model, v0, s)
    
    return density_3d, density_2DXY, h1_model, h2_model, h3_model, h4_model, V_model, sigma_model

compute_model_single_GH_vmap = jax.vmap(compute_model_single_GH, in_axes=(0,None,None,None,None,None,None,None,None,None))

compute_model_single_vmap = jax.vmap(compute_model_single, in_axes=(0,None,None,None,None,None,None,None,None,None))

@partial(jax.jit, static_argnames=('density_func', 'potential_func', 'num_Vbin', 'Rzphi_n_tot', 'nnls_maxiter'))
def build_model(density_func, potential_func,
                params_halo_pot, params_disk_rho, dict_data, num_Vbin,
                Rzphi_n_tot=360, Rzphi_n_grid = jnp.array([10,6,6]), Rzphi_lim_grid = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]]),
                xy_lim_grid = jnp.array([[-10.,10.],[-3.,3.]]), xy_n_grid = jnp.array([60,40]),
                nnls_maxiter=200, regularization = 1.0
                ):

    w0 = dict_data['w0']
    n_particles = w0.shape[0]
    v0 = dict_data['v0']
    s = dict_data['s']
    num_per_bin = dict_data['num_per_bin']
    bin_mapping = dict_data['bin_mapping']

    #=========================================== Get potential parameters =====================================================

    params_baryon = {
        'logM_disc': params_disk_rho['logM_disc'],
        'Rs_disc': params_disk_rho['Rs_disc'],
        'Hs_disc': params_disk_rho['Hs_disc'],
        'logM_bar': params_disk_rho['logM_bar'],
        'L_bar': params_disk_rho['L_bar'],
        'a_bar': params_disk_rho['a_bar'],
        'b_bar': params_disk_rho['b_bar'],

        'x_origin': params_disk_rho['x_origin'],
        'y_origin': params_disk_rho['y_origin'],
        'z_origin': params_disk_rho['z_origin'],
        'dirx': params_disk_rho['dirx'],
        'diry': params_disk_rho['diry'],
        'dirz': params_disk_rho['dirz'],
    }

    light_to_mass_ratio = 1/params_disk_rho['mass_to_light_ratio']

    Omega_bar = params_disk_rho['Omega_bar']
    alpha, beta, gamma = params_disk_rho['alpha'], params_disk_rho['beta'], params_disk_rho['gamma']
    rotation_matrix = makeRotationMatrix(alpha, beta, gamma)

    #=========================================== GET initial condition ===================================================

    key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
    w0_new = get_w0_new(w0, key1, key2, key3, n_particles,
                        params_baryon, params_halo_pot, potential_func, density_func)

    #======================================== Calculate orbital timescale =====================================================
    _R = jnp.sqrt(w0_new[:,0]**2 + w0_new[:,1]**2)
    _z = w0_new[:,2]

    _Vc = jax.vmap(get_rotation_curve, in_axes=(0, None, None, 0))(
        _R,
        potential_func,
        (params_baryon, params_halo_pot),
        _z
    )

    n_realizations = 4
    key = jax.random.PRNGKey(911)
    keys = jax.random.split(key, 6)
    d_scale = 0.1 * jnp.ones(_R.shape)
    v_scale = 0.1 * _Vc
    v_scale = jnp.clip(v_scale, a_min=1, a_max = 15)
    noise_x = (jax.random.uniform(keys[0], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_y = (jax.random.uniform(keys[1], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_z = (jax.random.uniform(keys[2], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_vx = (jax.random.uniform(keys[3], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vy = (jax.random.uniform(keys[4], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vz = (jax.random.uniform(keys[5], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]

    w0_new_batch = w0_new[:, jnp.newaxis, :]
    w0_new_batch = w0_new_batch + jnp.stack([noise_x, noise_y, noise_z, noise_vx, noise_vy, noise_vz], axis=-1)
    T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0, None, None, 0))(
        _R,
        potential_func,
        (params_baryon, params_halo_pot),
        _z
    )
    T_orb_batch = T_orb[:, jnp.newaxis].repeat(n_realizations, axis=1)

    #=========================================== Construct orbital library =======================================================
    # Single combined potential + grad for acceleration — one forward + one backward pass
    @jax.jit
    def acc_fn(x, y, z):
        def _pot(pos):
            return potential_func(pos[0], pos[1], pos[2], params_baryon, params_halo_pot)
        grad_phi = jax.grad(_pot)(jnp.array([x, y, z]))
        return -grad_phi

    @jax.jit
    def pot_fn(x, y, z):
        return potential_func(x, y, z, params_baryon, params_halo_pot)

    N_step_per_orb = 100
    N_dynamical_time = 50
    N_max = N_step_per_orb * N_dynamical_time
    T_total_batch = T_orb_batch * N_dynamical_time
    dt_init_batch = T_orb_batch / N_step_per_orb
    atol, rtol = 1e-7, 1e-4
    dt_min, dt_max = 1e-5, 0.3

    Rzphi_bin_counts, surface_density, h0, h1, h2, h3, h4, _ = _integrate_adaptive_batch_chunked_vmap(
                        w0_new_batch, acc_fn, pot_fn, N_max, T_total_batch,
                        dt_init_batch, -Omega_bar,
                        atol, rtol,    
                        dt_min, dt_max,    
                        num_Vbin, bin_mapping, num_per_bin,
                        Rzphi_lim_grid, xy_lim_grid,
                        Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
                        v0, s, rotation_matrix,
                        100, # chunk size
    )
    A_Rzphi = Rzphi_bin_counts.T
    A_xy = surface_density.T
    A_h0 = h0.T
    A_h1 = h1.T
    A_h2 = h2.T
    A_h3 = h3.T
    A_h4 = h4.T

    #================================== Construct 3D density target ============================================

    @jax.jit
    def density_func_Rz(R, z, phi, params):
        x = R * jnp.cos(phi)
        y = R * jnp.sin(phi)
        return density_func(x, y, z, params)

    @partial(jax.jit, static_argnames=['rho_fct'])
    def get_mass(R_grid, z_grid, phi_grid, rho_fct, dict_params, dR, dz, dphi, sample):
        R_samples = R_grid + (sample[:,0] - 0.5) * dR
        z_samples = z_grid + (sample[:,1] - 0.5) * dz
        phi_samples = phi_grid + (sample[:,2] - 0.5) * dphi
        density_samples = rho_fct(R_samples, z_samples, phi_samples, dict_params)
        mass_tot = jnp.sum(density_samples * R_samples) / sample.shape[0]
        return mass_tot * dR * dz * dphi

    R_grid, dR = dict_data['R_grid'], dict_data['dR']
    z_grid, dz = dict_data['z_grid'], dict_data['dz']
    phi_grid, dphi = dict_data['phi_grid'], dict_data['dphi']
    y_Rzphi = jax.vmap(get_mass, in_axes=[0, 0, 0, None, None, None, None, None, None])(
                R_grid, z_grid, phi_grid, density_func_Rz, params_disk_rho, dR, dz, dphi, dict_data['sample_for_integration']
    )
    sig_Rzphi = 0.02 * y_Rzphi + 1e-10

    #================================== Read the surface density and kinematic targets ============================================

    y_xy = dict_data['XY_density_data'] / light_to_mass_ratio
    sig_xy = (dict_data['XY_density_data_err'] + EPSILON) / light_to_mass_ratio

    y_h1 = dict_data['h1_data']
    y_h2 = dict_data['h2_data']
    y_h3 = dict_data['h3_data']
    y_h4 = dict_data['h4_data']
    sig_A1 = dict_data['h1_data_err'] + EPSILON
    sig_A2 = dict_data['h2_data_err'] + EPSILON
    sig_A3 = dict_data['h3_data_err'] + EPSILON
    sig_A4 = dict_data['h4_data_err'] + EPSILON

    #================================== Renormalise the 3D and 2D density constrain ============================================

    mean_mass_per_orb = jnp.sum(y_Rzphi) / A_Rzphi.shape[1]
    y_xy = y_xy / mean_mass_per_orb
    sig_xy = sig_xy / mean_mass_per_orb
    y_Rzphi = y_Rzphi / mean_mass_per_orb
    sig_Rzphi = sig_Rzphi / mean_mass_per_orb

    #=========================================== Gamma_kin: base-Gaussian amplitude (option-b normalisation) ====================
    # gamma_kin_b = M_b / (norm_b * 2*sqrt(pi) * s_b), the data's h_0 in the bin's basis.
    # M_b is the per-bin photometric mass; norm_b absorbs the higher-order GH contribution to the LOSVD integral.  In the rescaled-by-mean_mass_per_orb units used here,
    # M_b = y_xy * num_per_bin * area_per_pixel_in_pc^2 (rescaled).
    X_minmax_arr = dict_data['X_minmax']
    Y_minmax_arr = dict_data['Y_minmax']
    nXY_arr = dict_data['nX_nY']
    area_per_pixel_pc2 = ((X_minmax_arr[1] - X_minmax_arr[0]) / nXY_arr[0]) * \
                        ((Y_minmax_arr[1] - Y_minmax_arr[0]) / nXY_arr[1]) * 1e6
    M_per_bin = y_xy * num_per_bin * area_per_pixel_pc2
    norm_GH = 1.0 + y_h4 * jnp.sqrt(6.0) / 4.0
    gamma_kin = M_per_bin / (norm_GH * 2.0 * jnp.sqrt(jnp.pi) * s + EPSILON)

    #=========================================== Bootstrap NNLS solver (vmapped) ============================================

    # # Bootstrapped observations: (N_boot, n_bins) — pre-computed and stored in dict_data

    y_xy_boot = y_xy[None, :] + dict_data['XY_standard_normal'] * sig_xy[None, :]
    y_h1_boot = y_h1[None, :] + dict_data['h1_standard_normal'] * sig_A1[None, :]
    y_h2_boot = y_h2[None, :] + dict_data['h2_standard_normal'] * sig_A2[None, :]
    y_h3_boot = y_h3[None, :] + dict_data['h3_standard_normal'] * sig_A3[None, :]
    y_h4_boot = y_h4[None, :] + dict_data['h4_standard_normal'] * sig_A4[None, :]

    weights_all = solve_nnls_admm_bootstrap(
                            A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                            y_Rzphi, y_xy, gamma_kin,
                            y_xy_boot, y_h1_boot, y_h2_boot, y_h3_boot, y_h4_boot,
                            sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
                            lambda_reg=regularization, maxiter=nnls_maxiter,
                            w_rzphi = jnp.sqrt(5.0), w_xy = jnp.sqrt(5.0), w_h = jnp.sqrt(1.0)
    )  # (N_boot, n_orb)

    #===================================== Compute model vectors + logL for each bootstrap ==================================

    density_3d_model, density_all, h1_all, h2_all, h3_all, h4_all, V_all, sigma_all = compute_model_single_vmap(weights_all, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4, gamma_kin, v0, s)
    density_all = density_all * mean_mass_per_orb * light_to_mass_ratio  # convert back to luminosity units for density

    results_dict = {
        'density_3d':y_Rzphi,
        'density_3d_model': density_3d_model,
        'surface_density': density_all,
        'h1': h1_all,
        'h2': h2_all,
        'h3': h3_all,
        'h4': h4_all,
        'V': V_all,
        'sigma': sigma_all,
        'weights': weights_all,
    }
    return results_dict



@partial(jax.jit, static_argnames=('density_func', 'potential_func', 'chi2_func', 'num_Vbin', 'Rzphi_n_tot', 'nnls_maxiter', 'n_groups', 'batch_size'))
def model_deltaChi2_jackknife(density_func, potential_func, chi2_func,
                params_halo_pot, params_disk_rho, dict_data, num_Vbin,
                Rzphi_n_tot=360, Rzphi_n_grid = jnp.array([10,6,6]), Rzphi_lim_grid = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]]),
                xy_lim_grid = jnp.array([[-10.,10.],[-3.,3.]]), xy_n_grid = jnp.array([60,40]),
                nnls_maxiter=200, regularization = 1.0,
                n_groups=100, batch_size=50,
                ):
    """Delete-d jackknife with a user-supplied chi^2 function.

    The user supplies `chi2_func` with the signature

        chi2_func(y_Rzphi, y_Rzphi_model, sigma_Rzphi,
                  y_xy,    y_xy_model,    sigma_xy,
                  y_h1,    y_h1_model,    sigma_h1,
                  y_h2,    y_h2_model,    sigma_h2,
                  y_h3,    y_h3_model,    sigma_h3,
                  y_h4,    y_h4_model,    sigma_h4)
        -> scalar

    Each replicate g drops one disjoint group of Voronoi bins by setting
    its `sigma_*` entries to 1e20 (true per-replicate masking). The NNLS
    is re-solved per replicate via `vmap(solve_nnls_admm)` -- with σ
    actually changing across replicates, U / Q / Cholesky all rebuild
    internally, which is what makes the masked bins drop out of the
    constraint set. After solving we evaluate the user's `chi2_func`
    per replicate (also vmapped) and return `jnp.std(chi2_all, axis=0)`.
    """

    w0 = dict_data['w0']
    n_particles = w0.shape[0]
    v0 = dict_data['v0']
    s = dict_data['s']
    num_per_bin = dict_data['num_per_bin']
    bin_mapping = dict_data['bin_mapping']

    #=========================================== Get potential parameters =====================================================

    params_baryon = {
        'logM_disc': params_disk_rho['logM_disc'],
        'Rs_disc': params_disk_rho['Rs_disc'],
        'Hs_disc': params_disk_rho['Hs_disc'],
        'logM_bar': params_disk_rho['logM_bar'],
        'L_bar': params_disk_rho['L_bar'],
        'a_bar': params_disk_rho['a_bar'],
        'b_bar': params_disk_rho['b_bar'],

        'x_origin': params_disk_rho['x_origin'],
        'y_origin': params_disk_rho['y_origin'],
        'z_origin': params_disk_rho['z_origin'],
        'dirx': params_disk_rho['dirx'],
        'diry': params_disk_rho['diry'],
        'dirz': params_disk_rho['dirz'],
    }

    light_to_mass_ratio = 1/params_disk_rho['mass_to_light_ratio']

    Omega_bar = params_disk_rho['Omega_bar']
    alpha, beta, gamma = params_disk_rho['alpha'], params_disk_rho['beta'], params_disk_rho['gamma']
    rotation_matrix = makeRotationMatrix(alpha, beta, gamma)

    #=========================================== GET initial condition ===================================================

    key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
    w0_new = get_w0_new(w0, key1, key2, key3, n_particles,
                        params_baryon, params_halo_pot, potential_func, density_func)

    #======================================== Calculate orbital timescale =====================================================
    _R = jnp.sqrt(w0_new[:,0]**2 + w0_new[:,1]**2)
    _z = w0_new[:,2]

    _Vc = jax.vmap(get_rotation_curve, in_axes=(0, None, None, 0))(
        _R,
        potential_func,
        (params_baryon, params_halo_pot),
        _z
    )

    n_realizations = 4
    key = jax.random.PRNGKey(911)
    keys = jax.random.split(key, 6)
    d_scale = 0.1 * jnp.ones(_R.shape)
    v_scale = 0.1 * _Vc
    v_scale = jnp.clip(v_scale, a_min=1, a_max = 15)
    noise_x = (jax.random.uniform(keys[0], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_y = (jax.random.uniform(keys[1], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_z = (jax.random.uniform(keys[2], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_vx = (jax.random.uniform(keys[3], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vy = (jax.random.uniform(keys[4], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vz = (jax.random.uniform(keys[5], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]

    w0_new_batch = w0_new[:, jnp.newaxis, :]
    w0_new_batch = w0_new_batch + jnp.stack([noise_x, noise_y, noise_z, noise_vx, noise_vy, noise_vz], axis=-1)
    T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0, None, None, 0))(
        _R,
        potential_func,
        (params_baryon, params_halo_pot),
        _z
    )
    T_orb_batch = T_orb[:, jnp.newaxis].repeat(n_realizations, axis=1)

    #=========================================== Construct orbital library =======================================================
    # Single combined potential + grad for acceleration — one forward + one backward pass
    @jax.jit
    def acc_fn(x, y, z):
        def _pot(pos):
            return potential_func(pos[0], pos[1], pos[2], params_baryon, params_halo_pot)
        grad_phi = jax.grad(_pot)(jnp.array([x, y, z]))
        return -grad_phi

    @jax.jit
    def pot_fn(x, y, z):
        return potential_func(x, y, z, params_baryon, params_halo_pot)

    N_step_per_orb = 100
    N_dynamical_time = 50
    N_max = N_step_per_orb * N_dynamical_time
    T_total_batch = T_orb_batch * N_dynamical_time
    dt_init_batch = T_orb_batch / N_step_per_orb
    atol, rtol = 1e-7, 1e-4
    dt_min, dt_max = 1e-5, 0.3

    Rzphi_bin_counts, surface_density, h0, h1, h2, h3, h4, _ = _integrate_adaptive_batch_chunked_vmap(
                        w0_new_batch, acc_fn, pot_fn, N_max, T_total_batch,
                        dt_init_batch, -Omega_bar,
                        atol, rtol,    
                        dt_min, dt_max,    
                        num_Vbin, bin_mapping, num_per_bin,
                        Rzphi_lim_grid, xy_lim_grid,
                        Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
                        v0, s, rotation_matrix,
                        100, # chunk size
    )
    A_Rzphi = Rzphi_bin_counts.T
    A_xy = surface_density.T
    A_h0 = h0.T
    A_h1 = h1.T
    A_h2 = h2.T
    A_h3 = h3.T
    A_h4 = h4.T

    #================================== Construct 3D density target ============================================

    @jax.jit
    def density_func_Rz(R, z, phi, params):
        x = R * jnp.cos(phi)
        y = R * jnp.sin(phi)
        return density_func(x, y, z, params)

    @partial(jax.jit, static_argnames=['rho_fct'])
    def get_mass(R_grid, z_grid, phi_grid, rho_fct, dict_params, dR, dz, dphi, sample):
        R_samples = R_grid + (sample[:,0] - 0.5) * dR
        z_samples = z_grid + (sample[:,1] - 0.5) * dz
        phi_samples = phi_grid + (sample[:,2] - 0.5) * dphi
        density_samples = rho_fct(R_samples, z_samples, phi_samples, dict_params)
        mass_tot = jnp.sum(density_samples * R_samples) / sample.shape[0]
        return mass_tot * dR * dz * dphi

    R_grid, dR = dict_data['R_grid'], dict_data['dR']
    z_grid, dz = dict_data['z_grid'], dict_data['dz']
    phi_grid, dphi = dict_data['phi_grid'], dict_data['dphi']
    y_Rzphi = jax.vmap(get_mass, in_axes=[0, 0, 0, None, None, None, None, None, None])(
                R_grid, z_grid, phi_grid, density_func_Rz, params_disk_rho, dR, dz, dphi, dict_data['sample_for_integration']
    )
    sig_Rzphi = 0.02 * y_Rzphi + 1e-10

    #================================== Read the surface density and kinematic targets ============================================

    y_xy = dict_data['XY_density_data'] / light_to_mass_ratio
    sig_xy = (dict_data['XY_density_data_err'] + EPSILON) / light_to_mass_ratio

    y_h1 = dict_data['h1_data']
    y_h2 = dict_data['h2_data']
    y_h3 = dict_data['h3_data']
    y_h4 = dict_data['h4_data']
    sig_A1 = dict_data['h1_data_err'] + EPSILON
    sig_A2 = dict_data['h2_data_err'] + EPSILON
    sig_A3 = dict_data['h3_data_err'] + EPSILON
    sig_A4 = dict_data['h4_data_err'] + EPSILON

    #================================== Renormalise the 3D and 2D density constrain ============================================

    mean_mass_per_orb = jnp.sum(y_Rzphi) / A_Rzphi.shape[1]
    y_xy = y_xy / mean_mass_per_orb
    sig_xy = sig_xy / mean_mass_per_orb
    y_Rzphi = y_Rzphi / mean_mass_per_orb
    sig_Rzphi = sig_Rzphi / mean_mass_per_orb

    #=========================================== Gamma_kin: base-Gaussian amplitude (option-b normalisation) ====================
    # gamma_kin_b = M_b / (norm_b * 2*sqrt(pi) * s_b), the data's h_0 in the bin's basis.
    # M_b is the per-bin photometric mass; norm_b absorbs the higher-order GH contribution to the LOSVD integral.  In the rescaled-by-mean_mass_per_orb units used here,
    # M_b = y_xy * num_per_bin * area_per_pixel_in_pc^2 (rescaled).
    X_minmax_arr = dict_data['X_minmax']
    Y_minmax_arr = dict_data['Y_minmax']
    nXY_arr = dict_data['nX_nY']
    area_per_pixel_pc2 = ((X_minmax_arr[1] - X_minmax_arr[0]) / nXY_arr[0]) * \
                        ((Y_minmax_arr[1] - Y_minmax_arr[0]) / nXY_arr[1]) * 1e6
    M_per_bin = y_xy * num_per_bin * area_per_pixel_pc2
    norm_GH = 1.0 + y_h4 * jnp.sqrt(6.0) / 4.0
    gamma_kin = M_per_bin / (norm_GH * 2.0 * jnp.sqrt(jnp.pi) * s + EPSILON)

    #=========================================== Jackknife: per-replicate sigma masks ========================================
    # Each replicate g drops the bins assigned to group g by raising their
    # sigma to BIG (1e20). The σ in the NNLS denominator U = A/σ then sends
    # those rows to ~0, so they no longer constrain the orbital weights.
    # Group assignment is just bin_index % n_groups -- the final statistic
    # is jnp.std(chi2, axis=0), invariant under reordering of replicates.
    n_bins       = sig_xy.shape[0]
    group_of_bin = jnp.arange(n_bins) % n_groups                          # (n_bins,)
    dropped      = (group_of_bin[None, :] == jnp.arange(n_groups)[:, None])  # (n_groups, n_bins)
    BIG = 1e20

    sig_xy_jk = jnp.where(dropped, BIG, sig_xy[None, :])                  # (n_groups, n_bins)
    sig_A1_jk = jnp.where(dropped, BIG, sig_A1[None, :])
    sig_A2_jk = jnp.where(dropped, BIG, sig_A2[None, :])
    sig_A3_jk = jnp.where(dropped, BIG, sig_A3[None, :])
    sig_A4_jk = jnp.where(dropped, BIG, sig_A4[None, :])
    # sig_Rzphi has no per-bin observation dimension to mask, so it is
    # shared across replicates.

    #=========================================== Batched NNLS + chi^2 (memory-bounded) ======================================
    # vmap'ing solve_nnls_admm across ALL n_groups replicates simultaneously
    # blows up memory because each replicate carries its own Cholesky and
    # ADMM-scan state. Instead, vmap within each batch of `batch_size`
    # replicates and use jax.lax.map across batches -- working memory is
    # bounded by batch_size, not n_groups.
    #
    # If n_groups is not divisible by batch_size, we pad the per-replicate
    # sigma arrays up to the next multiple by duplicating the last
    # replicate (any valid replicate would do); the padded entries are
    # trimmed before the final std. Because n_groups and batch_size are
    # both static, n_padded / n_pad / n_batches are all concrete at trace
    # time.
    n_padded  = -(-n_groups // batch_size) * batch_size   # round up
    n_pad     = n_padded - n_groups
    n_batches = n_padded // batch_size

    def _pad_replicates(arr):
        if n_pad == 0:
            return arr
        pad_block = jnp.broadcast_to(arr[-1:], (n_pad,) + arr.shape[1:])
        return jnp.concatenate([arr, pad_block], axis=0)

    sig_xy_b = _pad_replicates(sig_xy_jk).reshape(n_batches, batch_size, n_bins)
    sig_A1_b = _pad_replicates(sig_A1_jk).reshape(n_batches, batch_size, n_bins)
    sig_A2_b = _pad_replicates(sig_A2_jk).reshape(n_batches, batch_size, n_bins)
    sig_A3_b = _pad_replicates(sig_A3_jk).reshape(n_batches, batch_size, n_bins)
    sig_A4_b = _pad_replicates(sig_A4_jk).reshape(n_batches, batch_size, n_bins)

    # Per-replicate NNLS solver.
    def _solve_one(sxy_g, s1_g, s2_g, s3_g, s4_g):
        return solve_nnls_admm(
            A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
            y_Rzphi, y_xy, gamma_kin, y_h1, y_h2, y_h3, y_h4,
            sig_Rzphi, sxy_g, s1_g, s2_g, s3_g, s4_g,
            lambda_reg=regularization, maxiter=nnls_maxiter,
            w_rzphi = jnp.sqrt(5.0), w_xy = jnp.sqrt(5.0), w_h = jnp.sqrt(1.0)
        )

    # `chi2_func` expects every (data, model, sigma) triple to be in the
    # SAME units. The NNLS internal arrays (`y_xy`, `sig_xy`) were divided
    # by `light_to_mass_ratio * mean_mass_per_orb` for the solver — but
    # the per-replicate prediction `d2d_b` is multiplied BACK to
    # luminosity below. To keep units consistent we expose y_xy and
    # sig_xy in luminosity too. The 3-D density and h1..h4 are unchanged.
    lum_factor  = mean_mass_per_orb * light_to_mass_ratio
    y_xy_lum    = y_xy * lum_factor                                       # (n_bins,)

    # Pre-broadcast the shared data arrays once at batch size; cheap, and
    # avoids redoing the broadcast inside every batch.
    y_Rzphi_b   = jnp.broadcast_to(y_Rzphi,   (batch_size,) + y_Rzphi.shape)
    sig_Rzphi_b = jnp.broadcast_to(sig_Rzphi, (batch_size,) + sig_Rzphi.shape)
    y_xy_b      = jnp.broadcast_to(y_xy_lum,  (batch_size,) + y_xy_lum.shape)
    y_h1_b      = jnp.broadcast_to(y_h1,      (batch_size,) + y_h1.shape)
    y_h2_b      = jnp.broadcast_to(y_h2,      (batch_size,) + y_h2.shape)
    y_h3_b      = jnp.broadcast_to(y_h3,      (batch_size,) + y_h3.shape)
    y_h4_b      = jnp.broadcast_to(y_h4,      (batch_size,) + y_h4.shape)

    def _batch_pipeline(sigs_batch):
        sxy, s1, s2, s3, s4 = sigs_batch                                 # each: (batch_size, n_bins)

        # NNLS for batch_size replicates in parallel.
        weights_b = jax.vmap(_solve_one)(sxy, s1, s2, s3, s4)            # (batch_size, n_orb)

        # Forward model -> per-replicate predictions.
        d3d_b, d2d_b, h1_b, h2_b, h3_b, h4_b, _, _ = compute_model_single_vmap(
            weights_b, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4, gamma_kin, v0, s,
        )
        d2d_b      = d2d_b * lum_factor                                  # luminosity units
        sxy_lum    = sxy   * lum_factor                                  # match y_xy_b units

        # User's chi2_func, vmapped over the batch axis.
        return jax.vmap(chi2_func, in_axes=(0,)*18)(
            y_Rzphi_b, d3d_b, sig_Rzphi_b,
            y_xy_b,    d2d_b, sxy_lum,
            y_h1_b,    h1_b,  s1,
            y_h2_b,    h2_b,  s2,
            y_h3_b,    h3_b,  s3,
            y_h4_b,    h4_b,  s4,
        )                                                                # (batch_size,)

    # Sequentially stream batches; only batch_size replicates live in
    # memory at a time.
    chi2_batched = jax.lax.map(
        _batch_pipeline,
        (sig_xy_b, sig_A1_b, sig_A2_b, sig_A3_b, sig_A4_b),
    )                                                                     # (n_batches, batch_size)
    # Flatten back to (n_padded,) then trim the duplicated tail down to the
    # true number of replicates the user asked for.
    chi2_all = chi2_batched.reshape(n_padded)[:n_groups]

    delta_chi2 = jnp.std(chi2_all, axis=0)

    return delta_chi2


@partial(jax.jit, static_argnames=('density_func', 'potential_func', 'num_Vbin', 'Rzphi_n_tot', 'nnls_maxiter'))
def get_model_with_orbit(density_func, potential_func,
                        params_halo_pot, params_disk_rho, dict_data, num_Vbin,
                        Rzphi_n_tot=360, Rzphi_n_grid = jnp.array([10,6,6]), Rzphi_lim_grid = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]]),
                        xy_lim_grid = jnp.array([[-10.,10.],[-3.,3.]]), xy_n_grid = jnp.array([60,40]),
                        nnls_maxiter=200, regularization = 1.0,
                        ):
    """
    Same forward model as `build_model`, but the integrator also stores
    every orbit's full 6-D phase-space trajectory. Returns the orbital
    weights + trajectories for post-processing (velocity maps, orbit
    classification, etc.).

    Uses n_realizations=1 (one orbit per particle, no random jittering)
    to keep the trajectory tensor a reasonable size; the integrator's
    withtraj variant returns y_traj of shape
        (n_particles, n_realizations, N_max, 6)
    and t_traj of shape
        (n_particles, n_realizations, N_max).
    With n_realizations=1 the orbits-axis is therefore length n_particles.

    All input conventions match `build_model` exactly: `density_func` and
    `potential_func` are static, `params_disk_rho` carries
    `mass_to_light_ratio` (not its inverse), bootstrap perturbations come
    from `dict_data['*_standard_normal']`.

    Returns
    -------
    dict with keys:
        weights              : (n_particles,) unperturbed NNLS weights
        weights_all_boot     : (N_boot, n_particles) per-bootstrap weights
        y_traj               : (n_particles, n_realizations, N_max, 6)
        t_traj               : (n_particles, n_realizations, N_max)
        rotation_matrix      : (3, 3)
        Omega_bar            : scalar (kpc-Gyr units)
        n_particles          : int
        n_realizations       : int (= 1)
        mass_per_orbit       : scalar
        density_3d, surface_density, h1, h2, h3, h4, V, sigma : per-bootstrap model maps
    """

    w0 = dict_data['w0']
    n_particles = w0.shape[0]
    v0 = dict_data['v0']
    s = dict_data['s']
    num_per_bin = dict_data['num_per_bin']
    bin_mapping = dict_data['bin_mapping']

    #=========================================== Get potential parameters =====================================================

    params_baryon = {
        'logM_disc': params_disk_rho['logM_disc'],
        'Rs_disc': params_disk_rho['Rs_disc'],
        'Hs_disc': params_disk_rho['Hs_disc'],
        'logM_bar': params_disk_rho['logM_bar'],
        'L_bar': params_disk_rho['L_bar'],
        'a_bar': params_disk_rho['a_bar'],
        'b_bar': params_disk_rho['b_bar'],

        'x_origin': params_disk_rho['x_origin'],
        'y_origin': params_disk_rho['y_origin'],
        'z_origin': params_disk_rho['z_origin'],
        'dirx': params_disk_rho['dirx'],
        'diry': params_disk_rho['diry'],
        'dirz': params_disk_rho['dirz'],
    }

    light_to_mass_ratio = 1/params_disk_rho['mass_to_light_ratio']

    Omega_bar = params_disk_rho['Omega_bar']
    alpha, beta, gamma = params_disk_rho['alpha'], params_disk_rho['beta'], params_disk_rho['gamma']
    rotation_matrix = makeRotationMatrix(alpha, beta, gamma)

    #=========================================== GET initial condition ===================================================

    key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
    w0_new = get_w0_new(w0, key1, key2, key3, n_particles,
                        params_baryon, params_halo_pot, potential_func, density_func)

    #======================================== Calculate orbital timescale =====================================================
    _R = jnp.sqrt(w0_new[:,0]**2 + w0_new[:,1]**2)
    _z = w0_new[:,2]

    _Vc = jax.vmap(get_rotation_curve, in_axes=(0, None, None, 0))(
        _R, potential_func, (params_baryon, params_halo_pot), _z,
    )

    # n_realizations = 1 here (vs 4 in build_model). One trajectory per
    # particle keeps the y_traj tensor small enough to fit in memory for
    # typical n_particles ~ 5000, N_max = 5000.
    n_realizations = 1
    key = jax.random.PRNGKey(911)
    keys = jax.random.split(key, 6)
    d_scale = 0.1 * jnp.ones(_R.shape)
    v_scale = 0.1 * _Vc
    v_scale = jnp.clip(v_scale, a_min=1, a_max = 15)
    noise_x  = (jax.random.uniform(keys[0], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_y  = (jax.random.uniform(keys[1], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_z  = (jax.random.uniform(keys[2], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_vx = (jax.random.uniform(keys[3], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vy = (jax.random.uniform(keys[4], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vz = (jax.random.uniform(keys[5], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]

    w0_new_batch = w0_new[:, jnp.newaxis, :]
    w0_new_batch = w0_new_batch + jnp.stack([noise_x, noise_y, noise_z, noise_vx, noise_vy, noise_vz], axis=-1)
    T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0, None, None, 0))(
        _R, potential_func, (params_baryon, params_halo_pot), _z,
    )
    T_orb_batch = T_orb[:, jnp.newaxis].repeat(n_realizations, axis=1)

    #=========================================== Construct orbital library (with trajectories) =================================
    @jax.jit
    def acc_fn(x, y, z):
        def _pot(pos):
            return potential_func(pos[0], pos[1], pos[2], params_baryon, params_halo_pot)
        grad_phi = jax.grad(_pot)(jnp.array([x, y, z]))
        return -grad_phi

    @jax.jit
    def pot_fn(x, y, z):
        return potential_func(x, y, z, params_baryon, params_halo_pot)

    N_step_per_orb = 100
    N_dynamical_time = 50
    N_max = N_step_per_orb * N_dynamical_time
    T_total_batch = T_orb_batch * N_dynamical_time
    dt_init_batch = T_orb_batch / N_step_per_orb
    atol, rtol = 1e-7, 1e-4
    dt_min, dt_max = 1e-5, 0.3

    # Same option-b return signature as the chunked variant, plus the
    # per-orbit trajectory + time arrays at the end.
    Rzphi_bin_counts, surface_density, h0, h1, h2, h3, h4, _, y_traj, t_traj = \
        _integrate_adaptive_withtraj_batch_vmap(
            w0_new_batch, acc_fn, pot_fn, N_max, T_total_batch,
            dt_init_batch, -Omega_bar,
            atol, rtol,
            dt_min, dt_max,
            num_Vbin, bin_mapping, num_per_bin,
            Rzphi_lim_grid, xy_lim_grid,
            Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
            v0, s, rotation_matrix,
        )
    A_Rzphi = Rzphi_bin_counts.T
    A_xy    = surface_density.T
    A_h0    = h0.T
    A_h1    = h1.T
    A_h2    = h2.T
    A_h3    = h3.T
    A_h4    = h4.T

    #================================== Construct 3D density target ============================================
    @jax.jit
    def density_func_Rz(R, z, phi, params):
        x = R * jnp.cos(phi)
        y = R * jnp.sin(phi)
        return density_func(x, y, z, params)

    @partial(jax.jit, static_argnames=['rho_fct'])
    def get_mass(R_grid, z_grid, phi_grid, rho_fct, dict_params, dR, dz, dphi, sample):
        R_samples = R_grid + (sample[:,0] - 0.5) * dR
        z_samples = z_grid + (sample[:,1] - 0.5) * dz
        phi_samples = phi_grid + (sample[:,2] - 0.5) * dphi
        density_samples = rho_fct(R_samples, z_samples, phi_samples, dict_params)
        mass_tot = jnp.sum(density_samples * R_samples) / sample.shape[0]
        return mass_tot * dR * dz * dphi

    R_grid, dR = dict_data['R_grid'], dict_data['dR']
    z_grid, dz = dict_data['z_grid'], dict_data['dz']
    phi_grid, dphi = dict_data['phi_grid'], dict_data['dphi']
    y_Rzphi = jax.vmap(get_mass, in_axes=[0, 0, 0, None, None, None, None, None, None])(
                R_grid, z_grid, phi_grid, density_func_Rz, params_disk_rho, dR, dz, dphi, dict_data['sample_for_integration']
    )
    sig_Rzphi = 0.02 * y_Rzphi + 1e-10

    #================================== Read surface density and kinematic targets ============================================
    y_xy   = dict_data['XY_density_data'] / light_to_mass_ratio
    sig_xy = (dict_data['XY_density_data_err'] + EPSILON) / light_to_mass_ratio

    y_h1 = dict_data['h1_data']
    y_h2 = dict_data['h2_data']
    y_h3 = dict_data['h3_data']
    y_h4 = dict_data['h4_data']
    sig_A1 = dict_data['h1_data_err'] + EPSILON
    sig_A2 = dict_data['h2_data_err'] + EPSILON
    sig_A3 = dict_data['h3_data_err'] + EPSILON
    sig_A4 = dict_data['h4_data_err'] + EPSILON

    #================================== Renormalise the 3D and 2D density constraint ===========================================
    mean_mass_per_orb = jnp.sum(y_Rzphi) / A_Rzphi.shape[1]
    y_xy      = y_xy / mean_mass_per_orb
    sig_xy    = sig_xy / mean_mass_per_orb
    y_Rzphi   = y_Rzphi / mean_mass_per_orb
    sig_Rzphi = sig_Rzphi / mean_mass_per_orb

    #=========================================== Gamma_kin (option-b normalisation) ============================================
    X_minmax_arr = dict_data['X_minmax']
    Y_minmax_arr = dict_data['Y_minmax']
    nXY_arr      = dict_data['nX_nY']
    area_per_pixel_pc2 = ((X_minmax_arr[1] - X_minmax_arr[0]) / nXY_arr[0]) * \
                        ((Y_minmax_arr[1] - Y_minmax_arr[0]) / nXY_arr[1]) * 1e6
    M_per_bin = y_xy * num_per_bin * area_per_pixel_pc2
    norm_GH   = 1.0 + y_h4 * jnp.sqrt(6.0) / 4.0
    gamma_kin = M_per_bin / (norm_GH * 2.0 * jnp.sqrt(jnp.pi) * s + EPSILON)

    #=========================================== Bootstrap NNLS solver (shared with build_model) ===============================
    y_xy_boot = y_xy[None, :] + dict_data['XY_standard_normal'] * sig_xy[None, :]
    y_h1_boot = y_h1[None, :] + dict_data['h1_standard_normal'] * sig_A1[None, :]
    y_h2_boot = y_h2[None, :] + dict_data['h2_standard_normal'] * sig_A2[None, :]
    y_h3_boot = y_h3[None, :] + dict_data['h3_standard_normal'] * sig_A3[None, :]
    y_h4_boot = y_h4[None, :] + dict_data['h4_standard_normal'] * sig_A4[None, :]

    weights_all = solve_nnls_admm_bootstrap(
                            A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                            y_Rzphi, y_xy, gamma_kin,
                            y_xy_boot, y_h1_boot, y_h2_boot, y_h3_boot, y_h4_boot,
                            sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
                            lambda_reg=regularization, maxiter=nnls_maxiter,
    )  # (N_boot, n_orb)

    #===================================== Compute model maps per bootstrap ==================================================
    density_3d_model, density_all, h1_all, h2_all, h3_all, h4_all, V_all, sigma_all = \
        compute_model_single_vmap(weights_all, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4, gamma_kin, v0, s)
    density_all = density_all * mean_mass_per_orb * light_to_mass_ratio   # back to luminosity units

    return {
        'weights':           weights_all[0],          # unperturbed (row 0 of bootstrap)
        'weights_all_boot':  weights_all,
        'y_traj':            y_traj,
        't_traj':            t_traj,
        'rotation_matrix':   rotation_matrix,
        'Omega_bar':         Omega_bar,
        'n_particles':       n_particles,
        'n_realizations':    n_realizations,
        'mass_per_orbit':    mean_mass_per_orb,
        'density_3d':        y_Rzphi,
        'density_3d_model':  density_3d_model,
        'surface_density':   density_all,
        'h1':                h1_all,
        'h2':                h2_all,
        'h3':                h3_all,
        'h4':                h4_all,
        'V':                 V_all,
        'sigma':             sigma_all,
    }

# ==========================================================================
# build_model_acc — copy of build_model() with a REQUIRED analytic acceleration_func.
# The original build_model() is deliberately left unchanged.
# ==========================================================================

# n_realizations is STATIC: it sets array shapes (the jitter arrays and T_orb_batch), so it must
# be a concrete Python int at trace time, not a traced value.
@partial(jax.jit, static_argnames=('density_func', 'potential_func', 'acceleration_func', 'potential_func_axi', 'density_func_axi', 'num_Vbin', 'Rzphi_n_tot', 'nnls_maxiter', 'n_realizations', 'return_design', 'anisotropy_b'))
def build_model_acc(density_func, potential_func, acceleration_func, potential_func_axi,
                    density_func_axi,
                params_halo_pot, params_disk_rho, dict_data, num_Vbin,
                jeans_z_int_max=5.0, sigma_frac_3d=0.02, v_sys=0.0,
                Rzphi_n_tot=360, Rzphi_n_grid = jnp.array([10,6,6]), Rzphi_lim_grid = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]]),
                xy_lim_grid = jnp.array([[-10.,10.],[-3.,3.]]), xy_n_grid = jnp.array([60,40]),
                nnls_maxiter=200, regularization = 1.0, n_realizations = 4, return_design = False,
                anisotropy_b = 1.0
                ):
    """build_model_acc: variant of build_model() taking an explicit `acceleration_func`.

    n_realizations : int, default 4 (the historical hard-coded value -- callers that do not
        pass it are unaffected).  Each seed orbit is jittered into this many copies, which are
        then AVERAGED into ONE library column (integrates.py:288-297).  So the parameter buys
        smoothing, not extra columns for the NNLS: n_realizations=1 turns the jitter off and
        makes each library entry a single orbit, integrating 4x faster for the same 7500/10000
        columns.  Worth testing because averaging orbits can only BROADEN a library entry's
        LOSVD, never narrow it -- see ngc1512_pipeline/code/diagnose_orbit_library.py.

    acceleration_func(x, y, z, params_baryon, params_halo) -> acceleration
        Required.  Same call signature as potential_func.  Used directly instead of
        differentiating potential_func, so spline / MGE potentials supply their own
        analytic gradient.

        Return shape: anything that unpacks into three components, since the
        integrator does `ax, ay, az = acc_fn(x, y, z)` (integrates.py).  Both a
        (3,) array -- as returned by CylindricalSpline.get_acc and by the autodiff
        path (-grad_phi) -- and a 3-tuple work; batched inputs give (3, n).

    density_func_axi(x, y, z, params_baryon) -> rho_m0
        The m = 0 (azimuthally averaged) density, used ONLY by the Jeans velocity
        seeding, which solves the AXISYMMETRIC Jeans equations.  It is the density
        counterpart of potential_func_axi.  For an MGE or any axisymmetric model this
        is just density_func.  For a barred parametric density it must be the m = 0
        component: the Jeans differentiates through it, and handing it the full barred
        density is both inconsistent and the dominant memory cost of the model.

    build_model() is left unchanged for book-keeping / regression comparison.
    """

    w0 = dict_data['w0']
    n_particles = w0.shape[0]
    # Systemic-velocity residual.  The model's orbits have zero mean LOS velocity, so a
    # pedestal left in the data's GH reference cannot be reproduced and shows up as a
    # uniform h1 offset.  Since the GH weight is w = (v_los - v0)/s, subtracting v_sys
    # from v0 is IDENTICAL to adding it to the model velocity, so h1..h4 all see it.
    v0 = dict_data['v0'] - v_sys
    s = dict_data['s']
    num_per_bin = dict_data['num_per_bin']
    bin_mapping = dict_data['bin_mapping']

    #=========================================== Get potential parameters =====================================================

    # params_baryon is only forwarded to density_func / potential_func / acceleration_func.
    # The analytic disc+bar model reads these keys; an MGE model closes over its own
    # Gaussians and ignores them, so they default to 0.0 rather than being required.
    params_baryon = {
        'logM_disc': params_disk_rho.get('logM_disc', 0.0),
        'Rs_disc': params_disk_rho.get('Rs_disc', 0.0),
        'Hs_disc': params_disk_rho.get('Hs_disc', 0.0),
        'logM_bar': params_disk_rho.get('logM_bar', 0.0),
        'L_bar': params_disk_rho.get('L_bar', 0.0),
        'a_bar': params_disk_rho.get('a_bar', 0.0),
        'b_bar': params_disk_rho.get('b_bar', 0.0),

        'x_origin': params_disk_rho.get('x_origin', 0.0),
        'y_origin': params_disk_rho.get('y_origin', 0.0),
        'z_origin': params_disk_rho.get('z_origin', 0.0),
        'dirx': params_disk_rho.get('dirx', 0.0),
        'diry': params_disk_rho.get('diry', 0.0),
        'dirz': params_disk_rho.get('dirz', 0.0),
    }

    light_to_mass_ratio = 1/params_disk_rho['mass_to_light_ratio']

    Omega_bar = params_disk_rho['Omega_bar']
    alpha, beta, gamma = params_disk_rho['alpha'], params_disk_rho['beta'], params_disk_rho['gamma']
    rotation_matrix = makeRotationMatrix(alpha, beta, gamma)

    #=========================================== GET initial condition ===================================================

    key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
    # Clockwise seeding (Lz<0) paired with -Omega_bar below: the bar co-rotates with
    # the stars.  An apparent rotation of either sign on the sky comes from alpha.
    # The Jeans equations are AXISYMMETRIC, so they get the m=0 density and the m=0
    # potential -- not the full barred ones.  Passing the barred density here is both
    # physically inconsistent and expensive: jax.grad(vertical_pressure) differentiates
    # through it, which for a Dehnen bar dominates the model's memory (measured: the
    # Jeans fell from 0.580 to 0.296 GB when density_func_axi was cheap).  With an MGE
    # the two coincide and this costs nothing.
    w0_new = get_w0_new_improved(w0, key1, key2, key3, n_particles,
                                 params_baryon, params_halo_pot, potential_func_axi, density_func_axi,
                                 z_int_max=jeans_z_int_max, anisotropy_b=anisotropy_b)

    #======================================== Calculate orbital timescale =====================================================
    _R = jnp.sqrt(w0_new[:,0]**2 + w0_new[:,1]**2)
    _z = w0_new[:,2]

    _Vc = jax.vmap(get_rotation_curve, in_axes=(0, None, None, 0))(
        _R,
        potential_func,
        (params_baryon, params_halo_pot),
        _z
    )

    # n_realizations comes from the signature now (default 4 -> unchanged for every
    # caller that does not pass it).  It was hard-coded here.
    key = jax.random.PRNGKey(911)
    keys = jax.random.split(key, 6)
    d_scale = 0.1 * jnp.ones(_R.shape)
    v_scale = 0.1 * _Vc
    v_scale = jnp.clip(v_scale, a_min=1, a_max = 15)
    noise_x = (jax.random.uniform(keys[0], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_y = (jax.random.uniform(keys[1], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_z = (jax.random.uniform(keys[2], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_vx = (jax.random.uniform(keys[3], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vy = (jax.random.uniform(keys[4], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vz = (jax.random.uniform(keys[5], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]

    w0_new_batch = w0_new[:, jnp.newaxis, :]
    w0_new_batch = w0_new_batch + jnp.stack([noise_x, noise_y, noise_z, noise_vx, noise_vy, noise_vz], axis=-1)
    T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0, None, None, 0))(
        _R,
        potential_func,
        (params_baryon, params_halo_pot),
        _z
    )
    T_orb_batch = T_orb[:, jnp.newaxis].repeat(n_realizations, axis=1)

    #=========================================== Construct orbital library =======================================================
    # Acceleration comes from the supplied analytic acceleration_func (same call signature
    # as potential_func).  This replaces autodiff of the potential: for spline / MGE
    # potentials the spline's own analytic gradient is both faster and less noisy.
    @jax.jit
    def acc_fn(x, y, z):
        return acceleration_func(x, y, z, params_baryon, params_halo_pot)

    @jax.jit
    def pot_fn(x, y, z):
        return potential_func(x, y, z, params_baryon, params_halo_pot)

    N_step_per_orb = 100
    N_dynamical_time = 50
    N_max = N_step_per_orb * N_dynamical_time
    T_total_batch = T_orb_batch * N_dynamical_time
    dt_init_batch = T_orb_batch / N_step_per_orb
    atol, rtol = 1e-7, 1e-4
    dt_min, dt_max = 1e-5, 0.3

    Rzphi_bin_counts, surface_density, h0, h1, h2, h3, h4, _ = _integrate_adaptive_batch_chunked_vmap(
                        w0_new_batch, acc_fn, pot_fn, N_max, T_total_batch,
                        dt_init_batch, Omega_bar,
                        atol, rtol,    
                        dt_min, dt_max,    
                        num_Vbin, bin_mapping, num_per_bin,
                        Rzphi_lim_grid, xy_lim_grid,
                        Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
                        v0, s, rotation_matrix,
                        100, # chunk size
    )
    A_Rzphi = Rzphi_bin_counts.T
    A_xy = surface_density.T
    A_h0 = h0.T
    A_h1 = h1.T
    A_h2 = h2.T
    A_h3 = h3.T
    A_h4 = h4.T

    #================================== Construct 3D density target ============================================

    @jax.jit
    def density_func_Rz(R, z, phi, params):
        x = R * jnp.cos(phi)
        y = R * jnp.sin(phi)
        return density_func(x, y, z, params)

    @partial(jax.jit, static_argnames=['rho_fct'])
    def get_mass(R_grid, z_grid, phi_grid, rho_fct, dict_params, dR, dz, dphi, sample):
        R_samples = R_grid + (sample[:,0] - 0.5) * dR
        z_samples = z_grid + (sample[:,1] - 0.5) * dz
        phi_samples = phi_grid + (sample[:,2] - 0.5) * dphi
        density_samples = rho_fct(R_samples, z_samples, phi_samples, dict_params)
        mass_tot = jnp.sum(density_samples * R_samples) / sample.shape[0]
        return mass_tot * dR * dz * dphi

    R_grid, dR = dict_data['R_grid'], dict_data['dR']
    z_grid, dz = dict_data['z_grid'], dict_data['dz']
    phi_grid, dphi = dict_data['phi_grid'], dict_data['dphi']
    y_Rzphi = jax.vmap(get_mass, in_axes=[0, 0, 0, None, None, None, None, None, None])(
                R_grid, z_grid, phi_grid, density_func_Rz, params_disk_rho, dR, dz, dphi, dict_data['sample_for_integration']
    )
    # Fractional uncertainty on the 3-D density.  Stated directly rather than as a
    # "relax" multiplier on a hidden 0.02: sigma_frac_3d IS the fraction.
    # (The 2-D density error needs no such knob - it comes straight from the data via
    #  dict_data['XY_density_data_err'], so set it when building the data dict.)
    sig_Rzphi = sigma_frac_3d * y_Rzphi + 1e-10

    #================================== Read the surface density and kinematic targets ============================================

    y_xy = dict_data['XY_density_data'] / light_to_mass_ratio
    sig_xy = (dict_data['XY_density_data_err'] + EPSILON) / light_to_mass_ratio

    y_h1 = dict_data['h1_data']
    y_h2 = dict_data['h2_data']
    y_h3 = dict_data['h3_data']
    y_h4 = dict_data['h4_data']
    sig_A1 = dict_data['h1_data_err'] + EPSILON
    sig_A2 = dict_data['h2_data_err'] + EPSILON
    sig_A3 = dict_data['h3_data_err'] + EPSILON
    sig_A4 = dict_data['h4_data_err'] + EPSILON

    #================================== Renormalise the 3D and 2D density constrain ============================================

    mean_mass_per_orb = jnp.sum(y_Rzphi) / A_Rzphi.shape[1]
    y_xy = y_xy / mean_mass_per_orb
    sig_xy = sig_xy / mean_mass_per_orb
    y_Rzphi = y_Rzphi / mean_mass_per_orb
    sig_Rzphi = sig_Rzphi / mean_mass_per_orb

    #=========================================== Gamma_kin: base-Gaussian amplitude (option-b normalisation) ====================
    # gamma_kin_b = M_b / (norm_b * 2*sqrt(pi) * s_b), the data's h_0 in the bin's basis.
    # M_b is the per-bin photometric mass; norm_b absorbs the higher-order GH contribution to the LOSVD integral.  In the rescaled-by-mean_mass_per_orb units used here,
    # M_b = y_xy * num_per_bin * area_per_pixel_in_pc^2 (rescaled).
    X_minmax_arr = dict_data['X_minmax']
    Y_minmax_arr = dict_data['Y_minmax']
    nXY_arr = dict_data['nX_nY']
    area_per_pixel_pc2 = ((X_minmax_arr[1] - X_minmax_arr[0]) / nXY_arr[0]) * \
                        ((Y_minmax_arr[1] - Y_minmax_arr[0]) / nXY_arr[1]) * 1e6
    M_per_bin = y_xy * num_per_bin * area_per_pixel_pc2
    norm_GH = 1.0 + y_h4 * jnp.sqrt(6.0) / 4.0
    gamma_kin = M_per_bin / (norm_GH * 2.0 * jnp.sqrt(jnp.pi) * s + EPSILON)

    #=========================================== Bootstrap NNLS solver (vmapped) ============================================

    # # Bootstrapped observations: (N_boot, n_bins) — pre-computed and stored in dict_data

    y_xy_boot = y_xy[None, :] + dict_data['XY_standard_normal'] * sig_xy[None, :]
    y_h1_boot = y_h1[None, :] + dict_data['h1_standard_normal'] * sig_A1[None, :]
    y_h2_boot = y_h2[None, :] + dict_data['h2_standard_normal'] * sig_A2[None, :]
    y_h3_boot = y_h3[None, :] + dict_data['h3_standard_normal'] * sig_A3[None, :]
    y_h4_boot = y_h4[None, :] + dict_data['h4_standard_normal'] * sig_A4[None, :]

    weights_all = solve_nnls_admm_bootstrap(
                            A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                            y_Rzphi, y_xy, gamma_kin,
                            y_xy_boot, y_h1_boot, y_h2_boot, y_h3_boot, y_h4_boot,
                            sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
                            lambda_reg=regularization, maxiter=nnls_maxiter,
                            w_rzphi = jnp.sqrt(1.0), w_xy = jnp.sqrt(1.0), w_h = jnp.sqrt(1.0)
    )  # (N_boot, n_orb)

    #===================================== Compute model vectors + logL for each bootstrap ==================================

    density_3d_model, density_all, h1_all, h2_all, h3_all, h4_all, V_all, sigma_all = compute_model_single_GH_vmap(weights_all, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4, gamma_kin, v0, s)
    density_all = density_all * mean_mass_per_orb * light_to_mass_ratio  # convert back to luminosity units for density

    results_dict = {
        'density_3d':y_Rzphi,
        'density_3d_model': density_3d_model,
        'surface_density': density_all,
        'h1': h1_all,
        'h2': h2_all,
        'h3': h3_all,
        'h4': h4_all,
        'V': V_all,
        'sigma': sigma_all,
        'weights': weights_all,
    }
    if return_design:
        # The design matrices, for library diagnostics only.  Guarded by a STATIC flag because
        # A_xy alone is (n_bins x n_orb) = 2324 x 7500 floats, and returning all six from every
        # likelihood evaluation would add ~350 MB per call to an MCMC that is already
        # memory-bound.  Default False leaves every existing caller untouched.
        #   A_h0  : zeroth GH moment  -> the light each orbit puts in each bin
        #   A_h2  : second GH moment; h2_model = (A_h2 @ w)/gamma_kin, so an orbit whose LOSVD
        #           is NARROWER than the bin's reference Gaussian (which IS the observed LOSVD,
        #           because the data have h1=h2=0 by construction) contributes A_h2 < 0.
        #           "Cold orbit" therefore means A_h2 < 0, exactly and linearly.
        results_dict.update({'A_xy': A_xy, 'A_h0': A_h0, 'A_h1': A_h1, 'A_h2': A_h2,
                             'A_h3': A_h3, 'A_h4': A_h4, 'A_Rzphi': A_Rzphi,
                             'gamma_kin': gamma_kin, 'y_xy': y_xy, 'sig_xy': sig_xy,
                             'mean_mass_per_orb': mean_mass_per_orb})
    return results_dict




# ==========================================================================
# model_deltaChi2_jackknife_acc — copy of model_deltaChi2_jackknife() with a REQUIRED analytic acceleration_func.
# The original model_deltaChi2_jackknife() is deliberately left unchanged.
# ==========================================================================

@partial(jax.jit, static_argnames=('density_func', 'potential_func', 'acceleration_func', 'potential_func_axi', 'density_func_axi', 'chi2_func', 'num_Vbin', 'Rzphi_n_tot', 'nnls_maxiter', 'jeans_z_int_max', 'n_groups', 'batch_size'))
def model_deltaChi2_jackknife_acc(density_func, potential_func, acceleration_func, potential_func_axi,
                                  density_func_axi, chi2_func,
                params_halo_pot, params_disk_rho, dict_data, num_Vbin,
                jeans_z_int_max=5.0,
                Rzphi_n_tot=360, Rzphi_n_grid = jnp.array([10,6,6]), Rzphi_lim_grid = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]]),
                xy_lim_grid = jnp.array([[-10.,10.],[-3.,3.]]), xy_n_grid = jnp.array([60,40]),
                nnls_maxiter=200, regularization = 1.0,
                n_groups=100, batch_size=50,
                ):
    """model_deltaChi2_jackknife_acc: variant of model_deltaChi2_jackknife() taking an explicit `acceleration_func`.

    acceleration_func(x, y, z, params_baryon, params_halo) -> acceleration
        Required.  Same call signature as potential_func.  Used directly instead of
        differentiating potential_func, so spline / MGE potentials supply their own
        analytic gradient.

        Return shape: anything that unpacks into three components, since the
        integrator does `ax, ay, az = acc_fn(x, y, z)` (integrates.py).  Both a
        (3,) array -- as returned by CylindricalSpline.get_acc and by the autodiff
        path (-grad_phi) -- and a 3-tuple work; batched inputs give (3, n).

    model_deltaChi2_jackknife() is left unchanged for book-keeping / regression comparison.
    """
    """Delete-d jackknife with a user-supplied chi^2 function.

    The user supplies `chi2_func` with the signature

        chi2_func(y_Rzphi, y_Rzphi_model, sigma_Rzphi,
                  y_xy,    y_xy_model,    sigma_xy,
                  y_h1,    y_h1_model,    sigma_h1,
                  y_h2,    y_h2_model,    sigma_h2,
                  y_h3,    y_h3_model,    sigma_h3,
                  y_h4,    y_h4_model,    sigma_h4)
        -> scalar

    Each replicate g drops one disjoint group of Voronoi bins by setting
    its `sigma_*` entries to 1e20 (true per-replicate masking). The NNLS
    is re-solved per replicate via `vmap(solve_nnls_admm)` -- with σ
    actually changing across replicates, U / Q / Cholesky all rebuild
    internally, which is what makes the masked bins drop out of the
    constraint set. After solving we evaluate the user's `chi2_func`
    per replicate (also vmapped) and return `jnp.std(chi2_all, axis=0)`.
    """

    w0 = dict_data['w0']
    n_particles = w0.shape[0]
    v0 = dict_data['v0']
    s = dict_data['s']
    num_per_bin = dict_data['num_per_bin']
    bin_mapping = dict_data['bin_mapping']

    #=========================================== Get potential parameters =====================================================

    # params_baryon is only forwarded to density_func / potential_func / acceleration_func.
    # The analytic disc+bar model reads these keys; an MGE model closes over its own
    # Gaussians and ignores them, so they default to 0.0 rather than being required.
    params_baryon = {
        'logM_disc': params_disk_rho.get('logM_disc', 0.0),
        'Rs_disc': params_disk_rho.get('Rs_disc', 0.0),
        'Hs_disc': params_disk_rho.get('Hs_disc', 0.0),
        'logM_bar': params_disk_rho.get('logM_bar', 0.0),
        'L_bar': params_disk_rho.get('L_bar', 0.0),
        'a_bar': params_disk_rho.get('a_bar', 0.0),
        'b_bar': params_disk_rho.get('b_bar', 0.0),

        'x_origin': params_disk_rho.get('x_origin', 0.0),
        'y_origin': params_disk_rho.get('y_origin', 0.0),
        'z_origin': params_disk_rho.get('z_origin', 0.0),
        'dirx': params_disk_rho.get('dirx', 0.0),
        'diry': params_disk_rho.get('diry', 0.0),
        'dirz': params_disk_rho.get('dirz', 0.0),
    }

    light_to_mass_ratio = 1/params_disk_rho['mass_to_light_ratio']

    Omega_bar = params_disk_rho['Omega_bar']
    alpha, beta, gamma = params_disk_rho['alpha'], params_disk_rho['beta'], params_disk_rho['gamma']
    rotation_matrix = makeRotationMatrix(alpha, beta, gamma)

    #=========================================== GET initial condition ===================================================

    key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
    # Improved Jeans seeder: the m=0 average required by the axisymmetric Jeans equations
    # is taken inside get_w0_new_improved, so the FULL potential is passed here.
    # The Jeans equations are AXISYMMETRIC, so they get the m=0 density and the m=0
    # potential -- not the full barred ones.  Passing the barred density here is both
    # physically inconsistent and expensive: jax.grad(vertical_pressure) differentiates
    # through it, which for a Dehnen bar dominates the model's memory (measured: the
    # Jeans fell from 0.580 to 0.296 GB when density_func_axi was cheap).  With an MGE
    # the two coincide and this costs nothing.
    w0_new = get_w0_new_improved(w0, key1, key2, key3, n_particles,
                                 params_baryon, params_halo_pot, potential_func_axi, density_func_axi,
                                 z_int_max=jeans_z_int_max)

    #======================================== Calculate orbital timescale =====================================================
    _R = jnp.sqrt(w0_new[:,0]**2 + w0_new[:,1]**2)
    _z = w0_new[:,2]

    _Vc = jax.vmap(get_rotation_curve, in_axes=(0, None, None, 0))(
        _R,
        potential_func,
        (params_baryon, params_halo_pot),
        _z
    )

    n_realizations = 4
    key = jax.random.PRNGKey(911)
    keys = jax.random.split(key, 6)
    d_scale = 0.1 * jnp.ones(_R.shape)
    v_scale = 0.1 * _Vc
    v_scale = jnp.clip(v_scale, a_min=1, a_max = 15)
    noise_x = (jax.random.uniform(keys[0], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_y = (jax.random.uniform(keys[1], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_z = (jax.random.uniform(keys[2], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_vx = (jax.random.uniform(keys[3], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vy = (jax.random.uniform(keys[4], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vz = (jax.random.uniform(keys[5], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]

    w0_new_batch = w0_new[:, jnp.newaxis, :]
    w0_new_batch = w0_new_batch + jnp.stack([noise_x, noise_y, noise_z, noise_vx, noise_vy, noise_vz], axis=-1)
    T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0, None, None, 0))(
        _R,
        potential_func,
        (params_baryon, params_halo_pot),
        _z
    )
    T_orb_batch = T_orb[:, jnp.newaxis].repeat(n_realizations, axis=1)

    #=========================================== Construct orbital library =======================================================
    # Acceleration comes from the supplied analytic acceleration_func (same call signature
    # as potential_func).  This replaces autodiff of the potential: for spline / MGE
    # potentials the spline's own analytic gradient is both faster and less noisy.
    @jax.jit
    def acc_fn(x, y, z):
        return acceleration_func(x, y, z, params_baryon, params_halo_pot)

    @jax.jit
    def pot_fn(x, y, z):
        return potential_func(x, y, z, params_baryon, params_halo_pot)

    N_step_per_orb = 100
    N_dynamical_time = 50
    N_max = N_step_per_orb * N_dynamical_time
    T_total_batch = T_orb_batch * N_dynamical_time
    dt_init_batch = T_orb_batch / N_step_per_orb
    atol, rtol = 1e-7, 1e-4
    dt_min, dt_max = 1e-5, 0.3

    Rzphi_bin_counts, surface_density, h0, h1, h2, h3, h4, _ = _integrate_adaptive_batch_chunked_vmap(
                        w0_new_batch, acc_fn, pot_fn, N_max, T_total_batch,
                        dt_init_batch, Omega_bar,
                        atol, rtol,    
                        dt_min, dt_max,    
                        num_Vbin, bin_mapping, num_per_bin,
                        Rzphi_lim_grid, xy_lim_grid,
                        Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
                        v0, s, rotation_matrix,
                        100, # chunk size
    )
    A_Rzphi = Rzphi_bin_counts.T
    A_xy = surface_density.T
    A_h0 = h0.T
    A_h1 = h1.T
    A_h2 = h2.T
    A_h3 = h3.T
    A_h4 = h4.T

    #================================== Construct 3D density target ============================================

    @jax.jit
    def density_func_Rz(R, z, phi, params):
        x = R * jnp.cos(phi)
        y = R * jnp.sin(phi)
        return density_func(x, y, z, params)

    @partial(jax.jit, static_argnames=['rho_fct'])
    def get_mass(R_grid, z_grid, phi_grid, rho_fct, dict_params, dR, dz, dphi, sample):
        R_samples = R_grid + (sample[:,0] - 0.5) * dR
        z_samples = z_grid + (sample[:,1] - 0.5) * dz
        phi_samples = phi_grid + (sample[:,2] - 0.5) * dphi
        density_samples = rho_fct(R_samples, z_samples, phi_samples, dict_params)
        mass_tot = jnp.sum(density_samples * R_samples) / sample.shape[0]
        return mass_tot * dR * dz * dphi

    R_grid, dR = dict_data['R_grid'], dict_data['dR']
    z_grid, dz = dict_data['z_grid'], dict_data['dz']
    phi_grid, dphi = dict_data['phi_grid'], dict_data['dphi']
    y_Rzphi = jax.vmap(get_mass, in_axes=[0, 0, 0, None, None, None, None, None, None])(
                R_grid, z_grid, phi_grid, density_func_Rz, params_disk_rho, dR, dz, dphi, dict_data['sample_for_integration']
    )
    sig_Rzphi = 0.02 * y_Rzphi + 1e-10

    #================================== Read the surface density and kinematic targets ============================================

    y_xy = dict_data['XY_density_data'] / light_to_mass_ratio
    sig_xy = (dict_data['XY_density_data_err'] + EPSILON) / light_to_mass_ratio

    y_h1 = dict_data['h1_data']
    y_h2 = dict_data['h2_data']
    y_h3 = dict_data['h3_data']
    y_h4 = dict_data['h4_data']
    sig_A1 = dict_data['h1_data_err'] + EPSILON
    sig_A2 = dict_data['h2_data_err'] + EPSILON
    sig_A3 = dict_data['h3_data_err'] + EPSILON
    sig_A4 = dict_data['h4_data_err'] + EPSILON

    #================================== Renormalise the 3D and 2D density constrain ============================================

    mean_mass_per_orb = jnp.sum(y_Rzphi) / A_Rzphi.shape[1]
    y_xy = y_xy / mean_mass_per_orb
    sig_xy = sig_xy / mean_mass_per_orb
    y_Rzphi = y_Rzphi / mean_mass_per_orb
    sig_Rzphi = sig_Rzphi / mean_mass_per_orb

    #=========================================== Gamma_kin: base-Gaussian amplitude (option-b normalisation) ====================
    # gamma_kin_b = M_b / (norm_b * 2*sqrt(pi) * s_b), the data's h_0 in the bin's basis.
    # M_b is the per-bin photometric mass; norm_b absorbs the higher-order GH contribution to the LOSVD integral.  In the rescaled-by-mean_mass_per_orb units used here,
    # M_b = y_xy * num_per_bin * area_per_pixel_in_pc^2 (rescaled).
    X_minmax_arr = dict_data['X_minmax']
    Y_minmax_arr = dict_data['Y_minmax']
    nXY_arr = dict_data['nX_nY']
    area_per_pixel_pc2 = ((X_minmax_arr[1] - X_minmax_arr[0]) / nXY_arr[0]) * \
                        ((Y_minmax_arr[1] - Y_minmax_arr[0]) / nXY_arr[1]) * 1e6
    M_per_bin = y_xy * num_per_bin * area_per_pixel_pc2
    norm_GH = 1.0 + y_h4 * jnp.sqrt(6.0) / 4.0
    gamma_kin = M_per_bin / (norm_GH * 2.0 * jnp.sqrt(jnp.pi) * s + EPSILON)

    #=========================================== Jackknife: per-replicate sigma masks ========================================
    # Each replicate g drops the bins assigned to group g by raising their
    # sigma to BIG (1e20). The σ in the NNLS denominator U = A/σ then sends
    # those rows to ~0, so they no longer constrain the orbital weights.
    # Group assignment is just bin_index % n_groups -- the final statistic
    # is jnp.std(chi2, axis=0), invariant under reordering of replicates.
    n_bins       = sig_xy.shape[0]
    group_of_bin = jnp.arange(n_bins) % n_groups                          # (n_bins,)
    dropped      = (group_of_bin[None, :] == jnp.arange(n_groups)[:, None])  # (n_groups, n_bins)
    BIG = 1e20

    sig_xy_jk = jnp.where(dropped, BIG, sig_xy[None, :])                  # (n_groups, n_bins)
    sig_A1_jk = jnp.where(dropped, BIG, sig_A1[None, :])
    sig_A2_jk = jnp.where(dropped, BIG, sig_A2[None, :])
    sig_A3_jk = jnp.where(dropped, BIG, sig_A3[None, :])
    sig_A4_jk = jnp.where(dropped, BIG, sig_A4[None, :])
    # sig_Rzphi has no per-bin observation dimension to mask, so it is
    # shared across replicates.

    #=========================================== Batched NNLS + chi^2 (memory-bounded) ======================================
    # vmap'ing solve_nnls_admm across ALL n_groups replicates simultaneously
    # blows up memory because each replicate carries its own Cholesky and
    # ADMM-scan state. Instead, vmap within each batch of `batch_size`
    # replicates and use jax.lax.map across batches -- working memory is
    # bounded by batch_size, not n_groups.
    #
    # If n_groups is not divisible by batch_size, we pad the per-replicate
    # sigma arrays up to the next multiple by duplicating the last
    # replicate (any valid replicate would do); the padded entries are
    # trimmed before the final std. Because n_groups and batch_size are
    # both static, n_padded / n_pad / n_batches are all concrete at trace
    # time.
    n_padded  = -(-n_groups // batch_size) * batch_size   # round up
    n_pad     = n_padded - n_groups
    n_batches = n_padded // batch_size

    def _pad_replicates(arr):
        if n_pad == 0:
            return arr
        pad_block = jnp.broadcast_to(arr[-1:], (n_pad,) + arr.shape[1:])
        return jnp.concatenate([arr, pad_block], axis=0)

    sig_xy_b = _pad_replicates(sig_xy_jk).reshape(n_batches, batch_size, n_bins)
    sig_A1_b = _pad_replicates(sig_A1_jk).reshape(n_batches, batch_size, n_bins)
    sig_A2_b = _pad_replicates(sig_A2_jk).reshape(n_batches, batch_size, n_bins)
    sig_A3_b = _pad_replicates(sig_A3_jk).reshape(n_batches, batch_size, n_bins)
    sig_A4_b = _pad_replicates(sig_A4_jk).reshape(n_batches, batch_size, n_bins)

    # Per-replicate NNLS solver.
    def _solve_one(sxy_g, s1_g, s2_g, s3_g, s4_g):
        return solve_nnls_admm(
            A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
            y_Rzphi, y_xy, gamma_kin, y_h1, y_h2, y_h3, y_h4,
            sig_Rzphi, sxy_g, s1_g, s2_g, s3_g, s4_g,
            lambda_reg=regularization, maxiter=nnls_maxiter,
            w_rzphi = jnp.sqrt(1.0), w_xy = jnp.sqrt(1.0), w_h = jnp.sqrt(1.0)
        )

    # `chi2_func` expects every (data, model, sigma) triple to be in the
    # SAME units. The NNLS internal arrays (`y_xy`, `sig_xy`) were divided
    # by `light_to_mass_ratio * mean_mass_per_orb` for the solver — but
    # the per-replicate prediction `d2d_b` is multiplied BACK to
    # luminosity below. To keep units consistent we expose y_xy and
    # sig_xy in luminosity too. The 3-D density and h1..h4 are unchanged.
    lum_factor  = mean_mass_per_orb * light_to_mass_ratio
    y_xy_lum    = y_xy * lum_factor                                       # (n_bins,)

    # Pre-broadcast the shared data arrays once at batch size; cheap, and
    # avoids redoing the broadcast inside every batch.
    y_Rzphi_b   = jnp.broadcast_to(y_Rzphi,   (batch_size,) + y_Rzphi.shape)
    sig_Rzphi_b = jnp.broadcast_to(sig_Rzphi, (batch_size,) + sig_Rzphi.shape)
    y_xy_b      = jnp.broadcast_to(y_xy_lum,  (batch_size,) + y_xy_lum.shape)
    y_h1_b      = jnp.broadcast_to(y_h1,      (batch_size,) + y_h1.shape)
    y_h2_b      = jnp.broadcast_to(y_h2,      (batch_size,) + y_h2.shape)
    y_h3_b      = jnp.broadcast_to(y_h3,      (batch_size,) + y_h3.shape)
    y_h4_b      = jnp.broadcast_to(y_h4,      (batch_size,) + y_h4.shape)

    def _batch_pipeline(sigs_batch):
        sxy, s1, s2, s3, s4 = sigs_batch                                 # each: (batch_size, n_bins)

        # NNLS for batch_size replicates in parallel.
        weights_b = jax.vmap(_solve_one)(sxy, s1, s2, s3, s4)            # (batch_size, n_orb)

        # Forward model -> per-replicate predictions.
        d3d_b, d2d_b, h1_b, h2_b, h3_b, h4_b, _, _ = compute_model_single_GH_vmap(
            weights_b, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4, gamma_kin, v0, s,
        )
        d2d_b      = d2d_b * lum_factor                                  # luminosity units
        sxy_lum    = sxy   * lum_factor                                  # match y_xy_b units

        # User's chi2_func, vmapped over the batch axis.
        return jax.vmap(chi2_func, in_axes=(0,)*18)(
            y_Rzphi_b, d3d_b, sig_Rzphi_b,
            y_xy_b,    d2d_b, sxy_lum,
            y_h1_b,    h1_b,  s1,
            y_h2_b,    h2_b,  s2,
            y_h3_b,    h3_b,  s3,
            y_h4_b,    h4_b,  s4,
        )                                                                # (batch_size,)

    # Sequentially stream batches; only batch_size replicates live in
    # memory at a time.
    chi2_batched = jax.lax.map(
        _batch_pipeline,
        (sig_xy_b, sig_A1_b, sig_A2_b, sig_A3_b, sig_A4_b),
    )                                                                     # (n_batches, batch_size)
    # Flatten back to (n_padded,) then trim the duplicated tail down to the
    # true number of replicates the user asked for.
    chi2_all = chi2_batched.reshape(n_padded)[:n_groups]

    delta_chi2 = jnp.std(chi2_all, axis=0)

    return delta_chi2




# ==========================================================================
# get_model_with_orbit_acc — copy of get_model_with_orbit() with a REQUIRED analytic acceleration_func.
# The original get_model_with_orbit() is deliberately left unchanged.
# ==========================================================================

@partial(jax.jit, static_argnames=('density_func', 'potential_func', 'acceleration_func', 'potential_func_axi', 'density_func_axi', 'num_Vbin', 'Rzphi_n_tot', 'nnls_maxiter', 'jeans_z_int_max'))
def get_model_with_orbit_acc(density_func, potential_func, acceleration_func, potential_func_axi,
                             density_func_axi,
                        params_halo_pot, params_disk_rho, dict_data, num_Vbin,
                jeans_z_int_max=5.0,
                        Rzphi_n_tot=360, Rzphi_n_grid = jnp.array([10,6,6]), Rzphi_lim_grid = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]]),
                        xy_lim_grid = jnp.array([[-10.,10.],[-3.,3.]]), xy_n_grid = jnp.array([60,40]),
                        nnls_maxiter=200, regularization = 1.0,
                        ):
    """get_model_with_orbit_acc: variant of get_model_with_orbit() taking an explicit `acceleration_func`.

    acceleration_func(x, y, z, params_baryon, params_halo) -> acceleration
        Required.  Same call signature as potential_func.  Used directly instead of
        differentiating potential_func, so spline / MGE potentials supply their own
        analytic gradient.

        Return shape: anything that unpacks into three components, since the
        integrator does `ax, ay, az = acc_fn(x, y, z)` (integrates.py).  Both a
        (3,) array -- as returned by CylindricalSpline.get_acc and by the autodiff
        path (-grad_phi) -- and a 3-tuple work; batched inputs give (3, n).

    get_model_with_orbit() is left unchanged for book-keeping / regression comparison.
    """
    """
    Same forward model as `build_model`, but the integrator also stores
    every orbit's full 6-D phase-space trajectory. Returns the orbital
    weights + trajectories for post-processing (velocity maps, orbit
    classification, etc.).

    Uses n_realizations=1 (one orbit per particle, no random jittering)
    to keep the trajectory tensor a reasonable size; the integrator's
    withtraj variant returns y_traj of shape
        (n_particles, n_realizations, N_max, 6)
    and t_traj of shape
        (n_particles, n_realizations, N_max).
    With n_realizations=1 the orbits-axis is therefore length n_particles.

    All input conventions match `build_model` exactly: `density_func` and
    `potential_func` are static, `params_disk_rho` carries
    `mass_to_light_ratio` (not its inverse), bootstrap perturbations come
    from `dict_data['*_standard_normal']`.

    Returns
    -------
    dict with keys:
        weights              : (n_particles,) unperturbed NNLS weights
        weights_all_boot     : (N_boot, n_particles) per-bootstrap weights
        y_traj               : (n_particles, n_realizations, N_max, 6)
        t_traj               : (n_particles, n_realizations, N_max)
        rotation_matrix      : (3, 3)
        Omega_bar            : scalar (kpc-Gyr units)
        n_particles          : int
        n_realizations       : int (= 1)
        mass_per_orbit       : scalar
        density_3d, surface_density, h1, h2, h3, h4, V, sigma : per-bootstrap model maps
    """

    w0 = dict_data['w0']
    n_particles = w0.shape[0]
    v0 = dict_data['v0']
    s = dict_data['s']
    num_per_bin = dict_data['num_per_bin']
    bin_mapping = dict_data['bin_mapping']

    #=========================================== Get potential parameters =====================================================

    # params_baryon is only forwarded to density_func / potential_func / acceleration_func.
    # The analytic disc+bar model reads these keys; an MGE model closes over its own
    # Gaussians and ignores them, so they default to 0.0 rather than being required.
    params_baryon = {
        'logM_disc': params_disk_rho.get('logM_disc', 0.0),
        'Rs_disc': params_disk_rho.get('Rs_disc', 0.0),
        'Hs_disc': params_disk_rho.get('Hs_disc', 0.0),
        'logM_bar': params_disk_rho.get('logM_bar', 0.0),
        'L_bar': params_disk_rho.get('L_bar', 0.0),
        'a_bar': params_disk_rho.get('a_bar', 0.0),
        'b_bar': params_disk_rho.get('b_bar', 0.0),

        'x_origin': params_disk_rho.get('x_origin', 0.0),
        'y_origin': params_disk_rho.get('y_origin', 0.0),
        'z_origin': params_disk_rho.get('z_origin', 0.0),
        'dirx': params_disk_rho.get('dirx', 0.0),
        'diry': params_disk_rho.get('diry', 0.0),
        'dirz': params_disk_rho.get('dirz', 0.0),
    }

    light_to_mass_ratio = 1/params_disk_rho['mass_to_light_ratio']

    Omega_bar = params_disk_rho['Omega_bar']
    alpha, beta, gamma = params_disk_rho['alpha'], params_disk_rho['beta'], params_disk_rho['gamma']
    rotation_matrix = makeRotationMatrix(alpha, beta, gamma)

    #=========================================== GET initial condition ===================================================

    key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
    # Improved Jeans seeder: the m=0 average required by the axisymmetric Jeans equations
    # is taken inside get_w0_new_improved, so the FULL potential is passed here.
    # The Jeans equations are AXISYMMETRIC, so they get the m=0 density and the m=0
    # potential -- not the full barred ones.  Passing the barred density here is both
    # physically inconsistent and expensive: jax.grad(vertical_pressure) differentiates
    # through it, which for a Dehnen bar dominates the model's memory (measured: the
    # Jeans fell from 0.580 to 0.296 GB when density_func_axi was cheap).  With an MGE
    # the two coincide and this costs nothing.
    w0_new = get_w0_new_improved(w0, key1, key2, key3, n_particles,
                                 params_baryon, params_halo_pot, potential_func_axi, density_func_axi,
                                 z_int_max=jeans_z_int_max)

    #======================================== Calculate orbital timescale =====================================================
    _R = jnp.sqrt(w0_new[:,0]**2 + w0_new[:,1]**2)
    _z = w0_new[:,2]

    _Vc = jax.vmap(get_rotation_curve, in_axes=(0, None, None, 0))(
        _R, potential_func, (params_baryon, params_halo_pot), _z,
    )

    # n_realizations = 1 here (vs 4 in build_model). One trajectory per
    # particle keeps the y_traj tensor small enough to fit in memory for
    # typical n_particles ~ 5000, N_max = 5000.
    n_realizations = 1
    key = jax.random.PRNGKey(911)
    keys = jax.random.split(key, 6)
    d_scale = 0.1 * jnp.ones(_R.shape)
    v_scale = 0.1 * _Vc
    v_scale = jnp.clip(v_scale, a_min=1, a_max = 15)
    noise_x  = (jax.random.uniform(keys[0], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_y  = (jax.random.uniform(keys[1], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_z  = (jax.random.uniform(keys[2], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_vx = (jax.random.uniform(keys[3], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vy = (jax.random.uniform(keys[4], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vz = (jax.random.uniform(keys[5], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]

    w0_new_batch = w0_new[:, jnp.newaxis, :]
    w0_new_batch = w0_new_batch + jnp.stack([noise_x, noise_y, noise_z, noise_vx, noise_vy, noise_vz], axis=-1)
    T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0, None, None, 0))(
        _R, potential_func, (params_baryon, params_halo_pot), _z,
    )
    T_orb_batch = T_orb[:, jnp.newaxis].repeat(n_realizations, axis=1)

    #=========================================== Construct orbital library (with trajectories) =================================
    # Acceleration comes from the supplied analytic acceleration_func (same call signature
    # as potential_func).  This replaces autodiff of the potential: for spline / MGE
    # potentials the spline's own analytic gradient is both faster and less noisy.
    @jax.jit
    def acc_fn(x, y, z):
        return acceleration_func(x, y, z, params_baryon, params_halo_pot)

    @jax.jit
    def pot_fn(x, y, z):
        return potential_func(x, y, z, params_baryon, params_halo_pot)

    N_step_per_orb = 100
    N_dynamical_time = 50
    N_max = N_step_per_orb * N_dynamical_time
    T_total_batch = T_orb_batch * N_dynamical_time
    dt_init_batch = T_orb_batch / N_step_per_orb
    atol, rtol = 1e-7, 1e-4
    dt_min, dt_max = 1e-5, 0.3

    # Same option-b return signature as the chunked variant, plus the
    # per-orbit trajectory + time arrays at the end.
    Rzphi_bin_counts, surface_density, h0, h1, h2, h3, h4, _, y_traj, t_traj = \
        _integrate_adaptive_withtraj_batch_vmap(
            w0_new_batch, acc_fn, pot_fn, N_max, T_total_batch,
            dt_init_batch, Omega_bar,
            atol, rtol,
            dt_min, dt_max,
            num_Vbin, bin_mapping, num_per_bin,
            Rzphi_lim_grid, xy_lim_grid,
            Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
            v0, s, rotation_matrix,
        )
    A_Rzphi = Rzphi_bin_counts.T
    A_xy    = surface_density.T
    A_h0    = h0.T
    A_h1    = h1.T
    A_h2    = h2.T
    A_h3    = h3.T
    A_h4    = h4.T

    #================================== Construct 3D density target ============================================
    @jax.jit
    def density_func_Rz(R, z, phi, params):
        x = R * jnp.cos(phi)
        y = R * jnp.sin(phi)
        return density_func(x, y, z, params)

    @partial(jax.jit, static_argnames=['rho_fct'])
    def get_mass(R_grid, z_grid, phi_grid, rho_fct, dict_params, dR, dz, dphi, sample):
        R_samples = R_grid + (sample[:,0] - 0.5) * dR
        z_samples = z_grid + (sample[:,1] - 0.5) * dz
        phi_samples = phi_grid + (sample[:,2] - 0.5) * dphi
        density_samples = rho_fct(R_samples, z_samples, phi_samples, dict_params)
        mass_tot = jnp.sum(density_samples * R_samples) / sample.shape[0]
        return mass_tot * dR * dz * dphi

    R_grid, dR = dict_data['R_grid'], dict_data['dR']
    z_grid, dz = dict_data['z_grid'], dict_data['dz']
    phi_grid, dphi = dict_data['phi_grid'], dict_data['dphi']
    y_Rzphi = jax.vmap(get_mass, in_axes=[0, 0, 0, None, None, None, None, None, None])(
                R_grid, z_grid, phi_grid, density_func_Rz, params_disk_rho, dR, dz, dphi, dict_data['sample_for_integration']
    )
    sig_Rzphi = 0.02 * y_Rzphi + 1e-10

    #================================== Read surface density and kinematic targets ============================================
    y_xy   = dict_data['XY_density_data'] / light_to_mass_ratio
    sig_xy = (dict_data['XY_density_data_err'] + EPSILON) / light_to_mass_ratio

    y_h1 = dict_data['h1_data']
    y_h2 = dict_data['h2_data']
    y_h3 = dict_data['h3_data']
    y_h4 = dict_data['h4_data']
    sig_A1 = dict_data['h1_data_err'] + EPSILON
    sig_A2 = dict_data['h2_data_err'] + EPSILON
    sig_A3 = dict_data['h3_data_err'] + EPSILON
    sig_A4 = dict_data['h4_data_err'] + EPSILON

    #================================== Renormalise the 3D and 2D density constraint ===========================================
    mean_mass_per_orb = jnp.sum(y_Rzphi) / A_Rzphi.shape[1]
    y_xy      = y_xy / mean_mass_per_orb
    sig_xy    = sig_xy / mean_mass_per_orb
    y_Rzphi   = y_Rzphi / mean_mass_per_orb
    sig_Rzphi = sig_Rzphi / mean_mass_per_orb

    #=========================================== Gamma_kin (option-b normalisation) ============================================
    X_minmax_arr = dict_data['X_minmax']
    Y_minmax_arr = dict_data['Y_minmax']
    nXY_arr      = dict_data['nX_nY']
    area_per_pixel_pc2 = ((X_minmax_arr[1] - X_minmax_arr[0]) / nXY_arr[0]) * \
                        ((Y_minmax_arr[1] - Y_minmax_arr[0]) / nXY_arr[1]) * 1e6
    M_per_bin = y_xy * num_per_bin * area_per_pixel_pc2
    norm_GH   = 1.0 + y_h4 * jnp.sqrt(6.0) / 4.0
    gamma_kin = M_per_bin / (norm_GH * 2.0 * jnp.sqrt(jnp.pi) * s + EPSILON)

    #=========================================== Bootstrap NNLS solver (shared with build_model) ===============================
    y_xy_boot = y_xy[None, :] + dict_data['XY_standard_normal'] * sig_xy[None, :]
    y_h1_boot = y_h1[None, :] + dict_data['h1_standard_normal'] * sig_A1[None, :]
    y_h2_boot = y_h2[None, :] + dict_data['h2_standard_normal'] * sig_A2[None, :]
    y_h3_boot = y_h3[None, :] + dict_data['h3_standard_normal'] * sig_A3[None, :]
    y_h4_boot = y_h4[None, :] + dict_data['h4_standard_normal'] * sig_A4[None, :]

    weights_all = solve_nnls_admm_bootstrap(
                            A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                            y_Rzphi, y_xy, gamma_kin,
                            y_xy_boot, y_h1_boot, y_h2_boot, y_h3_boot, y_h4_boot,
                            sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
                            lambda_reg=regularization, maxiter=nnls_maxiter,
    )  # (N_boot, n_orb)

    #===================================== Compute model maps per bootstrap ==================================================
    density_3d_model, density_all, h1_all, h2_all, h3_all, h4_all, V_all, sigma_all = \
        compute_model_single_GH_vmap(weights_all, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4, gamma_kin, v0, s)
    density_all = density_all * mean_mass_per_orb * light_to_mass_ratio   # back to luminosity units

    return {
        'weights':           weights_all[0],          # unperturbed (row 0 of bootstrap)
        'weights_all_boot':  weights_all,
        'y_traj':            y_traj,
        't_traj':            t_traj,
        'rotation_matrix':   rotation_matrix,
        'Omega_bar':         Omega_bar,
        'n_particles':       n_particles,
        'n_realizations':    n_realizations,
        'mass_per_orbit':    mean_mass_per_orb,
        'density_3d':        y_Rzphi,
        'density_3d_model':  density_3d_model,
        'surface_density':   density_all,
        'h1':                h1_all,
        'h2':                h2_all,
        'h3':                h3_all,
        'h4':                h4_all,
        'V':                 V_all,
        'sigma':             sigma_all,
    }


# ==========================================================================================
# build_model_acc_meff — build_model_acc PLUS the effective number of free parameters m_eff
# ==========================================================================================
# ADDED.  build_model_acc and nnls.py are NOT modified, so every existing result stays
# reproducible.  This is a WRAPPER, not a fork: it calls build_model_acc(return_design=True)
# and reuses its design matrices and the EXISTING solve_nnls_admm_bootstrap, so there is no
# duplicated linear algebra to drift out of sync.
#
# WHY.  Judging orbit models by chi2 alone biases the recovered galaxy properties, because
# models of different flexibility fit different amounts of noise (Lipka & Thomas 2021).  The
# correction is the generalized AIC of Thomas & Lipka (2022, eq. 15):
#
#     AIC_p = chi2 + 2 * m_eff
#     m_eff = mean_k sum_i [(f_i(Th_z^k) - f_i(Th_y))/eps_i] * [(z_i^k - f_i(Th_y))/eps_i]
#     z^k   = f(Th_y) + N(0, eps)            <- noise on the FIT, not on the data
#
# NOTE the difference from dict_data's own bootstraps, which perturb the DATA and are used to
# marginalise the likelihood.  Here the noise is added to the model's own fit, and the SAME
# standard normals are reused so that two different mass models see PAIRED noise realisations
# (needed if a dAIC threshold is ever calibrated across two models, Wang+2026 sec. 5.3).
#
# COST.  solve_nnls_admm_bootstrap is simply called TWICE -- once on the unperturbed data,
# once on the batch of perturbed sets.  Both calls build the same U and the same Cholesky from
# identical inputs, so XLA's CSE collapses them; the measured cost is in the ADMM iterations
# (0.168 s per bootstrap against 45.6 s for a full evaluation on CPU).  Every returned model
# vector is recomputed from w0, which lets XLA dead-code-eliminate build_model_acc's own
# internal bootstrap solve -- two live solves, not three.
#
# JIT.  `regularization` arrives as a TRACED value and is used only in arithmetic, so ONE
# compiled kernel serves every lambda and the MCMC never recompiles.  It must NEVER become a
# static argname.  n_boot_meff IS static: it sets an array shape.  And note jeans_z_int_max
# below -- forwarding an argument a jitted callee declares static turns a concrete default
# into a tracer.


@partial(jax.jit, static_argnames=(
    'density_func', 'potential_func', 'acceleration_func', 'potential_func_axi',
    'density_func_axi', 'num_Vbin', 'Rzphi_n_tot', 'nnls_maxiter', 'n_realizations',
    'anisotropy_b', 'n_boot_meff', 'jeans_z_int_max', 'err_den_meff'))
def build_model_acc_meff(density_func, potential_func, acceleration_func, potential_func_axi,
                         density_func_axi,
                         params_halo_pot, params_disk_rho, dict_data, num_Vbin,
                         jeans_z_int_max=5.0, sigma_frac_3d=0.02, v_sys=0.0,
                         Rzphi_n_tot=360, Rzphi_n_grid=jnp.array([10, 6, 6]),
                         Rzphi_lim_grid=jnp.array([[0, 10.], [-3, 3], [-jnp.pi, jnp.pi]]),
                         xy_lim_grid=jnp.array([[-10., 10.], [-3., 3.]]),
                         xy_n_grid=jnp.array([60, 40]),
                         nnls_maxiter=200, regularization=1.0, n_realizations=4,
                         anisotropy_b=1.0,
                         err_den_meff=None, n_boot_meff=None):
    """build_model_acc + m_eff.

    err_den_meff : fractional error on the surface density used for the m_eff bootstrap.
        Pass the SAME value the likelihood scores with (ERR_DEN_LOGL = 0.03), so chi2 and
        m_eff refer to one error model and AIC = chi2 + 2*m_eff is internally consistent.
        The NNLS keeps its own 1% internally -- that is an up-weighting of the density block,
        not a claim about the error, and it is deliberately left alone.
        None -> fall back to the NNLS's own sig_xy.
    n_boot_meff : number of m_eff bootstraps (STATIC, it sets a shape).  None -> all rows of
        the stored standard normals after row 0.  ~10 is enough: only dm_eff/dlambda needs to
        be accurate (Thomas & Lipka 2022, sec. 8).

    Adds to the returned dict:
        m_eff       scalar, summed over ALL bins and the 5 data channels
        m_eff_bins  (5, n_xy_bins) per-channel per-bin contributions, already averaged over
                    the bootstraps.  Sum these under the likelihood's own KEEP mask so chi2
                    and m_eff cover the SAME data points.
        weights     (1, n_orb) from the UNPERTURBED fit -- this path does not marginalise.
    """
    # jeans_z_int_max is deliberately NOT forwarded.  jeans.get_w0_new_improved declares
    # z_int_max STATIC, and build_model_acc only gets away with passing it because callers
    # never supply it -- an unpassed default stays a concrete Python float, whereas an
    # explicitly passed one becomes a tracer and the static-argument hash then fails with
    # "unhashable type: DynamicJaxprTracer".
    assert jeans_z_int_max == 5.0, (
        'jeans_z_int_max must stay at build_model_acc default 5.0 on the m_eff path: '
        'forwarding it explicitly makes it a tracer, and get_w0_new_improved needs it static')
    res = build_model_acc(
        density_func, potential_func, acceleration_func, potential_func_axi, density_func_axi,
        params_halo_pot, params_disk_rho, dict_data, num_Vbin,
        sigma_frac_3d=sigma_frac_3d, v_sys=v_sys,
        Rzphi_n_tot=Rzphi_n_tot, Rzphi_n_grid=Rzphi_n_grid, Rzphi_lim_grid=Rzphi_lim_grid,
        xy_lim_grid=xy_lim_grid, xy_n_grid=xy_n_grid, nnls_maxiter=nnls_maxiter,
        regularization=regularization, n_realizations=n_realizations,
        return_design=True, anisotropy_b=anisotropy_b)

    # ---- the ingredients, all already built by build_model_acc ---------------------------
    A_Rzphi, A_xy = res['A_Rzphi'], res['A_xy']
    A_h1, A_h2, A_h3, A_h4 = res['A_h1'], res['A_h2'], res['A_h3'], res['A_h4']
    gamma_kin, y_xy, sig_xy = res['gamma_kin'], res['y_xy'], res['sig_xy']
    mmpo = res['mean_mass_per_orb']
    y_Rzphi = res['density_3d']                     # already divided by mean_mass_per_orb
    sig_Rzphi = sigma_frac_3d * y_Rzphi + 1e-10 / mmpo      # as model.py:1113 + :1135
    EPS0 = 1e-10
    y_h = [dict_data['h%d_data' % k] for k in (1, 2, 3, 4)]
    sig_A = [dict_data['h%d_data_err' % k] + EPS0 for k in (1, 2, 3, 4)]
    v0 = dict_data['v0'] - v_sys
    s = dict_data['s']
    _args = (A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4)
    _sigs = (sig_Rzphi, sig_xy, sig_A[0], sig_A[1], sig_A[2], sig_A[3])

    # ---- pass 1: the EXISTING solver, on the unperturbed data -----------------------------
    w0 = solve_nnls_admm_bootstrap(
        *_args, y_Rzphi, y_xy, gamma_kin,
        y_xy[None, :], y_h[0][None, :], y_h[1][None, :], y_h[2][None, :], y_h[3][None, :],
        *_sigs, lambda_reg=regularization, maxiter=nnls_maxiter)

    # f(Th_y).  compute_model_single_GH_vmap returns the density in the SAME rescaled units
    # as y_xy (build_model_acc converts to luminosity only afterwards, model.py:1172).
    d3_0, den0, h10, h20, h30, h40, V0, sg0 = compute_model_single_GH_vmap(
        w0, *_args, gamma_kin, v0, s)
    f0 = [den0[0], h10[0], h20[0], h30[0], h40[0]]

    # ---- bootstrap data drawn around the FIT ---------------------------------------------
    nb = dict_data['XY_standard_normal'].shape[0] - 1 if n_boot_meff is None else n_boot_meff
    sn = [dict_data['XY_standard_normal'][1:nb + 1]] + \
         [dict_data['h%d_standard_normal' % k][1:nb + 1] for k in (1, 2, 3, 4)]
    # err_den_meff = 0.0 -> THE SURFACE DENSITY IS A HARD CONSTRAINT, not data being fitted.
    # Its mock is the fit itself with NO noise added, so every bootstrap solves against the same
    # density and the channel contributes nothing to m_eff or to chi2_boot.  This is Lipka &
    # Thomas's structure (2021, eq. 3): the light is in neither the criterion nor the flexibility
    # count.  MEASURED consequence (aic_diagnostics/daic_dataset.py, S/N 230): rho rises
    # 0.872 -> 0.957 and sigma_dAIC falls 43.8 -> 23.7, with sigma_AIC still exactly on its
    # Gaussian floor -- the best-conditioned configuration we have.
    HARD_DEN = (err_den_meff is not None) and (float(err_den_meff) == 0.0)
    eps_xy = sig_xy if err_den_meff is None else err_den_meff * y_xy
    eps = [eps_xy] + sig_A
    if HARD_DEN:
        # broadcast the unperturbed fit to every bootstrap row; eps_xy is 0 and is never divided by
        z = [jnp.broadcast_to(f0[0][None, :], sn[0].shape)] + \
            [f0[c][None, :] + sn[c] * eps[c][None, :] for c in range(1, 5)]
    else:
        z = [f0[c][None, :] + sn[c] * eps[c][None, :] for c in range(5)]

    # ---- pass 2: the SAME solver, on the perturbed batch -----------------------------------
    w_z = solve_nnls_admm_bootstrap(
        *_args, y_Rzphi, y_xy, gamma_kin, z[0], z[1], z[2], z[3], z[4],
        *_sigs, lambda_reg=regularization, maxiter=nnls_maxiter)
    _, den_z, h1z, h2z, h3z, h4z, _, _ = compute_model_single_GH_vmap(
        w_z, *_args, gamma_kin, v0, s)
    fz = [den_z, h1z, h2z, h3z, h4z]

    # ---- m_eff (Thomas & Lipka 2022, eqs. 6-7) -------------------------------------------
    per_bin = []
    for c in range(5):
        if HARD_DEN and c == 0:
            # 0/0 would be NaN and would poison m_eff and every downstream sum.  A constraint
            # that is never perturbed has, by definition, zero effective freedom.
            per_bin.append(jnp.zeros_like(f0[0]))
            continue
        a = (fz[c] - f0[c][None, :]) / eps[c][None, :]
        b = (z[c] - f0[c][None, :]) / eps[c][None, :]
        per_bin.append(jnp.mean(a * b, axis=0))          # mean over bootstraps -> (n_bins,)
    m_eff_bins = jnp.stack(per_bin, axis=0)              # (5, n_bins)

    # ---- per-realisation chi2, for the dAIC calibration ----------------------------------
    # Wang+2026 sec. 5.3 calibrate the model-selection threshold by refitting mock data drawn
    # around the best fit and looking at the SCATTER of AIC differences between two models.
    # The refits already exist here (pass 2), so the chi2 of each refit against its own mock
    # costs nothing extra.  Both models see the SAME standard normals (they are fixed in
    # dict_data), so the realisations are PAIRED and the common-mode noise cancels in the
    # difference -- which is the whole point (their sigma_AIC 42 -> sigma_dAIC 11).
    _c = [jnp.zeros(fz[0].shape[0]) if (HARD_DEN and c == 0) else
          jnp.sum(((fz[c] - z[c]) / eps[c][None, :])**2, axis=1) for c in range(5)]
    chi2_boot = sum(_c)                                  # (n_boot,) all 5 channels
    chi2_boot_kin = sum(_c[1:])                          # (n_boot,) h1-h4 only

    light_to_mass_ratio = 1 / params_disk_rho['mass_to_light_ratio']
    out = dict(res)
    for k in ('A_xy', 'A_h0', 'A_h1', 'A_h2', 'A_h3', 'A_h4', 'A_Rzphi',
              'gamma_kin', 'y_xy', 'sig_xy', 'mean_mass_per_orb'):
        out.pop(k, None)     # never return the design matrices: ~490 MB per call
    # EVERY model vector is rebuilt from w0, so build_model_acc's own bootstrap solve has no
    # live consumer and XLA can eliminate it.
    # m_eff_kin EXCLUDES the surface-density channel.  The density is still scored in chi2;
    # only its flexibility charge is dropped.  NOTE this is NOT the structure of the papers:
    # Lipka & Thomas (2021, eq. 3) minimise chi2 over the LOSVDs alone and treat the light as a
    # hard constraint that is in NEITHER chi2 nor m_eff, while Wang+2026's lensing term (eq. 13)
    # is in chi2 with no m_eff charge only because it depends on the MASS parameters and not on
    # the orbital weights.  Our surface density IS fitted by the weights (the A_xy block), so
    # charging no flexibility for it credits the chi2 it buys without paying for it.  Kept as a
    # deliberate, reversible choice -- both scalars are returned so either can be used.
    out.update({
        'density_3d_model': d3_0,
        'surface_density': den0 * mmpo * light_to_mass_ratio,
        'h1': h10, 'h2': h20, 'h3': h30, 'h4': h40, 'V': V0, 'sigma': sg0,
        'weights': w0,
        'm_eff': jnp.sum(m_eff_bins),            # all 5 channels
        'm_eff_kin': jnp.sum(m_eff_bins[1:]),    # h1-h4 only -- the one the likelihood uses
        'm_eff_bins': m_eff_bins,                # (5, n_bins): logSig, h1, h2, h3, h4
        'chi2_boot': chi2_boot,                  # (n_boot,) refit vs its own mock, all channels
        'chi2_boot_kin': chi2_boot_kin,          # (n_boot,) h1-h4 only
    })
    return out
