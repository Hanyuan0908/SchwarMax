import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial

from utils import makeRotationMatrix, get_rotation_curve, estimate_orbital_timescale
from constants import EPSILON
from integrates import _integrate_adaptive_batch_chunked_vmap
from nnls import solve_nnls_admm_bootstrap, solve_nnls_admm
from jeans import get_w0_new
from ghmoments import h_to_V_sigma

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

    # Pre-broadcast the shared data arrays once at batch size; cheap, and
    # avoids redoing the broadcast inside every batch.
    y_Rzphi_b   = jnp.broadcast_to(y_Rzphi,   (batch_size,) + y_Rzphi.shape)
    sig_Rzphi_b = jnp.broadcast_to(sig_Rzphi, (batch_size,) + sig_Rzphi.shape)
    y_xy_b      = jnp.broadcast_to(y_xy,      (batch_size,) + y_xy.shape)
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
        d2d_b = d2d_b * mean_mass_per_orb * light_to_mass_ratio          # luminosity units

        # User's chi2_func, vmapped over the batch axis.
        return jax.vmap(chi2_func, in_axes=(0,)*18)(
            y_Rzphi_b, d3d_b, sig_Rzphi_b,
            y_xy_b,    d2d_b, sxy,
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