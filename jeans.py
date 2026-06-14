
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from utils import getCartesianFromCylindrical_clockwise
from functools import partial

# @jax.jit
@partial(jax.jit, static_argnames=('potential_func', 'density_func', 'anisotropy_b'))
def get_jeans_moments(x_star, y_star, z_star, params_baryon, params_halo, potential_func, density_func, anisotropy_b=1.0):
    """
    Computes (v_mean, sigma_R, sigma_z, sigma_phi) for a star at (R, z).
    """
    R_star = jnp.sqrt(x_star**2 + y_star**2)

    @jax.jit
    def dPhi_dz(x, y, z, params_baryon, params_halo):
        # Numerical derivative of Potential w.r.t z
        d = 5e-3
        return (potential_func(x, y, z+d, params_baryon, params_halo) - potential_func(x, y, z-d, params_baryon, params_halo)) / (2*d)

    @jax.jit
    def dPhi_dR(x, y, z, params_baryon, params_halo):
        # Numerical derivative of Potential w.r.t R
        d = 5e-3
        R = jnp.sqrt(x**2 + y**2)
        return (potential_func(R+d, 0, z, params_baryon, params_halo) - potential_func(R-d, 0, z, params_baryon, params_halo)) / (2*d)


    # --- Step 1: Compute Sigma_z (Vertical Integration) ---
    def integrand(z_prime):
        return density_func(x_star, y_star, z_prime, params_baryon) * dPhi_dz(x_star, y_star, z_prime, params_baryon, params_halo)

    pts = jnp.linspace(jnp.abs(z_star), 10.0, 1000)
    dx = pts[1] - pts[0]
    integrand_val = jax.vmap(integrand, in_axes = (0))(pts)
    integral_val = jsp.integrate.trapezoid(integrand_val, pts, dx)

    nu_val = density_func(x_star, y_star, z_star, params_baryon)

    sigma_z2 = (1.0 / nu_val) * integral_val
    sigma_z2 = jnp.maximum(sigma_z2, 0.0)
    sigma_z = jnp.sqrt(sigma_z2)

    # --- Step 2: Compute Sigma_R (Anisotropy assumption) ---
    sigma_R2 = anisotropy_b * sigma_z2
    sigma_R = jnp.sqrt(sigma_R2)

    # --- Step 3: Compute v_phi_total^2 (Radial Equation) ---
    def vertical_pressure(r_in):
        def integrand_r(z_prime):
            return density_func(r_in, 0, z_prime, params_baryon) * dPhi_dz(r_in, 0, z_prime, params_baryon, params_halo)

        pts = jnp.linspace(jnp.abs(z_star), 10.0, 1000)
        dx = pts[1] - pts[0]
        integrand_val = jax.vmap(integrand_r, in_axes = (0))(pts)
        integral_val = jsp.integrate.trapezoid(integrand_val, pts, dx)
        return integral_val

    dR = 5e-3
    Pzz_plus = vertical_pressure(R_star + dR)
    Pzz_minus = vertical_pressure(R_star - dR)
    d_nu_sigR2_dR = anisotropy_b * (Pzz_plus - Pzz_minus) / (2*dR)

    term1 = sigma_R2
    term2 = (R_star / nu_val) * d_nu_sigR2_dR
    term3 = R_star * dPhi_dR(x_star, y_star, z_star, params_baryon, params_halo)

    v_phi_total_sq = term1 + term2 + term3

    # --- Step 4: Separate Rotation vs Dispersion ---
    sigma_phi = sigma_R
    v_streaming_sq = v_phi_total_sq - sigma_phi**2
    v_streaming_sq = jnp.maximum(v_streaming_sq, 0.0)
    v_mean_phi = jnp.sqrt(v_streaming_sq)

    output = jax.lax.cond(nu_val<=0, lambda: (0.0, 0.0, 0.0, 0.0), lambda: (v_mean_phi, sigma_R, sigma_z, sigma_phi))

    return output

get_jeans_moments_vmap = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None,None))

@partial(jax.jit, static_argnames=('potential_func', 'density_func'))
def get_w0_new(w0, key1, key2, key3, n_particles,
               params_baryon, params_halo, potential_func, density_func):
    jeans_moments = get_jeans_moments_vmap(w0[:,0], w0[:,1], w0[:,2], params_baryon, params_halo, potential_func, density_func, 1.)
    v_rot, sig_R, sig_z, sig_phi = jeans_moments
    n = w0.shape[0]
    g1, g2, g3 = jax.random.normal(key1, (n,)), jax.random.normal(key2, (n,)), jax.random.normal(key3, (n,))
    vR = g1 * sig_R
    vz = g2 * sig_z
    vphi = v_rot + g3 * sig_phi
    x, y, vx, vy = getCartesianFromCylindrical_clockwise(jnp.sqrt(w0[:,0]**2 + w0[:,1]**2), jnp.arctan2(w0[:,1], w0[:,0]), vR, vphi)
    return jnp.array([x, y, w0[:,2], vx, vy, vz]).T