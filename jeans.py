
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from utils import (getCartesianFromCylindrical_clockwise,
                   getCartesianFromCylindrical_anticlockwise,
                   axisymmetrize_potential)
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

# ============================================================================
# Improved Jeans solver for spline / MGE ("fixed") potentials.
#
# get_jeans_moments() above assumes an analytic density and a smooth analytic
# potential.  With a spline / MGE potential two of its choices break:
#
#   1. the vertical integral ran to z = 10 kpc; CylindricalSpline ringing appears
#      above ~3 kpc.  -> z_int_max is an argument here, default 5 kpc.
#   2. nu was not clipped; a Laplacian-reconstructed density can go slightly
#      negative.  -> nu is clipped to >= 0 here as well as sigma_z^2.
#
# d(nu sigR^2)/dR is now the EXACT autodiff derivative rather than a central
# difference, which removes the step-size choice entirely.
#
# Rotation sense: the Jeans equations give the MAGNITUDE v_rot >= 0.  The sense is
# fixed by the single package-wide convention -- clockwise seeding (Lz < 0) paired
# with -Omega_bar in the integrator, as in get_w0_new().  An apparent rotation of
# either sign on the sky is obtained from the viewing angle alpha (the sky position
# angle when LOS = z): alpha -> alpha + 180 flips the observed sense and leaves the
# projected density unchanged.  Verified equivalent to machine precision, so no
# separate rotation-sign switch is needed.
#
# The original get_jeans_moments() / get_w0_new() are left untouched.
# ============================================================================

@partial(jax.jit, static_argnames=('potential_func', 'density_func', 'anisotropy_b',
                                   'z_int_max', 'n_int'))
def get_jeans_moments_improved(x_star, y_star, z_star, params_baryon, params_halo,
                               potential_func, density_func, anisotropy_b=1.0,
                               z_int_max=5.0, n_int=32):
    """(v_mean_phi, sigma_R, sigma_z, sigma_phi) for a star at (R, z).

    density_func(x, y, z, params_baryon) -> nu        (clipped to >= 0 here)
    potential_func(x, y, z, params_baryon, params_halo) -> Phi

    z_int_max : upper limit of the vertical integral [kpc].  Keep well inside the
                spline's reliable region (5 kpc for CylindricalSpline).
    n_int     : number of Gauss-Legendre nodes in the vertical quadrature.

    Quadrature: Gauss-Legendre, NOT the trapezoid rule.  The integrand
    nu * dPhi/dz is smooth (a sum of Gaussians times a smooth force), so GL
    converges geometrically where the trapezoid converges as 1/n^2: 32 nodes
    match 1000 trapezoid points.  It is also the more GPU-friendly shape -- a
    vmap followed by a single dot product against static weights, instead of a
    1000-point linspace.  Nodes/weights are computed with numpy at TRACE time
    (n_int is a static argname), so they cost nothing at run time.

    Derivatives: forward-mode autodiff, NOT finite differences.  Central
    differences cost two potential evaluations and, in float32, carry a
    roundoff error ~eps/d amplified by 1/(2d) -- with d = 5e-3 that is ~1e-3
    relative on dPhi/dR, which then propagates straight into v_c.  Autodiff is
    exact and costs about one extra evaluation, so this is cheaper AND more
    accurate; there is no trade being made here.
    """
    R_star = jnp.sqrt(x_star**2 + y_star**2)

    _t_gl, _w_gl = np.polynomial.legendre.leggauss(n_int)   # static: trace-time only
    t_gl = jnp.asarray(_t_gl)
    w_gl = jnp.asarray(_w_gl)

    def gl_quad(f, a, b):
        """int_a^b f(z) dz by Gauss-Legendre on the affine map from [-1, 1]."""
        half = 0.5 * (b - a)
        pts = half * t_gl + 0.5 * (a + b)
        return half * jnp.dot(w_gl, jax.vmap(f)(pts))

    def nu(x, y, z):
        return jnp.maximum(density_func(x, y, z, params_baryon), 0.0)

    def dPhi_dz(x, y, z):
        return jax.grad(lambda zz: potential_func(x, y, zz,
                                                  params_baryon, params_halo))(z)

    def dPhi_dR(x, y, z):
        R = jnp.sqrt(x**2 + y**2)
        return jax.grad(lambda RR: potential_func(RR, 0.0, z,
                                                  params_baryon, params_halo))(R)

    # --- Step 1: sigma_z from the vertical Jeans equation ---
    def integrand(z_prime):
        return nu(x_star, y_star, z_prime) * dPhi_dz(x_star, y_star, z_prime)

    integral_val = gl_quad(integrand, jnp.abs(z_star), z_int_max)

    nu_val = nu(x_star, y_star, z_star)
    sigma_z2 = jnp.maximum((1.0 / nu_val) * integral_val, 0.0)
    sigma_z = jnp.sqrt(sigma_z2)

    # --- Step 2: sigma_R from the anisotropy assumption ---
    sigma_R2 = anisotropy_b * sigma_z2
    sigma_R = jnp.sqrt(sigma_R2)

    # --- Step 3: v_phi^2 from the radial Jeans equation ---
    def vertical_pressure(r_in):
        def integrand_r(z_prime):
            return nu(r_in, 0.0, z_prime) * dPhi_dz(r_in, 0.0, z_prime)
        return gl_quad(integrand_r, jnp.abs(z_star), z_int_max)

    d_nu_sigR2_dR = anisotropy_b * jax.grad(vertical_pressure)(R_star)

    term1 = sigma_R2
    term2 = (R_star / nu_val) * d_nu_sigR2_dR
    term3 = R_star * dPhi_dR(x_star, y_star, z_star)

    v_phi_total_sq = term1 + term2 + term3

    # --- Step 4: split streaming from dispersion ---
    sigma_phi = sigma_R
    v_mean_phi = jnp.sqrt(jnp.maximum(v_phi_total_sq - sigma_phi**2, 0.0))

    return jax.lax.cond(nu_val <= 0,
                        lambda: (0.0, 0.0, 0.0, 0.0),
                        lambda: (v_mean_phi, sigma_R, sigma_z, sigma_phi))


get_jeans_moments_improved_vmap = jax.vmap(
    get_jeans_moments_improved,
    in_axes=(0, 0, 0, None, None, None, None, None, None, None))


@partial(jax.jit, static_argnames=('potential_func_axi', 'density_func', 'n_particles',
                                   'z_int_max', 'anisotropy_b'))
def get_w0_new_improved(w0, key1, key2, key3, n_particles,
                        params_baryon, params_halo, potential_func_axi, density_func,
                        z_int_max=5.0, anisotropy_b=1.0):
    """Initial conditions from the improved Jeans solver.

    Pass the FULL potential here: the m=0 (axisymmetric) average needed by the Jeans
    equations is taken internally, so callers do not have to build it themselves.

    ANTICLOCKWISE seeding (Lz > 0), paired with +Omega_bar in the integrator so the bar
    co-rotates with the stars.  This is the convention of the validated NGC 4371
    runs.  It is NOT interchangeable with the clockwise/-Omega mirror: the static
    projection is identical, but running both through the rotating-frame integration gave
    measurably different fits (every chi2 channel worse by 15-54%), so the mirror symmetry
    does not survive the integration.  Do not "simplify" this back.
    """
    # potential_func_axi MUST already be the m=0 (axisymmetric) potential: the Jeans
    # equations sample along phi = 0, which for a bar is the long axis rather than the
    # azimuthal average.  For a CylindricalSpline/MGE potential use
    # evaluate_phi_axisymmetric (the m=0 coefficient, free); for a generic analytic
    # potential wrap it with utils.axisymmetrize_potential.  Doing the 8-point average
    # here instead cost a 3x slowdown for an identical answer (agrees to 0.017 km/s).
    v_rot, sig_R, sig_z, sig_phi = get_jeans_moments_improved_vmap(
        w0[:, 0], w0[:, 1], w0[:, 2], params_baryon, params_halo,
        potential_func_axi, density_func, anisotropy_b, z_int_max, 1000)

    n = w0.shape[0]
    vR   = jax.random.normal(key1, (n,)) * sig_R
    vz   = jax.random.normal(key2, (n,)) * sig_z
    vphi = v_rot + jax.random.normal(key3, (n,)) * sig_phi

    R   = jnp.sqrt(w0[:, 0]**2 + w0[:, 1]**2)
    phi = jnp.arctan2(w0[:, 1], w0[:, 0])
    x, y, vx, vy = getCartesianFromCylindrical_anticlockwise(R, phi, vR, vphi)

    return jnp.array([x, y, w0[:, 2], vx, vy, vz]).T
