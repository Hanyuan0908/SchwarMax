"""
Template log-likelihood for a real barred galaxy modelled with an MGE ("fixed") potential.

DESIGN
------
`build_model_acc` is NOT modified and never needs to be.  It takes three callables plus
two parameter dicts; everything specific to *how the potential is built* lives here, in
the outer layer.  Choosing a different mass model (analytic disc+bar, MGE, CylSpline,
something else) therefore only changes this file.

    density_func(x, y, z, params_baryon)                  -> nu
    potential_func(x, y, z, params_baryon, params_halo)   -> Phi
    acceleration_func(x, y, z, params_baryon, params_halo)-> (3,) a

The three callables are defined INSIDE the jitted likelihood.  That matters: they are
static arguments of `build_model_acc`, so jit caches on their identity.  Defining them at
the top level of a *jitted* likelihood means they are created once, during that
likelihood's single trace -- verified: 6 different parameter vectors give 1 outer trace
and 1 inner trace, i.e. compiled once and reused, no per-sample recompilation.  (Building
them in a plain Python helper called per sample WOULD recompile every sample.)

Because they are defined inside the trace they can simply close over the traced
deprojection results (p, q, dphi); nothing has to be smuggled through params_baryon.

PARAMETER SPLIT
---------------
fixed, per galaxy (closed over, never traced)
    MGE:      sigma_x, q_obs, twist, light fractions, L_tot, is_bar, bar q_obs / PA
    geometry: alpha (sky PA), the gamma(i) relation
    halo:     R_s
    numerics: MMAX, Nphi, NR/NZ/R/Z limits, sigma_frac_3d

free, sampled (the theta vector, unpacked in ONE place)
    inclination i, log10 halo scale mass, log10 M/L, log10 Omega, V_sys

Unpacking theta once is deliberate: M/L enters both the MGE mass and
`mass_to_light_ratio`, and i enters the deprojection, `beta` and (via the slaving) gamma.
Splitting that across several places is exactly how those get out of step.
"""

import jax
import jax.numpy as jnp

from model import build_model_acc
from mge_model import get_phi_m_exact, mge_density
from deprojection import deproject_components
from potentials import NFW_potential, NFW_acceleration
from CylindricalSpline import evaluate_phi, evaluate_phi_axisymmetric, get_acc


def make_mge_likelihood(mge, dict_data, num_Vbin,
                        alpha=0.0, halo_Rs=14.68,
                        gamma_a=0.0, gamma_b=0.0,
                        sigma_frac_3d=0.02, sigma_amplify=1.0, Mmax=2, Nphi=32,
                        NR=50, NZ=30, Rmin=1e-3, Rmax=50.0, Zmin=1e-3, Zmax=20.0,
                        **build_kwargs):
    """Build a jitted logL(theta) for an MGE model.

    Parameters
    ----------
    mge : dict with the FIXED observed 2-D MGE
        'sigma_x'  (ng,) major-axis sigma [kpc]
        'q_obs'    (ng,) observed axis ratio
        'is_bar'   (ng,) 1 for bar Gaussians, 0 for bulge/disc
        'light'    (ng,) luminosity per Gaussian [Lsun]  (= light_frac * L_tot)
        'bar_qobs', 'bar_pa'  the bar component's observed axis ratio and sky PA
    alpha : on-sky position angle [deg].  0 if the data is already derotated so that the
        kinematic line of nodes lies along Sky-X.  NOTE the package seeds ANTICLOCKWISE
        (Lz>0) with +Omega_bar; the clockwise/-Omega mirror is NOT equivalent through the
        rotating-frame integration (verified: every chi2 channel 15-54% worse).
    gamma_a, gamma_b : bar azimuth slaved to inclination, gamma = gamma_a*i + gamma_b.
        Fit this line to the deprojectable band; sampling gamma freely tends to propose
        points off that thin ridge, which the gate then rejects.

    Returns
    -------
    logL(theta) : jitted callable.  theta = [i, log10 M_halo, log10 M/L, log10 Omega, V_sys]
        Returns -inf where the MGE is not deprojectable.
    """
    SIG      = jnp.asarray(mge['sigma_x'])
    QOBS     = jnp.asarray(mge['q_obs'])
    ISBAR    = jnp.asarray(mge['is_bar'])
    LIGHT    = jnp.asarray(mge['light'])
    BAR_QOBS = float(mge['bar_qobs'])
    BAR_PA   = float(mge['bar_pa'])

    @jax.jit
    def logL(theta):
        inc, log_Mhalo, log_ML, log_Om, v_sys = theta

        # ---- free parameters unpacked ONCE, and everything derived from them here ----
        gam = gamma_a * inc + gamma_b                 # bar azimuth slaved to inclination
        ML = 10.0 ** log_ML

        p, q, physical = deproject_components(inc, gam, QOBS, ISBAR, BAR_QOBS, BAR_PA)

        mass = LIGHT * ML                              # M/L must scale the POTENTIAL
        dphi = get_phi_m_exact(mass, SIG, p, q, NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax, Nphi)

        # ---- the three callables: defined here, closing over the traced MGE ----
        def density_func(x, y, z, params_baryon):
            # mge_density is shape-safe: the Jeans solver calls this with SCALARS, while the
            # 3-D mass integration calls it with ARRAYS of Monte-Carlo samples.  A naive
            # scalar form broadcasts the (N,) points against the (ng,) Gaussians and fails.
            return mge_density(x, y, z, mass, SIG, p, q)

        def potential_func(x, y, z, params_baryon, params_halo):
            return evaluate_phi(x, y, z, dphi) + NFW_potential(x, y, z, params_halo)

        def acceleration_func(x, y, z, params_baryon, params_halo):
            return get_acc(x, y, z, dphi) + NFW_acceleration(x, y, z, params_halo)

        def potential_func_axi(x, y, z, params_baryon, params_halo):
            # m=0 term ONLY, for the axisymmetric Jeans equations.  For a CylindricalSpline
            # this is a free lookup; averaging the full potential over 8 azimuths gives the
            # same answer (0.017 km/s in v_rot) at 3x the cost.
            return evaluate_phi_axisymmetric(x, y, z, dphi) + NFW_potential(x, y, z, params_halo)

        # ---- the two parameter dicts build_model_acc expects ----
        params_halo_pot = {'logM': log_Mhalo, 'Rs': halo_Rs,
                           'a': 1.0, 'b': 1.0, 'c': 1.0,
                           'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
                           'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}
        params_disk_rho = {'mass_to_light_ratio': ML,       # build_model_acc -> 1/(M/L)
                           'Omega_bar': (10.0 ** log_Om) / 0.97779,   # km/s/kpc -> rad/Gyr
                           'alpha': alpha, 'beta': inc, 'gamma': gam}

        res = build_model_acc(density_func, potential_func, acceleration_func, potential_func_axi,
                              params_halo_pot, params_disk_rho, dict_data, num_Vbin,
                              sigma_frac_3d=sigma_frac_3d, v_sys=v_sys,
                              **build_kwargs)

        # build_model_acc returns the model observables; the likelihood is formed here
        # (log-mean-exp over the bootstrap realizations, as in likelihood.py).
        O_den = dict_data['XY_density_data']
        E_den = dict_data['XY_density_data_err']
        den = res['surface_density']
        mden = den * (jnp.nansum(O_den) / jnp.nansum(den, axis=1, keepdims=True))
        sig_lden = E_den / (O_den * jnp.log(10) + 1e-30) * sigma_amplify
        r = jnp.nansum(((jnp.log10(mden + 1e-30) - jnp.log10(O_den + 1e-30)) / sig_lden)**2, axis=1)
        for hh, key in ((res['h1'], 'h1_data'), (res['h2'], 'h2_data'),
                        (res['h3'], 'h3_data'), (res['h4'], 'h4_data')):
            r = r + jnp.nansum(((hh - dict_data[key])
                                / (dict_data[key.replace('_data', '_data_err')] * sigma_amplify))**2, axis=1)
        ll = -0.5 * r
        m = jnp.max(ll)
        out = m + jnp.log(jnp.mean(jnp.exp(ll - m)))

        # deprojectability is a hard prior: a non-deprojectable geometry is not a worse
        # fit, it is not a galaxy.
        return jnp.where(physical, out, -jnp.inf)

    return logL
