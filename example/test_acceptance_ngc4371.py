"""Acceptance test: NGC 4371 against the validated fixed-potential pipeline.

Runs `build_model_acc` on the published NGC 4371 best-fit milestone (581 bins coarsened
from 1442, exact-MGE potential, 10 000 orbits) and compares the per-channel chi2 against
the numbers recorded in that milestone.

THE BENCHMARK IS `NGC4371_bestfit_milestone_2026-07-22/` -- NOT `ngc4371_pipeline/run_fit.py`,
which is superseded: it pairs with the old clockwise / -Omega_bar code AND mirror-flipped data
registration. **h1 chi2/bin ~ 10 is the diagnostic signature of that mirror flip.** If this test
ever reports h1 ~ 10, suspect the input data before suspecting the model.

Handedness: the milestone uses ACW seeding with +Omega_bar at angles (0, 59, 78) -- the same
convention `build_model_acc` uses, so NO angle flip is needed. The milestone records the exact
parity partner as (180, 59, -101) with CW/-Omega, i.e. the mirror flips BOTH alpha and gamma.

float64 is enabled here, matching the validated run. That is fine on CPU. Do NOT enable
it for GPU work -- it is both far slower and far more memory-hungry there.

Usage:  python test_acceptance_ngc4371.py
Exit status 0 if every channel is within TOL of the reference, 1 otherwise.
"""

import os
import sys
import pickle
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import jax
import jax.numpy as jnp

jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

from CylindricalSpline import evaluate_phi, evaluate_phi_axisymmetric, get_acc
from potentials import NFW_potential, NFW_acceleration
from model import build_model_acc

# Cache written by prepare_ngc4371_acceptance.py.  Override with NGC4371_CACHE.
CACHE = os.environ.get(
    'NGC4371_CACHE',
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'ngc4371_milestone_acceptance.pkl'))

# Target from MILESTONE.md.  (run_bestfit.py's own header reports a neighbouring
# realisation, 0.08 / 0.97 / 0.59 / 1.19 / 0.79.)
REFERENCE = {'logSig': 0.07, 'h1': 0.91, 'h2': 0.62, 'h3': 1.19, 'h4': 0.80}

# build_model_acc reproduces the milestone to 3.5% on every channel (measured:
# 1.035 / 1.007 / 0.995 / 1.007 / 1.029).  10% leaves room for JAX-version and
# platform drift while still catching a real regression -- a handedness error or a
# mismatched convention moves these by factors, not percent.
TOL = 0.10


def chi2_per_channel(res, dd):
    """Reduced chi2 per channel. Surface density is compared in log space after
    renormalising the model to the data total, since the NNLS fixes only the shape."""
    obs = np.asarray(dd['XY_density_data'])
    err = np.asarray(dd['XY_density_data_err'])
    model = np.asarray(res['surface_density'])[0]
    model = model * np.nansum(obs) / np.nansum(model)
    log_err = err / (obs * np.log(10) + 1e-30)
    out = {'logSig': float(np.nanmean(
        ((np.log10(model + 1e-30) - np.log10(obs + 1e-30)) / log_err) ** 2))}
    for name in ('h1', 'h2', 'h3', 'h4'):
        resid = np.asarray(res[name])[0] - np.asarray(dd[name + '_data'])
        out[name] = float(np.nanmean((resid / np.asarray(dd[name + '_data_err'])) ** 2))
    return out


def main():
    blob = pickle.load(open(CACHE, 'rb'))
    dd = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
          for k, v in blob['dd'].items()}
    dphi = {k: jnp.asarray(v) for k, v in blob['dphi'].items()}
    assert 'mge_S' in dphi, "potential must carry its MGE tracer"

    # MGE density, evaluated directly from the Gaussians. Never reconstruct it from the
    # Laplacian of the potential -- that is numerically unstable and can go negative.
    S, P, Q, rho0 = dphi['mge_S'], dphi['mge_P'], dphi['mge_Q'], dphi['mge_rho0']

    def dens(x, y, z, params_baryon):
        xf = jnp.asarray(x).ravel()
        yf = jnp.asarray(y).ravel()
        zf = jnp.asarray(z).ravel()
        r2 = (xf[:, None] ** 2 + yf[:, None] ** 2 / P[None, :] ** 2
              + zf[:, None] ** 2 / Q[None, :] ** 2) / S[None, :] ** 2
        return jnp.sum(rho0[None, :] * jnp.exp(-0.5 * r2), axis=-1).reshape(jnp.shape(x))

    def pot(x, y, z, pb, ph):
        return evaluate_phi(x, y, z, dphi) + NFW_potential(x, y, z, ph)

    def accf(x, y, z, pb, ph):
        # The spline's own analytic gradient. The integrator calls this ~1e9 times per
        # model evaluation, so differentiating the potential here is not an option.
        return get_acc(x, y, z, dphi) + NFW_acceleration(x, y, z, ph)

    def potaxi(x, y, z, pb, ph):
        # m=0 only: the Jeans equations assume Phi = Phi(R, z). The integrator above
        # still gets the FULL barred potential.
        return evaluate_phi_axisymmetric(x, y, z, dphi) + NFW_potential(x, y, z, ph)

    p_halo = {'logM': blob['halo']['logM'], 'Rs': blob['halo']['Rs'],
              'a': 1.0, 'b': 1.0, 'c': 1.0,
              'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
              'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}
    # Angles used as-is: the milestone is already in this package's ACW / +Omega_bar
    # convention.  Do NOT add 180 to alpha here -- doing so reproduces the superseded
    # mirror-flipped result instead of the published one.
    p_disk = {'mass_to_light_ratio': 1.0,           # M/L = 1
              'Omega_bar': blob['Omega'],           # 20 km/s/kpc -> rad/Gyr
              'alpha': float(dd['alpha']), 'beta': float(dd['beta']),
              'gamma': float(dd['gamma'])}

    t0 = time.time()
    # density_func_axi = dens here, reproducing the behaviour from before that argument
    # existed, so this regression test still measures the same thing.
    res = build_model_acc(
        dens, pot, accf, potaxi, dens, p_halo, p_disk, dd, int(dd['total_bins']),
        sigma_frac_3d=0.05, v_sys=0.0,   # milestone: relax_3d=2.5 -> 0.02*2.5
        Rzphi_n_tot=int(dd['Rzphi_n_tot']),
        Rzphi_n_grid=jnp.asarray(dd['Rzphi_n_grid']),
        Rzphi_lim_grid=jnp.asarray([dd['R_minmax'], dd['z_minmax'], dd['phi_minmax']]),
        xy_lim_grid=jnp.asarray([dd['X_minmax'], dd['Y_minmax']]),
        xy_n_grid=jnp.asarray(list(blob['dd']['nX_nY'])))
    jax.block_until_ready(res['surface_density'])
    elapsed = time.time() - t0

    got = chi2_per_channel(res, dd)
    print("\n=== NGC 4371 acceptance test ===")
    print("  %d bins, %d orbits, %.0f s\n" % (
        int(dd['total_bins']), dd['w0'].shape[0], elapsed))
    print("  channel      new    milestone    ratio")
    failures = []
    for name in ('logSig', 'h1', 'h2', 'h3', 'h4'):
        ratio = got[name] / REFERENCE[name]
        flag = 'ok' if abs(ratio - 1.0) <= TOL else 'FAIL'
        if flag == 'FAIL':
            failures.append(name)
        print("  %-8s %8.4f  %9.4f  %7.3f  %s" % (
            name, got[name], REFERENCE[name], ratio, flag))

    if failures:
        print("\n  FAILED on %s (tolerance %.0f%%)" % (', '.join(failures), 100 * TOL))
        return 1
    print("\n  PASS: every channel within %.0f%% of the validated pipeline" % (100 * TOL))
    return 0


if __name__ == '__main__':
    sys.exit(main())
