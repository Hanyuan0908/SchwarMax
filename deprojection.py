"""
Deprojection of an observed 2-D MGE into intrinsic 3-D Gaussians, for real barred galaxies.

The input is a component-labelled 2-D MGE (e.g. from a GALFIT bulge + bar + disc
decomposition): per Gaussian a major-axis sigma, an observed axis ratio q_obs, an
isophotal twist relative to the disc line of nodes, and a flag saying whether it belongs
to the bar.

Conventions (must match `integrates.py`: sky = (x, y), LOS = z, R = Rz(-a) Rx(-b) Rz(-g)):

    alpha = on-sky position angle   (rotation about the LOS)
    beta  = inclination
    gamma = bar azimuth             (rotation about the galaxy's symmetry axis)

Intrinsic frame: bar long axis along +x, disc plane x-y, symmetry axis z.  All Gaussians
are axis-aligned there, so the 3-D density is a plain sum of aligned triaxial Gaussians.

Two routines:

    solve_bar_pq(inc, gam, q_obs, pa)   -- intrinsic (p, q) of the bar, SOLVED so that
                                           reprojecting reproduces the observed ellipse
    deproject_components(...)           -- full MGE -> (p, q, physical)

`physical` is the deprojectability gate: every Gaussian must satisfy 0 < q <= p <= 1.
Feed it to the likelihood as a hard -inf prior; a non-deprojectable geometry is not a
worse fit, it is not a galaxy.
"""

import jax
import jax.numpy as jnp
from functools import partial


# ----------------------------------------------------------------------------------
# Projected covariance of one aligned triaxial Gaussian (1, p, q) at alpha = 0.
#
#   S00 = cg^2 + sg^2 u
#   S01 = cb cg sg (u - 1)
#   S11 = cb^2 (sg^2 + cg^2 u) + sb^2 v          with u = p^2, v = q^2
#
# (alpha only rotates the sky plane, so it does not change q_obs; it shifts the sky PA.)
# ----------------------------------------------------------------------------------
def _bar_uv(inc, gam, u, bar_pa):
    b = jnp.radians(inc)
    g = jnp.radians(gam)
    cb, sb, cg, sg = jnp.cos(b), jnp.sin(b), jnp.cos(g), jnp.sin(g)
    tp = jnp.tan(jnp.radians(2.0 * bar_pa))
    S00 = cg**2 + sg**2 * u
    S01 = cb * cg * sg * (u - 1.0)
    S11 = S00 - 2.0 * S01 / tp                       # imposed by the observed sky PA
    v = (S11 - cb**2 * (sg**2 + cg**2 * u)) / (sb**2 + 1e-12)
    t = jnp.sqrt((S00 - S11)**2 + 4.0 * S01**2)
    q_proj = jnp.sqrt(jnp.clip((S00 + S11 - t) / 2.0, 0.0, None)
                      / jnp.clip((S00 + S11 + t) / 2.0, 1e-12, None))
    return q_proj, v


@partial(jax.jit, static_argnames=('n_iter',))
def solve_bar_pq(inc, gam, bar_qobs, bar_pa, n_iter=80):
    """Intrinsic (p, q) of the bar at this inclination and bar azimuth.

    Solved, not assumed: (p, q) are chosen so that projecting the triaxial Gaussian at
    (alpha=0, beta=inc, gamma=gam) reproduces the OBSERVED bar ellipse (bar_qobs, bar_pa).
    Bisection on u = p^2 -- jittable, no gradients needed (RMH is derivative-free).

    Returns (p, q, q_fit) where q_fit is the reprojected axis ratio, so the caller can
    check the solve converged.
    """
    def body(k, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        rm = _bar_uv(inc, gam, mid, bar_pa)[0] - bar_qobs
        rl = _bar_uv(inc, gam, lo, bar_pa)[0] - bar_qobs
        same = (rm > 0) == (rl > 0)
        return (jnp.where(same, mid, lo), jnp.where(same, hi, mid))

    lo, hi = jax.lax.fori_loop(0, n_iter, body, (1e-6, 1.0))
    u = 0.5 * (lo + hi)
    q_fit, v = _bar_uv(inc, gam, u, bar_pa)
    return (jnp.sqrt(jnp.clip(u, 1e-8, 1.0)),
            jnp.sqrt(jnp.clip(v, 1e-8, 1.0)),
            q_fit)


@partial(jax.jit, static_argnames=('n_iter',))
def deproject_components(inc, gam, q_obs, is_bar, bar_qobs, bar_pa, n_iter=80):
    """Deproject a component-labelled 2-D MGE.

    inc, gam : inclination and bar azimuth [deg]
    q_obs    : (ng,) observed axis ratio per Gaussian
    is_bar   : (ng,) 1 for bar Gaussians (triaxial), 0 for bulge/disc (oblate)
    bar_qobs, bar_pa : the bar component's observed axis ratio and sky PA (its twist
               relative to the disc line of nodes).  The bar is treated as ONE coherent
               component, so a single (p, q) applies to all of its Gaussians.

    Returns (p, q, physical):
        p, q     : (ng,) intrinsic sigma_y/sigma_x and sigma_z/sigma_x
        physical : scalar bool, the deprojectability gate

    Bulge/disc -> axisymmetric oblate, q_int^2 = (q_obs^2 - cos^2 i) / sin^2 i, p = 1.
    Bar        -> triaxial, from solve_bar_pq.
    """
    ci = jnp.cos(jnp.radians(inc))
    si = jnp.sin(jnp.radians(inc))
    q2_obl = (q_obs**2 - ci**2) / si**2

    pb, qb, q_fit = solve_bar_pq(inc, gam, bar_qobs, bar_pa, n_iter)

    p = jnp.where(is_bar > 0, pb, 1.0)
    q = jnp.where(is_bar > 0, qb, jnp.sqrt(jnp.clip(q2_obl, 1e-8, 1.0)))
    q = jnp.minimum(q, p)

    ok_oblate = jnp.all(jnp.where(is_bar > 0, True, q2_obl > 1e-4))   # disc/bulge deprojectable
    ok_bar = (qb > 1e-3) & (qb <= pb + 1e-6) & (pb <= 1.0 + 1e-6)     # bar physical
    ok_fit = jnp.abs(q_fit - bar_qobs) < 0.01                         # bar solve converged

    return p, q, (ok_oblate & ok_bar & ok_fit)


def inclination_floor(q_obs_disc_min):
    """Minimum inclination at which the roundest disc/bulge Gaussian still deprojects.

    i >= arccos(q_obs_min).  NOTE: only meaningful if q_obs was measured on CLEAN outer
    isophotes -- a bar- or arm-contaminated q_obs inflates this floor and biases i high.
    """
    import numpy as np
    return float(np.degrees(np.arccos(q_obs_disc_min)))
