"""Template: NFW halo + CylSpline axisymmetric disc + analytic triaxial bar.

The case this answers: the baryons are a double-exponential disc (no closed-form
potential, so it needs a CylindricalSpline) plus a Dehnen-like analytic bar (closed
form, but with no straightforward axisymmetric version), on top of an NFW halo.

THE ONE IDEA THAT MATTERS -- Phi is additive, so axisymmetrise ONLY the bar:

    Phi_m0 = Phi_disc(spline, m=0)  +  <Phi_bar>_phi  +  Phi_NFW
             ^ already axisymmetric    ^ the only term    ^ spherical, m=0 is itself
               (free, exact)             needing an average   (free, exact)

Do NOT wrap the whole potential in axisymmetrize_potential. That evaluates the
expensive spline n_phi times for nothing -- measured on a real galaxy, wrapping the full
spline cost 41 s of Jeans time versus 0.21 s using its own m=0.

TWO WAYS to get the bar's m=0 term, and which wins depends ENTIRELY on how expensive
your Phi_bar is per call.  Measure it; do not assume.

  (a) average at evaluation time -- axisymmetrize_potential(phi_bar, n_phi)
      costs n_phi * (one Phi_bar call), every call, forever.
  (b) expand once -- get_phi_m(rho_bar, ..., Mmax=0, Nphi=32) then
      evaluate_phi_axisymmetric.  Costs one spline build per sample (~0.8 s),
      after which each call is a spline lookup.  Exact in the m=0 sense: no
      n_phi aliasing at evaluation time, and Nphi is paid once so it can be large.

Measured here with a CHEAP closed-form Phi_bar (a few flops):
    average n_phi=16     0.0002 s / 2e5 evaluations
    spline m=0 lookup    0.0023 s / 2e5 evaluations      <- 11x SLOWER
because a spline gather costs more than 16 cheap analytic evaluations.  So (a) wins
for a cheap bar.  But a real triaxial Dehnen/Ferrers potential needs an ellipsoidal
quadrature per call, and then n_phi of them dominates and (b) wins by a wide margin.

Decision rule: time one Phi_bar call against one evaluate_phi_axisymmetric lookup.
If Phi_bar costs more than ~1/n_phi of a lookup, expand it.  Note (b) also keeps the
bar's Phi consistent with its rho by construction, which (a) does not check.

Build the disc spline with Mmax=0: the disc is axisymmetric, so every m != 0
harmonic is zero by construction and computing them is pure waste. With Mmax=0,
evaluate_phi and evaluate_phi_axisymmetric agree identically.

n_phi for the bar: averaging over n_phi equally spaced azimuths annihilates every
harmonic EXCEPT multiples of n_phi, so n_phi = 8 leaves m = 8, 16, ... intact.
MEASURED on the Dehnen-like bar below, <Phi>_phi at R = 8 kpc:

    n_phi =  4   -3413.8      <- badly wrong
    n_phi =  8   -3051.2      <- still 1.5% off
    n_phi = 16   -3007.9      <- 0.03%
    n_phi = 32   -3007.0

So the default here is 16, NOT 8. The band-limited argument that makes n_phi = 4
exact for an MGE (which carries only m = 0 and m = 2) does NOT carry over: a cusped
analytic bar has real power at m >= 8. ALWAYS check convergence against 2x n_phi --
it is cheap, because the bar is analytic.
"""

import jax
import jax.numpy as jnp

from CylindricalSpline import get_phi_m, evaluate_phi_axisymmetric, get_acc
from potentials import NFW_potential, NFW_acceleration
from utils import axisymmetrize_potential
from model import build_model_acc


# ---------------------------------------------------------------- components --
def rho_disc(x, y, z, p):
    """Double-exponential disc: rho = rho0 exp(-R/Rd) sech^2(z/2 hz)-ish."""
    R = jnp.sqrt(x ** 2 + y ** 2)
    Md, Rd, hz = 10 ** p['logM_disc'], p['Rd'], p['hz']
    rho0 = Md / (4.0 * jnp.pi * Rd ** 2 * hz)
    return rho0 * jnp.exp(-R / Rd) * jnp.exp(-jnp.abs(z) / hz)


def rho_bar(x, y, z, p):
    """Dehnen-like triaxial bar; replace with whatever profile you use.

    Only two things matter for the pipeline: it is ANALYTIC (cheap per call, and
    safe to differentiate) and it is TRIAXIAL (so it carries m != 0 and is the
    single term that needs azimuthal averaging).
    """
    Mb, a, q_b, p_b, gam = 10 ** p['logM_bar'], p['a_bar'], p['q_bar'], p['p_bar'], p['gamma_bar']
    m = jnp.sqrt(x ** 2 + (y / p_b) ** 2 + (z / q_b) ** 2) + 1e-8
    return (Mb * (3.0 - gam) / (4.0 * jnp.pi * p_b * q_b)) * a / (
        m ** gam * (m + a) ** (4.0 - gam))


def phi_bar(x, y, z, p):
    """Closed-form bar potential.

    *** WARNING -- THIS IS A PLACEHOLDER AND IS **NOT** THE POTENTIAL OF rho_bar
    ABOVE.  It is the SPHERICAL Hernquist form -G M / (m + a); the potential of a
    genuinely TRIAXIAL Dehnen density is not that, and requires an ellipsoidal
    quadrature.  Substituting your own consistent (rho, Phi) pair is mandatory --
    an inconsistent pair silently mismatches the mass in the potential against the
    mass in the 3D density constraint.  Checked numerically: the m=0 of this
    placeholder disagrees with the Poisson solution of rho_bar by 70-110%. ***
    """
    Mb, a, q_b, p_b = 10 ** p['logM_bar'], p['a_bar'], p['q_bar'], p['p_bar']
    m = jnp.sqrt(x ** 2 + (y / p_b) ** 2 + (z / q_b) ** 2) + 1e-8
    G = 4.498502151469553e-06                       # kpc (km/s)^2 / Msun
    return -G * Mb / (m + a)


# ------------------------------------------------------------- the likelihood -
def make_logL(dict_data, num_Vbin, grids, n_phi_bar=16):
    """Everything is built INSIDE the jitted logL so the callables close over the
    traced parameters of the current sample. Nothing model-specific has to travel
    through params_baryon, and there is no `if` anywhere -- so this jits and vmaps.
    """

    @jax.jit
    def logL(theta):
        inc, logMh, logML, logOm, v_sys, logMd, Rd, hz, logMb, a_bar = theta

        p_bary = {'logM_disc': logMd, 'Rd': Rd, 'hz': hz,
                  'logM_bar': logMb, 'a_bar': a_bar,
                  'q_bar': 0.3, 'p_bar': 0.4, 'gamma_bar': 1.0}
        p_halo = {'logM': logMh, 'Rs': 14.68, 'a': 1., 'b': 1., 'c': 1.,
                  'x_origin': 0., 'y_origin': 0., 'z_origin': 0.,
                  'dirx': 0., 'diry': 0., 'dirz': 1.}

        # Disc spline. Mmax=0: the disc is axisymmetric, so m != 0 is zero by
        # construction. Rebuilt every sample because Rd/hz/Md are free; if you FIX the
        # disc, build it once outside make_logL and close over it instead.
        dphi_disc = get_phi_m(rho_disc, p_bary, NR=40, NZ=30,
                              Rmin=1e-3, Rmax=40., Zmin=1e-3, Zmax=20.,
                              Mmax=0, Nphi=1, N_int=40)

        # --- the four callables build_model_acc needs -------------------------
        def dens(x, y, z, pb):                       # total baryonic MASS density
            return rho_disc(x, y, z, p_bary) + rho_bar(x, y, z, p_bary)

        def pot(x, y, z, pb, ph):                    # FULL potential (bar present)
            return (evaluate_phi_axisymmetric(x, y, z, dphi_disc)   # Mmax=0 => full
                    + phi_bar(x, y, z, p_bary)
                    + NFW_potential(x, y, z, ph))

        def accf(x, y, z, pb, ph):
            # Spline: its own analytic gradient. NEVER autodiff a CylindricalSpline
            # here -- the integrator calls this ~1e9 times per model evaluation.
            # Bar: analytic and cheap, so autodiff is fine and exact.
            a_bar_vec = -jnp.stack(jax.grad(lambda X, Y, Z: phi_bar(X, Y, Z, p_bary),
                                            argnums=(0, 1, 2))(x, y, z))
            return get_acc(x, y, z, dphi_disc) + a_bar_vec + NFW_acceleration(x, y, z, ph)

        # ONLY the bar is averaged. Disc and halo contribute their exact m=0.
        _bar_axi = axisymmetrize_potential(
            lambda X, Y, Z, pb, ph: phi_bar(X, Y, Z, p_bary), n_phi=n_phi_bar)

        def potaxi(x, y, z, pb, ph):
            return (evaluate_phi_axisymmetric(x, y, z, dphi_disc)
                    + _bar_axi(x, y, z, pb, ph)
                    + NFW_potential(x, y, z, ph))

        # params_disk_rho carries ONLY what build_model_acc itself reads; every
        # component parameter reaches the physics through the closures above.
        p_disk = {'mass_to_light_ratio': 10 ** logML,
                  'Omega_bar': (10 ** logOm) / 0.97779,
                  'alpha': 0.0, 'beta': inc, 'gamma': 54.5}

        res = build_model_acc(dens, pot, accf, potaxi, p_halo, p_disk,
                              dict_data, num_Vbin, sigma_frac_3d=0.05,
                              v_sys=v_sys, **grids)
        return res

    return logL
