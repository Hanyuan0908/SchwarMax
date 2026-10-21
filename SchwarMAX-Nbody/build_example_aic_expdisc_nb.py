#!/usr/bin/env python
"""Build the DOUBLE-EXPONENTIAL-DISC variant of the mock AIC notebook.

Input:  SchwarMAX_Nbody_AIC.ipynb, the Miyamoto-Nagai notebook built by
        ngc1512_pipeline/code/build_example_aic_nb.py --dest SchwarMAX-Nbody --out SchwarMAX_Nbody_AIC.ipynb
Output: SchwarMAX_Nbody_AIC_expdisc.ipynb

The variant is made by EDITING the MN notebook cell by cell, so every cell that is not about the
disc or the sampler stays byte-identical (install, data + handedness flip, Fourier/AIC config,
likelihood, minimiser, dAIC calibration, corner).  Two things change:

  THE DISC   Miyamoto-Nagai -> NGC 1512's double exponential, rho ~ exp(-R/Rd) exp(-|z|/hz),
             with its potential solved by the CylSpline.  The get_phi_m hyperparameters are
             READ OUT of the NGC 1512 notebook's own call, so they cannot drift from it.  Bar,
             bulge, halo, the ties (a_bar = L/5, b_bar = hz, V4 mass = M_bar), the stellar-mass
             tie to the light and the 12 parameters are unchanged; log Rd / log hz take the
             slots of log Rs / log Hs.
  THE SAMPLER  BlackJAX adaptive RMH -> emcee (DE + snooker moves, vectorize=True, chunked
             vmap, HDF5 backend with RESUME), NGC 1512's cell.

Every output file carries RUN_TAG = 'aic_exp', so nothing of the MN chain is read or overwritten.

    python build_example_aic_expdisc_nb.py
"""
import argparse, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
A = argparse.ArgumentParser()
A.add_argument('--base', default=os.path.join(HERE, 'SchwarMAX_Nbody_AIC.ipynb'))
A.add_argument('--out', default=os.path.join(HERE, 'SchwarMAX_Nbody_AIC_expdisc.ipynb'))
A.add_argument('--ngc1512-nb', dest='n1512', default=os.path.join(
    ROOT, '..', 'PHANGS_SchwarMAX', 'NGC1512_parametric_v1', 'PHANGS_NGC1512_parametric_F3DSQ4_MCMC.ipynb'),
               help='the NGC 1512 notebook whose CylSpline call is copied; if it is not there, the '
                    'recorded settings below are used')
A = A.parse_args()

nb = json.load(open(A.base))
CELLS = [[c['cell_type'], ''.join(c['source'])] for c in nb['cells']]
assert len(CELLS) == 19, 'the MN notebook changed its cell count (%d)' % len(CELLS)
for i, head in ((0, '# SchwarMAX — the AIC pipeline'), (2, '# ======'), (3, '# Use the Schwarzschild'),
                (4, '# ---- Fixed V4 bulge parameters ----'), (7, '# The AIC likelihood'),
                (8, '# ================================================================= CONFIGURATION'),
                (9, '# Example code for calling'), (10, '# example_schwarmax.ipynb cell 7'),
                (13, '# Multi-start optimisation'), (16, '# MCMC'), (17, '"""\nMCMC sampling using BlackJAX'),
                (18, '# Corner, with the TRUTH marked')):
    assert CELLS[i][1].startswith(head), 'cell %d is not what this builder expects: %r' % (i, CELLS[i][1][:60])


def put(i, text):
    CELLS[i][1] = text


def edit(i, old, new):
    assert CELLS[i][1].count(old) == 1, 'cell %d: anchor not found exactly once: %r' % (i, old[:70])
    CELLS[i][1] = CELLS[i][1].replace(old, new)


# ---------------------------------------------------------------- NGC 1512's CylSpline disc
# The get_phi_m settings are NGC 1512's.  When its notebook is reachable (the PHANGS workspace beside
# this repository) they are READ from it, so they cannot drift; otherwise the values recorded here
# (read 2026-10-20) are used and the build says so.
CYL_RECORDED = ['50', '30', '1e-3', '50.', '1e-3', '20.', '0', '1', '10000']
RHO_1512 = ("(10**p['logM']/(4*jnp.pi*p['Rd']**2*p['hz']))*jnp.exp(-R/p['Rd'])"
            "*jnp.exp(-jnp.abs(z)/p['hz'])")
if os.path.isfile(A.n1512):
    n15 = json.load(open(A.n1512))
    S11 = ''.join(n15['cells'][11]['source'])
    m = re.search(r'dphi = get_phi_m\(rho_disc, p_disc, ([^)]*)\)', S11)
    assert m, 'the NGC 1512 notebook lost its disc get_phi_m call'
    CYL = [v.strip() for v in m.group(1).split(',')]
    assert RHO_1512 in S11, 'the NGC 1512 rho_disc changed'      # ... it must be the one written below
    if CYL != CYL_RECORDED:
        print('NOTE: NGC 1512 now uses get_phi_m(%s); the recorded values were (%s)'
              % (', '.join(CYL), ', '.join(CYL_RECORDED)))
    CYL_SRC = os.path.basename(A.n1512)
else:
    CYL = list(CYL_RECORDED)
    CYL_SRC = 'the recorded values -- %s not found' % A.n1512
assert len(CYL) == 9, CYL
NR, NZ, RMIN, RMAX, ZMIN, ZMAX, MMAX, NPHI, NINT = CYL
assert (MMAX, NPHI) == ('0', '1'), 'NGC 1512 disc spline is no longer axisymmetric (Mmax=0, Nphi=1)'

# ================================================================= cell 0: the introduction
edit(0, '# SchwarMAX — the AIC pipeline, demonstrated on a mock N-body galaxy\n',
     r'''# SchwarMAX — the AIC pipeline on the mock N-body galaxy: double-exponential disc

**This is the exponential-disc variant of `SchwarMAX_Nbody_AIC.ipynb`.** Two things differ:

1. **The disc** is NGC 1512's double exponential, $\rho\propto e^{-R/R_d}\,e^{-|z|/h_z}$, in
   place of the Miyamoto–Nagai disc. Its potential is solved from the density by the CylSpline
   (`get_phi_m`) with NGC 1512's settings. The bar, the bulge, the halo, the ties
   ($a_{\rm bar}=L_{\rm bar}/5$, $b_{\rm bar}=h_z$, bulge mass $=M_{\rm bar}$), the stellar mass
   tied to the light and the 12 parameters are unchanged: $\log R_d$ and $\log h_z$ take the slots
   of $\log R_s$ and $\log H_s$.
2. **The sampler** is emcee (differential-evolution + snooker moves, vectorised), NGC 1512's
   cell, in place of BlackJAX.

The Miyamoto–Nagai chain recovered a disc that was too compact ($R_s$ 5.3 against 12 kpc) and a
bar that was too short and too weak. This notebook runs the same experiment with a disc of NGC
1512's form. Every file it writes carries `RUN_TAG = 'aic_exp'`.

---
''')
edit(0, '| sampler | BlackJAX adaptive RMH | **unchanged** |',
     '| stellar disc | Miyamoto–Nagai | **double exponential, CylSpline potential** |\n'
     '| sampler | BlackJAX adaptive RMH | **emcee** (affine-invariant ensemble, NGC 1512\'s cell) |')

edit(0, 'The same mock and the same density as `example_schwarmax.ipynb`.',
     'The same mock as `example_schwarmax.ipynb`.')

# ================================================================= cell 2: imports
edit(2, 'from functools import partial\nimport blackjax\n', 'from functools import partial\n')
edit(2, '''from potentials import (NFW_potential, MiyamotoNagai_potential, MiyamotoNagai_density,
                        T3_density, T3_potential, V4_density, V4_potential)
''', '''from potentials import NFW_potential, T3_density, T3_potential, V4_density, V4_potential
from CylindricalSpline import get_phi_m, evaluate_phi, evaluate_phi_axisymmetric, get_acc
''')

# ================================================================= cell 3: the model, in words
put(3, r'''# Use the Schwarzschild function in SchwarMAX to build model likelihood

The stellar density is a **double-exponential disc** (NGC 1512's form) plus the parent's T3 bar
and V4 bulge; the halo is the parent's NFW. The bar, bulge and halo potentials are analytic. The
disc potential has no closed form, so it is **solved from the density by the CylSpline**,
`get_phi_m`, at every $\theta$, with the settings of the NGC 1512 notebooks (this builder reads
them out of that notebook's own call).

`build_model_acc_meff_fourier` takes five callables: the density, the potential, the
acceleration, and the $m=0$ potential and density for the Jeans seeding. They depend on the disc
spline of the current $\theta$, so `make_model_funcs` builds all five **inside** `run_model`, as
NGC 1512 does. The disc acceleration is the spline's analytic gradient (`get_acc`); the analytic
part is $-\nabla\Phi$ by autodiff, as in the parent.

**Accuracy of the disc potential**, measured against AGAMA's exact `Disk` potential (float32, as
on the GPU, $R_d=6.47$ kpc):

| $h_z$ | disc $v_c$, max error | $a_z(R=4)$ at $z=h_z/2$ |
|---|---|---|
| 0.75 kpc (the truth) | 3.6% (−2.2 km/s at 3 kpc) | 6% |
| 0.25 kpc | 8.7% | 86% |
| 0.75 kpc, `CYL_NINT = 40000` | 1.0% | 0.0% |
| 0.25 kpc, `CYL_NINT = 40000` | 2.4% | 0.3% |

The quadrature (`N_int`), not the grid, sets the error. The NGC 1512 value is kept; raising
`CYL_NINT` in cell 4 is the knob if the chain wanders to thin discs.''')

# ================================================================= cell 4: the density and the five callables
put(4, r'''# ---- Fixed V4 bulge and bar parameters (as in example_schwarmax.ipynb) ----
V4_A, V4_B, V4_L, V4_GAMMA = 0.5, 0.5, 0.1, 0.0
GAMMA_BAR = 1.0


# ---- the DOUBLE-EXPONENTIAL disc: NGC 1512's rho_disc, in place of the Miyamoto-Nagai disc ----
def rho_expdisc(x, y, z, p):
    """M / (4 pi Rd^2 hz) exp(-R/Rd) exp(-|z|/hz), with p = {'logM', 'Rd', 'hz'}."""
    R = jnp.sqrt(x**2 + y**2)
    return __RHO__


# Its potential is solved by the CylSpline with NGC 1512's settings, which that notebook
# benchmarked against an exact Hankel solution:
#     get_phi_m(rho, p, NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax, Nphi, N_int)
# Mmax = 0: the disc is axisymmetric.  DO NOT LOWER N_int -- the quadrature is base x base with
# base = max(9, int(sqrt(N_int))), so every N_int below 81 collapses to the same 9x9 grid.
# Measured against AGAMA's 'Disk' in float32 (Rd 6.47 kpc), max error in the disc's own force:
#     hz 0.75 (the truth)  v_c 3.6% (-2.2 km/s at 3 kpc)   a_z(R=4, z=hz/2) 6%
#     hz 0.25              v_c 8.7%                        a_z(R=4, z=hz/2) 86%
#     CYL_NINT = 40000     1.0% / 0.0% at hz 0.75;  2.4% / 0.3% at hz 0.25  (more memory)
CYL_NR, CYL_NZ = __NR__, __NZ__
CYL_RMIN, CYL_RMAX, CYL_ZMIN, CYL_ZMAX = __RMIN__, __RMAX__, __ZMIN__, __ZMAX__
CYL_MMAX, CYL_NPHI = __MMAX__, __NPHI__
CYL_NINT = __NINT__       # <<< the accuracy knob: see the table above


def disc_spline(p_disc):
    """The disc's CylSpline potential (a dict of spline coefficients) for p = {'logM', 'Rd', 'hz'}."""
    return get_phi_m(rho_expdisc, p_disc, CYL_NR, CYL_NZ, CYL_RMIN, CYL_RMAX,
                     CYL_ZMIN, CYL_ZMAX, CYL_MMAX, CYL_NPHI, CYL_NINT)


# SHAPE SAFETY.  The Fourier target quadrature evaluates the density on a 3-D broadcast grid,
# and `potentials._shift` (inside NFW_potential) builds jnp.array([x-x0, ...]) and needs three
# IDENTICALLY shaped arrays, while `_rotate` does `R @ vec` with R of shape (3,3).  Broadcasting
# AND flattening fixes both and changes no physics.
def _flat(f):
    @jax.jit
    def g(x, y, z, *a):
        x, y, z = jnp.broadcast_arrays(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z))
        return jnp.reshape(f(x.ravel(), y.ravel(), z.ravel(), *a), x.shape)
    return g


# ================================================================= the m=0 tabulation
# The Jeans seeding needs the AXISYMMETRIC (m=0) part of the density and potential.  The disc
# spline is axisymmetric (Mmax = 0), the V4 bulge nearly so and the NFW halo exactly.
#
# TRAP.  get_w0_new_improved DIFFERENTIATES through these, so computing the bar's m=0 part as an
# azimuthal quadrature here puts n_phi copies of the integrand in the reverse-mode tape:
# MEASURED, 400 orbits, 0.98 GB -> 5.60 GB and 5.1 s -> 17.5 s.  The bar's m=0 part is
# therefore TABULATED ONCE on an (R,|z|) grid, outside the tape, and looked up by linear
# interpolation.
#
# SECOND TRAP.  The V4 bulge is deliberately NOT tabulated.  It has a = b = 0.5 kpc against a
# grid step dR = 0.48 kpc, so the grid cannot resolve it -- the same mistake, on NGC 1512's
# NSD, drove the surface-density chi2 from 13.3 to 140.8.  It also needs no averaging: it
# varies only 2-6% in azimuth, against 106-479% for the T3 bar.
M0_NR, M0_NZ, M0_RMAX, M0_ZMAX = 64, 33, 30.0, 10.0
M0_NPHI = 16          # only the BAR is averaged, and its m=0 converges fast; 16 >> 2*mmax


def make_model_funcs(params_baryon, params_halo):
    """The five callables build_model_acc_meff_fourier takes, at ONE theta:
        (density, potential, acceleration, potential_axi, density_axi)

    Called inside run_model, so they close over THIS theta's disc spline and bar table.  The
    params dicts the builder later passes back to them are accepted and ignored: model.py
    rebuilds params_baryon with the Miyamoto-Nagai key names and would drop Rd and hz.
    Safe inside a jitted function -- the closures capture tracers (NGC 1512 does the same).
    """
    p_disc = {'logM': params_baryon['logM_disc'], 'Rd': params_baryon['Rd_disc'],
              'hz': params_baryon['hz_disc']}
    dphi = disc_spline(p_disc)
    M_bar = 10.0 ** params_baryon['logM_bar']
    BAR = (M_bar, params_baryon['a_bar'], params_baryon['b_bar'], params_baryon['L_bar'], GAMMA_BAR)
    BUL = (M_bar, V4_A, V4_B, V4_L, V4_GAMMA)          # the bulge carries M_bar, as in the parent

    def density(x, y, z, pb):
        return rho_expdisc(x, y, z, p_disc) + T3_density(x, y, z, *BAR) + V4_density(x, y, z, *BUL)

    def _analytic(x, y, z, ph):                         # bar + bulge + halo
        return NFW_potential(x, y, z, ph) + T3_potential(x, y, z, *BAR) + V4_potential(x, y, z, *BUL)

    def potential(x, y, z, pb, ph):
        return evaluate_phi(x, y, z, dphi) + _analytic(x, y, z, ph)

    def acceleration(x, y, z, pb, ph):
        # disc: the spline's analytic gradient; the rest: -grad by autodiff, as in the parent
        g = jax.grad(lambda q: _analytic(q[0], q[1], q[2], ph))(jnp.array([x, y, z]))
        return get_acc(x, y, z, dphi) - g

    # the bar's m=0 part, tabulated once (TRAP above)
    Rt = jnp.linspace(0.0, M0_RMAX, M0_NR)
    zt = jnp.linspace(0.0, M0_ZMAX, M0_NZ)
    at = jnp.linspace(0.0, 2 * jnp.pi, M0_NPHI, endpoint=False)
    RR, ZZ = jnp.meshgrid(Rt, zt, indexing='ij')
    rho_tab = jnp.mean(jax.vmap(lambda a: T3_density(RR * jnp.cos(a), RR * jnp.sin(a), ZZ, *BAR))(at), 0)
    pot_tab = jnp.mean(jax.vmap(lambda a: T3_potential(RR * jnp.cos(a), RR * jnp.sin(a), ZZ, *BAR))(at), 0)
    dR, dz = M0_RMAX / (M0_NR - 1), M0_ZMAX / (M0_NZ - 1)

    def look(T, R, z):
        fR = jnp.clip(jnp.asarray(R) / dR, 0.0, M0_NR - 1.0)
        fz = jnp.clip(jnp.abs(jnp.asarray(z)) / dz, 0.0, M0_NZ - 1.0)
        return jax.scipy.ndimage.map_coordinates(
            T, [fR.ravel(), fz.ravel()], order=1, mode='nearest').reshape(jnp.shape(fR))

    def density_axi(x, y, z, pb):
        R = jnp.sqrt(x ** 2 + y ** 2)
        return rho_expdisc(x, y, z, p_disc) + look(rho_tab, R, z) + V4_density(x, y, z, *BUL)

    def potential_axi(x, y, z, pb, ph):
        R = jnp.sqrt(x ** 2 + y ** 2)
        return (NFW_potential(x, y, z, ph) + evaluate_phi_axisymmetric(x, y, z, dphi)
                + look(pot_tab, R, z) + V4_potential(x, y, z, *BUL))

    return _flat(density), _flat(potential), acceleration, potential_axi, density_axi
'''.replace('__RHO__', RHO_1512).replace('__NR__', NR).replace('__NZ__', NZ)
       .replace('__RMIN__', RMIN).replace('__RMAX__', RMAX).replace('__ZMIN__', ZMIN)
       .replace('__ZMAX__', ZMAX).replace('__MMAX__', MMAX).replace('__NPHI__', NPHI)
       .replace('__NINT__', NINT))

# ================================================================= cell 7: the box fraction text
edit(7, r'''  the Miyamoto–Nagai disc has a long tail, and at the generating parameters only 63% of
  $M_{\rm disc}$ projects inside the box, while every N-body star lies within $R=20$ kpc.''',
     r'''  at the generating parameters 93% of $M_{\rm disc}$ projects inside the box (the parent's
  Miyamoto–Nagai disc, with its long tail, managed 63%), while every N-body star lies within
  $R=20$ kpc.''')

# ================================================================= cell 8: normalisation + run_model
C8 = CELLS[8][1]
edit(8, '''# Every file this notebook writes carries RUN_TAG, so a chain of the previous 13-parameter
# layout (the *_aic.pkl / *_aic.npy files) is never resumed into, or overwritten by, this one.
RUN_TAG = 'aic_tied'
''', '''# Every file this notebook writes carries RUN_TAG, so neither the Miyamoto-Nagai chain
# (*_aic_tied*) nor the 13-parameter one (*_aic*) is ever resumed into, or overwritten by, this one.
RUN_TAG = 'aic_exp'
''')
C8 = CELLS[8][1]
_a = C8.index('# ================================================================= ONE stellar normalisation')
_b = C8.index("@partial(jax.jit, static_argnames=('num_Vbin', 'Rzphi_n_tot'))\ndef model_likelihood")
_k0 = C8.index("    X_minmax, Y_minmax = dict_data['X_minmax'], dict_data['Y_minmax']")
_k1 = C8.index('    return dict(out, M_star=M_star')
CALL = C8[_k0:_k1]
_old_args = '''        density_func_bc, potential_func_bc, acceleration_func,
        potential_func_axi, density_func_axi,
'''
assert CALL.count(_old_args) == 1, 'the builder call in cell 8 changed shape'
CALL = CALL.replace(_old_args, '        dens, pot, acc, pot_axi, dens_axi,\n')

MIDDLE = r'''# ================================================================= ONE stellar normalisation
# The stellar mass is DERIVED (markdown above):  M_star * p_box(theta) = M/L * L_BOX.
BOX_HALF = 20.0           # kpc: the sky box |X|, |Y| < BOX_HALF, centred on the galaxy
L_BOX    = 6.4078e10      # Msun at M/L = 1: the light inside it, 99.96% of the total
                          # (make_mock_box_light.py; for a real galaxy, sum the image)
BOX_NXY, BOX_NLOS, BOX_SMAX = 128, 256, 200.0   # the exponential disc: within 4.4e-4 of a
                          # 512 x 2048, s_max 400 reference for hz 0.1-2 kpc, i 40-80 deg (measured)


def _sinh_nodes(half, a, n):
    """Midpoint nodes of x = half*sinh(a*u)/sinh(a), u uniform in (-1, 1), and their dx: dense
    at the centre (the V4 bulge has a = 0.5 kpc), coarse in the smooth outskirts."""
    u = (jnp.arange(n) + 0.5) / n * 2.0 - 1.0
    return (half * jnp.sinh(a * u) / jnp.sinh(a),
            half * a * jnp.cosh(a * u) / jnp.sinh(a) * (2.0 / n))


_BX, _BW  = _sinh_nodes(BOX_HALF, 4.0, BOX_NXY)      # sky X (and Y)
_BS, _BSW = _sinh_nodes(BOX_SMAX, 7.0, BOX_NLOS)     # line of sight
_BY  = jnp.broadcast_to(_BX[:, None], (BOX_NXY, BOX_NLOS)).ravel()
_BSS = jnp.broadcast_to(_BS[None, :], (BOX_NXY, BOX_NLOS)).ravel()
_BWT = (_BW[:, None] * _BSW[None, :]).ravel()


def box_fractions(Rd_disc, hz_disc, L_bar, a_bar, b_bar, alpha, beta, gamma):
    """(p_disc, p_bar): the fraction of a UNIT-MASS disc, and of the unit-mass bar + bulge
    pair, that projects inside the sky box.  Angles in DEGREES, the convention of model.py:
    X = R @ x with R = makeRotationMatrix(alpha, beta, gamma), sky (X0, X1), line of sight X2.
    One sky column per lax.map step (~33k points), so it adds ~0.1 GB, not ~4 GB."""
    R    = makeRotationMatrix(alpha, beta, gamma)
    unit = {'logM': 0.0, 'Rd': Rd_disc, 'hz': hz_disc}

    def column(X):
        x, y, z = R.T @ jnp.stack([jnp.full_like(_BY, X), _BY, _BSS])
        rd = rho_expdisc(x, y, z, unit)
        rb = 0.5 * (T3_density(x, y, z, 1.0, a_bar, b_bar, L_bar, GAMMA_BAR)
                    + V4_density(x, y, z, 1.0, V4_A, V4_B, V4_L, V4_GAMMA))
        return jnp.stack([jnp.sum(rd * _BWT), jnp.sum(rb * _BWT)])

    p = jnp.sum(jax.lax.map(column, _BX) * _BW[:, None], axis=0)
    return p[0], p[1]


def stellar_masses(f_bar, log_mass_to_light_ratio, Rd_disc, hz_disc, L_bar, a_bar, b_bar,
                   alpha, beta, gamma):
    """-> (logM_disc, logM_bar, M_star, p_box).  M_star * p_box = M/L * L_BOX, then split:
    M_disc = (1 - f_bar) M_star, and M_bar = f_bar M_star / 2 because the T3 bar and the V4
    bulge EACH carry M_bar."""
    p_disc, p_bar = box_fractions(Rd_disc, hz_disc, L_bar, a_bar, b_bar, alpha, beta, gamma)
    p_box  = (1.0 - f_bar) * p_disc + f_bar * p_bar
    M_star = 10.0 ** log_mass_to_light_ratio * L_BOX / p_box
    return (jnp.log10((1.0 - f_bar) * M_star), jnp.log10(0.5 * f_bar * M_star), M_star, p_box)


# ================================================================= the 12 parameters
# The same 12 as the Miyamoto-Nagai notebook; the disc scale length and scale height take the
# slots of Rs and Hs.
param_names = ['logM_10kpc', 'f_bar', 'logC_halo', 'logRd_disk',
               'loghz_disk', 'logL_bar', 'alpha', 'beta', 'gamma',
               'log_mass_to_light_ratio', 'log_Omega', 'log10_lambda']
NDIM  = len(param_names)
J_LAM = param_names.index('log10_lambda')      # NEVER hard-code this index


@partial(jax.jit, static_argnames=('num_Vbin', 'Rzphi_n_tot', 'n_boot'))
def run_model(theta, dict_data, num_Vbin, Rzphi_n_tot, n_boot=N_BOOT_MEFF):
    """The forward model: the parent's bar, bulge and halo, NGC 1512's double-exponential disc
    with a CylSpline potential, and the stellar mass derived from M/L."""
    logM_enc, log_c = theta[0], theta[2]
    logM_halo, logRs_halo = logMenc_logc_to_logM_logRs(logM_enc, log_c, r_enc=10.0,
                                                       Delta=200., rho_crit=277.54)
    f_bar = theta[1]
    logRd_disk, loghz_disk, logL_bar = theta[3], theta[4], theta[5]
    alpha = theta[6] * 180 / jnp.pi
    beta  = theta[7] * 180 / jnp.pi
    gamma = theta[8] * 180 / jnp.pi
    log_mass_to_light_ratio, log_Omega_bar = theta[9], theta[10]

    L_bar   = 10.0 ** logL_bar
    a_bar   = L_bar / 5.0
    Rd_disc = 10.0 ** logRd_disk
    hz_disc = 10.0 ** loghz_disk
    b_bar   = hz_disc                   # the bar is as thick as the disc, as in the parent

    logM_disc, logM_bar, M_star, p_box = stellar_masses(
        f_bar, log_mass_to_light_ratio, Rd_disc, hz_disc, L_bar, a_bar, b_bar,
        alpha, beta, gamma)

    params_halo_pot = {'logM': logM_halo, 'Rs': 10 ** logRs_halo, 'a': 1.0, 'b': 1.0, 'c': 1.0,
                       'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
                       'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}
    params_baryon_rho = {'logM_disc': logM_disc, 'Rd_disc': Rd_disc, 'hz_disc': hz_disc,
                         'logM_bar': logM_bar, 'L_bar': L_bar,
                         'a_bar': a_bar, 'b_bar': b_bar,
                         'mass_to_light_ratio': 10 ** log_mass_to_light_ratio,
                         'Omega_bar': 10 ** log_Omega_bar,
                         'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
                         'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
                         'alpha': alpha, 'beta': beta, 'gamma': gamma}

    # the five callables, closing over THIS theta's disc spline and bar table (cell 4)
    dens, pot, acc, pot_axi, dens_axi = make_model_funcs(params_baryon_rho, params_halo_pot)

''' + CALL + '''    return dict(out, M_star=M_star, p_box=p_box, logM_disc=logM_disc, logM_bar=logM_bar)


'''
put(8, C8[:_a] + MIDDLE + C8[_b:])

# ================================================================= cells 9-10: the ground truth
put(9, r'''# Example code for calling the likelihood function

The mock's generating parameters are known, so these scans have a right answer. The halo, the
viewing angles, $\Omega$ and $M/L$ are the mock's own, exactly as in the Miyamoto–Nagai notebook.
The **disc and bar shapes and $f_{\rm bar}$** are this family's: the parent's cell 7 is a
Miyamoto–Nagai fit and does not describe an exponential disc. They come from a direct
least-squares fit of **this** family — double-exponential disc + T3 bar + V4 bulge, with
`run_model`'s ties — to the N-body star particles: the stellar $v_c(R)$ at 1%, the face-on
$\Sigma(R)$ at 0.05 dex and $A_2/A_0(R)$ at 0.02 (`fit_family_to_nbody.py`, family `exp_tied`).

**One exception: $L_{\rm bar}$ is the published example's 4.87 kpc** (its cell 7, which the
Miyamoto–Nagai notebook also uses), not the 5.34 kpc of this fit, so all three notebooks share one
bar length. $L_{\rm bar}$ is a parameter of the T3 model, not a measured bar length. With this fit's
other shapes, 4.87 kpc puts the $A_2/A_0$ peak (0.37) at 2.6 kpc, and it falls to half by 4.8 kpc.
The N-body peaks at 0.40 at 3.3 kpc and falls to half by 5.4 kpc.

$\log M/L=0$ is exact: the mock's surface density *is* the N-body's stellar mass. The shapes are a
parametric description, and an imperfect one: no single smooth disc follows the N-body's disc,
whose local scale length grows from 2.4 kpc (2–6 kpc) to 9.4 kpc (4–9 kpc). At these shapes the
model needs 28% more mass in the box than the light ($\log M/L=+0.108$) to match the N-body's
stellar potential. `chi2_report` prints the derived $M_\star$ and $p_{\rm box}$.''')

put(10, r'''# The disc and bar shapes and f_bar: a direct fit of THIS family (exponential disc + T3 bar +
# V4 bulge, run_model's ties) to the N-body star particles -- v_c,*(R) 1%, face-on Sigma 0.05 dex,
# A2/A0 0.02 (fit_family_to_nbody.py, family exp_tied).  Halo, angles, Omega, M/L: the mock's own.
# EXCEPT L_bar: the published example's 4.87 kpc (example_schwarmax.ipynb cell 7), shared by all three
# notebooks; this fit gave 5.343.  L_bar barely moves the derived masses (p_box, p_field change < 1e-3).
_Md, _Mb = 5.37301e10, 1.60154e10    # disc; bar (= bulge)  ->  f_bar = 2 M_bar / M_star
ground_truth = [
  10.76,                       # logM_10kpc
  2 * _Mb / (_Md + 2 * _Mb),   # f_bar = 0.3735
  np.log10(7.76),              # logC_halo
  np.log10(6.466),             # logRd_disk  [kpc]
  np.log10(0.7477),            # loghz_disk  [kpc]
  np.log10(4.87),              # logL_bar    [kpc]  (the published example's; this fit gave 5.343)
  np.radians(25.0),            # alpha
  np.radians(65.0),            # beta   (inclination)
  np.radians(45.0),            # gamma
  0.0,                         # log_mass_to_light_ratio
  1.40,                        # log_Omega
]
theta_true = jnp.asarray(ground_truth + [0.0])      # log10(lambda) = 0 -> lambda = 1

t0 = time.time()
rep = chi2_report(theta_true)
print('chi2_report at GROUND TRUTH  (%.0f s)' % (time.time() - t0))
for k, v in rep.items():
    print('   %-12s %10.4f' % (k, v))
print('\nlogL at truth (S2 = %.2f): %.4f'
      % (S2, float(model_likelihood(theta_true, dict_data, NUM_VBIN, RZ_TOT, jnp.sqrt(S2)))))
''')

# ================================================================= cell 13: comments that named BlackJAX
edit(13, '''The MCMC below
#   vmaps 16 chains at a time, so a serially-optimised start''', '''emcee below
#   evaluates its walkers in vmapped chunks, so a serially-optimised start''')
edit(13, '''VMAP_BATCH = 8      # <<< TUNE ME: evaluations per batched call -- the same budget as the
                    # sampler's N_PER below, which is measured to run on a 24 GB L4.  Lower
                    # it on a smaller card.''', '''VMAP_BATCH = 8      # <<< TUNE ME: evaluations per batched call -- the same budget as
                    # EMCEE_CHUNK below.  The Miyamoto-Nagai notebook ran 8 at once on a 24 GB
                    # L4; the disc spline adds to that (not yet measured on a GPU).  Lower it
                    # on a smaller card or on an out-of-memory error.''')
edit(13, '''# Per-parameter step.  These are the MCMC's tested `rmh_sigma_init` (see the sampler cell), so
# the hill climb and the sampler move on the same scales.''', '''# Per-parameter step: the Miyamoto-Nagai notebook's tested BlackJAX `rmh_sigma_init`.''')

# ================================================================= cells 16-17: emcee
put(16, r'''# MCMC

**emcee**, the affine-invariant ensemble sampler, with differential-evolution (80%) and snooker
(20%) moves — the cell of the NGC 1512 notebooks. It replaces BlackJAX's adaptive random-walk
Metropolis, which stalled on NGC 1512 (acceptance 0.41 → 0.04, $\hat R$ 1.3–2.3, a proposal 3–9×
wider than the posterior) and ran at acceptance 0.125 with $\hat R$ 1.19–1.49 after 690 steps on
this mock. Affine-invariant moves need no tuned covariance, which is the part that failed.

`vectorize=True` hands each half-ensemble to `log_prob_vec` in one call, evaluated in vmapped
chunks of `EMCEE_CHUNK` (the memory knob). The chain is written to an HDF5 file after every step:
with `RESUME = True` a re-run continues it up to `EMCEE_STEPS` in total, so a Colab disconnect
loses nothing.''')

put(17, r'''# ===========================================================================
# MCMC: emcee affine-invariant ensemble (NGC 1512's cell).
# vectorize=True gives log_prob_vec the whole half-ensemble in ONE call, so jax.vmap works; the
# call is CHUNKED so peak memory is set by EMCEE_CHUNK, not by the walker count.
# ===========================================================================
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'emcee'], check=False)
import emcee
print('emcee', emcee.__version__)

EMCEE_WALKERS = 32        # >= 2*NDIM = 24 (emcee refuses fewer); a multiple of 4*EMCEE_CHUNK
EMCEE_STEPS   = 700      # TOTAL chain length, not steps-per-run: with RESUME the cell tops the
                          # stored chain up to this number
EMCEE_CHUNK   = 8         # <<< TUNE ME: evaluations held in flight at once (memory knob).  Keep
                          # it a divisor of EMCEE_WALKERS // 4: the snooker move updates the
                          # walkers in 4 groups, and a short group is padded to a full chunk.
                          # Lower it on an out-of-memory error.
EMCEE_FILE    = os.path.join(path, 'example/emcee_%s.h5' % RUN_TAG)
OUTPUT_FILE   = os.path.join(path, 'example/emcee_results_%s.pkl' % RUN_TAG)
RESUME        = True      # True  -> continue the chain in EMCEE_FILE (an empty or missing file
                          #          starts at step 0).  The backend flushes every step.
                          # False -> start over, DELETING that file's contents.

res = np.load(MINIMISER_RESULT)

# ---- Prior bounds (12D): identical to the Miyamoto-Nagai notebook -------------------------
# f_bar: uniform on [0.01, 0.99], kept off 0 and 1 where log M_bar or log M_disc is -inf.
BOUNDS_LO = jnp.array([res[0] - 3, 0.01, res[2] - 1, res[3] - 1,
                       res[4] - 1, max(0, res[5] - 1),
                       0., 0., 0., -2., 0., -1.])
BOUNDS_HI = jnp.array([res[0] + 3, 0.99, res[2] + 1, res[3] + 1,
                       res[4] + 1, min(1, res[5] + 1),
                       float(jnp.pi), float(jnp.pi / 2), float(jnp.pi), 2., 2., 4.])
assert BOUNDS_LO.size == NDIM and BOUNDS_HI.size == NDIM


def logdensity_fn(theta):
    in_bounds = jnp.all((theta >= BOUNDS_LO) & (theta <= BOUNDS_HI))
    log_vol   = jnp.sum(jnp.log(BOUNDS_HI - BOUNDS_LO))
    logprior  = jnp.where(in_bounds, -log_vol, -jnp.inf)
    ll = model_likelihood(theta, dict_data, NUM_VBIN, RZ_TOT, sigma_amplify=SIGMA_AMPLIFY)
    ll = jnp.where(jnp.isfinite(ll), ll, -1e30)
    return logprior + ll


_lik = jax.jit(jax.vmap(logdensity_fn))


def log_prob_vec(P):
    """P is (n, NDIM) from emcee (vectorize=True) -> (n,) log-probabilities, in chunks of
    EMCEE_CHUNK.  A short last chunk is PADDED to EMCEE_CHUNK, so there is one compiled shape
    and no recompilation.  A model that failed (-1e30) is returned as -inf."""
    P = np.atleast_2d(np.asarray(P, dtype=np.float32))
    out = np.empty(P.shape[0], dtype=np.float64)
    for a in range(0, P.shape[0], EMCEE_CHUNK):
        blk = P[a:a + EMCEE_CHUNK]
        n = blk.shape[0]
        if n < EMCEE_CHUNK:
            blk = np.concatenate([blk, np.repeat(blk[-1:], EMCEE_CHUNK - n, axis=0)])
        out[a:a + n] = np.asarray(_lik(jnp.asarray(blk)), dtype=np.float64)[:n]
    return np.where(np.isfinite(out) & (out > -1e29), out, -np.inf)


print('initial parameters (the minimiser result):')
for i in range(NDIM):
    print('   %-26s %.6f' % (param_names[i], res[i]))

_lo, _hi = np.asarray(BOUNDS_LO, float), np.asarray(BOUNDS_HI, float)
_moves   = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
_backend = emcee.backends.HDFBackend(EMCEE_FILE)


def _stored_steps(b):
    """Steps already in the backend.  emcee's .iteration RAISES on a file that does not exist
    yet (the first run) or that reset() never initialised, rather than returning 0."""
    try:
        return b.iteration
    except (OSError, KeyError):
        return 0


if RESUME and _stored_steps(_backend) > 0:
    # get_last_sample() carries the coordinates, the log-probabilities AND the RNG state, so this
    # is a true continuation.  The walker count comes from the FILE: emcee refuses a sampler
    # whose shape disagrees with its backend.
    _done = _backend.iteration
    EMCEE_WALKERS = _backend.shape[0]
    _p0 = _backend.get_last_sample()
    print('RESUME: %d steps already stored (%d walkers), topping up to EMCEE_STEPS = %d'
          % (_done, EMCEE_WALKERS, EMCEE_STEPS))
else:
    _done = 0
    # Start in a ball around the minimiser result, REFLECTED off the prior box rather than
    # clipped (clipping stacks walkers on a bound, where the prior returns -inf).  The ball is
    # the Miyamoto-Nagai notebook's BlackJAX initial_guess_sigma.
    EMCEE_BALL = np.array([0.05, 0.03, 0.02, 0.02, 0.02, 0.02,
                           0.02, 0.02, 0.10, 0.05, 0.10, 0.30])
    assert EMCEE_BALL.size == NDIM
    _start = np.asarray(res, dtype=np.float64)
    _eps = 1e-4 * (_hi - _lo)         # survives the float32 cast in log_prob_vec
    _oob = np.where((_start <= _lo) | (_start >= _hi))[0]
    assert _oob.size == 0, ('the start point is outside the prior in ' + ', '.join(
        '%s=%.4f not in [%.4f, %.4f]' % (param_names[k], _start[k], _lo[k], _hi[k]) for k in _oob))

    def _reflect(x, lo, hi):
        """Fold x back into [lo, hi] by repeated mirror reflection (a triangle wave)."""
        w = hi - lo
        y = np.mod(x - lo, 2.0 * w)
        return lo + np.where(y > w, 2.0 * w - y, y)

    _rng = np.random.default_rng(0)
    _p0  = np.clip(_reflect(_start[None, :] + EMCEE_BALL[None, :]
                            * _rng.standard_normal((EMCEE_WALKERS, NDIM)), _lo, _hi),
                   _lo + _eps, _hi - _eps)
    _chk = log_prob_vec(_p0)
    # a walker whose model will not solve is re-drawn from a ball that halves each round
    for _try in range(1, 21):
        _bad = ~np.isfinite(_chk)
        if not _bad.any():
            break
        _cand = np.clip(_reflect(_start[None, :] + (0.5 ** _try * EMCEE_BALL)[None, :]
                                 * _rng.standard_normal((int(_bad.sum()), NDIM)), _lo, _hi),
                        _lo + _eps, _hi - _eps)
        _p0[_bad] = _cand
        _chk[_bad] = log_prob_vec(_cand)
        print('   re-drew %d walker(s) at %.1f%% of the ball' % (_bad.sum(), 100 * 0.5 ** _try))
    assert np.isfinite(_chk).all(), 'walkers still failing after 20 re-draws -- inspect _p0[~np.isfinite(_chk)]'
    print('initial ensemble: %d walkers, logP %.2f .. %.2f' % (EMCEE_WALKERS, _chk.min(), _chk.max()))
    _backend.reset(EMCEE_WALKERS, NDIM)          # DELETES whatever was in the file

_sampler = emcee.EnsembleSampler(EMCEE_WALKERS, NDIM, log_prob_vec,
                                 vectorize=True, moves=_moves, backend=_backend)

# The bar is driven here rather than by progress=True so that it CONTINUES from the stored step
# count.  Status lines go through tqdm.write: a bare print inside the loop tears the bar apart.
t0 = time.time()
_todo = max(0, EMCEE_STEPS - _done)
if _todo == 0:
    print('chain already has %d steps >= EMCEE_STEPS = %d -- nothing to run' % (_done, EMCEE_STEPS))
_pbar = tqdm(initial=_done, total=EMCEE_STEPS)
for _ in (_sampler.sample(_p0, iterations=_todo, progress=False) if _todo else []):
    _pbar.update(1)
    if _sampler.iteration % 100:
        continue
    try:
        _tau = _sampler.get_autocorr_time(tol=0)
        _msg = 'tau %.0f-%.0f' % (np.nanmin(_tau), np.nanmax(_tau))
    except Exception:
        _msg = 'tau not yet estimable'
    tqdm.write('  step %5d | acceptance %.3f | logP max %.2f | %s'
               % (_sampler.iteration, np.mean(_sampler.acceptance_fraction),
                  np.max(_sampler.get_log_prob()[-1]), _msg))
_pbar.close()
print('done in %.0f s' % (time.time() - t0))

# ---- results ------------------------------------------------------------------------------
_tau = _sampler.get_autocorr_time(tol=0)
_tau_ok = bool(np.all(np.isfinite(_tau)))
BURNIN = int(3 * np.nanmax(_tau)) if _tau_ok else max(1, _sampler.iteration // 5)
BURNIN = min(BURNIN, _sampler.iteration - 1)
_thin  = max(1, int(0.5 * np.nanmin(_tau))) if _tau_ok else 1
chain   = _sampler.get_chain()                 # (steps, walkers, NDIM)
logprob = _sampler.get_log_prob()
flat_samples = _sampler.get_chain(discard=BURNIN, thin=_thin, flat=True)
print('\nautocorrelation time %.0f - %.0f steps | burn-in %d, thin %d -> %d samples | acceptance %.3f'
      % (np.nanmin(_tau), np.nanmax(_tau), BURNIN, _thin, len(flat_samples),
         np.mean(_sampler.acceptance_fraction)))
if _sampler.iteration < 50 * np.nanmax(_tau):
    print('WARNING: the chain is shorter than 50 tau, below which emcee calls tau unreliable.')

print('\n%30s %10s %10s %10s %10s %10s %7s' % ('', 'mean', 'std', '2.5%', '97.5%', 'truth', 'pull'))
for i, name in enumerate(param_names):
    s = flat_samples[:, i]
    truth = np.nan if i == J_LAM else float(np.asarray(theta_true)[i])
    pull = (s.mean() - truth) / s.std() if (s.std() > 0 and np.isfinite(truth)) else np.nan
    print('%30s %10.4f %10.4f %10.4f %10.4f %10s %7s'
          % (name, s.mean(), s.std(), np.percentile(s, 2.5), np.percentile(s, 97.5),
             '--' if not np.isfinite(truth) else '%10.4f' % truth,
             '--' if not np.isfinite(pull) else '%7.2f' % pull))
_q = np.percentile(10 ** flat_samples[:, param_names.index('log_Omega')] * KPCGYR_TO_KMS, [16, 50, 84])
print('   Omega_bar = %.2f +%.2f / -%.2f km/s/kpc  (10**log_Omega is in 1/Gyr)'
      % (_q[1], _q[2] - _q[1], _q[1] - _q[0]))

pickle.dump({'chain': chain, 'logprob': logprob, 'flat_samples': flat_samples,
             'param_names': param_names, 'tau': _tau, 'burnin': BURNIN, 'thin': _thin,
             'acceptance': _sampler.acceptance_fraction}, open(OUTPUT_FILE, 'wb'))
print('saved -> %s  (and the live HDF5 backend %s)' % (OUTPUT_FILE, EMCEE_FILE))
''')

# ================================================================= write + guards
cells = []
for t, s in CELLS:
    lines = s.split('\n')
    c = {'cell_type': t, 'metadata': {}, 'source': [l + '\n' for l in lines[:-1]] + [lines[-1]]}
    if t == 'code':
        c['execution_count'] = None
        c['outputs'] = []
    cells.append(c)
out = {'cells': cells, 'metadata': nb.get('metadata', {}), 'nbformat': 4, 'nbformat_minor': 5}

flat = '\n'.join(s for t, s in CELLS if t == 'code')
for lit in re.findall(r'(?:theta|th|thA|thB|best|res)\[(\d+)\]', flat):
    assert int(lit) < 12, 'hard-coded parameter index %s >= NDIM' % lit
assert 'MiyamotoNagai' not in flat and 'Rs_disc' not in flat and 'Hs_disc' not in flat, \
    'a Miyamoto-Nagai piece survived'
assert 'get_phi_m(rho_expdisc, p_disc, CYL_NR, CYL_NZ, CYL_RMIN, CYL_RMAX,' in flat
assert ('CYL_NR, CYL_NZ = %s, %s' % (NR, NZ)) in flat and ('CYL_NINT = %s' % NINT) in flat, \
    'the CylSpline settings are not NGC 1512\'s'
assert RHO_1512 in flat, 'the disc density is not NGC 1512\'s'
assert 'get_acc(x, y, z, dphi)' in flat and 'evaluate_phi_axisymmetric(x, y, z, dphi)' in flat
assert "RUN_TAG = 'aic_exp'" in flat and "'aic_tied'" not in flat
assert flat.count('RUN_TAG') >= 7, 'every output file must carry RUN_TAG'
assert "'logRd_disk'" in flat and "'loghz_disk'" in flat and "'f_bar'" in flat
assert 'M_star = 10.0 ** log_mass_to_light_ratio * L_BOX / p_box' in flat
assert 'import blackjax' not in flat and 'run_mcmc' not in flat and 'N_GROUPS' not in flat, \
    'a BlackJAX piece survived'
assert 'emcee.EnsembleSampler' in flat and 'vectorize=True' in flat and 'HDFBackend' in flat
assert 'get_last_sample()' in flat and 'RESUME' in flat
assert 'chain   = _sampler.get_chain()' in flat and 'BURNIN = ' in flat, 'cell 18 needs chain and BURNIN'
assert 'build_model_acc_meff_fourier' in flat and 'dens, pot, acc, pot_axi, dens_axi,' in flat
assert 'anisotropy_b=ANISO_B' in flat and 'err_den_meff=ERR_DEN_MEFF' in flat
assert '_stray_cuda_plugins' in flat and 'FLIP_HANDEDNESS' in flat and 'parallel_minimize' in flat
assert "ADOPT  = 'gaussian'" in flat
json.dump(out, open(A.out, 'w'), indent=1)
print('CylSpline disc: get_phi_m(rho, p, %s)  [from %s]' % (', '.join(CYL), CYL_SRC))
print('wrote %s  (%d cells)' % (A.out, len(cells)))
