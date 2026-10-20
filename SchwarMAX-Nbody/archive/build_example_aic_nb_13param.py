#!/usr/bin/env python
"""Build the AIC-pipeline demonstration notebook from the old one.

Parent: `schwarmax_merge_branch/example/example_schwarmax.ipynb`
Output: `schwarmax_merge_branch/example/example_schwarmax_AIC.ipynb`

It keeps the parent's STRUCTURE (imports -> likelihood -> load -> call it -> optimise ->
calibrate -> MCMC) because that structure is what other users follow, and swaps the pipeline
underneath.  What is deliberately carried over, and what is deliberately replaced:

  CARRIED OVER
    * `density_func` / `potential_func` -- COPIED VERBATIM from the parent, asserted below
    * the 12 physical parameters, same order, same meaning
    * the TRACER PARTICLE POSITIONS: `load_data_bootstrap`'s own w0 and NOTHING else.  No
      nuclear reseeding -- that is an NGC 1512 choice, not part of the pipeline.
    * the sampler: BlackJAX adaptive RMH, vmapped in two batches, with checkpointing.
      NOT emcee -- an emcee move cannot vmap a JAX likelihood.

  REPLACED
    * `build_model` -> `build_model_acc_meff_fourier`
    * the bootstrap-marginalised log-mean-exp -> the generalized AIC of Thomas & Lipka (2022)
    * `calculate_delta_chi2` (the jackknife) -> sigma_dAIC from PAIRED noise realisations
    * the regular (R,z,phi) 3-D block -> the Fourier ring block
    * 12 parameters -> 13: log10(lambda) is sampled
    * the setup cell -> the NGC 1512 one, which pins `jax[cuda12]==0.7.2` as a MATCHED set

  AND ONE THING THE USER OF THIS NOTEBOOK MUST KNOW
    the mock has to be FLIPPED.  `get_w0_new` (old) seeds clockwise, `get_w0_new_improved`
    (new) seeds anticlockwise; the mock was built for the first.  Measured both ways, the two
    pipelines swap: chi2_total OLD 14940 / NEW 62599 on the mock as delivered, OLD 93522 /
    NEW 10187 once flipped.  Cell 6 does the flip and explains it.

Run:  python build_example_aic_nb.py
"""
import argparse, json, os, re

A = argparse.ArgumentParser()
A.add_argument('--parent', default='example/example_schwarmax.ipynb')
A.add_argument('--out', default='example/example_schwarmax_AIC.ipynb')
A.add_argument('--repo', default='schwarmax_merge_branch')
A.add_argument('--dest', default=None,
               help='folder to write --out into; default = --repo (the package checkout)')
A.add_argument('--drive', default='SchwarMAX-Nbody',
               help='the Google Drive folder name the notebook looks for FIRST')
A.add_argument('--ngc1512-nb', dest='n1512',
               default='NGC1512_parametric_v1/PHANGS_NGC1512_parametric_F3DSQ4_MCMC.ipynb',
               help='source of the jax 0.7.2 install cell and the AIC constants')
A = A.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
REPO = os.path.join(ROOT, A.repo)
nb = json.load(open(os.path.join(REPO, A.parent)))
src = lambda i: ''.join(nb['cells'][i]['source'])

# ---------------------------------------------------------------- lift from the parent
P3 = src(3)
for anchor in ('V4_A, V4_B, V4_L, V4_GAMMA = 0.5, 0.5, 0.1, 0.0',
               'def density_func(x, y, z, params):',
               'def potential_func(x, y, z, params_baryon, params_halo):'):
    assert anchor in P3, 'parent cell 3 lost its anchor: %r' % anchor
DENSITY_BLOCK = P3[P3.index('# ---- Fixed V4 bulge parameters ----'):
                   P3.index('# An example function that use SchwarMAX')].rstrip()
assert DENSITY_BLOCK.count('def ') == 2

# the BlackJAX sampler body, lifted from the parent so its logic is not re-derived
P12 = src(12)
for anchor in ('import blackjax', 'def build_sampler(proposal)', 'def run_mcmc(resume=True)',
               'blackjax.mcmc.random_walk.normal', 'OPTIMAL_SCALE = 2.38'):
    assert anchor in P12, 'parent cell 12 lost its anchor: %r' % anchor
_a = P12.index('# ── Helper: build sampler from proposal')
_b = P12.index("if __name__ == '__main__':")
SAMPLER_BODY = P12[_a:_b].rstrip()

# ONE correction to the lifted body.  Its posterior table has a column headed "truth" that
# actually prints `res[i]`, the MINIMISER result -- harmless mislabelling in the parent, but
# actively misleading in a notebook whose whole point is a mock with a known answer.  Point it
# at the real generating parameters, and show the pull.
_old = '''        f"{'2.5%':>10s} {'97.5%':>10s} {'truth':>10s}")
    print("-" * 82)
    for i, name in enumerate(param_names):
        s = flat_samples[:, i]
        truth = res[i]
        print(f"{name:>30s} {s.mean():10.4f} {s.std():10.4f} "
              f"{np.percentile(s, 2.5):10.4f} {np.percentile(s, 97.5):10.4f} "
              f"{truth:10.4f}")'''
_new = '''        f"{'2.5%':>10s} {'97.5%':>10s} {'truth':>10s} {'pull':>7s}")
    print("-" * 90)
    for i, name in enumerate(param_names):
        s = flat_samples[:, i]
        # the GENERATING parameters, not the minimiser result; lambda has no truth
        truth = np.nan if i == J_LAM else float(np.asarray(theta_true)[i])
        pull = (s.mean() - truth) / s.std() if (s.std() > 0 and np.isfinite(truth)) else np.nan
        t_s = '--' if not np.isfinite(truth) else '%10.4f' % truth
        p_s = '--' if not np.isfinite(pull) else '%7.2f' % pull
        print(f"{name:>30s} {s.mean():10.4f} {s.std():10.4f} "
              f"{np.percentile(s, 2.5):10.4f} {np.percentile(s, 97.5):10.4f} "
              f"{t_s:>10s} {p_s:>7s}")'''
assert _old in SAMPLER_BODY, 'the parent posterior table changed shape'
SAMPLER_BODY = SAMPLER_BODY.replace(_old, _new)

# SECOND correction.  The parent splits the 32 chains into exactly TWO vmapped halves of 16.
# That is sized for the OLD likelihood.  The AIC likelihood costs ~2.3 GB per evaluation, so 16
# at once asks for ~37 GB and dies on a T4 with
#     XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 37094461736
# (37094461736/16 = 2.318 GB per chain -- the same ~2.3 GB/eval the DGDP chains measured for the
# Mmax=4 Fourier potential).  Replace the hard-coded halves with the N_GROUPS chunking those
# chains use: peak memory scales as N_CHAINS/N_GROUPS, and the result is INDEPENDENT of
# N_GROUPS because each chain keeps its own key and state slice.
_old2 = """N_HALF = N_CHAINS // 2  # each vmap batch size

def _split_pytree(tree):
    \"\"\"Split a pytree of (N_CHAINS, ...) arrays into two halves.\"\"\"
    return (jax.tree.map(lambda x: x[:N_HALF], tree),
            jax.tree.map(lambda x: x[N_HALF:], tree))

def _merge_pytree(tree0, tree1):
    \"\"\"Merge two pytree halves back into (N_CHAINS, ...).\"\"\"
    return jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), tree0, tree1)

def batched_init(vmap_init, positions):
    \"\"\"Initialise states in two batches to avoid OOM.\"\"\"
    states_0 = vmap_init(positions[:N_HALF])
    states_1 = vmap_init(positions[N_HALF:])
    return _merge_pytree(states_0, states_1)

def batched_step(vmap_step, keys, states):
    \"\"\"Run one MCMC step in two batches to avoid OOM.\"\"\"
    states_0, states_1 = _split_pytree(states)
    keys_0, keys_1 = keys[:N_HALF], keys[N_HALF:]
    new_states_0, infos_0 = vmap_step(keys_0, states_0)
    new_states_1, infos_1 = vmap_step(keys_1, states_1)
    return _merge_pytree(new_states_0, new_states_1), _merge_pytree(infos_0, infos_1)"""

_new2 = """# Chains are stepped in N_GROUPS SEQUENTIAL groups; peak memory scales with N_CHAINS/N_GROUPS.
# The parent's two halves of 16 is sized for the OLD likelihood.  This one needs about 1.5x the
# memory per chain, so 16 at once fails on a 24 GB L4 with
#     XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 37094461736
# while 8 at once runs.  The samples are IDENTICAL for any N_GROUPS -- every chain keeps its own
# key and its own state slice, so the grouping is pure memory bookkeeping (verified bit-for-bit
# for N_GROUPS = 1, 2, 4, 8, 16).
#
# WHERE THE 1.5x COMES FROM -- it is NOT the AIC.  Counting every array >= 8 MB in the traced
# graph at 7500 orbits (backend-independent; an upper bound, since XLA reuses buffers):
#     build_model (OLD)                 59.3 GB
#     + accelerations, n_real=1        124.5 GB   +65.2 GB   <-- 73% of the growth
#     + n_realizations 1 -> 4          136.1 GB   +11.6 GB
#     + Fourier 3-D block              135.0 GB    -1.0 GB   (CHEAPER than the regular grid)
#     + m_eff second NNLS pass (NEW)   148.8 GB   +13.7 GB   <-- the AIC is only 15%
# The dominant step is the acceleration-based integrator, not the AIC: arrays of shape
# (n_orb, N_max) = (7500, 1000) go from 942 of them to 3388 (26.3 -> 94.7 GB), because the force
# is acceleration_func = -grad(potential_func) differentiated through NFW + Miyamoto-Nagai + T3
# bar + V4 bulge with their rotations and shifts.  Both paths use the same chunk size of 100.
# Since the dominant shape is (n_orb, N_max), memory is LINEAR in N_ORBIT: halve it to halve
# this.  N_REALIZATIONS would save 13% but it is a tested hyperparameter -- leave it.
N_GROUPS = 4                        # <<< TUNE ME.  MEASURED ON AN L4 (24 GB): 4 groups of 8
                                    # runs; 2 groups of 16 fails, asking 37094461736 B.
                                    # Raise to 8 or 16 on a 16 GB T4.
assert N_CHAINS % N_GROUPS == 0, 'N_GROUPS must divide N_CHAINS'
N_PER = N_CHAINS // N_GROUPS
print('stepping %d chains as %d groups of %d (~%.1f GB/group at 2.3 GB per evaluation)'
      % (N_CHAINS, N_GROUPS, N_PER, 2.3 * N_PER))

def _sl(g):
    return slice(g * N_PER, (g + 1) * N_PER)

def _cat(parts):
    \"\"\"Concatenate a list of pytree groups back into (N_CHAINS, ...).\"\"\"
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *parts)

def batched_init(vmap_init, positions):
    \"\"\"Initialise states in N_GROUPS groups to bound peak memory.\"\"\"
    return _cat([vmap_init(positions[_sl(g)]) for g in range(N_GROUPS)])

def batched_step(vmap_step, keys, states):
    \"\"\"Run one MCMC step in N_GROUPS groups to bound peak memory.\"\"\"
    outs = [vmap_step(keys[_sl(g)], jax.tree.map(lambda x: x[_sl(g)], states))
            for g in range(N_GROUPS)]
    return _cat([o[0] for o in outs]), _cat([o[1] for o in outs])"""

assert _old2 in SAMPLER_BODY, 'the parent two-half chunking changed shape'
SAMPLER_BODY = SAMPLER_BODY.replace(_old2, _new2)

# the parent prints "in 2 batches of {N_HALF}" at the fresh start
_old3 = 'print(f"Initialising {N_CHAINS} chains in 2 batches of {N_HALF}...")'
_new3 = 'print(f"Initialising {N_CHAINS} chains in {N_GROUPS} groups of {N_PER}...")'
assert _old3 in SAMPLER_BODY, 'the parent init print changed shape'
SAMPLER_BODY = SAMPLER_BODY.replace(_old3, _new3)
assert 'N_HALF' not in SAMPLER_BODY, 'a two-half split survived'
assert SAMPLER_BODY.count('N_GROUPS') >= 4

# the install cell, lifted from the NGC 1512 notebook
n15 = json.load(open(os.path.join(ROOT, A.n1512)))
S3 = ''.join(n15['cells'][3]['source'])
assert "'jax[cuda12]==%s' % TARGET" in S3, 'the NGC 1512 setup cell lost its jax pin'
INSTALL = S3[:S3.index("import os, sys\nBASE")].rstrip()

C = []

C.append(('markdown', """# SchwarMAX — the AIC pipeline, demonstrated on a mock N-body galaxy

The same mock, the same density and the same 12 physical parameters as
`example_schwarmax.ipynb`. The structure of this notebook is deliberately identical, so the two
can be read side by side; **only the pipeline underneath changes**.

| | `example_schwarmax.ipynb` | this notebook |
|---|---|---|
| objective | log-mean-exp over data bootstraps of $-\\tfrac12\\sum\\chi^2$ | $\\log L=-\\tfrac12(\\chi^2_{\\rm kin}+2m_{\\rm eff,kin}+\\chi^2_{\\rm den})/S^2$ |
| bootstraps | marginalised over | spent on $m_{\\rm eff}$; the fit itself is unperturbed |
| model builder | `build_model` | `build_model_acc_meff_fourier` |
| 3-D block | regular $(R,z,\\phi)$ cells | Fourier rings, even $m\\le6$ |
| the $\\chi^2$ scale | jackknife $\\Delta\\chi^2$ | **$\\sqrt{2N_{\\rm kin}}$, validated by $\\sigma_{\\rm AIC}$** (with $\\sigma_{\\Delta\\rm AIC}$ measured alongside) |
| parameters | 12 | 13 — $\\log_{10}\\lambda$ is sampled |
| sampler | BlackJAX adaptive RMH | **unchanged** |
| tracer positions | `load_data_bootstrap` | **unchanged** |

$m_{\\rm eff}$ is the effective number of free parameters the orbit solution actually uses
(Thomas & Lipka 2022, eq. 15): noise is added to the FIT, the fit is repeated at the same
$\\lambda$, and the response is measured. Judging orbit models by $\\chi^2$ alone rewards
flexibility, and the parameter it corrupts most is the inclination — which this mock knows the
truth of, so this is a test with an answer.

> **Read cell 6 before running.** The mock must be flipped in handedness for this pipeline.
> `get_w0_new` (used by `build_model`) seeds clockwise; `get_w0_new_improved` (used by every
> `build_model_acc*`) seeds anticlockwise. Measured both ways, the two pipelines swap places."""))

C.append(('markdown', '# Import functions'))
C.append(('code', INSTALL + '''

import os, sys
# This folder IS the package: model.py, utils.py ... sit at its top level, next to example/.
# Upload the whole folder to My Drive and the first candidate below is found automatically.
# Edit DRIVE_FOLDER if you renamed it; or put this notebook beside model.py and run it locally.
DRIVE_FOLDER = '/content/drive/MyDrive/__DRIVE__'
path = None
for _c in (DRIVE_FOLDER, '/content/drive/MyDrive/__DRIVE__',
           '/content/drive/MyDrive/SchwarMAX', os.getcwd(), os.path.dirname(os.getcwd())):
    _c = _c.rstrip('/') + '/'
    if os.path.isfile(_c + 'model.py') and os.path.isdir(_c + 'example'):
        path = _c
        break
assert path is not None, (
    'could not find the SchwarMAX folder.  Set DRIVE_FOLDER to the uploaded folder -- the one '
    'that contains model.py and example/.  Looked in: %s' % DRIVE_FOLDER)
print('SchwarMAX root:', path)
sys.path.insert(0, path)

import jax, jax.numpy as jnp
import numpy as np, pandas as pd, pickle, time, corner
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from functools import partial
import blackjax

from utils import logMenc_logc_to_logM_logRs, load_data_bootstrap
from constants import EPSILON, KPCGYR_TO_KMS
from model import build_model_acc_meff_fourier
from potentials import (NFW_potential, MiyamotoNagai_potential, MiyamotoNagai_density,
                        T3_density, T3_potential, V4_density, V4_potential)

# The Fourier 3-D block and m_eff are recent additions; fail loudly if the uploaded schwarmax
# predates them rather than halfway through a 12-hour chain.
import model as _m
_need = ('fourier3d', 'meff_fourier', 'sigma_scheme', 'pow_spacing')
_miss = [f for f in _need if f not in getattr(_m, 'SCHWARMAX_FEATURES', ())]
assert not _miss, ('the SchwarMAX on Drive is OLDER than this notebook: missing %s. '
                   'Re-upload the package to %s' % (_miss, path))
print(jax.devices(), '| x64 =', jax.config.jax_enable_x64)
'''))

C.append(('markdown', """# Use the Schwarzschild function in SchwarMAX to build model likelihood

`density_func` and `potential_func` are **copied verbatim** from `example_schwarmax.ipynb`.
The AIC builder additionally needs an explicit `acceleration_func` and the $m=0$ counterparts
used by the Jeans seeding; all three are derived from those two functions rather than written
independently, so the density being modelled is unchanged."""))

C.append(('code', DENSITY_BLOCK + '''


# ================================================================= the three NEW ingredients
# build_model computed its accelerations internally as -grad(potential_func).  build_model_acc*
# takes them explicitly, so supply exactly that.
@jax.jit
def acceleration_func(x, y, z, params_baryon, params_halo):
    def _pot(pos):
        return potential_func(pos[0], pos[1], pos[2], params_baryon, params_halo)
    return -jax.grad(_pot)(jnp.array([x, y, z]))


# SHAPE SAFETY.  The Fourier target quadrature evaluates the density on a 3-D broadcast grid,
# and the parent's density cannot take that: `potentials._shift` builds jnp.array([x-x0, ...])
# and needs three IDENTICALLY shaped arrays, while `_rotate` does `R @ vec` with R of shape
# (3,3), which matmul only reads correctly for vec of shape (3,) or (3,N).  Broadcasting AND
# flattening fixes both and changes no physics.
def _flat(f):
    @jax.jit
    def g(x, y, z, *a):
        x, y, z = jnp.broadcast_arrays(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z))
        return jnp.reshape(f(x.ravel(), y.ravel(), z.ravel(), *a), x.shape)
    return g


density_func_bc, potential_func_bc = _flat(density_func), _flat(potential_func)


# ================================================================= the m=0 callables
# The Jeans seeding needs the AXISYMMETRIC (m=0) part of the density and potential.
#
# TRAP.  get_w0_new_improved DIFFERENTIATES through these, so computing the m=0 part as an
# azimuthal quadrature here puts n_phi copies of the integrand in the reverse-mode tape:
# MEASURED, 400 orbits, 0.98 GB -> 5.60 GB and 5.1 s -> 17.5 s.  The bar's m=0 part is
# therefore TABULATED ONCE on an (R,|z|) grid, outside the tape, and looked up by linear
# interpolation.  MN and NFW are axisymmetric already and are used as they are.
#
# SECOND TRAP.  The V4 bulge is deliberately NOT tabulated.  It has a = b = 0.5 kpc against a
# grid step dR = 0.48 kpc, so the grid cannot resolve it -- the same mistake, on NGC 1512's
# NSD, drove the surface-density chi2 from 13.3 to 140.8.  It also needs no averaging: it
# varies only 2-6% in azimuth, against 106-479% for the T3 bar.
M0_NR, M0_NZ, M0_RMAX, M0_ZMAX = 64, 33, 30.0, 10.0
M0_NPHI = 16          # only the BAR is averaged, and its m=0 converges fast; 16 >> 2*mmax


def make_m0_callables(params_baryon, params_halo):
    """(density_func_axi, potential_func_axi): the m=0 part of density_func/potential_func.

    Safe to call inside a jitted function -- the returned closures capture tracers.
    """
    M_bar = 10.0 ** params_baryon['logM_bar']
    a_bar, b_bar, L_bar = params_baryon['a_bar'], params_baryon['b_bar'], params_baryon['L_bar']

    Rt = jnp.linspace(0.0, M0_RMAX, M0_NR)
    zt = jnp.linspace(0.0, M0_ZMAX, M0_NZ)
    at = jnp.linspace(0.0, 2 * jnp.pi, M0_NPHI, endpoint=False)
    RR, ZZ = jnp.meshgrid(Rt, zt, indexing='ij')
    rho_tab = jnp.mean(jax.vmap(lambda a: T3_density(
        RR * jnp.cos(a), RR * jnp.sin(a), ZZ, M_bar, a_bar, b_bar, L_bar, GAMMA_BAR))(at), 0)
    pot_tab = jnp.mean(jax.vmap(lambda a: T3_potential(
        RR * jnp.cos(a), RR * jnp.sin(a), ZZ, M_bar, a_bar, b_bar, L_bar, GAMMA_BAR))(at), 0)
    dR, dz = M0_RMAX / (M0_NR - 1), M0_ZMAX / (M0_NZ - 1)

    def look(T, R, z):
        fR = jnp.clip(jnp.asarray(R) / dR, 0.0, M0_NR - 1.0)
        fz = jnp.clip(jnp.abs(jnp.asarray(z)) / dz, 0.0, M0_NZ - 1.0)
        return jax.scipy.ndimage.map_coordinates(
            T, [fR.ravel(), fz.ravel()], order=1, mode='nearest').reshape(jnp.shape(fR))

    MNK = ('logM_disc', 'Rs_disc', 'Hs_disc', 'x_origin', 'y_origin', 'z_origin',
           'dirx', 'diry', 'dirz')

    def density_func_axi(x, y, z, p):
        R = jnp.sqrt(x ** 2 + y ** 2)
        return (MiyamotoNagai_density(x, y, z, {k: p[k] for k in MNK})
                + look(rho_tab, R, z)
                + V4_density(x, y, z, 10.0 ** p['logM_bar'], V4_A, V4_B, V4_L, V4_GAMMA))

    def potential_func_axi(x, y, z, pb, ph):
        R = jnp.sqrt(x ** 2 + y ** 2)
        return (NFW_potential(x, y, z, ph)
                + MiyamotoNagai_potential(x, y, z, {k: pb[k] for k in MNK})
                + look(pot_tab, R, z)
                + V4_potential(x, y, z, 10.0 ** pb['logM_bar'], V4_A, V4_B, V4_L, V4_GAMMA))

    return density_func_axi, potential_func_axi
'''))

C.append(('markdown', """# Load data

The tracer particle POSITIONS come from `load_data_bootstrap` and nothing else — exactly as in
`example_schwarmax.ipynb`. They are drawn once from $p(R)\\propto R\\,e^{-R/3\\,{\\rm kpc}}$
(truncated at 12 kpc) and $p(z)\\propto e^{-z/1.2\\,{\\rm kpc}}$, $z\\ge0$, and are **fixed for the
whole MCMC** — the NNLS weights are only comparable between samples if the library does not
move. The velocities are then set by the Jeans solver at each seed's own $(R,z)$.

## The handedness flip — read this

`build_model` seeds the library **clockwise** (`getCartesianFromCylindrical_clockwise`,
$\\langle v_\\phi\\rangle=-125$ km/s here, 14% of orbits with $L_z>0$). Every `build_model_acc*`
seeds **anticlockwise** ($+128$ km/s, 87% with $L_z>0$) — deliberately, it is the convention of
the validated NGC 4371 runs, paired with $+\\Omega_{\\rm bar}$ so the bar co-rotates.

This mock was generated for the first convention: it has ${\\rm corr}(X,v_0)=+0.93$, negative
$v_0$ on the left and positive on the right. Run the new pipeline on it unflipped and the
library counter-rotates against the data. Measured, same $\\theta$, 7500 orbits:

| mock | old pipeline | this pipeline |
|---|---|---|
| as delivered | **14 940** | 62 599 |
| flipped | 93 522 | **10 187** |

So the flip below is not optional **for this mock**. It is the substitution
$v_{\\rm los}\\to-v_{\\rm los}$, which negates every quantity that is **odd** in $v$ and nothing
else: $v_0$, $V$, $h_1$, $h_3$. Even in $v$, and therefore untouched: $s$, $\\sigma$, $h_2$,
$h_4$. Untouched for other reasons: the uncertainties (magnitudes), the stored standard normals
(symmetric), and the grid geometry and tracer positions. The cell classifies **all 47 keys** the
data dict carries and asserts the classification is complete, so nothing is left behind and a
future key cannot slip through unnoticed.

(In this mock $h_1=h_2=0$ identically — pPXF defines the Gauss-Hermite moments about $v_0$ — so
all the rotation information lives in $v_0$.)

**`FLIP_HANDEDNESS` is a manual switch and there is no automatic check.** Deciding the handedness
from fit quality is not stable enough to automate: $\\rm corr(V_{\\rm model},V_{\\rm data})$ stays
**positive either way** (measured $+0.996$ right, $+0.893$ wrong), because the NNLS can build a
rotation field out of a counter-rotating library by up-weighting the minority of co-rotating
orbits. Set it from how your data were constructed, not from the output."""))

C.append(('code', """filename = '/example/mock_Nbody_galaxy_beta65_gamma45_D50.pkl'

N_ORBIT = 7_500
dict_data = load_data_bootstrap(path, filename, n_samples=N_ORBIT)
NUM_VBIN = int(dict_data['total_bins'])

# ---------------------------------------------------------------- the handedness flip
# Set FLIP_HANDEDNESS = False only if your data already rotates anticlockwise on the sky.
# A quick way to tell: project a co-rotating test disc through your own geometry and compare
# the SIGN of v_los against the data.
FLIP_HANDEDNESS = True

# Every key the dict carries is classified, so the flip is complete by construction rather than
# by a hand-kept list: v_los -> -v_los negates everything ODD in v and nothing else.  The assert
# fails if a future load_data_bootstrap adds a key this cell has not been told about, instead of
# silently leaving it unflipped.
_ODD   = ('v0', 'V_data', 'h1_data', 'h3_data')            # <- negated
_EVEN  = ('s', 'sigma_data', 'h2_data', 'h4_data')         # even in v: unchanged
_MAG   = tuple(k for k in dict_data if k.endswith('_err'))              # uncertainties: positive
_NOISE = tuple(k for k in dict_data if k.endswith('_standard_normal'))  # N(0,1): symmetric
_GEOM  = ('XY_density_data', 'num_per_bin', 'bin_mapping', 'total_bins',
          'R_grid', 'z_grid', 'phi_grid', 'R_minmax', 'z_minmax', 'phi_minmax',
          'Rzphi_n_tot', 'Rzphi_n_grid', 'dR', 'dz', 'dphi', 'sample_for_integration',
          'X_regular_grid', 'Y_regular_grid', 'dX', 'dY', 'sample_for_integration_XY',
          'X_minmax', 'Y_minmax', 'nX_nY', 'w0')          # geometry, counts, tracer POSITIONS
_unclassified = set(dict_data) - set(_ODD + _EVEN + _MAG + _NOISE + _GEOM)
assert not _unclassified, ('cell 6 does not know whether these are odd or even in v: %s'
                           % sorted(_unclassified))

if FLIP_HANDEDNESS:
    for _k in _ODD:
        dict_data[_k] = -dict_data[_k]
    print('handedness FLIPPED: %s negated' % ', '.join(_ODD))
    print('                    %d other keys unchanged (%d even in v, %d uncertainties, '
          '%d noise draws, %d geometry/positions)'
          % (len(_EVEN) + len(_MAG) + len(_NOISE) + len(_GEOM),
             len(_EVEN), len(_MAG), len(_NOISE), len(_GEOM)))
else:
    print('handedness NOT flipped (FLIP_HANDEDNESS = False)')

_R = np.hypot(np.asarray(dict_data['w0'])[:, 0], np.asarray(dict_data['w0'])[:, 1])
print('mock: %d Voronoi bins | %d orbits, <R> = %.2f kpc, median %.2f, R<2 kpc frac %.3f'
      % (NUM_VBIN, N_ORBIT, _R.mean(), np.median(_R), (_R < 2).mean()))
"""))

C.append(('markdown', """# The AIC likelihood

Every constant that is a CHOICE is set here and marked, so the notebook states its own
configuration instead of hiding it inside a call."""))

C.append(('code', '''# ================================================================= CONFIGURATION
# --- the 3-D Fourier block --------------------------------------------------------------
F3D_MMAX   = 6            # even harmonics only: the 4-fold symmetry makes odd m identically 0
F3D_R      = (0.2, 10.0, 15)   # R_lo, R_hi [kpc], n rings            <<< TUNE ME
F3D_Z      = (0.05, 5.0, 6)    # |z|_lo, |z|_hi [kpc], n floors       <<< TUNE ME
F3D_POW_R  = 0.5          # rings uniform in R**0.5.  NOT log: log gives the centre 53% of the
F3D_POW_Z  = 0.5          # constraint budget for 25% of the mass
F3D_LOG    = False
F3D_CLOSE_INNER = True    # bin 0 runs 0 -> e_1 in BOTH R and |z|, so no unconstrained midplane
F3D_NPHI_T = 32           # phi quadrature for the target; MUST exceed 2*mmax (Nyquist)
F3D_NSUB_T = 8            # n_sub x n_sub midpoint rule per (R,z) cell; converges as n_sub^-2
F3D_FLOOR  = 0.06
F3D_SCHEME = 'a0'         # sigma must NOT depend on the mode amplitudes -- see below
F3D_NMODE  = 1 + 2 * (F3D_MMAX // 2)
RZ_TOT     = F3D_R[2] * F3D_Z[2] * F3D_NMODE

# --- the density errors -----------------------------------------------------------------
# TWO different numbers, deliberately.  ERR_DEN_NNLS is what the NNLS SOLVES at -- it is an
# up-weighting of the density block, not a claim about the error.  ERR_DEN_LOGL is what chi2 is
# SCORED at.  Mixing them silently rescales chi2_logSig by (5/1)^2.
ERR_DEN_NNLS = 0.01       # READ-ONLY here: hard-coded at utils.py:619 as 0.01*surface_density
ERR_DEN_LOGL = 0.01       # <<< TUNE ME.  The mock carries 1%; NGC 1512 production uses 5%.
# And a THIRD number, for m_eff only.  err_den_meff = 0.0 makes the surface density a HARD
# CONSTRAINT in the flexibility count: its mock is the fit itself with no noise added, so the
# channel is charged no m_eff -- while still being scored in chi2_den.  This is Lipka & Thomas
# (2021, eq. 3)'s structure, and it is the configuration MEASURED to be best conditioned
# (NGC 1512 S/N 230: rho 0.872 -> 0.957, sigma_dAIC 43.8 -> 23.7, sigma_AIC exactly on its
# Gaussian floor).  Do not set it to ERR_DEN_LOGL without re-running that calibration.
ERR_DEN_MEFF = 0.0
# The 3-D errors are sigma(A_0) = SIG_FRAC_A0 * A_0 for the ring mass and
# sigma(c_m) = sigma(s_m) = SIG_FRAC_An * A_0 for the harmonics -- both tied to A_0, the RING
# MASS, NOT to |A_m|.  A tolerance proportional to |A_m| shrinks wherever the model is more
# axisymmetric, so the likelihood would systematically reward a stronger bar.
SIG_FRAC_A0 = 0.02        # <<< TUNE ME
SIG_FRAC_An = 0.04        # <<< TUNE ME

# --- the orbit library ------------------------------------------------------------------
N_REALIZATIONS = 4        # jitter realisations averaged into the design matrix
ANISO_B        = 1.0      # sigma_R^2 = ANISO_B * sigma_z^2.  1.0 = isotropic, matching the
                          # old pipeline's get_w0_new.  NGC 1512 production intends 1.5.
N_BOOT_MEFF    = 9        # m_eff bootstraps.  ~10 is enough: only dm_eff/dlambda must be good.
# (jeans_z_int_max is NOT a knob on this path: build_model_acc_meff_fourier asserts it is 5.0
#  and does not forward it, because get_w0_new_improved declares z_int_max static.)

# --- the temperature --------------------------------------------------------------------
# S2 divides the WHOLE bracket.  sqrt(2*N_kin) is the single-model Gaussian floor and is the
# PROVISIONAL value; the calibration cell below measures sigma_dAIC and may adopt that instead.
N_BIN_FIT = NUM_VBIN              # this mock has no inner mask
N_KIN     = 4 * N_BIN_FIT
S2        = float(np.sqrt(2.0 * N_KIN))
print('fitted bins %d -> N_kin %d -> S2 = sqrt(2 N_kin) = %.2f  (provisional)'
      % (N_BIN_FIT, N_KIN, S2))
print('3-D block: %d rings x %d modes = %d constraints' % (F3D_R[2]*F3D_Z[2], F3D_NMODE, RZ_TOT))

# ================================================================= the 13 parameters
param_names = ['logM_10kpc', 'logM_disk', 'logM_bar', 'logC_halo', 'logRs_disk',
               'logHs_disk', 'logL_bar', 'alpha', 'beta', 'gamma',
               'log_mass_to_light_ratio', 'log_Omega', 'log10_lambda']
NDIM  = len(param_names)
J_LAM = param_names.index('log10_lambda')      # NEVER hard-code this index


@partial(jax.jit, static_argnames=('num_Vbin', 'Rzphi_n_tot', 'n_boot'))
def run_model(theta, dict_data, num_Vbin, Rzphi_n_tot, n_boot=N_BOOT_MEFF):
    """The forward model.  Same physics as example_schwarmax.ipynb; different builder."""
    logM_enc, log_c = theta[0], theta[3]
    logM_halo, logRs_halo = logMenc_logc_to_logM_logRs(logM_enc, log_c, r_enc=10.0,
                                                       Delta=200., rho_crit=277.54)
    logM_disc, logM_bar = theta[1], theta[2]
    logRs_disk, logHs_disk, logL_bar = theta[4], theta[5], theta[6]
    alpha = theta[7] * 180 / jnp.pi
    beta  = theta[8] * 180 / jnp.pi
    gamma = theta[9] * 180 / jnp.pi
    log_mass_to_light_ratio, log_Omega_bar = theta[10], theta[11]

    L_bar   = 10.0 ** logL_bar
    a_bar   = L_bar / 5.0
    Hs_disc = 10.0 ** logHs_disk
    b_bar   = Hs_disc

    params_halo_pot = {'logM': logM_halo, 'Rs': 10 ** logRs_halo, 'a': 1.0, 'b': 1.0, 'c': 1.0,
                       'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
                       'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}
    params_baryon_rho = {'logM_disc': logM_disc, 'Rs_disc': 10 ** logRs_disk,
                         'Hs_disc': Hs_disc, 'logM_bar': logM_bar, 'L_bar': L_bar,
                         'a_bar': a_bar, 'b_bar': b_bar,
                         'mass_to_light_ratio': 10 ** log_mass_to_light_ratio,
                         'Omega_bar': 10 ** log_Omega_bar,
                         'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
                         'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
                         'alpha': alpha, 'beta': beta, 'gamma': gamma}

    # m=0 density and potential for the Jeans seeding (cell 4; read its two traps)
    density_func_axi, potential_func_axi = make_m0_callables(params_baryon_rho, params_halo_pot)

    X_minmax, Y_minmax = dict_data['X_minmax'], dict_data['Y_minmax']
    nX, nY = dict_data['nX_nY']

    # anisotropy_b is passed as a KEYWORD.  Patching jeans.get_w0_new_improved does NOT work:
    # build_model_acc_fourier forwards `anisotropy_b=anisotropy_b` explicitly with a default of
    # 1.0, and an explicit keyword overrides a functools.partial binding.
    # lambda stays TRACED (it arrives inside theta); as a static argname it would recompile on
    # EVERY MCMC step.
    return build_model_acc_meff_fourier(
        density_func_bc, potential_func_bc, acceleration_func,
        potential_func_axi, density_func_axi,
        params_halo_pot, params_baryon_rho, dict_data, num_Vbin,
        anisotropy_b=ANISO_B, n_realizations=N_REALIZATIONS,
        sigma_frac_3d=SIG_FRAC_An, sigma_frac_m0=SIG_FRAC_A0, sigma_scheme=F3D_SCHEME,
        floor_frac_3d=F3D_FLOOR, v_sys=0.0,
        mmax=F3D_MMAX, nR_f=F3D_R[2], nZ_f=F3D_Z[2], log_R=F3D_LOG, log_z=F3D_LOG,
        R_lo_f=F3D_R[0], R_hi_f=F3D_R[1], z_lo_f=F3D_Z[0], z_hi_f=F3D_Z[1],
        n_phi_t=F3D_NPHI_T, n_sub_t=F3D_NSUB_T,
        pow_R=F3D_POW_R, pow_z=F3D_POW_Z, close_inner=F3D_CLOSE_INNER,
        Rzphi_n_tot=Rzphi_n_tot, Rzphi_n_grid=dict_data['Rzphi_n_grid'],
        Rzphi_lim_grid=jnp.array([dict_data['R_minmax'], dict_data['z_minmax'],
                                  dict_data['phi_minmax']]),
        xy_lim_grid=jnp.array([X_minmax, Y_minmax]), xy_n_grid=jnp.array([nX, nY]),
        nnls_maxiter=200, regularization=10.0 ** theta[J_LAM],
        err_den_meff=ERR_DEN_MEFF, n_boot_meff=n_boot)


@partial(jax.jit, static_argnames=('num_Vbin', 'Rzphi_n_tot'))
def model_likelihood(theta, dict_data, num_Vbin, Rzphi_n_tot, sigma_amplify=1.0):
    """logL = -0.5 * (chi2_kin + 2*m_eff_kin + chi2_den) / sigma_amplify**2.

    NO log-mean-exp: this is a SINGLE unperturbed fit.  The bootstraps are spent on m_eff.
    The SURFACE DENSITY is in chi2 but is NOT charged m_eff: measured on NGC 1512 it carries
    23% of the mean chi2 but 88% of its variance, and removing it from the AIC puts sigma_AIC
    on its Gaussian floor.  There is no gas term -- the mock has no rotation curve.
    """
    res = run_model(theta, dict_data, num_Vbin, Rzphi_n_tot)

    O_den = dict_data['XY_density_data']
    sig_l = (ERR_DEN_LOGL * O_den) / (O_den * jnp.log(10) + EPSILON)
    mden  = res['surface_density'][0]
    chi2_den = jnp.nansum(((jnp.log10(mden + EPSILON) - jnp.log10(O_den + EPSILON)) / sig_l) ** 2)

    chi2_kin = 0.0
    for k in (1, 2, 3, 4):
        chi2_kin = chi2_kin + jnp.nansum(
            ((res['h%d' % k][0] - dict_data['h%d_data' % k])
             / (dict_data['h%d_data_err' % k] + EPSILON)) ** 2)

    aic = chi2_kin + 2.0 * res['m_eff_kin']
    return -0.5 * (aic + chi2_den) / sigma_amplify ** 2


def chi2_report(theta):
    """Per-bin chi2 and the AIC terms, as plain numbers."""
    res = run_model(jnp.asarray(theta), dict_data, NUM_VBIN, RZ_TOT)
    O = np.asarray(dict_data['XY_density_data'])
    d0 = np.asarray(res['surface_density'])[0]
    sl = (ERR_DEN_LOGL * O) / (O * np.log(10) + 1e-30)
    out = {'logSig': float(np.nanmean(((np.log10(d0 + 1e-30) - np.log10(O + 1e-30)) / sl) ** 2))}
    kin = 0.0
    for k in (1, 2, 3, 4):
        r = ((np.asarray(res['h%d' % k])[0] - np.asarray(dict_data['h%d_data' % k]))
             / np.asarray(dict_data['h%d_data_err' % k])) ** 2
        out['h%d' % k] = float(np.nanmean(r)); kin += float(np.nansum(r))
    out['SUM']        = sum(out[k] for k in ('logSig', 'h1', 'h2', 'h3', 'h4'))
    out['m_eff_kin']  = float(res['m_eff_kin'])
    out['AIC']        = kin + 2 * out['m_eff_kin']
    out['flux_d/m']   = float(O.sum() / d0.sum())
    return out
'''))

C.append(('markdown', """# Example code for calling the likelihood function

The mock's generating parameters are known, so these scans have a right answer: each should
peak at the truth. (Note the `ground_truth` list below is the TRUTH column of the posterior
summary, not the rough starting point used in the old notebook's cell 7.)"""))

C.append(('code', """ground_truth = [
  10.6750,                     # logM_10kpc
  10.7179,                     # logM_disk
  10.1961,                     # logM_bar
   0.9876,                     # logC_halo
   0.9713,                     # logRs_disk
  -0.2564,                     # logHs_disk
   0.7457,                     # logL_bar
   0.4649,                     # alpha
   1.1576,                     # beta   (inclination)
   1.0316,                     # gamma  (bar azimuth)
  -0.1660,                     # log_mass_to_light_ratio
   1.3822,                     # log_Omega
]
theta_true = jnp.asarray(ground_truth + [0.0])      # log10(lambda) = 0 -> lambda = 1

t0 = time.time()
rep = chi2_report(theta_true)
print('chi2_report at GROUND TRUTH  (%.0f s)' % (time.time() - t0))
for k, v in rep.items():
    print('   %-12s %10.4f' % (k, v))
print('\\nlogL at truth (S2 = %.2f): %.4f'
      % (S2, float(model_likelihood(theta_true, dict_data, NUM_VBIN, RZ_TOT, jnp.sqrt(S2)))))

"""))

C.append(('code', '''# The old notebook scans M/L and Omega; same two here, plus the inclination, which is the
# parameter model-flexibility bias corrupts most and the reason for using AIC at all.
fig, ax = plt.subplots(1, 3, figsize=(16, 3.4), constrained_layout=True)
for a, (j, lo, hi, lab) in zip(ax, [(10, -0.6, 0.3, 'log M/L'),
                                    (11, 1.0, 1.8, 'log Omega'),
                                    (8, 0.9, 1.4, 'beta [rad]')]):
    xs = np.linspace(lo, hi, 9)
    ll = []
    for x in tqdm(xs, desc=lab, leave=False):
        th = np.asarray(theta_true).copy(); th[j] = x
        ll.append(float(model_likelihood(jnp.asarray(th), dict_data, NUM_VBIN, RZ_TOT,
                                         jnp.sqrt(S2))))
    ll = np.asarray(ll)
    a.plot(xs, ll - np.nanmax(ll), 'o-')
    a.axvline(float(np.asarray(theta_true)[j]), color='crimson', ls='--', label='truth')
    a.set_xlabel(lab); a.set_ylabel(r'$\\log L - \\max$'); a.legend(fontsize=8)
    a.set_title('peak at %.3f (truth %.3f)' % (xs[int(np.nanargmax(ll))],
                                               float(np.asarray(theta_true)[j])), fontsize=9)
plt.show()
'''))

C.append(('markdown', """# Optimisation pipeline

`N_PART` particles are hill-climbed together with a cooling step, exactly as in the tested
NGC 1512 chains. Every iteration evaluates all of them in **one vmapped call**, so the progress
bar below counts **iterations** (`N_ITER` of them), not starting points, and each tick is
`N_PART` likelihood evaluations.

The first batched call compiles the whole likelihood and is timed and printed *before* the bar
appears, so the initial wait is accounted for rather than looking like a hang. After that a tick
should arrive every `N_PART` x (seconds per evaluation) — if the bar sits at 0/`N_ITER` for much
longer than the compile line reported, something really is wrong."""))

C.append(('code', '''# Multi-start optimisation.  The whole ENSEMBLE is saved, not just the best point: the dAIC
# calibration below needs its spread to place the second model.
#
# This is `parallel_minimize` from the tested NGC 1512 chains: a cooled random-walk hill climb
# in which every iteration evaluates all N_PART particles in ONE vmapped call.  It replaces a
# serial scipy Nelder-Mead for two measured reasons.
#
#   PROGRESS.  Nelder-Mead is serial, and `maxiter = 40*NDIM` rails at 520 iterations for
#   672-705 function calls per start (measured, scipy 1.17; `maxfev` is left at infinity when
#   only `maxiter` is set, so nothing else bounds it).  A bar over 8 starts therefore advances
#   once per ~690 likelihood calls.  At the 2.0 s/eval this model costs on a T4 -- 63.84 s/step
#   for 32 chains, from the parent notebook\'s own saved output -- that is 23 min per tick with
#   nothing to watch, and 5498 evaluations for the cell.  Here the bar ticks once per iteration,
#   N_PART evaluations apart, and the cell costs N_PART*(N_ITER+1) = 968.
#
#   CONSISTENCY.  vmapped and serial evaluation are NOT the same number in float32.  Both are
#   bit-reproducible and batch-content-independent, but the NNLS lands on a different point of
#   its degenerate ridge: at the truth, 5 of 159 active orbits differ, sum(w) by 0.29%, and
#   chi2_kin +8.9% against chi2_den -10.0% (they partly cancel, to ~15 in logL).  The MCMC below
#   vmaps 16 chains at a time, so a serially-optimised start is the optimum of a surface the
#   sampler never evaluates.  Optimise on the surface you sample.
RUN_MINIMISER = True
N_PART     = 8      # particles, all evaluated in one batched call
N_ITER     = 120    # iterations; the tested NGC 1512 value.  Step cools as 0.97**it -> x0.026
VMAP_BATCH = 8      # <<< TUNE ME: evaluations per batched call -- the same budget as the
                    # sampler's N_PER below, which is measured to run on a 24 GB L4.  Lower
                    # it on a smaller card.  The bar still ticks once per ITERATION (N_PART
                    # evaluations) whatever this is set to.
MINIMISER_RESULT = os.path.join(path, 'example/minimisation_result_aic.npy')
MINIMISER_POS    = os.path.join(path, 'example/minimiser_pos_aic.npy')
MINIMISER_VAL    = os.path.join(path, 'example/minimiser_val_aic.npy')

# Per-parameter step.  These are the MCMC\'s tested `rmh_sigma_init` (see the sampler cell), so
# the hill climb and the sampler move on the same scales.
OPT_SPREAD = np.array([0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02,
                       0.02, 0.02, 0.07, 0.02, 0.02, 0.20])
# Search box: the shape of the MCMC prior, centred on the generating parameters rather than on
# the minimiser result (which does not exist yet).
_t = np.asarray(theta_true, float)
OPT_LO = np.array([_t[0] - 3, _t[1] - 3, _t[2] - 3, _t[3] - 1, _t[4] - 1, _t[5] - 1,
                   max(0.0, _t[6] - 1), 0., 0., 0., -2., 0., -1.])
OPT_HI = np.array([_t[0] + 3, _t[1] + 3, _t[2] + 3, _t[3] + 1, _t[4] + 1, _t[5] + 1,
                   min(1.0, _t[6] + 1), np.pi, np.pi / 2, np.pi, 2., 2., 4.])
assert OPT_SPREAD.size == NDIM and OPT_LO.size == NDIM and OPT_HI.size == NDIM

_lp      = lambda th: model_likelihood(th, dict_data, NUM_VBIN, RZ_TOT, jnp.sqrt(S2))
_lp_vmap = jax.vmap(_lp)

def parallel_minimize(x0s, spread, n_iter, lo, hi, vmap_batch=VMAP_BATCH, seed=0):
    """Cooled random-walk hill climb, batched over particles (tested NGC 1512 structure)."""
    lo, hi = jnp.asarray(lo), jnp.asarray(hi)
    def ev(p):
        return jnp.concatenate([_lp_vmap(p[i:i + vmap_batch])
                                for i in range(0, p.shape[0], vmap_batch)])
    key = jax.random.PRNGKey(seed)
    pos = jnp.clip(jnp.asarray(x0s), lo, hi)
    t0  = time.time(); val = ev(pos); val.block_until_ready()
    print('   first batched call (JIT compile + %d evaluations): %.1f s  -- the bar below ticks'
          % (pos.shape[0], time.time() - t0))
    print('   once per iteration, %d evaluations apart; %d iterations in total.'
          % (pos.shape[0], n_iter))
    sp  = jnp.asarray(spread)
    bar = tqdm(range(n_iter), desc='minimiser', unit='iter')
    for it in bar:
        key, k1 = jax.random.split(key)
        cand   = jnp.clip(pos + jax.random.normal(k1, pos.shape) * sp * (0.97 ** it), lo, hi)
        cv     = ev(cand)
        better = cv > val
        pos    = jnp.where(better[:, None], cand, pos)
        val    = jnp.where(better, cv, val)
        # float() forces a device sync, so the bar and its ETA are honest
        bar.set_postfix(best='%.2f' % float(jnp.max(val)),
                        moved='%d/%d' % (int(better.sum()), pos.shape[0]))
    return np.asarray(pos), np.asarray(val)

if RUN_MINIMISER:
    rng   = np.random.default_rng(0)
    start = _t[None, :] + OPT_SPREAD[None, :] * rng.standard_normal((N_PART, NDIM))
    t0    = time.time()
    pos, val = parallel_minimize(start, OPT_SPREAD, N_ITER, OPT_LO, OPT_HI)
    print('   %d iterations x %d particles = %d evaluations in %.0f s (%.2f s/eval)'
          % (N_ITER, N_PART, N_PART * (N_ITER + 1), time.time() - t0,
             (time.time() - t0) / (N_PART * (N_ITER + 1))))
    np.save(MINIMISER_POS, pos); np.save(MINIMISER_VAL, val)
    np.save(MINIMISER_RESULT, pos[int(np.argmax(val))])
else:
    pos = np.load(MINIMISER_POS); val = np.load(MINIMISER_VAL)

best = pos[int(np.argmax(val))]
print('best logL %.4f  (ensemble spread %.4f over %d particles)'
      % (val.max(), float(np.std(val)), len(val)))
print('%-26s %11s %11s %11s' % ('parameter', 'truth', 'best fit', 'offset'))
for j, n in enumerate(param_names):
    t = float(np.asarray(theta_true)[j])
    print('%-26s %11.4f %11.4f %+11.4f' % (n, t, best[j], best[j] - t))
print('\\n   i_true = %.2f deg | i_fit = %.2f deg'
      % (np.degrees(float(np.asarray(theta_true)[8])), np.degrees(best[8])))
'''))

C.append(('markdown', """# The $\\chi^2$ scale

The old notebook set `sigma_amplify` from a **jackknife** $\\Delta\\chi^2$. This cell replaces it
with the two quantities Wang, Thomas & Lipka (2026) measure. They answer different questions and
only one of them is a temperature.

**1. Is the noise model right?** $\\sigma_{\\rm AIC}$, the scatter of a *single* model's AIC over
noise realisations, should sit near the Gaussian floor $\\sqrt{2\\chi^2}$ — Wang+2026 measure a
ratio of 1.13. If it does, the familiar $S^2=\\sqrt{2N_{\\rm kin}}$ is justified rather than
assumed.

**2. How big a $\\Delta$AIC is significant?** Not $\\sigma_{\\rm AIC}$: two models evaluated on the
*same* noise realisations share most of it. The paired scatter is
$$\\sigma_{\\Delta\\rm AIC}=\\sigma_{\\rm AIC}\\sqrt{2(1-\\rho)}\\,,$$
which **vanishes as the two models approach each other** ($\\rho\\to1$). So there is no single
$\\sigma_{\\Delta\\rm AIC}$: it is a property of a model *pair*, and Wang+2026 accordingly quote
one number for one pair of competing mass models, as a model-**selection** threshold.

The cell walks one fixed direction in parameter space and tabulates $\\rho$, $|\\Delta{\\rm AIC}|$
and $\\sigma_{\\Delta\\rm AIC}$, so the pair-dependence is visible rather than hidden.

> **Why $\\sigma_{\\Delta\\rm AIC}$ is NOT adopted as the temperature here.** It is tempting to
> look for the displacement where $|\\Delta{\\rm AIC}|=\\sigma_{\\Delta\\rm AIC}$ and call that the
> $1\\sigma$ contour. **Measured on this mock (1000 orbits, 16 realisations, displacements
> 0.125–4 spreads), it does not work:** $\\sigma_{\\Delta\\rm AIC}$ rises smoothly, 11 → 28, but
> $|\\Delta{\\rm AIC}|$ does not grow at all — 25.5, 26.4, 41.7, 23.0, 10.8, 75.0, a fitted slope
> of $s^{0.09}$ where a quadratic would give $s^{2}$. At this orbit count the AIC landscape is
> rough on the scale being probed, so any crossing found by interpolation is noise. **Check
> your own table before trusting a measured $S^2$**: $|\\Delta{\\rm AIC}|$ must grow
> monotonically first. `ADOPT` defaults to the Gaussian value for this reason."""))

C.append(('code', """N_CAL  = 24                            # noise realisations per model
SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)    # displacements, in units of the minimiser spread
ADOPT  = 'gaussian'                    # 'gaussian' = sqrt(2 N_kin); 'measured' = sigma_dAIC,
                                       # which the markdown above says to verify FIRST

# ONE fixed random direction, then walked outwards.  A fresh direction per scale would confound
# the displacement with the direction.
spread = np.std(pos, axis=0) + 1e-6
unit   = np.random.default_rng(20260922).standard_normal(NDIM) * spread
unit[J_LAM] = 0.0      # lambda is IDENTICAL in every pair: it is a property of the estimator,
                       # not of the model, so comparing two lambdas is a different experiment.
thA = best.copy()

# The noise realisations are the ones ALREADY in dict_data (rows 1..N_CAL; row 0 is zeroed by
# load_data_bootstrap and is skipped by the m_eff path).  Reusing the stored rows rather than
# drawing fresh ones is what makes the models PAIRED -- the common-mode noise then cancels in
# aA - aB, which is the entire reason sigma_dAIC is smaller than sigma_AIC.
_NROW = int(np.asarray(dict_data['XY_standard_normal']).shape[0]) - 1
assert N_CAL <= _NROW, 'only %d stored noise rows; raise N_BOOTSTRAP in load_data_bootstrap' % _NROW


def _realisations(th):
    \"\"\"(chi2 per realisation, m_eff) -- kept apart so the Gaussian floor uses chi2 alone.\"\"\"
    r = run_model(jnp.asarray(th), dict_data, NUM_VBIN, RZ_TOT, n_boot=N_CAL)
    return np.asarray(r['chi2_boot_kin'], float), float(r['m_eff_kin'])


t0 = time.time()
c2A, meA = _realisations(thA)
aA      = c2A + 2.0 * meA
SIG_AIC = float(aA.std(ddof=1))
FLOOR   = float(np.sqrt(2.0 * c2A.mean()))      # on chi2, NOT on chi2 + 2 m_eff
print('QUESTION 1 -- is the noise model right?   (%d realisations)' % N_CAL)
print('   <chi2_kin> %.1f over %d points | m_eff_kin %.2f' % (c2A.mean(), N_KIN, meA))
print('   sigma_AIC (one model)        %9.2f' % SIG_AIC)
print('   Gaussian floor sqrt(2 chi2)  %9.2f' % FLOOR)
print('   RATIO                        %9.2f    [Wang+2026: 1.13]' % (SIG_AIC / FLOOR))
print('   sqrt(2 N_kin)                %9.2f    <- the provisional S2' % S2)

print('\\nQUESTION 2 -- how big a dAIC is significant?')
print('   %6s %10s %10s %11s %8s %10s' % ('scale', 'rho', '|dAIC|', 'sig_dAIC', 'ratio', 'predicted'))
rows = []
for sc in SCALES:
    thB = thA + sc * unit
    thB[J_LAM] = thA[J_LAM]
    c2B, meB = _realisations(thB)
    aB  = c2B + 2.0 * meB
    d   = abs(float(aA.mean() - aB.mean()))
    sd  = float((aA - aB).std(ddof=1))
    rho = float(np.corrcoef(aA, aB)[0, 1])
    rows.append((sc, rho, d, sd, d / max(sd, 1e-12)))
    print('   %6.2f %10.6f %10.2f %11.2f %8.3f %10.2f'
          % (rows[-1] + (SIG_AIC * np.sqrt(max(2 * (1 - rho), 0.0)),)))
rows = np.asarray(rows)
print('   the last column is sigma_AIC*sqrt(2(1-rho)) -- it should track column 4')
print('   (%.0f s for %d model evaluations)' % (time.time() - t0, 1 + len(SCALES)))

# ---- is |dAIC| even growing?  the markdown explains why this is the gate ---------------
_pl = lambda c: float(np.polyfit(np.log(rows[:, 0]), np.log(np.maximum(rows[:, c], 1e-12)), 1)[0])
p_d, p_s = _pl(2), _pl(3)
mono = bool(np.all(np.diff(rows[:, 2]) > 0))
print('\\n   |dAIC|   ~ s^%+.2f   (a quadratic landscape gives +2)' % p_d)
print('   sig_dAIC ~ s^%+.2f' % p_s)
print('   |dAIC| monotonically increasing: %s' % mono)

S2_MEASURED = np.nan
if rows[0, 1] > 0.9999:
    print('   -> rho = %.6f even at the smallest displacement: the two models are NOT distinct,'
          % rows[0, 1])
    print('      which is an unconverged minimiser rather than a real resolution.')
elif not mono or p_d < 1.0:
    print('   -> |dAIC| is NOT growing with displacement, so no crossing is meaningful.')
    print('      This is what we measured on this mock; see the markdown above.')
elif rows[:, 4].min() < 1.0 < rows[:, 4].max():
    _o = np.argsort(rows[:, 4])
    s_star = float(np.exp(np.interp(0.0, np.log(rows[_o, 4]), np.log(rows[_o, 0]))))
    S2_MEASURED = float(np.exp(np.interp(np.log(s_star), np.log(rows[:, 0]), np.log(rows[:, 3]))))
    print('   -> |dAIC| = sigma_dAIC at %.3f spreads  ->  sigma_dAIC = %.2f' % (s_star, S2_MEASURED))
    for j, n in enumerate(param_names):
        if abs(unit[j]) > 0:
            print('      1-sigma %-26s %+.4f' % (n, s_star * unit[j]))
else:
    print('   -> |dAIC|/sigma_dAIC never crosses 1 over %s; widen SCALES.' % (list(SCALES),))

print('\\nsqrt(2 N_kin) = %.2f   |   measured sigma_dAIC = %s'
      % (S2, 'n/a' if not np.isfinite(S2_MEASURED) else '%.2f' % S2_MEASURED))
if ADOPT == 'measured' and np.isfinite(S2_MEASURED):
    S2_ADOPTED = S2_MEASURED; print('ADOPTED: the measured sigma_dAIC (ADOPT = %r)' % ADOPT)
else:
    S2_ADOPTED = S2
    print('ADOPTED: sqrt(2 N_kin) (ADOPT = %r%s)'
          % (ADOPT, '' if ADOPT == 'gaussian' else '; no usable measured value'))
SIGMA_AMPLIFY = jnp.sqrt(S2_ADOPTED)
print('S2 in force for the chain: %.2f   (posterior widths scale as sqrt(S2))' % S2_ADOPTED)
"""))

C.append(('markdown', """# MCMC

The sampler is **unchanged** from `example_schwarmax.ipynb`: BlackJAX adaptive random-walk
Metropolis, 32 chains vmapped in two batches (vmap is several times faster than looping, and
two batches avoids OOM), covariance adapted from the running chain, with checkpointing. Only
the dimensionality (13) and the log-density (the AIC one, at the calibrated $S^2$) differ."""))

C.append(('code', '''"""
MCMC sampling using BlackJAX.  Other JAX/vmap-friendly samplers (numpyro, zeus) also work.
The first call is slow because of JIT compilation and gets much faster after warm-up.
"""
N_CHAINS = 32              # parallel chains (must fit in a GPU vmap)
N_STEPS = 1000             # total MCMC steps per chain
CHECKPOINT_EVERY = 5
BURNIN = 300
ADAPT_EVERY = 50
ADAPT_AFTER = 150

CHECKPOINT_FILE = os.path.join(path, 'example/mcmc_checkpoint_aic.pkl')
OUTPUT_FILE     = os.path.join(path, 'example/mcmc_results_aic.pkl')
OUTPUT_CSV      = os.path.join(path, 'example/mcmc_posterior_aic.csv')

res = np.load(MINIMISER_RESULT)

# ---- Prior bounds (13D) ---------------------------------------------------------------
BOUNDS_LO = jnp.array([res[0] - 3, res[1] - 3, res[2] - 3, res[3] - 1,
                       res[4] - 1, res[5] - 1, max(0, res[6] - 1),
                       0., 0., 0., -2., 0., -1.])
BOUNDS_HI = jnp.array([res[0] + 3, res[1] + 3, res[2] + 3, res[3] + 1,
                       res[4] + 1, res[5] + 1, min(1, res[6] + 1),
                       float(jnp.pi), float(jnp.pi / 2), float(jnp.pi), 2., 2., 4.])

def logdensity_fn(theta):
    in_bounds = jnp.all((theta >= BOUNDS_LO) & (theta <= BOUNDS_HI))
    log_vol   = jnp.sum(jnp.log(BOUNDS_HI - BOUNDS_LO))
    logprior  = jnp.where(in_bounds, -log_vol, -jnp.inf)
    ll = model_likelihood(theta, dict_data, NUM_VBIN, RZ_TOT, sigma_amplify=SIGMA_AMPLIFY)
    ll = jnp.where(jnp.isfinite(ll), ll, -1e30)
    return logprior + ll

print('initial parameters:')
for i in range(len(param_names)):
    print('   %-26s %.6f' % (param_names[i], res[i]))
print('begining logL', float(model_likelihood(jnp.asarray(res), dict_data, NUM_VBIN, RZ_TOT,
                                              SIGMA_AMPLIFY)))

OPTIMAL_SCALE = 2.38 / np.sqrt(NDIM)   # Roberts & Rosenthal 2001

initial_guess_sigma = jnp.array([0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02,
                                 0.02, 0.02, 0.10, 0.05, 0.10, 0.30])
rmh_sigma_init      = jnp.array([0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02,
                                 0.02, 0.02, 0.07, 0.02, 0.02, 0.20])

''' + SAMPLER_BODY + '''


chain, logprob = run_mcmc(resume=False)
'''))

C.append(('code', '''# Corner, with the TRUTH marked -- the point of using a mock.
flat = chain[BURNIN:].reshape(-1, NDIM)
truths = list(np.asarray(theta_true))
truths[J_LAM] = None          # lambda has no truth: it is an estimator setting

fig = corner.corner(flat, labels=param_names, truths=truths, truth_color='tab:red',
                    quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.3f',
                    title_kwargs={'fontsize': 8}, label_kwargs={'fontsize': 8})
fig.suptitle('AIC pipeline on the mock -- red = the generating truth,  $S^2$ = %.2f'
             % S2_ADOPTED, fontsize=13)
plt.show()

print('\\n%-26s %10s %10s %10s %8s' % ('parameter', 'truth', 'median', '+/-', 'pull'))
for j, n in enumerate(param_names):
    q16, q50, q84 = np.percentile(flat[:, j], [16, 50, 84])
    s = 0.5 * (q84 - q16)
    if j == J_LAM:
        print('%-26s %10s %10.4f %10.4f %8s' % (n, '--', q50, s, '--'))
    else:
        t = float(np.asarray(theta_true)[j])
        print('%-26s %10.4f %10.4f %10.4f %8.2f' % (n, t, q50, s, (q50 - t) / s if s > 0 else np.nan))
print('\\nTHE TEST: every |pull| should be <~ 2.')
'''))

C = [(t, src.replace('__DRIVE__', A.drive)) for t, src in C]
cells = [{'cell_type': t, 'metadata': {}, 'source': s.split('\n')} for t, s in C]
for c in cells:
    c['source'] = [l + '\n' for l in c['source'][:-1]] + [c['source'][-1]]
    if c['cell_type'] == 'code':
        c['execution_count'] = None; c['outputs'] = []

out = {'cells': cells, 'metadata': nb.get('metadata', {}), 'nbformat': 4, 'nbformat_minor': 5}

# ---------------------------------------------------------------- build-time guards
flat = '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code')
for lit in re.findall(r'(?:theta|th|thA|thB|best|res)\[(\d+)\]', flat):
    assert int(lit) < 13, 'hard-coded parameter index %s >= NDIM' % lit
assert 'build_model_acc_meff_fourier' in flat
assert 'logsumexp' not in flat and 'logl_marg' not in flat, 'a marginalisation survived'
assert 'emcee' not in flat.replace("'emcee'", ''), 'the sampler must stay BlackJAX'
assert 'blackjax' in flat and 'run_mcmc' in flat
assert 'calculate_delta_chi2' not in flat, 'the jackknife delta_chi2 must be gone'
assert 'anisotropy_b=ANISO_B' in flat, 'anisotropy must be passed as a KEYWORD, not patched'
assert 'err_den_meff=ERR_DEN_MEFF' in flat and 'ERR_DEN_MEFF = 0.0' in flat, \
    'the m_eff density error must be the tested 0.0 (hard constraint), not ERR_DEN_LOGL'
assert 'jeans_z_int_max=' not in flat, 'the meff path asserts jeans_z_int_max == 5.0; do not pass it'
assert 'dd_cal' not in flat, 'the calibration must reuse the STORED normals so A and B are paired'
assert 'DAIC_MIN_FRAC' not in flat, \
    'the old fractional rail rejected sigma_dAIC = 10.7 on this very mock -- measure, do not gate'
assert 'S2_MEASURED' in flat and "ADOPT" in flat, 'both temperatures must be reported'
assert "ADOPT  = 'gaussian'" in flat, \
    'the measured sigma_dAIC must NOT be the default: |dAIC| was measured flat (s^0.09)'
assert 'np.sqrt(2.0 * c2A.mean())' in flat, 'the Gaussian floor must use chi2, not chi2 + 2 m_eff'
assert 'truth = res[i]' not in flat, 'the posterior table must compare against theta_true, not the fit'
assert flat.count('theta_true') >= 4
assert 'make_m0_callables' in flat and flat.count('make_m0_callables') >= 2, \
    'the m=0 tabulation must be a NAMED factory, so the smoke test can check it independently'
assert '_bar_rho_tab' not in flat, 'the old inlined tabulation survived inside run_model'
assert "jax[cuda12]==%s' % TARGET" in flat, 'the jax 0.7.2 pin is missing'
assert 'FLIP_HANDEDNESS' in flat, 'the handedness flip is missing'
assert "'V_data'" in flat and "'h3_data'" in flat, 'the flip must cover every ODD quantity'
assert 'sigma_data' in flat, 'the flip must say which quantities are EVEN and left alone'
assert 'corr_V' not in flat and 'WRONG WAY ROUND' not in flat, \
    'no automatic handedness trigger: it is flipped by hand, and the output numbers do not'\
    ' judge it stably (corr_V was +0.893 even when wrong)'
assert '_unclassified' in flat, 'the flip must be complete by construction over every dict key'
assert '__DRIVE__' not in flat, 'the Drive folder placeholder was not substituted'
assert "_m.get_w0_new_improved" not in flat, 'the monkey-patch route must not be used'
assert flat.count('J_LAM') >= 3, 'lambda must be indexed BY NAME'
assert 'parallel_minimize' in flat, \
    'the minimiser must be the tested NGC 1512 vmapped parallel_minimize'
assert "method='Nelder-Mead'" not in flat, \
    'a serial Nelder-Mead optimises a surface the vmapped sampler never evaluates (chi2_kin' \
    ' differs by 8.9% in float32, 1.6% in float64) and its bar ticks once per ~690 calls'
assert 'from scipy.optimize import minimize' not in flat, 'the scipy minimiser is no longer used'
assert "tqdm(range(n_iter)" in flat, 'the progress bar must tick once per ITERATION'
assert 'VMAP_BATCH = 8' in flat, 'the minimiser batch must match the sampler N_PER measured on the L4'
assert 'N_GROUPS = 4' in flat and 'N_HALF' not in flat, \
    'the sampler must chunk into N_GROUPS, not two halves of 16 (~37 GB, OOMs a T4)'
assert 'VMAP_BATCH' in flat and '_lp_vmap = jax.vmap' in flat, \
    'the minimiser must evaluate its particles in ONE batched call'
assert 'OPT_SPREAD' in flat and 'OPT_LO' in flat and 'OPT_HI' in flat, \
    'the hill climb needs a per-parameter step and a search box'
assert 'N_ITER     = 120' in flat, 'n_iter is the tested NGC 1512 value; 0.97**120 -> x0.026'
assert 'nucseed' not in flat.lower() and 'N_NUC' not in flat, \
    'the tracer positions must be load_data_bootstrap\'s, with no nuclear reseed'

dst = os.path.join(os.path.join(ROOT, A.dest) if A.dest else REPO, A.out)
os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
json.dump(out, open(dst, 'w'), indent=1)
print('wrote %s  (%d cells: %d code, %d markdown)'
      % (dst, len(cells), sum(c['cell_type'] == 'code' for c in cells),
         sum(c['cell_type'] == 'markdown' for c in cells)))
