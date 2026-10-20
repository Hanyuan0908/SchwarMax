#!/usr/bin/env python
"""Execute the AIC test notebook's OWN cells locally, on a small orbit library.

The notebook is written for Colab (it mounts Drive and pip-installs), so cell 2 cannot run
here.  This supplies an equivalent local header and then exec's the notebook's real cells --
4 (density + the three new ingredients), 6 (load + the handedness flip), 8 (config + the AIC
likelihood), 10 (ground truth) -- so a mistake in the notebook fails HERE rather than on a GPU
an hour in.  Cells 13/15/17 (minimiser, dAIC calibration, MCMC) are too slow for a smoke test;
cell 15's logic is exercised instead by the dAIC control at the bottom, on the truth.

It also runs the one control that matters for the port: the explicit `acceleration_func` must
reproduce what `build_model` computes internally as -grad(potential_func).

    python test_example_aic_nb.py [--nsamples 600] [--cells 4,6,8,10,12]
"""
import argparse, json, os, sys, time
import numpy as np

A = argparse.ArgumentParser()
A.add_argument('--nb', default='schwarmax_merge_branch/example/example_schwarmax_AIC.ipynb')
A.add_argument('--root', default='schwarmax_merge_branch',
               help='the SchwarMAX folder to import from and load the mock out of')
A.add_argument('--nsamples', type=int, default=600)
A.add_argument('--cells', default='4,6,8,10')   # add 13 for the minimiser
A.add_argument('--nreal', type=int, default=2, help='n_realizations, reduced for the test')
A.add_argument('--nboot', type=int, default=3, help='n_boot_meff, reduced for the test')
A.add_argument('--ncal', type=int, default=8, help='N_CAL for the dAIC control')
A.add_argument('--niter', type=int, default=4,
               help='N_ITER for cell 7, reduced for the test')
A.add_argument('--noflip', action='store_true',
               help='force FLIP_HANDEDNESS = False -- the control that the notebook DETECTS it')
A = A.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
REPO = os.path.join(ROOT, A.root)

HEADER = '''
import os, sys, time
sys.path.insert(0, %r)
import warnings; warnings.filterwarnings("ignore")
import jax, jax.numpy as jnp
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from functools import partial
from tqdm import tqdm
from utils import logMenc_logc_to_logM_logRs, load_data_bootstrap
from constants import EPSILON, KPCGYR_TO_KMS
from model import build_model, build_model_acc_meff_fourier
from potentials import (NFW_potential, MiyamotoNagai_potential, MiyamotoNagai_density,
                        T3_density, T3_potential, V4_density, V4_potential)
path = %r + "/"          # cell 2 sets this on Colab; cell 2 cannot run here
''' % (REPO, REPO)

G = {'__name__': '__main__'}
exec(compile(HEADER, '<header>', 'exec'), G)

nb = json.load(open(os.path.join(ROOT, A.nb)))
want = [int(v) for v in A.cells.split(',')]
print('notebook %s\nroot     %s\nrunning cells %s on a %d-orbit library\n'
      % (A.nb, REPO, want, A.nsamples))
assert os.path.isfile(os.path.join(REPO, 'model.py')), 'no model.py in %s' % REPO

for i in want:
    s = ''.join(nb['cells'][i]['source'])
    # local substitutions: the Drive path, a small library, and cheap m_eff settings
    s = s.replace('N_ORBIT = 7_500', 'N_ORBIT = %d' % A.nsamples)
    s = s.replace('N_REALIZATIONS = 4', 'N_REALIZATIONS = %d' % A.nreal)
    s = s.replace('N_BOOT_MEFF    = 9', 'N_BOOT_MEFF    = %d' % A.nboot)
    s = s.replace('N_ITER     = 120', 'N_ITER     = %d' % A.niter)
    if A.noflip:
        s = s.replace('FLIP_HANDEDNESS = True', 'FLIP_HANDEDNESS = False')
    t0 = time.time()
    try:
        exec(compile(s, '<cell %d>' % i, 'exec'), G)
    except Exception as e:
        print('\n*** CELL %d FAILED: %s: %s' % (i, type(e).__name__, e))
        raise
    print('   cell %-3d OK  (%.1f s)' % (i, time.time() - t0))

# ---------------------------------------------------------------- the port control
print('\nCONTROL: does the explicit acceleration_func reproduce build_model\'s internal -grad?')
th = np.asarray(G['theta_true'])
jnp_ = G['jnp']; jax_ = G['jax']
logM_halo, logRs_halo = G['logMenc_logc_to_logM_logRs'](th[0], th[3], r_enc=10.0,
                                                        Delta=200., rho_crit=277.54)
ph = {'logM': float(logM_halo), 'Rs': float(10 ** logRs_halo), 'a': 1.0, 'b': 1.0, 'c': 1.0,
      'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0, 'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}
Lb = 10.0 ** th[6]; Hs = 10.0 ** th[5]
pb = {'logM_disc': th[1], 'Rs_disc': 10 ** th[4], 'Hs_disc': Hs, 'logM_bar': th[2],
      'L_bar': Lb, 'a_bar': Lb / 5.0, 'b_bar': Hs,
      'mass_to_light_ratio': 10 ** th[10], 'Omega_bar': 10 ** th[11],
      'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0, 'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
      'alpha': th[7] * 180 / np.pi, 'beta': th[8] * 180 / np.pi, 'gamma': th[9] * 180 / np.pi}

def ref_acc(x, y, z):
    def _pot(pos):
        return G['potential_func'](pos[0], pos[1], pos[2], pb, ph)
    return -jax_.grad(_pot)(jnp_.array([x, y, z]))

rng = np.random.default_rng(1)
pts = rng.uniform(-8, 8, size=(40, 3)); pts[:, 2] *= 0.3
d = [float(np.max(np.abs(np.asarray(G['acceleration_func'](*p, pb, ph)) - np.asarray(ref_acc(*p)))))
     for p in pts]
scale = float(np.median([np.max(np.abs(np.asarray(ref_acc(*p)))) for p in pts]))
print('   max|diff| over 40 points = %.3e   (typical |a| = %.3e)  -> relative %.2e'
      % (max(d), scale, max(d) / scale))
assert max(d) / scale < 1e-5, 'acceleration_func does NOT match build_model\'s internal gradient'
print('   PASS')

print('\nCONTROL: is make_m0_callables the m=0 part of density_func/potential_func?')
# The notebook's factory against a brute-force 256-angle average of the FULL density.  The
# bulge is not azimuthally averaged by design (it varies 2-6% in phi), so a few per cent is
# expected and the bar term, which varies by hundreds of per cent, is what is being tested.
dax, pax = G['make_m0_callables'](pb, ph)
ang = np.linspace(0, 2 * np.pi, 256, endpoint=False)
print('   %-14s %13s %13s %8s | %13s %13s %8s'
      % ('', 'rho notebook', 'rho 256-avg', 'diff', 'phi notebook', 'phi 256-avg', 'diff'))
worst = 0.0
for (R, z) in ((0.5, 0.0), (1.0, 0.0), (2.0, 0.2), (3.0, 0.5), (6.0, 1.0), (9.0, 2.0)):
    rr = float(np.mean([float(G['density_func'](R*np.cos(a), R*np.sin(a), z, pb)) for a in ang]))
    rg = float(dax(R, 0.0, z, pb))
    pr = float(np.mean([float(G['potential_func'](R*np.cos(a), R*np.sin(a), z, pb, ph)) for a in ang]))
    pg = float(pax(R, 0.0, z, pb, ph))
    print('   R=%4.1f z=%4.1f  %13.5e %13.5e %7.2f%% | %13.5e %13.5e %7.2f%%'
          % (R, z, rg, rr, 100*(rg/rr - 1), pg, pr, 100*(pg/pr - 1)))
    worst = max(worst, abs(pg/pr - 1))
assert worst < 0.05, 'the m=0 POTENTIAL is off by %.1f%% -- the tabulation grid is too coarse' % (100*worst)
print('   PASS (worst |d phi_m0| = %.2f%%)' % (100 * worst))

print('\nALL CELLS RAN.  NDIM = %d, RZ_TOT = %d, S2 = %.2f'
      % (G['NDIM'], G['RZ_TOT'], G['S2']))

# ---------------------------------------------------------------- the dAIC control
# Cell 15's calibration, run once at the truth: the two diagnostics Wang+2026 quote.
print('\nCONTROL: the dAIC calibration (cell 15 logic) at the truth, N_CAL = %d' % A.ncal)
rm = G['run_model']
thB = th.copy(); thB[8] += 0.03                 # a genuinely different model: 1.7 deg in i
aic = lambda t: (np.asarray(rm(G['jnp'].asarray(t), G['dict_data'], G['NUM_VBIN'],
                               G['RZ_TOT'], n_boot=A.ncal)['chi2_boot_kin'], float)
                 + 2.0 * float(rm(G['jnp'].asarray(t), G['dict_data'], G['NUM_VBIN'],
                                  G['RZ_TOT'], n_boot=A.ncal)['m_eff_kin']))
aA, aB = aic(th), aic(thB)
sig1  = 0.5 * (aA.std(ddof=1) + aB.std(ddof=1))
floor = float(np.sqrt(2.0 * 0.5 * (aA.mean() + aB.mean())))
print('   sigma_AIC %.2f | floor sqrt(2 chi2) %.2f | RATIO %.2f   [Wang+2026: 1.13]'
      % (sig1, floor, sig1 / floor))
print('   sigma_dAIC (paired) %.2f | rho %.4f   [Wang+2026: 11, ~0.97]'
      % ((aA - aB).std(ddof=1), float(np.corrcoef(aA, aB)[0, 1])))
assert np.all(np.isfinite(aA)) and np.all(np.isfinite(aB)), 'the AIC realisations are not finite'
print('   PASS')
