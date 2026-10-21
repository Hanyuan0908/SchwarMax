#!/usr/bin/env python
"""Execute the CHEAP (dAIC + MCMC only) notebook's cells 13, 15 and 17 locally.

test_example_aic_expdisc_nb.py --nb <cheap notebook> already covers cells 4, 6, 8, 10 and 17-18;
what it cannot see is the part that is new: cell 13 LOADS an earlier run's minimiser ensemble
instead of minimising.  Here a fake ensemble is written to a scratch example/ folder under
PREV_TAG, then:
  1. cell 13 refuses to run without the files, then loads them: best = the argmax, |pos| = NDIM
  2. cell 15 (the dAIC calibration) runs on the LOADED best and spread, on a small library and
     fewer realisations, and leaves S2_ADOPTED / SIGMA_AMPLIFY for the chain
  3. cell 17 (emcee, stub likelihood) starts its walkers around the LOADED best and writes the
     chain under the notebook's own RUN_TAG, not PREV_TAG

    python test_example_mcmc_only_nb.py [--nsamples 600] [--ncal 3]
"""
import argparse, json, os, re, sys, threading, time, resource, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
A = argparse.ArgumentParser()
A.add_argument('--nb', default=os.path.join(HERE, 'SchwarMAX_Nbody_AIC_expdisc_field_den3_mcmc_only.ipynb'))
A.add_argument('--nsamples', type=int, default=600)
A.add_argument('--nreal', type=int, default=2)
A.add_argument('--nboot', type=int, default=3)
A.add_argument('--ncal', type=int, default=3, help='cell 15 noise realisations (the notebook uses 24)')
A.add_argument('--scratch', default=os.path.join(HERE, '_test_mcmc_only_scratch'))
A.add_argument('--cap', type=float, default=8.0, help='abort if RSS exceeds this many GB')
A = A.parse_args()


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 3


def _watch():
    while True:
        time.sleep(0.25)
        if _rss_gb() > A.cap:
            sys.stderr.write('\n*** WATCHDOG: RSS %.2f GB > cap %.1f GB -- ABORTING\n' % (_rss_gb(), A.cap))
            os._exit(42)


threading.Thread(target=_watch, daemon=True).start()
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

HEADER = '''
import os, sys, time
sys.path.insert(0, %r)
import warnings; warnings.filterwarnings("ignore")
import jax, jax.numpy as jnp
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle, corner
from functools import partial
from tqdm import tqdm
from utils import logMenc_logc_to_logM_logRs, load_data_bootstrap, makeRotationMatrix
from constants import EPSILON, KPCGYR_TO_KMS
from model import build_model_acc_meff_fourier
from potentials import NFW_potential, T3_density, T3_potential, V4_density, V4_potential
from CylindricalSpline import get_phi_m, evaluate_phi, evaluate_phi_axisymmetric, get_acc
path = %r + "/"
''' % (ROOT, A.scratch)
G = {'__name__': '__main__'}
exec(compile(HEADER, '<header>', 'exec'), G)
nb = json.load(open(A.nb))
cell = lambda i: ''.join(nb['cells'][i]['source'])
assert len(nb['cells']) == 19

# cell 6 reads the mock from the REPOSITORY's example/, so it runs before path moves to the scratch
G['path'] = ROOT + '/'
for i in (4, 6, 8, 10):
    s = cell(i)
    s = s.replace('N_ORBIT = 7_500', 'N_ORBIT = %d' % A.nsamples)
    s = s.replace('N_REALIZATIONS = 4', 'N_REALIZATIONS = %d' % A.nreal)
    s = s.replace('N_BOOT_MEFF    = 9', 'N_BOOT_MEFF    = %d' % A.nboot)
    t0 = time.time()
    exec(compile(s, '<cell %d>' % i, 'exec'), G)
    print('   cell %-3d OK  (%.1f s, peak RSS %.2f GB)' % (i, time.time() - t0, _rss_gb()), flush=True)
assert 'chi2_report' not in cell(10).split('# (the full notebook')[0].split('\n')[-1]
jnp = G['jnp']
th = np.asarray(G['theta_true'], float)
NDIM, TAG = int(G['NDIM']), G['RUN_TAG']
PREV = re.search(r"PREV_TAG = '([^']+)'", cell(13)).group(1)
assert PREV != TAG

# ---------------------------------------------------------------- 1. the loader
shutil.rmtree(A.scratch, ignore_errors=True)
os.makedirs(os.path.join(A.scratch, 'example'))
G['path'] = A.scratch + '/'
try:
    exec(compile(cell(13), '<cell 13, no files>', 'exec'), G)
    raise SystemExit('FAIL: cell 13 ran without the minimiser files')
except AssertionError as e:
    print('\n1. without the files cell 13 stops: %s' % str(e)[:90] + '...')
rng = np.random.default_rng(3)
spread = np.full(NDIM, 0.02)
pos = th + rng.standard_normal((16, NDIM)) * spread
pos[:, 1] = np.clip(pos[:, 1], 0.05, 0.95)
val = -200.0 - rng.random(16) * 10
k = int(np.argmax(val))
ex = os.path.join(A.scratch, 'example')
np.save(os.path.join(ex, 'minimiser_pos_%s.npy' % PREV), pos)
np.save(os.path.join(ex, 'minimiser_val_%s.npy' % PREV), val)
np.save(os.path.join(ex, 'minimisation_result_%s.npy' % PREV), pos[k])
exec(compile(cell(13), '<cell 13>', 'exec'), G)
assert np.array_equal(G['best'], pos[k]) and G['pos'].shape == (16, NDIM)
assert G['MINIMISER_RESULT'].endswith('minimisation_result_%s.npy' % PREV)
print('   with them: best = particle %d of 16, MINIMISER_RESULT -> %s' % (k, os.path.basename(G['MINIMISER_RESULT'])))
print('   PASS')

# ---------------------------------------------------------------- 2. the dAIC calibration
s15 = cell(15)
s15, n1 = re.subn(r'^N_CAL\s*=\s*\d+', 'N_CAL  = %d' % A.ncal, s15, flags=re.M)
s15, n2 = re.subn(r'^SCALES = \([^)]*\)', 'SCALES = (0.5, 2.0)', s15, flags=re.M)
assert n1 == n2 == 1
t0 = time.time()
exec(compile(s15, '<cell 15>', 'exec'), G)
assert np.array_equal(G['thA'], pos[k]), 'cell 15 did not start from the loaded best'
assert float(G['S2_ADOPTED']) == float(G['S2']) and np.isclose(float(G['SIGMA_AMPLIFY']), np.sqrt(float(G['S2'])))
print('\n2. cell 15 on the loaded best + spread: %d realisations x 3 models in %.0f s (peak RSS %.2f GB); '
      'S2_ADOPTED %.2f' % (A.ncal, time.time() - t0, _rss_gb(), float(G['S2_ADOPTED'])))
print('   PASS')

# ---------------------------------------------------------------- 3. emcee from the loaded best
_t0 = jnp.asarray(pos[k])
_s0 = jnp.asarray([0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.1, 0.05, 0.1, 0.3])
G['model_likelihood'] = lambda t, dd, nv, nr, sigma_amplify=1.0: -0.5 * jnp.sum(((t - _t0) / _s0) ** 2)
s17 = cell(17).replace("subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'emcee'], check=False)", '')
s17, n = re.subn(r'^EMCEE_STEPS(\s*)=\s*\d+', r'EMCEE_STEPS\1= 20', s17, flags=re.M)
assert n == 1 and 'pip' not in s17
exec(compile(s17, '<cell 17>', 'exec'), G)
c = np.array(G['chain'])
assert np.array_equal(G['_start'], pos[k]), 'cell 17 did not centre its start ball on the loaded best'
assert os.path.isfile(os.path.join(ex, 'emcee_%s.h5' % TAG))
assert os.path.isfile(os.path.join(ex, 'emcee_results_%s.pkl' % TAG))
assert not os.path.exists(os.path.join(ex, 'emcee_%s.h5' % PREV)), 'the chain was written under PREV_TAG'
print('\n3. cell 17: the start ball of %d walkers is centred on the loaded best; chain written as emcee_%s.h5'
      % (c.shape[1], TAG))
print('   PASS')
shutil.rmtree(A.scratch)
print('\nALL CONTROLS PASS.  peak RSS %.2f GB' % _rss_gb())
