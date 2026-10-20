#!/usr/bin/env python
"""Execute the exponential-disc notebook's OWN cells locally, on a small orbit library.

Cell 2 (Colab install + Drive) cannot run here, so an equivalent local header stands in for it,
then cells 4 (density + the five callables), 6 (load + flip), 8 (config + likelihood) and 10
(ground truth) are exec'd as written.  Controls:
  1. the derived stellar masses invert to the ground truth's disc and bar masses
  2. the disc CylSpline against AGAMA's exact 'Disk' at the truth
  3. the acceleration closure against -grad of the potential closure
  4. the m=0 closures against a 256-angle average of the full density / potential
  5. the likelihood VMAPPED over two thetas (what the minimiser and emcee do) is finite
  6. cell 17 (emcee) + cell 18 (corner) on a STUB likelihood: the start ball, the HDF5 backend,
     RESUME continuing a chain bit-for-bit, the results pickle, the corner cell

    python test_example_aic_expdisc_nb.py [--nsamples 600]
"""
import argparse, json, os, sys, threading, time, resource, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # the package (model.py, utils.py, ...); notebooks, data and the N-body snapshot sit here in example/
A = argparse.ArgumentParser()
A.add_argument('--nb', default=os.path.join(HERE, 'SchwarMAX_Nbody_AIC_expdisc.ipynb'))
A.add_argument('--nsamples', type=int, default=600)
A.add_argument('--nreal', type=int, default=2)
A.add_argument('--nboot', type=int, default=3)
A.add_argument('--scratch', default=os.path.join(HERE, '_test_expdisc_scratch'))
A.add_argument('--cap', type=float, default=12.0, help='abort if RSS exceeds this many GB')
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
path = %r + "/"          # cell 2 sets this on Colab
''' % (ROOT, ROOT)
G = {'__name__': '__main__'}
exec(compile(HEADER, '<header>', 'exec'), G)
nb = json.load(open(A.nb))
cell = lambda i: ''.join(nb['cells'][i]['source'])

for i in (4, 6, 8, 10):
    s = cell(i)
    s = s.replace('N_ORBIT = 7_500', 'N_ORBIT = %d' % A.nsamples)
    s = s.replace('N_REALIZATIONS = 4', 'N_REALIZATIONS = %d' % A.nreal)
    s = s.replace('N_BOOT_MEFF    = 9', 'N_BOOT_MEFF    = %d' % A.nboot)
    t0 = time.time()
    exec(compile(s, '<cell %d>' % i, 'exec'), G)
    print('   cell %-3d OK  (%.1f s, peak RSS %.2f GB)' % (i, time.time() - t0, _rss_gb()), flush=True)

jax, jnp = G['jax'], G['jnp']
th = np.asarray(G['theta_true'])
pn = list(G['param_names'])
P = lambda n: float(th[pn.index(n)])
assert G['NDIM'] == 12 and len(pn) == 12

# ---------------------------------------------------------------- 1. the derived masses
Rd, hz, Lb = 10 ** P('logRd_disk'), 10 ** P('loghz_disk'), 10 ** P('logL_bar')
al, be, ga = [np.degrees(P(n)) for n in ('alpha', 'beta', 'gamma')]
_Md, _Mb = G['_Md'], G['_Mb']
lMd, lMb, Mst, pbox = [float(v) for v in G['stellar_masses'](P('f_bar'), P('log_mass_to_light_ratio'),
                                                             Rd, hz, Lb, Lb / 5, hz, al, be, ga)]
pdisc, pbar = [float(v) for v in G['box_fractions'](Rd, hz, Lb, Lb / 5, hz, al, be, ga)]
_lml = np.log10((_Md + 2 * _Mb) * pbox / G['L_BOX'])
d_, b_, _, _ = [float(v) for v in G['stellar_masses'](P('f_bar'), _lml, Rd, hz, Lb, Lb / 5, hz, al, be, ga)]
print('\n1. derived masses: p_disc %.4f p_bar %.4f p_box %.4f -> M_star %.4e at M/L = 1; box log M/L of '
      'the direct fit %+.4f' % (pdisc, pbar, pbox, Mst, _lml))
print('   split at M_star = Md + 2 Mb: logM_disc %.4f (%.4f), logM_bar %.4f (%.4f)'
      % (d_, np.log10(_Md), b_, np.log10(_Mb)))
assert abs(d_ - np.log10(_Md)) < 1e-4 and abs(b_ - np.log10(_Mb)) < 1e-4
print('   PASS')

# ---------------------------------------------------------------- 2-4. the five callables
lM_h, lR_h = G['logMenc_logc_to_logM_logRs'](P('logM_10kpc'), P('logC_halo'), r_enc=10.0, Delta=200.,
                                             rho_crit=277.54)
ph = {'logM': float(lM_h), 'Rs': float(10 ** lR_h), 'a': 1.0, 'b': 1.0, 'c': 1.0, 'x_origin': 0.0,
      'y_origin': 0.0, 'z_origin': 0.0, 'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}
pb = {'logM_disc': np.log10(_Md), 'Rd_disc': Rd, 'hz_disc': hz, 'logM_bar': np.log10(_Mb), 'L_bar': Lb,
      'a_bar': Lb / 5, 'b_bar': hz}
dens, pot, acc, pax, dax = G['make_model_funcs'](pb, ph)

import agama
agama.setUnits(length=1, velocity=1, mass=1)
pa = agama.Potential(type='Disk', surfaceDensity=_Md / (2 * np.pi * Rd ** 2), scaleRadius=Rd, scaleHeight=hz)
dphi = G['disc_spline']({'logM': np.log10(_Md), 'Rd': Rd, 'hz': hz})
K = float(G['KPCGYR_TO_KMS'])
R = np.array([0.5, 1, 2, 3, 5, 8, 12, 20])
vA = np.sqrt(-R * pa.force(np.column_stack([R, 0 * R, 0 * R]))[:, 0])
vS = np.array([np.sqrt(max(-r * float(np.asarray(G['get_acc'](r, 0.0, 0.0, dphi))[0]), 0)) * K for r in R])
print('\n2. disc CylSpline vs AGAMA Disk at the truth: v_c %s km/s -> rel %s'
      % (np.round(vA, 1).tolist(), ' '.join('%+.3f' % v for v in vS / vA - 1)))
assert np.max(np.abs(vS / vA - 1)) < 0.05
print('   PASS')

rng = np.random.default_rng(1)
pts = rng.uniform(-8, 8, size=(30, 3)); pts[:, 2] *= 0.3
gpot = jax.grad(lambda q: pot(q[0], q[1], q[2], pb, ph))
d = [float(np.max(np.abs(np.asarray(acc(*p, pb, ph)) + np.asarray(gpot(jnp.asarray(p, jnp.float32))))))
     for p in pts]
sc = float(np.median([np.max(np.abs(np.asarray(gpot(jnp.asarray(p, jnp.float32))))) for p in pts]))
print('\n3. acceleration closure vs -grad(potential closure): max |diff| / typical |a| = %.2e' % (max(d) / sc))
assert max(d) / sc < 1e-3
print('   PASS')

ang = np.linspace(0, 2 * np.pi, 256, endpoint=False)
worst = 0.0
print('\n4. m=0 closures vs a 256-angle average of the full density / potential')
for (Rr, z) in ((0.5, 0.0), (1.0, 0.0), (2.0, 0.2), (3.0, 0.5), (6.0, 1.0), (9.0, 2.0)):
    xs, ys = jnp.asarray(Rr * np.cos(ang)), jnp.asarray(Rr * np.sin(ang))
    zs = jnp.full_like(xs, z)
    rr = float(jnp.mean(dens(xs, ys, zs, pb)))
    pr = float(jnp.mean(pot(xs, ys, zs, pb, ph)))
    rg, pg = float(dax(Rr, 0.0, z, pb)), float(pax(Rr, 0.0, z, pb, ph))
    print('   R=%4.1f z=%4.1f  rho %+.2f%%   phi %+.3f%%' % (Rr, z, 100 * (rg / rr - 1), 100 * (pg / pr - 1)))
    worst = max(worst, abs(pg / pr - 1))
assert worst < 0.05
print('   PASS (worst |d phi_m0| = %.3f%%)' % (100 * worst))

# ---------------------------------------------------------------- 5. vmapped likelihood
thB = th.copy(); thB[pn.index('beta')] += 0.03
t0 = time.time()
lv = np.asarray(jax.jit(jax.vmap(lambda t: G['model_likelihood'](t, G['dict_data'], G['NUM_VBIN'], G['RZ_TOT'],
                                                                 jnp.sqrt(G['S2']))))(jnp.asarray(np.stack([th, thB]))))
print('\n5. vmapped logL over 2 thetas: %s  (%.0f s, peak RSS %.2f GB)' % (lv, time.time() - t0, _rss_gb()))
assert np.all(np.isfinite(lv))
print('   PASS')

# ---------------------------------------------------------------- 6. emcee + corner, stub likelihood
shutil.rmtree(A.scratch, ignore_errors=True)
os.makedirs(os.path.join(A.scratch, 'example'))
G['path'] = A.scratch + '/'
G['MINIMISER_RESULT'] = os.path.join(A.scratch, 'example', 'minimisation_result_%s.npy' % G['RUN_TAG'])
np.save(G['MINIMISER_RESULT'], th)
G['SIGMA_AMPLIFY'] = 1.0
G['S2_ADOPTED'] = G['S2']
_t0, _s0 = jnp.asarray(th), jnp.asarray([0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.1, 0.05, 0.1, 0.3])
G['model_likelihood'] = lambda t, dd, nv, nr, sigma_amplify=1.0: -0.5 * jnp.sum(((t - _t0) / _s0) ** 2)
s17 = cell(17).replace("subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'emcee'], check=False)", '')
assert 'pip' not in s17
import re
# the user tunes EMCEE_STEPS, so match whatever value the cell carries
_steps = lambda n: re.subn(r'^EMCEE_STEPS(\s*)=\s*\d+', r'EMCEE_STEPS\1= %d' % n, s17, flags=re.M)
_s300, _n = _steps(300)
assert _n == 1, 'cell 17 has no single EMCEE_STEPS line'
exec(compile(_s300, '<cell 17 a>', 'exec'), G)
c300 = np.array(G['chain'])
exec(compile(_steps(500)[0], '<cell 17 b, RESUME>', 'exec'), G)
c500 = np.array(G['chain'])
_W = int(G['EMCEE_WALKERS'])
assert c300.shape == (300, _W, 12) and c500.shape == (500, _W, 12), (c300.shape, c500.shape)
assert np.array_equal(c500[:300], c300), 'RESUME did not continue the stored chain'
exec(compile(cell(18), '<cell 18>', 'exec'), G)
fs = G['flat_samples']
z = (fs.mean(0) - th) / np.asarray(_s0)
print('\n6. emcee on a stub Gaussian: 300 + 500 resumed steps, chain continued bit-for-bit; '
      'posterior mean offset / sigma max %.2f, std / sigma %.2f-%.2f'
      % (np.max(np.abs(z)), np.min(fs.std(0) / np.asarray(_s0)), np.max(fs.std(0) / np.asarray(_s0))))
assert os.path.isfile(os.path.join(A.scratch, 'example', 'emcee_results_%s.pkl' % G['RUN_TAG']))
shutil.rmtree(A.scratch)
print('   PASS')
print('\nALL CONTROLS PASS.  peak RSS %.2f GB' % _rss_gb())
