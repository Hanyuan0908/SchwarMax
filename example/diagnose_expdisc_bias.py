#!/usr/bin/env python
"""Why does the exponential-disc chain (emcee_aic_exp.h5) put Rd and L_bar below the truth?

Runs the notebook's OWN forward model (cells 4, 6, 8 of SchwarMAX_Nbody_AIC_expdisc.ipynb, the
full orbit library) at the truth, at the chain median, and at the median with the truth's disc
and bar shapes swapped in, and splits the likelihood into its channels:

    logL = -0.5 * (chi2_den + chi2_h1 + chi2_h2 + chi2_h3 + chi2_h4 + 2 m_eff_kin) / S2

chi2_den is the sky surface density at ERR_DEN_LOGL; the 3-D Fourier block is NOT in logL (it
only constrains the NNLS), so its chi2 against the parametric target is reported separately as
chi2_3d: how well the orbits reproduce the model's own 3-D density.

Each point is one serial run_model call (CPU, float32): not bit-identical to the chain's vmapped
GPU numbers, so compare points with each other, not with the chain's logP.

    python diagnose_expdisc_bias.py --points T,M,M+shape [--norbit 7500] [--out example/diag.json]
"""
import argparse, json, os, sys, threading, time, resource

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # the package (model.py, utils.py, ...); notebooks, data and the N-body snapshot sit here in example/
P = argparse.ArgumentParser()
P.add_argument('--chain', default=os.path.join(HERE, 'emcee_aic_exp.h5'),
               help='the chain whose median is the point M')
P.add_argument('--nb', default=os.path.join(HERE, 'SchwarMAX_Nbody_AIC_expdisc.ipynb'),
               help='the notebook whose likelihood is evaluated, e.g. the IFU-field-normalised '
                    'SchwarMAX_Nbody_AIC_expdisc_field.ipynb')
P.add_argument('--points', default='T,M,M+shape,M+Rd,M+L,T+shape')
P.add_argument('--norbit', type=int, default=7500)
P.add_argument('--nreal', type=int, default=4, help="the notebook's N_REALIZATIONS")
P.add_argument('--nboot', type=int, default=9, help="the notebook's N_BOOT_MEFF")
P.add_argument('--out', default=os.path.join(HERE, 'diagnose_expdisc_bias.json'))
P.add_argument('--cap', type=float, default=16.0, help='abort if RSS exceeds this many GB')
P.add_argument('--param-only', action='store_true',
               help='no orbits: project the PARAMETRIC density into the 609 bins at every point '
                    'and add it to the existing --out file')
A = P.parse_args()


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 3


def _watch():
    while True:
        time.sleep(0.25)
        if _rss_gb() > A.cap:
            sys.stderr.write('\n*** WATCHDOG: RSS %.2f GB > cap %.1f GB -- ABORTING\n' % (_rss_gb(), A.cap))
            sys.stderr.flush()
            os._exit(42)


threading.Thread(target=_watch, daemon=True).start()
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np

# ---------------------------------------------------------------- the notebook
HEADER = '''
import os, sys, time, pickle
sys.path.insert(0, %r)
import warnings; warnings.filterwarnings("ignore")
import jax, jax.numpy as jnp
import numpy as np
from functools import partial
from utils import logMenc_logc_to_logM_logRs, load_data_bootstrap, makeRotationMatrix
from constants import EPSILON, KPCGYR_TO_KMS
from model import build_model_acc_meff_fourier
from potentials import NFW_potential, T3_density, T3_potential, V4_density, V4_potential
from CylindricalSpline import get_phi_m, evaluate_phi, evaluate_phi_axisymmetric, get_acc
path = %r + "/"
''' % (ROOT, ROOT)
G = {'__name__': '__main__'}
exec(compile(HEADER, '<header>', 'exec'), G)
nb = json.load(open(A.nb))
for i in (4, 6, 8):
    s = ''.join(nb['cells'][i]['source'])
    for old, new in (('N_ORBIT = 7_500', 'N_ORBIT = %d' % A.norbit),
                     ('N_REALIZATIONS = 4', 'N_REALIZATIONS = %d' % A.nreal),
                     ('N_BOOT_MEFF    = 9', 'N_BOOT_MEFF    = %d' % A.nboot)):
        s = s.replace(old, new)
    exec(compile(s, '<notebook cell %d>' % i, 'exec'), G)
c10 = ''.join(nb['cells'][10]['source'])
exec(c10[:c10.index('theta_true')], G)
jnp = G['jnp']
names = list(G['param_names'])
J = {n: names.index(n) for n in names}
T = np.asarray(G['ground_truth'] + [0.0], float)

import emcee
_b = emcee.backends.HDFBackend(A.chain, read_only=True)
_S = _b.get_chain()
_burn = int(3 * np.nanmax(_b.get_autocorr_time(tol=0)))
M = np.median(_S[_burn:].reshape(-1, len(names)), axis=0)
SHAPE = ('f_bar', 'logRd_disk', 'logL_bar')


def swap(base, src, keys):
    th = base.copy()
    for k in keys:
        th[J[k]] = src[J[k]]
    return th


POINTS = {'T': T, 'M': M,
          'M+shape': swap(M, T, SHAPE),            # chain solution, truth's f_bar/Rd/L_bar
          'M+Rd': swap(M, T, ('logRd_disk',)),
          'M+L': swap(M, T, ('logL_bar',)),
          'M+fbar': swap(M, T, ('f_bar',)),
          'T+shape': swap(T, M, SHAPE),            # truth, chain's f_bar/Rd/L_bar
          'T+ML': swap(T, T, ()), }                # truth at the direct fit's box M/L (below)
# the exp-family fit to the N-body needs log M/L = +0.108 in the box at its own masses
# (test_example_aic_expdisc_nb.py control 1): the truth's MASS, with the light still L_BOX
POINTS['T+ML'][J['log_mass_to_light_ratio']] = 0.1079


# ---------------------------------------------------------------- one evaluation, all channels
def decompose(theta):
    """model_likelihood's arithmetic (cell 8), kept channel by channel."""
    res = G['run_model'](jnp.asarray(theta, dtype=jnp.float32), G['dict_data'], G['NUM_VBIN'], G['RZ_TOT'])
    dd = G['dict_data']
    O = np.asarray(dd['XY_density_data'])
    sig_l = (G['ERR_DEN_LOGL'] * O) / (O * np.log(10) + float(G['EPSILON']))
    mden = np.asarray(res['surface_density'])[0]
    rden = (np.log10(mden + float(G['EPSILON'])) - np.log10(O + float(G['EPSILON']))) / sig_l
    out = {'chi2_den': float(np.nansum(rden ** 2))}
    for k in (1, 2, 3, 4):
        r = ((np.asarray(res['h%d' % k])[0] - np.asarray(dd['h%d_data' % k]))
             / (np.asarray(dd['h%d_data_err' % k]) + float(G['EPSILON'])))
        out['chi2_h%d' % k] = float(np.nansum(r ** 2))
    out['m_eff_kin'] = float(res['m_eff_kin'])
    out['chi2_kin'] = sum(out['chi2_h%d' % k] for k in (1, 2, 3, 4))
    out['logL'] = -0.5 * (out['chi2_kin'] + 2 * out['m_eff_kin'] + out['chi2_den']) / G['S2']
    d3t = np.asarray(res['density_3d']); d3m = np.asarray(res['density_3d_model'])[0]
    s3 = np.asarray(res['sig_Rzphi'])
    out['chi2_3d'] = float(np.sum(((d3m - d3t) / s3) ** 2))
    out['logM_disc'], out['logM_bar'] = float(res['logM_disc']), float(res['logM_bar'])
    out['M_star'], out['p_box'] = float(res['M_star']), float(res['p_box'])
    out['flux_d/m'] = float(O.sum() / mden.sum())
    out['den_resid'] = rden.tolist()          # per sky pixel, in sigma units, for later maps
    return out


# ---------------------------------------------------------------- the parametric model on the sky
# Sky pixel (X, Y), line of sight s:  (x, y, z) = R^T (X, Y, s), R = makeRotationMatrix -- the
# convention of box_fractions (cell 8).  3 x 3 sub-pixels, the box_fractions line-of-sight nodes.
_dd = G['dict_data']
_nX, _nY = [int(v) for v in np.asarray(_dd['nX_nY'])]
_dXp = float(np.diff(np.asarray(_dd['X_minmax']))[0]) / _nX
_dYp = float(np.diff(np.asarray(_dd['Y_minmax']))[0]) / _nY
_off = (np.arange(3) - 1) / 3.0
_PX = (np.asarray(_dd['X_regular_grid'])[:, None, None] + _off[None, :, None] * _dXp
       + 0 * _off[None, None, :]).ravel()
_PY = (np.asarray(_dd['Y_regular_grid'])[:, None, None] + 0 * _off[None, :, None]
       + _off[None, None, :] * _dYp).ravel()
_BMAP = np.asarray(_dd['bin_mapping'])[:_nX * _nY].astype(int)


def param_sigma_bins(theta):
    """Mean projected density of the PARAMETRIC model per Voronoi bin [Lsun/pc^2 at the
    theta's M/L], with the masses derived exactly as run_model does."""
    jax = G['jax']
    th = np.asarray(theta, float)
    Rd, hz, Lb = 10 ** th[J['logRd_disk']], 10 ** th[J['loghz_disk']], 10 ** th[J['logL_bar']]
    ang = [np.degrees(th[J[n]]) for n in ('alpha', 'beta', 'gamma')]
    lMd, lMb, _, _ = [float(v) for v in G['stellar_masses'](
        th[J['f_bar']], th[J['log_mass_to_light_ratio']], Rd, hz, Lb, Lb / 5.0, hz, *ang)]
    R = G['makeRotationMatrix'](*ang)
    Mb = 10.0 ** lMb
    pdisc = {'logM': lMd, 'Rd': Rd, 'hz': hz}
    S, W = G['_BS'], G['_BSW']

    def column(XY):
        X, Y = XY
        x, y, z = R.T @ jnp.stack([jnp.full_like(S, X), jnp.full_like(S, Y), S])
        rho = (G['rho_expdisc'](x, y, z, pdisc)
               + G['T3_density'](x, y, z, Mb, Lb / 5.0, hz, Lb, G['GAMMA_BAR'])
               + G['V4_density'](x, y, z, Mb, G['V4_A'], G['V4_B'], G['V4_L'], G['V4_GAMMA']))
        return jnp.sum(rho * W)

    sig_pix = np.asarray(jax.lax.map(column, (jnp.asarray(_PX), jnp.asarray(_PY))))
    sig_pix = sig_pix.reshape(_nX * _nY, 9).mean(axis=1) / 1e6 / 10 ** th[J['log_mass_to_light_ratio']]
    nb_ = int(G['NUM_VBIN'])
    ok = (_BMAP >= 0) & (_BMAP < nb_)
    return np.bincount(_BMAP[ok], weights=sig_pix[ok], minlength=nb_) / np.maximum(
        np.bincount(_BMAP[ok], minlength=nb_), 1)


if A.param_only:
    results = json.load(open(A.out)) if os.path.isfile(A.out) else {}
    O = np.asarray(_dd['XY_density_data'])
    for lab, th in POINTS.items():
        sb = param_sigma_bins(th)
        r = np.log10(sb / O)
        results.setdefault(lab, {'theta': th.tolist()})
        results[lab]['param_logratio'] = r.tolist()
        print('%-8s parametric projection vs data: median %+.4f dex, rms %.4f dex, rms after the '
              'median %.4f dex, chi2 at 1%% %.0f'
              % (lab, np.median(r), np.sqrt(np.mean(r ** 2)), np.std(r),
                 np.sum((r * np.log(10) / G['ERR_DEN_LOGL']) ** 2)), flush=True)
    json.dump(results, open(A.out, 'w'))

    # The LIGHT inside the IFU field, as a fraction of L_BOX.  The tie M_* p_box = M/L L_BOX fixes
    # the model's light in the 20 kpc box whatever M/L is (light = mass / (M/L)), so the field
    # fraction is set by the SHAPES alone.  Scan Rd with everything else held.
    _npix = np.bincount(_BMAP[(_BMAP >= 0) & (_BMAP < len(O))], minlength=len(O))
    _apc2 = _dXp * _dYp * 1e6
    Lf = lambda sb: float(np.sum(sb * _npix) * _apc2)
    print('\nIFU-field light / L_BOX:  N-body (data) %.4f' % (Lf(O) / G['L_BOX']))
    for base in ('T', 'M'):
        for Rd in (3.0, 3.5, 3.93, 4.5, 5.0, 5.5, 6.0, 6.47):
            th = POINTS[base].copy(); th[J['logRd_disk']] = np.log10(Rd)
            sb = param_sigma_bins(th); r = np.log10(sb / O)
            print('   %s, Rd %5.2f kpc: field light / L_BOX %.4f | log(model/data) median %+.4f, '
                  'scatter %.4f dex' % (base, Rd, Lf(sb) / G['L_BOX'], np.median(r), np.std(r)), flush=True)
    sys.exit(0)

results = json.load(open(A.out)) if os.path.isfile(A.out) else {}
meta = dict(nb=os.path.basename(A.nb), norbit=A.norbit, nreal=A.nreal, nboot=A.nboot, burnin=_burn, S2=float(G['S2']),
            names=names, T=T.tolist(), M=M.tolist())
if (results.get('_meta', {}).get('norbit') not in (None, A.norbit)
        or results.get('_meta', {}).get('nb') not in (None, os.path.basename(A.nb))):
    results = {}                               # a different library or likelihood: start over
results['_meta'] = meta
for lab in A.points.split(','):
    th = POINTS[lab]
    t0 = time.time()
    r = decompose(th)
    r['theta'] = th.tolist()
    r['seconds'] = time.time() - t0
    results[lab] = r
    json.dump(results, open(A.out, 'w'))
    print('%-8s logL %9.3f | chi2_den %9.1f  h1 %7.1f  h2 %7.1f  h3 %7.1f  h4 %7.1f  2m_eff %6.1f | '
          'chi2_3d %8.1f | M* %.3e  (%.0f s, peak RSS %.2f GB)'
          % (lab, r['logL'], r['chi2_den'], r['chi2_h1'], r['chi2_h2'], r['chi2_h3'], r['chi2_h4'],
             2 * r['m_eff_kin'], r['chi2_3d'], r['M_star'], r['seconds'], _rss_gb()), flush=True)
