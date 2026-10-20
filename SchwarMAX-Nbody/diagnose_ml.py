#!/usr/bin/env python
"""Why does the mock fit come out at a low M/L while photometry and kinematics both fit well?

In the AIC pipeline M/L enters in ONE place (model.py, build_model_acc_fourier):
    y_xy      = light * M/L                       2-D surface-density target    (line 2290)
    gamma_kin ~ y_xy per bin                      kinematic normalisation       (line 2326)
while the POTENTIAL and the 3-D density target come from logM_disk / logM_bar through
density_func and potential_func, which never see M/L.  So a joint rescaling
(M/L, all orbit weights) -> f*(M/L, weights) leaves chi2_den and h1-h4 unchanged, and the only
thing tying M/L to the potential's stellar mass is the 3-D block inside the NNLS -- whose chi2
is NOT in logL.

  --part A   (cheap, no orbits)  project density_func(theta) on the sky exactly as the
             integrator does (X = R @ x, sky = X[0], X[1], LOS = X[2]), area-average it over
             each Voronoi bin, and divide by the observed Sigma.  That ratio is the M/L the
             POTENTIAL'S stars imply; compare it with the M/L PARAMETER.
  --part B   (orbit model)  fixed-theta scan of log M/L at the best fit, plus the generating
             truth: logL and its terms, and the UNSCORED 3-D mismatch.
  --part C   the same scan in the CURRENT 12-parameter notebook, where the stellar mass is
             derived from M/L (M_star * p_box = M/L * L_BOX).  Parts A and B run on the archived
             13-parameter notebook they were written for.

The generating truth is `example_schwarmax.ipynb` cell 7 (= example_analysis GROUND_TRUTH):
log M/L = 0.  NOT the AIC notebook's `ground_truth`, which is the old pipeline's minimiser.

    python diagnose_ml.py --part A
    python diagnose_ml.py --part B --cap 9
    python diagnose_ml.py --part C --cap 9.5
"""
import argparse, json, os, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # the repository: the shared package (model.py, ...) and example/ (mock data, N-body snapshot, run outputs)
P = argparse.ArgumentParser()
P.add_argument('--part', required=True, choices=('A', 'B', 'C'))
P.add_argument('--notebook', default=None,
               help='default: archive/SchwarMAX_Nbody_AIC_13param.ipynb for A/B, the live one for C')
P.add_argument('--cache', default=os.path.join(ROOT, 'example', 'datavsmodel_bestfit_aic.npz'),
               help='best-fit theta is read from the data-vs-model cache')
P.add_argument('--scan', default=None,
               help='log M/L values to scan; default -0.60,...,0.15 (B, best fit added) or '
                    '-0.475,...,0.30 (C)')
P.add_argument('--cap', type=float, default=9.0, help='abort if RSS exceeds this many GB')
A = P.parse_args()

PEAK = [0.0]


def _rss_gb():
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024.0 ** 3 if sys.platform == 'darwin' else r / 1024.0 ** 2


def _watch():
    while True:
        time.sleep(0.25)
        g = _rss_gb()
        if g > A.cap:
            sys.stderr.write('\n*** WATCHDOG: RSS %.2f GB > cap %.1f GB -- ABORTING\n' % (g, A.cap))
            sys.stderr.flush()
            os._exit(42)


threading.Thread(target=_watch, daemon=True).start()
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np

# ---------------------------------------------------------------- the three parameter vectors
C = np.load(A.cache)
TH_BEST = np.asarray(C['theta'], float)
LAM = float(TH_BEST[12])
D2R = np.pi / 180
TH_TRUE = np.array([10.76, 10.83, 10.2, np.log10(7.76), np.log10(12.0), np.log10(0.9),
                    np.log10(4.87), 25 * D2R, 65 * D2R, 45 * D2R, 0.0, 1.40, LAM])
TH_OLDMIN = np.array([10.6750, 10.7179, 10.1961, 0.9876, 0.9713, -0.2564, 0.7457, 0.4649,
                      1.1576, 1.0316, -0.1660, 1.3822, LAM])   # the AIC notebook's 'ground_truth'
NAMES = ['logM_10kpc', 'logM_disk', 'logM_bar', 'logC_halo', 'logRs_disk', 'logHs_disk',
         'logL_bar', 'alpha', 'beta', 'gamma', 'logML', 'logOmega', 'log10_lam']

# ---------------------------------------------------------------- exec the notebook's own cells
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
from potentials import (NFW_potential, MiyamotoNagai_potential, MiyamotoNagai_density,
                        T3_density, T3_potential, V4_density, V4_potential)
path = %r + "/"
''' % (ROOT, ROOT)
G = {'__name__': '__main__'}
exec(compile(HEADER, '<header>', 'exec'), G)
NB = A.notebook or os.path.join(HERE, *(('SchwarMAX_Nbody_AIC.ipynb',) if A.part == 'C' else
                                         ('archive', 'SchwarMAX_Nbody_AIC_13param.ipynb')))
nb = json.load(open(NB))
print('notebook: %s' % os.path.relpath(NB, HERE))
for i in ((4, 6) if A.part == 'A' else (4, 6, 8)):
    exec(compile(''.join(nb['cells'][i]['source']), '<notebook cell %d>' % i, 'exec'), G)
jax, jnp = G['jax'], G['jnp']
dd = G['dict_data']
O = np.asarray(dd['XY_density_data'], float)         # observed Sigma per bin [Msun/pc^2, "light"]
npb = np.asarray(dd['num_per_bin'], float)


def baryon_params(th):
    """run_model's own theta -> params_baryon_rho mapping (cell 8)."""
    L_bar, Hs = 10.0 ** th[6], 10.0 ** th[5]
    return {'logM_disc': th[1], 'Rs_disc': 10 ** th[4], 'Hs_disc': Hs, 'logM_bar': th[2],
            'L_bar': L_bar, 'a_bar': L_bar / 5.0, 'b_bar': Hs,
            'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0, 'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}


# ================================================================= PART A
def project(th, n_los=401, s_max=40.0, n_sub=2):
    """Sigma of density_func(theta) per Voronoi bin, in Msun/pc^2, area-averaged like the
    integrator: X = R @ x with R = makeRotationMatrix(alpha, beta, gamma), sky = (X0, X1),
    LOS = X2.  So x = R.T @ (X0, X1, s)."""
    pb = baryon_params(th)
    R = np.asarray(G['makeRotationMatrix'](th[7] / D2R, th[8] / D2R, th[9] / D2R))
    Xg, Yg = np.asarray(dd['X_regular_grid'], float), np.asarray(dd['Y_regular_grid'], float)
    nX, nY = [int(v) for v in dd['nX_nY']]
    dX = (dd['X_minmax'][1] - dd['X_minmax'][0]) / nX
    dY = (dd['Y_minmax'][1] - dd['Y_minmax'][0]) / nY
    a = 5.0
    s = s_max * np.sinh(np.linspace(-a, a, n_los)) / np.sinh(a)
    off = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    rho = jax.jit(lambda x, y, z: G['density_func'](x, y, z, pb))
    pix = np.zeros(Xg.size)
    for ox in off:
        for oy in off:
            X = np.broadcast_to((Xg + ox * dX)[:, None], (Xg.size, n_los))
            Y = np.broadcast_to((Yg + oy * dY)[:, None], (Xg.size, n_los))
            S = np.broadcast_to(s[None, :], (Xg.size, n_los))
            sky = np.stack([X.ravel(), Y.ravel(), S.ravel()])          # (3, N)
            x = R.T @ sky
            r = np.asarray(rho(jnp.asarray(x[0]), jnp.asarray(x[1]), jnp.asarray(x[2])))
            pix += np.trapezoid(r.reshape(Xg.size, n_los), s, axis=1)
    pix /= n_sub ** 2
    bm = np.asarray(dd['bin_mapping'])[:-1]
    per_bin = np.bincount(bm, weights=pix, minlength=O.size) / np.maximum(npb, 1)
    return per_bin / 1e6                                                # kpc^-2 -> pc^-2


def part_A():
    print('PART A -- the M/L the POTENTIAL\'S stellar density implies, vs the M/L PARAMETER')
    print('   implied M/L = sum(projected density_func) / sum(observed Sigma) over the field\n')
    # convergence of the projection itself, at the truth
    t0 = time.time()
    p1 = project(TH_TRUE)
    p2 = project(TH_TRUE, n_los=801, s_max=80.0, n_sub=3)
    print('   projection converged: finer LOS/longer LOS/3x3 sub-pixels change the total by '
          '%.2e (%.0f s)\n' % (p2.sum() / p1.sum() - 1, time.time() - t0))
    print('   %-22s %9s %9s %10s %10s %12s' % ('theta', 'log M/L', 'log M/L', 'ratio', 'shape rms',
                                              'logM_disk'))
    print('   %-22s %9s %9s %10s %10s %12s' % ('', 'PARAM', 'IMPLIED', 'impl/param', '[dex]', ''))
    for lab, th in (('generating TRUTH', TH_TRUE), ('AIC best fit', TH_BEST),
                    ('old minimiser', TH_OLDMIN)):
        p = project(th)
        w = npb                                       # area weights: num_per_bin pixels
        implied = float(np.sum(p * w) / np.sum(O * w))
        lr = np.log10(p / O) - np.log10(implied)
        print('   %-22s %+9.3f %+9.3f %10.3f %10.4f %12.3f'
              % (lab, th[10], np.log10(implied), implied / 10 ** th[10],
                 np.sqrt(np.mean(lr ** 2)), th[1]))
    print('\n   "shape rms" = rms of log10(projected/observed) after removing the global factor.')
    print('   If IMPLIED != PARAM, the 2-D target (light*M/L) and the potential/3-D target')
    print('   (density_func) describe DIFFERENT stellar masses, and the NNLS must arbitrate.')


# ================================================================= PART B
def make_evaluate():
    rm = G['run_model']
    NMODE, S2 = G['F3D_NMODE'], G['S2']
    EPS = float(G['EPSILON'])
    sig_l = (G['ERR_DEN_LOGL'] * O) / (O * np.log(10) + EPS)

    def evaluate(th):
        res = rm(jnp.asarray(th), dd, G['NUM_VBIN'], G['RZ_TOT'])
        d0 = np.asarray(res['surface_density'])[0]
        c_den = float(np.nansum(((np.log10(d0 + EPS) - np.log10(O + EPS)) / sig_l) ** 2))
        c_kin = 0.0
        for k in (1, 2, 3, 4):
            c_kin += float(np.nansum(((np.asarray(res['h%d' % k])[0] - np.asarray(dd['h%d_data' % k]))
                                      / (np.asarray(dd['h%d_data_err' % k]) + EPS)) ** 2))
        me = float(res['m_eff_kin'])
        logL = -0.5 * (c_kin + 2 * me + c_den) / S2
        # the 3-D block: target vs what the solved weights put there (both / mean_mass_per_orb)
        t3 = np.asarray(res['density_3d']); m3 = np.asarray(res['density_3d_model'])[0]
        box = float(m3[0::NMODE].sum() / t3[0::NMODE].sum())   # m=0 column = ring masses
        ring = m3[0::NMODE] / t3[0::NMODE]
        out = dict(logL=logL, den=c_den, kin=c_kin, meff=me, box=box,
                   ring_lo=float(np.percentile(ring, 16)), ring_hi=float(np.percentile(ring, 84)),
                   flux=float(O.sum() / d0.sum()))
        if 'M_star' in res:                       # the 12-parameter notebook derives the masses
            out.update(logMs=float(np.log10(res['M_star'])), p_box=float(res['p_box']))
        return out
    return evaluate, S2


def part_B():
    evaluate, S2 = make_evaluate()

    rows = []
    scan = sorted(set([float(v) for v in (A.scan or '-0.60,-0.45,-0.20,-0.10,0.00,0.15').split(',')]
                      + [round(float(TH_BEST[10]), 4)]))
    print('PART B -- fixed-theta scan of log M/L at the AIC best fit (everything else frozen)')
    print('   %d orbits, n_boot_meff %d, S2 %.2f.  "3-D box" = orbit mass / target mass in the '
          'Fourier box (UNSCORED)\n' % (G['N_ORBIT'], G['N_BOOT_MEFF'], S2))
    hdr = '   %-18s %8s %9s %9s %9s %8s %8s %16s %8s' % (
        'point', 'log M/L', 'logL', 'chi2_den', 'chi2_kin', 'm_eff', '3-D box', 'ring 16-84%', 'flux')
    print(hdr)
    pts = [('best-fit scan', v, np.r_[TH_BEST[:10], v, TH_BEST[11:]]) for v in scan]
    pts += [('TRUTH', 0.0, TH_TRUE), ('TRUTH, M/L=best', float(TH_BEST[10]),
                                       np.r_[TH_TRUE[:10], TH_BEST[10], TH_TRUE[11:]])]
    for lab, v, th in pts:
        t0 = time.time()
        r = evaluate(th)
        rows.append((lab, v, r))
        print('   %-18s %+8.3f %9.3f %9.1f %9.1f %8.1f %8.3f   %6.3f-%6.3f %8.4f   (%.0f s, peak %.1f GB)'
              % (lab, v, r['logL'], r['den'], r['kin'], r['meff'], r['box'], r['ring_lo'],
                 r['ring_hi'], r['flux'], time.time() - t0, _rss_gb()), flush=True)
    np.save(os.path.join(ROOT, 'example', 'diagnose_ml_partB.npy'),
            np.array([(l, v, r['logL'], r['den'], r['kin'], r['meff'], r['box'], r['flux'])
                      for l, v, r in rows], dtype=object), allow_pickle=True)


# ================================================================= PART C
def field_ratio(th):
    """(model stellar mass projected into the IFU field) / (M/L * light in the field) for a
    12-parameter theta -- independent of M/L once the mass is tied to the box light.  1 means
    the 2-D and 3-D targets agree in the field; <1 means the 3-D target is too light there."""
    sm = G['stellar_masses']
    Lb, Hs, Rs = 10 ** th[5], 10 ** th[4], 10 ** th[3]
    lMd, lMb, _, _ = [float(v) for v in sm(th[1], th[9], Rs, Hs, Lb, Lb / 5, Hs,
                                            th[6] / D2R, th[7] / D2R, th[8] / D2R)]
    t13 = np.array([0, lMd, lMb, 0, th[3], th[4], th[5], th[6], th[7], th[8], 0, 0, 0])
    return float(np.sum(project(t13) * npb) / (10 ** th[9] * np.sum(O * npb)))


def part_C():
    assert G['NDIM'] == 12 and G['param_names'][1] == 'f_bar', 'part C needs the 12-parameter notebook'
    evaluate, S2 = make_evaluate()
    f = lambda md, mb: 2 * 10 ** mb / (10 ** md + 2 * 10 ** mb)
    true12 = np.array([10.76, f(10.83, 10.2), np.log10(7.76), np.log10(12.0), np.log10(0.9),
                       np.log10(4.87), 25 * D2R, 65 * D2R, 45 * D2R, 0.0, 1.40, LAM])
    b = TH_BEST                              # the 13-parameter chain's best fit, re-expressed
    best12 = np.r_[b[0], f(b[1], b[2]), b[3:10], 0.0, b[11], b[12]]
    scan = sorted(float(v) for v in (A.scan or '-0.475,-0.20,-0.10,0.00,0.10,0.20,0.30').split(','))
    print('PART C -- fixed-theta scan of log M/L with the stellar mass TIED to the light')
    print('   M_star * p_box = M/L * L_BOX (L_BOX %.4e, box +-%.0f kpc).  %d orbits, S2 %.2f'
          % (G['L_BOX'], G['BOX_HALF'], G['N_ORBIT'], S2))
    print('   "field" = model mass / (M/L * light) in the IFU field (fixed by the shapes; 1 = the')
    print('   2-D and 3-D targets agree).  "3-D box" = orbit / target mass in the Fourier box.\n')
    rows = []
    for lab, base in (('TRUTH shapes', true12), ('old-best shapes', best12)):
        fr = field_ratio(base)
        print('   %s: f_bar %.3f, logRs %.3f -> field ratio %.3f' % (lab, base[1], base[3], fr))
        print('   %-16s %8s %9s %9s %9s %8s %8s %9s %7s' % ('point', 'log M/L', 'logL', 'chi2_den',
              'chi2_kin', 'm_eff', '3-D box', 'logM_star', 'p_box'))
        for v in scan:
            t0 = time.time()
            th = base.copy(); th[9] = v
            r = evaluate(th)
            rows.append((lab, v, r))
            print('   %-16s %+8.3f %9.3f %9.1f %9.1f %8.1f %8.3f %9.3f %7.4f   (%.0f s, peak %.1f GB)'
                  % (lab, v, r['logL'], r['den'], r['kin'], r['meff'], r['box'], r['logMs'],
                     r['p_box'], time.time() - t0, _rss_gb()), flush=True)
        print()
    np.save(os.path.join(ROOT, 'example', 'diagnose_ml_partC.npy'),
            np.array([(l, v, r['logL'], r['den'], r['kin'], r['meff'], r['box'], r['logMs'],
                       r['p_box']) for l, v, r in rows], dtype=object), allow_pickle=True)


{'A': part_A, 'B': part_B, 'C': part_C}[A.part]()
print('\npeak RSS %.2f GB (cap %.1f)' % (_rss_gb(), A.cap))
