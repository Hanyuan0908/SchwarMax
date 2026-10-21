#!/usr/bin/env python
"""How far is the PARAMETRIC density (the 3-D target) from the N-body itself, mode by mode?

The likelihood asks the orbits to match two densities at once: the observed sky map (the N-body,
projected) at 1%, and the parametric model's own 3-D Fourier block at 2% (A_0) / 4% (harmonics)
of A_0.  If the parametric model is far from the N-body, those two targets disagree.  This script
measures that disagreement directly, with no orbits:

  3-D   the N-body star particles (centred, bar along x, as make_mock_box_light.py prepares them
        for the mock) are binned into the notebook's OWN Fourier cells with the package's binner,
        and compared with the target the builder computes (fourier3d.target_fourier_modes on the
        notebook's density, masses from its M/L tie).  chi2 uses the likelihood's 3-D errors and
        is split into A_0 (ring mass) and the m = 2, 4, 6 harmonics, and by ring.
  sky   the parametric projection / observed map (diagnose_expdisc_bias.py --param-only output),
        split into its ring-averaged (axisymmetric) part and the rest.

    python compare_target_vs_nbody.py [--chain ../example/emcee_aic_exp_field.h5]
           [--nb SchwarMAX_Nbody_AIC_expdisc_field.ipynb] [--param-json ...]
"""
import argparse, json, os, resource, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # the repository: the shared package and example/
P = argparse.ArgumentParser()
P.add_argument('--chain', default=os.path.join(ROOT, 'example', 'emcee_aic_exp_field.h5'))
P.add_argument('--nb', default=os.path.join(HERE, 'SchwarMAX_Nbody_AIC_expdisc_field.ipynb'))
P.add_argument('--nbody', default=os.path.join(ROOT, 'example', 'Bar_model_TG21', 'model', 't_t0_7'))
P.add_argument('--param-json', default=None,
               help='diagnose_expdisc_bias.py --param-only output, for the on-sky split (optional)')
P.add_argument('--cap', type=float, default=8.0)
A = P.parse_args()


def _rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3


def _watch():
    while True:
        time.sleep(0.25)
        if _rss() > A.cap:
            sys.stderr.write('WATCHDOG: RSS %.1f GB > %.1f\n' % (_rss(), A.cap)); os._exit(42)


threading.Thread(target=_watch, daemon=True).start()
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np, emcee

G = {'__name__': '__main__'}
exec('''
import os, sys, time
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
''' % (ROOT, ROOT), G)
nb = json.load(open(A.nb))
for i in (4, 6, 8, 10):
    s = ''.join(nb['cells'][i]['source']).replace('N_ORBIT = 7_500', 'N_ORBIT = 200')
    if i == 10:
        s = s[:s.index('t0 = time.time()')]
    exec(compile(s, '<cell %d>' % i, 'exec'), G)
jax, jnp = G['jax'], G['jnp']
from fourier3d import fourier_edges, target_fourier_modes, bin_fourier_rings_edges
names = list(G['param_names'])
J = {n: names.index(n) for n in names}

# ---------------------------------------------------------------- the points
b = emcee.backends.HDFBackend(A.chain, read_only=True)
burn = int(3 * np.nanmax(b.get_autocorr_time(tol=0)))
S, L = b.get_chain(discard=burn, flat=True), b.get_log_prob(discard=burn, flat=True)
B = S[int(np.argmax(L))].copy()
POINTS = {'B (best fit)': B}
for Lk in (4.87,):
    t = B.copy(); t[J['logL_bar']] = np.log10(Lk); POINTS['B, L_bar %g' % Lk] = t
POINTS['T (truth)'] = np.asarray(G['theta_true'], float)

# ---------------------------------------------------------------- the notebook's Fourier cells
Re = np.asarray(fourier_edges(G['F3D_R'][0], G['F3D_R'][1], G['F3D_R'][2], log=G['F3D_LOG'],
                              p=G['F3D_POW_R'], close_inner=G['F3D_CLOSE_INNER']))
ze = np.asarray(fourier_edges(G['F3D_Z'][0], G['F3D_Z'][1], G['F3D_Z'][2], log=G['F3D_LOG'],
                              p=G['F3D_POW_Z'], close_inner=G['F3D_CLOSE_INNER']))
nR, nZ, MM = len(Re) - 1, len(ze) - 1, G['F3D_MMAX']
NMODE = 1 + 2 * (MM // 2)
print('Fourier cells: R edges %s kpc\n               |z| edges %s kpc  -> %d cells x %d modes'
      % (np.round(Re, 2), np.round(ze, 2), nR * nZ, NMODE))


def target(theta):
    """fourier3d.target_fourier_modes on the notebook's density, masses from its M/L tie."""
    th = np.asarray(theta, float)
    Rd, hz, Lb = 10 ** th[J['logRd_disk']], 10 ** th[J['loghz_disk']], 10 ** th[J['logL_bar']]
    ang = [np.degrees(th[J[n]]) for n in ('alpha', 'beta', 'gamma')]
    lMd, lMb, Ms, _ = [float(v) for v in G['stellar_masses'](th[J['f_bar']], th[J['log_mass_to_light_ratio']],
                                                            Rd, hz, Lb, Lb / 5.0, hz, *ang)]
    Mb = 10.0 ** lMb
    pd = {'logM': lMd, 'Rd': Rd, 'hz': hz}

    def rho_Rz(R, z, phi, _):
        x, y = R * jnp.cos(phi), R * jnp.sin(phi)
        return (G['rho_expdisc'](x, y, z, pd) + G['T3_density'](x, y, z, Mb, Lb / 5.0, hz, Lb, G['GAMMA_BAR'])
                + G['V4_density'](x, y, z, Mb, G['V4_A'], G['V4_B'], G['V4_L'], G['V4_GAMMA']))
    return np.asarray(target_fourier_modes(jnp.asarray(Re), jnp.asarray(ze), rho_Rz, {}, MM,
                                           n_phi=G['F3D_NPHI_T'], n_sub=G['F3D_NSUB_T'])), Ms


# ---------------------------------------------------------------- the N-body, as the mock was made
import agama
from astropy.constants import G as Gc
import astropy.units as u
agama.setUnits(length=1, velocity=1, mass=1)
w0, m = agama.readSnapshot(A.nbody)
m = m * (1 / ((Gc * u.Msun).to(u.kpc * (u.km / u.s) ** 2))).value
st = m != np.unique(m)[-1]
for r_ap in (10.0, 5.0, 3.0, 2.0):
    mc = st & (np.hypot(w0[:, 0], w0[:, 1]) < r_ap)
    for col in range(3):
        w0[:, col] -= np.sum(w0[mc, col] * m[mc]) / np.sum(m[mc])
x, y, z, m = w0[st, 0], w0[st, 1], w0[st, 2], m[st]
del w0
Rc, ph = np.hypot(x, y), np.arctan2(y, x)
ann = np.arange(1, 5, 0.25)
bar = float(np.mean([0.5 * np.arctan2(np.mean(np.sin(2 * ph[s])), np.mean(np.cos(2 * ph[s])))
                     for s in ((Rc > ann[i]) & (Rc < ann[i + 1]) for i in range(len(ann) - 1))]))
c, s_ = np.cos(-bar), np.sin(-bar)
xb, yb = x * c - y * s_, x * s_ + y * c
NB = np.asarray(bin_fourier_rings_edges(jnp.asarray(xb, jnp.float32), jnp.asarray(yb, jnp.float32),
                                        jnp.asarray(z, jnp.float32), jnp.asarray(m, jnp.float32),
                                        jnp.asarray(Re), jnp.asarray(ze), MM, nR * nZ), float)
print('N-body: %d stars, %.3e Msun; %.3e Msun (%.1f%%) inside the Fourier box; peak RSS %.1f GB'
      % (len(m), m.sum(), NB[:, 0].sum(), 100 * NB[:, 0].sum() / m.sum(), _rss()))

# ---------------------------------------------------------------- compare
cols = {'A_0': [0], 'm=2': [1, 2], 'm=4': [3, 4], 'm=6': [5, 6]}
ring = np.arange(nR * nZ) % nR
for lab, th in POINTS.items():
    T, Ms = target(th)
    sig = np.concatenate([G['SIG_FRAC_A0'] * np.abs(T[:, :1]),
                          np.repeat(G['SIG_FRAC_An'] * np.abs(T[:, :1]), NMODE - 1, axis=1)], axis=1) + 1e-30
    chi = ((NB - T) / sig) ** 2
    print('\n=== %s: L_bar %.2f, R_d %.2f, f_bar %.3f; model M_* %.3e (in the box %.3e), N-body in the box %.3e'
          % (lab, 10 ** th[J['logL_bar']], 10 ** th[J['logRd_disk']], th[J['f_bar']], Ms, T[:, 0].sum(), NB[:, 0].sum()))
    print('   chi2 of the TARGET against the N-body, with the likelihood\'s 3-D errors (630 constraints):')
    print('   ' + '   '.join('%s %.0f' % (k, chi[:, v].sum()) for k, v in cols.items())
          + '   | total %.0f' % chi.sum())
    layer = np.arange(nR * nZ) // nR
    print('   by |z| layer (summed over R):   |z| range | A_0 model/N-body | chi2 A_0 | A_0 model/N-body in R<2, 2-5, 5-10 kpc')
    for l in range(nZ):
        k = layer == l
        sub = []
        for lo_, hi_ in ((0, 2), (2, 5), (5, 10)):
            kk = k & (Re[ring] >= lo_) & (Re[ring + 1] <= hi_ + 1e-6)
            sub.append(T[kk, 0].sum() / NB[kk, 0].sum())
        print('      %4.2f-%4.2f | %6.3f | %8.0f | %6.3f %6.3f %6.3f'
              % (ze[l], ze[l + 1], T[k, 0].sum() / NB[k, 0].sum(), chi[k, 0].sum(), *sub))
    print('   by ring (summed over |z|):  R range | A_0 model/N-body | M(<R) model/N-body | |A_2|/A_0 model, N-body | chi2 A_0, m=2, m=4+6')
    cumT = cumN = 0.0
    for r in range(nR):
        k = ring == r
        a0T, a0N = T[k, 0].sum(), NB[k, 0].sum()
        cumT += a0T; cumN += a0N
        a2T = np.hypot(T[k, 1].sum(), T[k, 2].sum()) / a0T
        a2N = np.hypot(NB[k, 1].sum(), NB[k, 2].sum()) / a0N
        print('      %5.2f-%5.2f | %6.3f | %6.3f | %5.3f %5.3f | %8.0f %8.0f %8.0f'
              % (Re[r], Re[r + 1], a0T / a0N, cumT / cumN, a2T, a2N,
                 chi[k][:, 0].sum(), chi[k][:, 1:3].sum(), chi[k][:, 3:].sum()))

# ---------------------------------------------------------------- the sky: ring-averaged part vs the rest
if A.param_json and os.path.isfile(A.param_json):
    pj = json.load(open(A.param_json))
    dd = G['dict_data']
    bm = np.asarray(dd['bin_mapping'])[:int(np.prod(np.asarray(dd['nX_nY'])))].astype(int)
    nb_ = int(G['NUM_VBIN'])
    Xg, Yg = np.asarray(dd['X_regular_grid']), np.asarray(dd['Y_regular_grid'])
    cnt = np.bincount(bm, minlength=nb_)
    Xb, Yb = np.bincount(bm, Xg, nb_) / cnt, np.bincount(bm, Yg, nb_) / cnt
    al, inc = POINTS['B (best fit)'][J['alpha']], POINTS['B (best fit)'][J['beta']]
    xp, yp = np.cos(al) * Xb - np.sin(al) * Yb, np.sin(al) * Xb + np.cos(al) * Yb
    Rdep = np.hypot(xp, yp / np.cos(inc))
    edges = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 13, 100])
    rr = np.digitize(Rdep, edges) - 1
    print('\n=== the SKY: log10(parametric projection / observed map), per Voronoi bin')
    print('   point            rms     ring-averaged part   the rest (non-axisymmetric)')
    for key in ('B', 'B+L4.87', 'T'):
        if key not in pj or 'param_logratio' not in pj[key]:
            continue
        r = np.asarray(pj[key]['param_logratio'])
        ringmean = np.array([r[rr == k].mean() if np.any(rr == k) else 0.0 for k in range(len(edges) - 1)])
        axi = ringmean[rr]
        print('   %-12s  %.4f dex   %.4f dex            %.4f dex'
              % (key, np.sqrt(np.mean(r ** 2)), np.sqrt(np.mean(axi ** 2)), np.sqrt(np.mean((r - axi) ** 2))))
    print('   ring means for B:', ' '.join('%.1f-%.1f:%+.3f' % (edges[k], edges[k + 1], m_)
                                         for k, m_ in enumerate(
        [np.asarray(pj['B']['param_logratio'])[rr == k].mean() if np.any(rr == k) else np.nan
         for k in range(len(edges) - 1)])))
print('\npeak RSS %.1f GB' % _rss())
