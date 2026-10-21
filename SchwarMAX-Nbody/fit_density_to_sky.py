#!/usr/bin/env python
"""Which bar does the parametric density want if it only has to match the SKY MAP?  No orbits.

The exponential-disc family of the field notebook (exp disc + T3 bar + V4 bulge, b_bar = h_z,
bulge mass = bar mass, V4 size fixed) is projected exactly as diagnose_expdisc_bias.py does
(3x3 sub-pixels, the notebook's line-of-sight nodes), normalised to the data's light in the IFU
field (the field tie), and fitted to the mock's 609 Voronoi bins in log Sigma at 1% -- the
likelihood's own 2-D term.  Viewing angles fixed at the generating values.

  tied   a_bar = L_bar / 5 (the notebook)      free: log R_d, log h_z, log L_bar, f_bar
  free   a_bar free                            free: + log a_bar

Each best fit is then compared with the N-body in 3-D, in the notebook's own Fourier cells:
in-plane ring masses (|z| < 2.45 kpc) and the face-on |A_2|/A_0 per ring.

    python fit_density_to_sky.py
"""
import json, os, resource, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CAP = 10.0


def _rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3


def _watch():
    while True:
        time.sleep(0.25)
        if _rss() > CAP:
            sys.stderr.write('WATCHDOG: RSS %.1f GB\n' % _rss()); os._exit(42)


threading.Thread(target=_watch, daemon=True).start()
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np
from scipy.optimize import minimize

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
nb = json.load(open(os.path.join(HERE, 'SchwarMAX_Nbody_AIC_expdisc_field.ipynb')))
for i in (4, 6, 8, 10):
    s = ''.join(nb['cells'][i]['source']).replace('N_ORBIT = 7_500', 'N_ORBIT = 200')
    if i == 10:
        s = s[:s.index('t0 = time.time()')]
    exec(compile(s, '<cell %d>' % i, 'exec'), G)
jax, jnp = G['jax'], G['jnp']
GT = np.asarray(G['ground_truth'], float)
ANG = [float(np.degrees(GT[k])) for k in (6, 7, 8)]          # alpha, beta, gamma: the mock's own

# ---------------------------------------------------------------- the sky: pixels, bins, data
dd = G['dict_data']
nX, nY = [int(v) for v in np.asarray(dd['nX_nY'])]
dXp = float(np.diff(np.asarray(dd['X_minmax']))[0]) / nX
dYp = float(np.diff(np.asarray(dd['Y_minmax']))[0]) / nY
off = (np.arange(3) - 1) / 3.0
PX = (np.asarray(dd['X_regular_grid'])[:, None, None] + off[None, :, None] * dXp + 0 * off[None, None, :]).ravel()
PY = (np.asarray(dd['Y_regular_grid'])[:, None, None] + 0 * off[None, :, None] + off[None, None, :] * dYp).ravel()
BMAP = np.asarray(dd['bin_mapping'])[:nX * nY].astype(int)
NBIN = int(G['NUM_VBIN'])
O = np.asarray(dd['XY_density_data'], float)
NPIX = np.bincount(BMAP, minlength=NBIN)
LFIELD = float(np.sum(O * NPIX))                  # in Sigma x pixel units: only the ratio matters
R = np.asarray(G['makeRotationMatrix'](*ANG))
S_, SW = G['_BS'], G['_BSW']
XS = jnp.asarray(PX); YS = jnp.asarray(PY)
BM9 = jnp.asarray(np.repeat(BMAP, 9))


@jax.jit
def unit_sky(Rd, hz, Lb, ab):
    """Per-bin mean projected density of a UNIT-mass disc and a unit-mass bar+bulge pair."""
    Rj = jnp.asarray(R)

    def column(XY):
        X, Y = XY
        x, y, z = Rj.T @ jnp.stack([jnp.full_like(S_, X), jnp.full_like(S_, Y), S_])
        rd = G['rho_expdisc'](x, y, z, {'logM': 0.0, 'Rd': Rd, 'hz': hz})
        rb = 0.5 * (G['T3_density'](x, y, z, 1.0, ab, hz, Lb, G['GAMMA_BAR'])
                    + G['V4_density'](x, y, z, 1.0, G['V4_A'], G['V4_B'], G['V4_L'], G['V4_GAMMA']))
        return jnp.stack([jnp.sum(rd * SW), jnp.sum(rb * SW)])
    pix = jax.lax.map(column, (XS, YS))                                   # (n_sub, 2)
    return (jax.ops.segment_sum(pix, BM9, NBIN) / (9.0 * jnp.asarray(NPIX))[:, None])


def model_bins(p, free_a):
    lRd, lhz, lL, fb = p[:4]
    Lb = 10 ** lL
    ab = 10 ** p[4] if free_a else Lb / 5.0
    u = np.asarray(unit_sky(10 ** lRd, 10 ** lhz, Lb, ab), float)
    sig = (1 - fb) * u[:, 0] + fb * u[:, 1]
    return sig * LFIELD / np.sum(sig * NPIX)          # the field tie: model light in field = data's


def chi2(p, free_a):
    if not (0.02 <= p[3] <= 0.9):
        return 1e12
    m = model_bins(p, free_a)
    return float(np.sum(((np.log10(m) - np.log10(O)) * np.log(10) / 0.01) ** 2))


def rms(p, free_a):
    r = np.log10(model_bins(p, free_a) / O)
    return np.sqrt(np.mean(r ** 2))


# ---------------------------------------------------------------- the fits
t0 = time.time()
chi2([np.log10(6.8), np.log10(0.78), np.log10(3.23), 0.373], False)
print('one model evaluation (incl. compile): %.1f s; peak RSS %.1f GB' % (time.time() - t0, _rss()), flush=True)
STARTS = {'chain best fit': (6.81, 0.78, 3.23, 0.373), 'truth': (6.466, 0.748, 4.87, 0.3735)}
FITS = {}
for free_a in (False, True):
    for lab, (Rd, hz, Lb, fb) in STARTS.items():
        p0 = [np.log10(Rd), np.log10(hz), np.log10(Lb), fb] + ([np.log10(Lb / 5.0)] if free_a else [])
        t0 = time.time()
        r = minimize(chi2, p0, args=(free_a,), method='Nelder-Mead',
                     options={'maxiter': 1500, 'xatol': 1e-4, 'fatol': 1.0, 'adaptive': True})
        p = r.x
        key = ('a_bar free' if free_a else 'a_bar = L/5') + ', from ' + lab
        FITS[key] = (p, free_a, r.fun)
        print('%-36s chi2 %9.0f  rms %.4f dex | R_d %.2f  h_z %.3f  L_bar %.2f  a_bar %.2f  f_bar %.3f   (%d evals, %.0f s)'
              % (key, r.fun, rms(p, free_a), 10 ** p[0], 10 ** p[1], 10 ** p[2],
                 10 ** p[4] if free_a else 10 ** p[2] / 5, p[3], r.nfev, time.time() - t0), flush=True)
REF = {'chain best fit (a = L/5)': (np.array([np.log10(6.81), np.log10(0.78), np.log10(3.23), 0.373]), False),
       'truth (a = L/5)':          (np.array([np.log10(6.466), np.log10(0.7477), np.log10(4.87), 0.3735]), False)}
for k, (p, fa) in REF.items():
    print('%-36s chi2 %9.0f  rms %.4f dex   (reference, not fitted)' % (k, chi2(p, fa), rms(p, fa)))

# ---------------------------------------------------------------- 3-D check against the N-body
from fourier3d import fourier_edges, target_fourier_modes, bin_fourier_rings_edges
Re = np.asarray(fourier_edges(G['F3D_R'][0], G['F3D_R'][1], G['F3D_R'][2], log=G['F3D_LOG'],
                              p=G['F3D_POW_R'], close_inner=G['F3D_CLOSE_INNER']))
ze = np.asarray(fourier_edges(G['F3D_Z'][0], G['F3D_Z'][1], G['F3D_Z'][2], log=G['F3D_LOG'],
                              p=G['F3D_POW_Z'], close_inner=G['F3D_CLOSE_INNER']))
nR, nZ, MM = len(Re) - 1, len(ze) - 1, G['F3D_MMAX']
import agama
from astropy.constants import G as Gc
import astropy.units as u
agama.setUnits(length=1, velocity=1, mass=1)
w0, m = agama.readSnapshot(os.path.join(ROOT, 'example', 'Bar_model_TG21', 'model', 't_t0_7'))
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
NB = np.asarray(bin_fourier_rings_edges(jnp.asarray(x * c - y * s_, jnp.float32), jnp.asarray(x * s_ + y * c, jnp.float32),
                                        jnp.asarray(z, jnp.float32), jnp.asarray(m, jnp.float32),
                                        jnp.asarray(Re), jnp.asarray(ze), MM, nR * nZ), float)
ring, layer = np.arange(nR * nZ) % nR, np.arange(nR * nZ) // nR
INPL = layer < 4                                                     # |z| < 2.45 kpc


def target3d(p, free_a):
    lRd, lhz, lL, fb = p[:4]
    Rd, hz, Lb = 10 ** lRd, 10 ** lhz, 10 ** lL
    ab = 10 ** p[4] if free_a else Lb / 5.0
    Mtot = NB[:, 0].sum()                       # normalise the model to the N-body's box mass

    def rho(Rr, zz, pp, _):
        X_, Y_ = Rr * jnp.cos(pp), Rr * jnp.sin(pp)
        return ((1 - fb) * G['rho_expdisc'](X_, Y_, zz, {'logM': 0.0, 'Rd': Rd, 'hz': hz})
                + fb * 0.5 * (G['T3_density'](X_, Y_, zz, 1.0, ab, hz, Lb, G['GAMMA_BAR'])
                              + G['V4_density'](X_, Y_, zz, 1.0, G['V4_A'], G['V4_B'], G['V4_L'], G['V4_GAMMA'])))
    T = np.asarray(target_fourier_modes(jnp.asarray(Re), jnp.asarray(ze), rho, {}, MM, n_phi=32, n_sub=8), float)
    return T * Mtot / T[:, 0].sum()


def a2_profile(T):
    out = []
    for r in range(nR):
        k = (ring == r) & INPL
        out.append(np.hypot(T[k, 1].sum(), T[k, 2].sum()) / T[k, 0].sum())
    return np.array(out)


def peak_half(prof):
    j = int(np.argmax(prof)); h = np.where((np.arange(nR) > j) & (prof < 0.5 * prof[j]))[0]
    return prof[j], 0.5 * (Re[j] + Re[j + 1]), (Re[h[0]] if h.size else np.nan)


a2N = a2_profile(NB)
pN = peak_half(a2N)
print('\n3-D, in the notebook\'s Fourier cells, |z| < %.2f kpc (models normalised to the N-body\'s box mass):' % ze[4])
print('   N-body face-on |A_2|/A_0: peak %.3f in ring at %.2f kpc, below half from %.2f kpc' % pN)
print('   %-36s %-9s %-34s %s' % ('model', 'chi2 A_0', 'in-plane ring mass model/N-body', '|A_2|/A_0 peak, ring R, half-by R'))
rows = [(k, v[0], v[1]) for k, v in FITS.items()] + [(k, v[0], v[1]) for k, v in REF.items()]
for lab, p, fa in rows:
    T = target3d(p, fa)
    sig = G['SIG_FRAC_A0'] * np.abs(T[INPL, 0])
    c0 = np.sum(((NB[INPL, 0] - T[INPL, 0]) / sig) ** 2)
    rr = [T[(ring == r) & INPL, 0].sum() / NB[(ring == r) & INPL, 0].sum() for r in range(nR)]
    pk = peak_half(a2_profile(T))
    print('   %-36s %8.0f  min %.2f max %.2f (1-2 kpc %.2f, 2.4-4.3 kpc %.2f)    %.3f  %.2f  %.2f'
          % (lab, c0, min(rr), max(rr), np.mean(rr[3:5]), np.mean(rr[6:9]), *pk))
print('\npeak RSS %.1f GB' % _rss())
