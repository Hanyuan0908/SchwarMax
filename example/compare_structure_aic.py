#!/usr/bin/env python
"""Recovered stellar structure vs the N-body: is the tied chain's compact disc / short bar
(R_s ~5 kpc, L_bar ~3 kpc against cell 7's 12 and 4.9) closer to the N-body or further from it?

Three things on every panel:
  N-body          the snapshot itself (t_t0_7), centred like make_mock_box_light.py
  cell-7 truth    the parametric model fitted to the N-body (example_schwarmax.ipynb cell 7)
  recovered       the tied chain: a band from --nsamp posterior draws + the median draw,
                  with the stellar masses DERIVED exactly as run_model does (stellar_masses)

Panels: face-on Sigma(R) with components; cylindrical M_*(<R); A2/A0(R); v_c(R) total and by
component (example_analysis Section 3, plus AGAMA potentials built from the star and the halo
particles separately so the N-body's own split is shown); M_DM(<r) (example_analysis cell 22);
and log Sigma_model / Sigma_N-body.

The forward model is not re-implemented: notebook cells 4, 6 and 8 are exec'd for density_func,
potential_func and stellar_masses.  No orbits are integrated.

    python compare_structure_aic.py [--chain ...] [--burnin 300] [--nsamp 100]
"""
import argparse, json, os, pickle, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # the package (model.py, utils.py, ...); notebooks, data and the N-body snapshot sit here in example/
P = argparse.ArgumentParser()
P.add_argument('--chain', default=None, help='default the newest *_aic_tied results/checkpoint')
P.add_argument('--burnin', type=int, default=300)
P.add_argument('--nsamp', type=int, default=100, help='posterior draws for the bands')
P.add_argument('--nbody', default=os.path.join(HERE,
                                                'Bar_model_TG21', 'model', 't_t0_7'))
P.add_argument('--outdir', default=os.path.join(HERE, 'figures'))
P.add_argument('--cap', type=float, default=6.0, help='abort if RSS exceeds this many GB')
A = P.parse_args()


def _rss_gb():
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024.0 ** 3 if sys.platform == 'darwin' else r / 1024.0 ** 2


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

# ---------------------------------------------------------------- the chain
if A.chain is None:
    for k in ('results', 'checkpoint'):
        A.chain = os.path.join(HERE, 'mcmc_%s_aic_tied.pkl' % k)
        if os.path.isfile(A.chain):
            break
d = pickle.load(open(A.chain, 'rb'))
S = np.asarray(d['chain']) if 'chain' in d else np.stack(d['all_samples'], 0)
assert S.shape[-1] == 12, 'needs the 12-parameter (tied) layout'
flat = S[A.burnin:].reshape(-1, 12)
rng = np.random.default_rng(0)
draws = flat[rng.choice(flat.shape[0], size=min(A.nsamp, flat.shape[0]), replace=False)]
med = np.median(flat, axis=0)
print('chain %s: %d steps, burn-in %d -> %d samples; %d draws for the bands'
      % (os.path.relpath(A.chain, HERE), S.shape[0], A.burnin, flat.shape[0], len(draws)))

# ---------------------------------------------------------------- the notebook's own functions
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
nb = json.load(open(os.path.join(HERE, 'SchwarMAX_Nbody_AIC.ipynb')))
for i in (4, 6, 8):
    src = ''.join(nb['cells'][i]['source'])
    if i == 6:                                   # only the bin geometry is needed, not a library
        assert 'N_ORBIT = 7_500' in src
        src = src.replace('N_ORBIT = 7_500', 'N_ORBIT = 500')
    exec(compile(src, '<notebook cell %d>' % i, 'exec'), G)
jax, jnp = G['jax'], G['jnp']
KPCGYR_TO_KMS = float(G['KPCGYR_TO_KMS'])
assert G['param_names'][1] == 'f_bar'


def _halo(logM_enc, log_c):
    lM, lR = G['logMenc_logc_to_logM_logRs'](logM_enc, log_c, r_enc=10.0, Delta=200., rho_crit=277.54)
    return {'logM': float(lM), 'Rs': float(10 ** lR), 'a': 1.0, 'b': 1.0, 'c': 1.0,
            'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0, 'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}


def _baryon(logM_disc, logM_bar, Rs, Hs, L_bar):
    return {'logM_disc': float(logM_disc), 'Rs_disc': float(Rs), 'Hs_disc': float(Hs),
            'logM_bar': float(logM_bar), 'L_bar': float(L_bar), 'a_bar': float(L_bar) / 5.0,
            'b_bar': float(Hs), 'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
            'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}


def unpack12(th):
    """run_model's mapping (cell 8), stellar masses DERIVED from M/L."""
    Rs, Hs, Lb = 10 ** th[3], 10 ** th[4], 10 ** th[5]
    lMd, lMb, Ms, pb = [float(v) for v in G['stellar_masses'](
        th[1], th[9], Rs, Hs, Lb, Lb / 5.0, Hs, th[6] * 180 / np.pi, th[7] * 180 / np.pi,
        th[8] * 180 / np.pi)]
    return _halo(th[0], th[2]), _baryon(lMd, lMb, Rs, Hs, Lb), Ms


# cell 7 of example_schwarmax.ipynb: the parametric fit to the N-body
PH_TRUE = _halo(10.76, np.log10(7.76))
PB_TRUE = _baryon(10.83, 10.2, 12.0, 0.9, 4.87)

# ---------------------------------------------------------------- face-on model profiles
EDGES = np.concatenate([[0.0], np.geomspace(0.1, 25.0, 46)])          # annuli [kpc]
RC = 0.5 * (EDGES[1:] + EDGES[:-1])
AREA = np.pi * (EDGES[1:] ** 2 - EDGES[:-1] ** 2)                     # kpc^2
NSUB, NPHI = 6, 32
_u = (np.arange(NSUB) + 0.5) / NSUB                                    # uniform in R^2 = area
RSUB = np.sqrt(EDGES[:-1, None] ** 2 + _u[None, :] * (EDGES[1:, None] ** 2 - EDGES[:-1, None] ** 2))
PHI = np.arange(NPHI) * 2 * np.pi / NPHI
_a = 7.0
ZS = 40.0 * np.sinh(np.linspace(-_a, _a, 241)) / np.sinh(_a)
RR, PP, ZZ = np.meshgrid(RSUB.ravel(), PHI, ZS, indexing='ij')         # (nR*NSUB, NPHI, nz)
XJ, YJ, ZJ = (jnp.asarray(v.ravel()) for v in (RR * np.cos(PP), RR * np.sin(PP), ZZ))
GB, V4 = G['GAMMA_BAR'], (G['V4_A'], G['V4_B'], G['V4_L'], G['V4_GAMMA'])


@jax.jit
def _sigma(logM_disc, Rs, Hs, M_bar, L_bar):
    """Face-on Sigma(R_sub, phi) [Msun/kpc^2] of disc and of bar+bulge, z-integrated."""
    mn = {'logM_disc': logM_disc, 'Rs_disc': Rs, 'Hs_disc': Hs, 'x_origin': 0.0, 'y_origin': 0.0,
          'z_origin': 0.0, 'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}
    rd = G['MiyamotoNagai_density'](XJ, YJ, ZJ, mn)
    rb = (G['T3_density'](XJ, YJ, ZJ, M_bar, L_bar / 5.0, Hs, L_bar, GB)
          + G['V4_density'](XJ, YJ, ZJ, M_bar, *V4))
    sh = (RSUB.size, NPHI, ZS.size)
    return (jnp.trapezoid(rd.reshape(sh), jnp.asarray(ZS), axis=2),
            jnp.trapezoid(rb.reshape(sh), jnp.asarray(ZS), axis=2))


def profiles(pb):
    sd, sb = (np.asarray(v).reshape(len(RC), NSUB, NPHI).mean(axis=1) for v in
              _sigma(pb['logM_disc'], pb['Rs_disc'], pb['Hs_disc'], 10 ** pb['logM_bar'], pb['L_bar']))
    tot = sd + sb
    A0 = tot.mean(axis=1)
    A2 = np.abs(np.mean(tot * np.exp(2j * PHI)[None, :], axis=1))
    return dict(Sig=A0 / 1e6, Sig_d=sd.mean(1) / 1e6, Sig_b=sb.mean(1) / 1e6,   # Msun/pc^2
                Menc=np.cumsum(A0 * AREA), A2A0=A2 / A0)


# ---------------------------------------------------------------- rotation curves (cell 17)
R_VC = np.linspace(0.05, 20.0, 80)
PHI_VC = np.linspace(0.0, 2 * np.pi, 10)          # example_analysis: 10 angles incl. both ends
NOHALO = dict(PH_TRUE, logM=-30.0)


@jax.jit
def _vc(pb, ph):
    def phi_at(r, ang):
        return G['potential_func'](r * jnp.cos(ang), r * jnp.sin(ang), 0.0, pb, ph)
    def one(r, ang):
        dR = 1e-3
        return (phi_at(r + dR, ang) - phi_at(r - dR, ang)) / (2 * dR)
    dphi = jax.vmap(lambda ang: jax.vmap(lambda r: one(r, ang))(jnp.asarray(R_VC)))(jnp.asarray(PHI_VC))
    return jnp.sqrt(jnp.maximum(jnp.asarray(R_VC) * dphi, 0.0)).mean(axis=0) * KPCGYR_TO_KMS


def vcurves(pb, ph):
    tot = np.asarray(_vc(pb, ph))
    star = np.asarray(_vc(pb, dict(ph, logM=-30.0)))
    halo = np.sqrt(np.maximum(tot ** 2 - star ** 2, 0.0))
    return tot, star, halo


R_ENC = np.logspace(np.log10(0.3), np.log10(30), 200)


def nfw_menc(ph):
    x = R_ENC / ph['Rs']
    return 10 ** ph['logM'] * (np.log1p(x) - x / (1 + x))


# ---------------------------------------------------------------- run the model side
t0 = time.time()
out = {}
for lab, (ph, pb) in (('true', (PH_TRUE, PB_TRUE)),):
    pr = profiles(pb); vt, vs, vh = vcurves(pb, ph)
    out[lab] = dict(pr, vc=vt, vc_star=vs, vc_halo=vh, Mdm=nfw_menc(ph),
                    Mstar=10 ** pb['logM_disc'] + 2 * 10 ** pb['logM_bar'])
ph, pb, Ms = unpack12(med)
pr = profiles(pb); vt, vs, vh = vcurves(pb, ph)
out['median'] = dict(pr, vc=vt, vc_star=vs, vc_halo=vh, Mdm=nfw_menc(ph), Mstar=Ms)
print('median draw: logM_disc %.3f logM_bar %.3f Rs %.2f Hs %.3f L_bar %.2f M_star %.3e'
      % (pb['logM_disc'], pb['logM_bar'], pb['Rs_disc'], pb['Hs_disc'], pb['L_bar'], Ms))
band = {k: [] for k in ('Sig', 'Menc', 'A2A0', 'vc', 'vc_star', 'vc_halo', 'Mdm', 'Mstar')}
for th in draws:
    ph, pb, Ms = unpack12(th)
    pr = profiles(pb); vt, vs, vh = vcurves(pb, ph)
    for k, v in (('Sig', pr['Sig']), ('Menc', pr['Menc']), ('A2A0', pr['A2A0']), ('vc', vt),
                 ('vc_star', vs), ('vc_halo', vh), ('Mdm', nfw_menc(ph)), ('Mstar', Ms)):
        band[k].append(v)
band = {k: np.asarray(v) for k, v in band.items()}
print('model side: %.0f s, peak RSS %.2f GB' % (time.time() - t0, _rss_gb()))

# ---------------------------------------------------------------- the N-body
import agama
from astropy.constants import G as Gc
import astropy.units as u
agama.setUnits(length=1, velocity=1, mass=1)
w0, m_int = agama.readSnapshot(A.nbody)
mu = (1 / ((Gc * u.Msun).to(u.kpc * (u.km / u.s) ** 2))).value
m = m_int * mu
st = m != np.unique(m)[-1]
for r_ap in (10.0, 5.0, 3.0, 2.0):                 # make_mock_box_light.py's centring
    mc = st & (np.hypot(w0[:, 0], w0[:, 1]) < r_ap)
    for col in range(3):
        w0[:, col] -= np.sum(w0[mc, col] * m[mc]) / np.sum(m[mc])
xs, ys, zs, ms = w0[st, 0], w0[st, 1], w0[st, 2], m[st]
Rn, phn = np.hypot(xs, ys), np.arctan2(ys, xs)
ib = np.digitize(Rn, EDGES) - 1
ok = (ib >= 0) & (ib < len(RC))
M_ann = np.bincount(ib[ok], weights=ms[ok], minlength=len(RC))
A2n = np.abs(np.bincount(ib[ok], weights=ms[ok] * np.cos(2 * phn[ok]), minlength=len(RC))
             + 1j * np.bincount(ib[ok], weights=ms[ok] * np.sin(2 * phn[ok]), minlength=len(RC)))
nbody = dict(Sig=M_ann / AREA / 1e6, Menc=np.cumsum(M_ann), A2A0=A2n / np.maximum(M_ann, 1e-30),
             Mstar=float(ms.sum()))
rh = np.sqrt(np.sum(w0[~st, :3] ** 2, axis=1))
o = np.argsort(rh)
nbody['Mdm'] = np.interp(R_ENC, rh[o], np.cumsum(m[~st][o]))
print('N-body: %d stars %.3e Msun, %d halo particles; peak RSS %.2f GB'
      % (st.sum(), ms.sum(), (~st).sum(), _rss_gb()))


def _vc_agama(pot):
    pos = np.column_stack([R_VC, R_VC * 0, R_VC * 0])
    return np.sqrt(np.maximum(-R_VC * pot.force(pos)[:, 0], 0.0))


pot_tot = agama.Potential(A.nbody + '.ini')                      # example_analysis cell 18
# masses in Msun: setUnits(mass=1) makes AGAMA's mass unit 1 Msun (G = 4.3e-6), while the
# snapshot column is NEMO's G*M -- passing it raw makes every particle 4.3e-6 too light
pot_star = agama.Potential(type='CylSpline', particles=(w0[st, :3], m[st]),
                           symmetry='axisymmetric', gridSizeR=50, gridSizez=50,
                           Rmin=0.05, Rmax=80.0, zmin=0.02, zmax=40.0)
pot_halo = agama.Potential(type='Multipole', particles=(w0[~st, :3], m[~st]),
                           symmetry='spherical', lmax=0, gridSizeR=60, rmin=0.05, rmax=500.0)
nbody.update(vc=_vc_agama(pot_tot), vc_star=_vc_agama(pot_star), vc_halo=_vc_agama(pot_halo))
del w0
print('AGAMA potentials: v_c(8 kpc) total %.1f = sqrt(stars %.1f^2 + halo %.1f^2) -> %.1f km/s'
      % (np.interp(8, R_VC, nbody['vc']), np.interp(8, R_VC, nbody['vc_star']),
         np.interp(8, R_VC, nbody['vc_halo']),
         np.hypot(np.interp(8, R_VC, nbody['vc_star']), np.interp(8, R_VC, nbody['vc_halo']))))

# ---------------------------------------------------------------- numbers
q = lambda a: np.percentile(a, [16, 50, 84], axis=0)
print('\n%-26s %12s %12s %24s' % ('', 'N-body', 'cell-7 truth', 'recovered 16/50/84'))
for Rx in (2.0, 5.0, 7.3, 10.0, 20.0):
    f = lambda v: float(np.interp(Rx, EDGES[1:], v))
    lo, mi, hi = q([f(v) for v in band['Menc']])
    print('M_*(<R=%4.1f kpc) [1e10]   %12.3f %12.3f %10.3f %6.3f %6.3f'
          % (Rx, f(nbody['Menc']) / 1e10, f(out['true']['Menc']) / 1e10, lo / 1e10, mi / 1e10, hi / 1e10))
lo, mi, hi = q(band['Mstar'])
print('M_* total [1e10]           %12.3f %12.3f %10.3f %6.3f %6.3f'
      % (nbody['Mstar'] / 1e10, out['true']['Mstar'] / 1e10, lo / 1e10, mi / 1e10, hi / 1e10))
for lab, prof in (('N-body', nbody['A2A0']), ('cell-7 truth', out['true']['A2A0']),
                  ('recovered median', out['median']['A2A0'])):
    sel = RC < 10
    j = int(np.argmax(prof[sel]))
    half = np.where((RC > RC[j]) & (prof < 0.5 * prof[j]))[0]
    print('A2/A0 %-18s peak %.3f at R = %.2f kpc; falls to half by R = %s kpc'
          % (lab, prof[j], RC[j], '%.2f' % RC[half[0]] if half.size else '>25'))
for Rx in (2.0, 5.0, 8.0, 12.0):
    f = lambda v: float(np.interp(Rx, R_VC, v))
    print('v_c(%4.1f) total/star/halo  N-body %5.1f/%5.1f/%5.1f  truth %5.1f/%5.1f/%5.1f  recovered %5.1f/%5.1f/%5.1f'
          % (Rx, f(nbody['vc']), f(nbody['vc_star']), f(nbody['vc_halo']),
             f(out['true']['vc']), f(out['true']['vc_star']), f(out['true']['vc_halo']),
             np.median([f(v) for v in band['vc']]), np.median([f(v) for v in band['vc_star']]),
             np.median([f(v) for v in band['vc_halo']])))

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R_FIELD, R_BOX = 7.27, 20.0                          # IFU major-axis half-width; sky box
CN, CT, CR = 'k', 'royalblue', 'tomato'
fig, ax = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)


def bandplot(a, x, key, scale=1.0):
    lo, mi, hi = q(band[key] * scale)
    a.fill_between(x, lo, hi, color=CR, alpha=0.25, lw=0, label='recovered 16-84%')
    a.plot(x, mi, color=CR, lw=2, ls='--', label='recovered median')


def marks(a, log=False):
    a.axvspan(0, R_FIELD, color='0.9', zorder=0)
    a.axvline(R_BOX, color='0.6', ls=':', lw=1)


a = ax[0, 0]
marks(a)
a.plot(RC, nbody['Sig'], color=CN, lw=2.5, label='N-body')
a.plot(RC, out['true']['Sig'], color=CT, lw=2, label='cell-7 truth (param. fit)')
a.plot(RC, out['true']['Sig_d'], color=CT, lw=1, ls='--', label='  disc / bar+bulge')
a.plot(RC, out['true']['Sig_b'], color=CT, lw=1, ls=':')
bandplot(a, RC, 'Sig')
a.plot(RC, out['median']['Sig_d'], color=CR, lw=1, ls='--')
a.plot(RC, out['median']['Sig_b'], color=CR, lw=1, ls=':')
a.set(xscale='log', yscale='log', xlim=(0.1, 25), ylim=(1e-1, 1e5), xlabel='R [kpc]',
      ylabel=r'face-on $\Sigma_\star$ [M$_\odot$ pc$^{-2}$]', title='surface density (face-on)')
a.legend(fontsize=8, frameon=False, loc='lower left')

a = ax[0, 1]
marks(a)
a.plot(EDGES[1:], nbody['Menc'] / 1e10, color=CN, lw=2.5, label='N-body')
a.plot(EDGES[1:], out['true']['Menc'] / 1e10, color=CT, lw=2, label='cell-7 truth')
bandplot(a, EDGES[1:], 'Menc', 1e-10)
a.set(xlim=(0, 25), ylim=(0, None), xlabel='R [kpc]', ylabel=r'$M_\star(<R)$ [$10^{10}$ M$_\odot$], cylinder',
      title='enclosed stellar mass')
a.legend(fontsize=8, frameon=False, loc='lower right')

a = ax[0, 2]
marks(a)
a.plot(RC, nbody['A2A0'], color=CN, lw=2.5, label='N-body')
a.plot(RC, out['true']['A2A0'], color=CT, lw=2, label='cell-7 truth')
bandplot(a, RC, 'A2A0')
a.set(xlim=(0, 15), ylim=(0, None), xlabel='R [kpc]', ylabel=r'$A_2/A_0$ (face-on)', title='bar strength')
a.legend(fontsize=8, frameon=False)

a = ax[1, 0]
marks(a)
a.plot(R_VC, nbody['vc'], color=CN, lw=2.5, label='N-body total (CylSpline .ini)')
a.plot(R_VC, nbody['vc_star'], color=CN, lw=1.2, ls='--', label='  N-body stars / halo')
a.plot(R_VC, nbody['vc_halo'], color=CN, lw=1.2, ls=':')
a.plot(R_VC, out['true']['vc'], color=CT, lw=2, label='cell-7 truth')
a.plot(R_VC, out['true']['vc_star'], color=CT, lw=1, ls='--')
a.plot(R_VC, out['true']['vc_halo'], color=CT, lw=1, ls=':')
bandplot(a, R_VC, 'vc')
a.plot(R_VC, np.median(band['vc_star'], 0), color=CR, lw=1, ls='--')
a.plot(R_VC, np.median(band['vc_halo'], 0), color=CR, lw=1, ls=':')
a.set(xlim=(0, 20), ylim=(0, None), xlabel='R [kpc]', ylabel=r'$V_c$ [km/s] (azimuthal mean, z=0)',
      title='rotation curve: total, stars (--), halo (:)')
a.legend(fontsize=8, frameon=False, loc='lower right')

a = ax[1, 1]
a.axvspan(0.3, R_FIELD, color='0.9', zorder=0)
a.plot(R_ENC, nbody['Mdm'], color=CN, lw=2.5, label='N-body halo')
a.plot(R_ENC, out['true']['Mdm'], color=CT, lw=2, label='cell-7 truth (NFW)')
bandplot(a, R_ENC, 'Mdm')
a.set(xscale='log', yscale='log', xlim=(R_ENC[0], R_ENC[-1]), ylim=(1e8, 1e12), xlabel='r [kpc]',
      ylabel=r'$M_{\rm DM}(<r)$ [M$_\odot$]', title='dark matter (example_analysis cell 22)')
a.legend(fontsize=8, frameon=False, loc='lower right')

a = ax[1, 2]
marks(a)
a.axhline(0, color=CN, lw=1)
a.plot(RC, np.log10(out['true']['Sig'] / nbody['Sig']), color=CT, lw=2, label='cell-7 truth / N-body')
lo, mi, hi = q(np.log10(band['Sig'] / nbody['Sig'][None, :]))
a.fill_between(RC, lo, hi, color=CR, alpha=0.25, lw=0, label='recovered / N-body 16-84%')
a.plot(RC, mi, color=CR, lw=2, ls='--')
a.set(xscale='log', xlim=(0.1, 25), ylim=(-1.0, 1.0), xlabel='R [kpc]',
      ylabel=r'$\log_{10}\Sigma_{\rm model}/\Sigma_{\rm N\!-\!body}$', title='face-on surface-density ratio')
a.legend(fontsize=8, frameon=False, loc='lower left')

fig.suptitle('Stellar structure vs the N-body -- %s, %d steps (burn-in %d, %d draws).  '
             'Grey: R < %.1f kpc (IFU major-axis half-width); dotted: the %.0f kpc sky box'
             % (os.path.basename(A.chain), S.shape[0], A.burnin, len(draws), R_FIELD, R_BOX), fontsize=11)
os.makedirs(A.outdir, exist_ok=True)
tag = os.path.basename(A.chain).replace('.pkl', '').replace('mcmc_', '')
outf = os.path.join(A.outdir, 'structure_vs_nbody_%s_%dsteps.png' % (tag, S.shape[0]))
fig.savefig(outf, dpi=110)
np.savez(os.path.join(HERE, 'structure_vs_nbody_%s.npz' % tag),
         RC=RC, EDGES=EDGES, R_VC=R_VC, R_ENC=R_ENC,
         **{'nbody_' + k: v for k, v in nbody.items()},
         **{'true_' + k: v for k, v in out['true'].items()},
         **{'median_' + k: v for k, v in out['median'].items()},
         **{'band_' + k: v for k, v in band.items()})
print('\nwrote %s\npeak RSS %.2f GB (cap %.1f)' % (os.path.relpath(outf, HERE), _rss_gb(), A.cap))
