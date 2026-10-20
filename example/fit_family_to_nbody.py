#!/usr/bin/env python
"""Which stellar family can describe the N-body -- its stellar potential AND its bar AND its light?

No orbits.  The N-body targets, all from the star particles of t_t0_7 (centred as in
make_mock_box_light.py):
    v_c,*(R)    from an AGAMA CylSpline built on the star particles (axisymmetric part)
    log Sigma(R), face-on, log annuli 0.1-15 kpc
    A2/A0(R),   face-on, 0.25 kpc annuli to 8 kpc
are fitted by least squares (sigma: 1% in v_c, 0.05 dex in Sigma, 0.02 in A2/A0).  Every family
has a T3 bar (a_bar = L_bar/5) and a V4 bulge (a = b, gamma 0, L 0.1, as in cell 4):

  Miyamoto-Nagai disc (the notebook's), b_bar = Hs:
    tied           logM_disc, logM_bar, logRs, logHs, logL_bar; V4 mass = M_bar, a = 0.5 kpc
    free_mass      + logM_bul
    free_bulge     + logM_bul, log a_bul
  double-exponential disc rho = M/(4 pi Rd^2 hz) exp(-R/Rd) exp(-|z|/hz) (NGC 1512's):
    exp_tied       logM_disc, logM_bar, logRd, loghz, logL_bar; b_bar = hz; V4 mass = M_bar
    exp_bbar       + log b_bar
    exp_bbar_bul   + log b_bar, logM_bul
  inner shapes free (does the residual live in the bar/bulge SHAPE ties?):
    mn_inner_free  MN disc + T3 with a_bar, b_bar free + V4 with mass, a and inner slope gamma free
    exp_inner_free the same with the double-exponential disc
  two bars (is the N-body bar a concentrated inner part + a long faint outer part?):
    mn_2bar / exp_2bar   disc + TWO aligned T3 bars (each a = L/5, b = h) + V4 bulge, free mass
  the BAR SHAPE free, everything else as in the notebook (bulge tied):
    mn_ab / exp_ab       a_bar, b_bar free (gamma_bar = 1, T3)
    mn_abg / exp_abg     + the needle slope gamma_bar in [-1, 1] (0 = uniform, 1 = triangular)
    mn_abg_T1/T2/T4      the same with the other Dehnen & Aly kernels (T1 = MN-like ... T4)

The exponential disc's potential has no closed form; here it comes from AGAMA's 'Disk'
(DiskAnsatz + Multipole), so this test does not depend on get_phi_m's grid settings.  Cell 7 of
example_schwarmax.ipynb is scored on the same objective.  For every best fit the script also
reports the stellar mass inside the +-20 kpc sky box at the generating angles, i.e. the log M/L
the notebook's tie would assign it (0 = consistent with the light).

    python fit_family_to_nbody.py
"""
import argparse, os, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # the package (model.py, utils.py, ...); notebooks, data and the N-body snapshot sit here in example/
P = argparse.ArgumentParser()
P.add_argument('--nbody', default=os.path.join(HERE,
                                                'Bar_model_TG21', 'model', 't_t0_7'))
P.add_argument('--cap', type=float, default=6.0)
P.add_argument('--outdir', default=os.path.join(HERE, 'figures'))
P.add_argument('--sig', default='0.01,0.05,0.02',
               help='sigma of v_c (fractional), log Sigma (dex), A2/A0; e.g. 0.01,1e9,1e9 = v_c only')
P.add_argument('--families', default=None, help='comma list; default all')
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
            os._exit(42)


threading.Thread(target=_watch, daemon=True).start()
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, ROOT)
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)        # CPU-only fit of a handful of numbers
from scipy.optimize import minimize
from potentials import (MiyamotoNagai_density, MiyamotoNagai_potential, T3_density, T3_potential,
                        V4_density, V4_potential)
from constants import KPCGYR_TO_KMS
from utils import makeRotationMatrix
import dehnen_bar as DB

V4_L, V4_GAMMA, GAMMA_BAR = 0.1, 0.0, 1.0        # cell 4's fixed values
L_BOX, BOX_HALF = 6.4078e10, 20.0                # the notebook's tie (make_mock_box_light.py)

# ---------------------------------------------------------------- N-body targets
import agama
from astropy.constants import G as Gc
import astropy.units as u
agama.setUnits(length=1, velocity=1, mass=1)
w0, m = agama.readSnapshot(A.nbody)
m = m * (1 / ((Gc * u.Msun).to(u.kpc * (u.km / u.s) ** 2))).value      # NEMO G*M -> Msun
mass_tags = np.unique(m)
st = m != mass_tags[-1]
for r_ap in (10.0, 5.0, 3.0, 2.0):
    mc = st & (np.hypot(w0[:, 0], w0[:, 1]) < r_ap)
    for col in range(3):
        w0[:, col] -= np.sum(w0[mc, col] * m[mc]) / np.sum(m[mc])
x, y, z, ms = w0[st, 0], w0[st, 1], w0[st, 2], m[st]
print('N-body stellar species:')
for tag in mass_tags[:-1]:
    k = m[st] == tag
    r3 = np.sqrt(x[k] ** 2 + y[k] ** 2 + z[k] ** 2)
    print('   particle mass %.4g: N = %7d, M = %.3e, r_half(3-D) = %.2f kpc, R_half = %.2f kpc, '
          'median |z| = %.3f kpc' % (tag, k.sum(), ms[k].sum(), np.median(r3),
                                     np.median(np.hypot(x[k], y[k])), np.median(np.abs(z[k]))))
pot_star = agama.Potential(type='CylSpline', particles=(np.column_stack([x, y, z]), ms),
                           symmetry='axisymmetric', gridSizeR=50, gridSizez=50,
                           Rmin=0.05, Rmax=80.0, zmin=0.02, zmax=40.0)
R_VC = np.linspace(0.5, 15.0, 30)
VC_T = np.sqrt(np.maximum(-R_VC * pot_star.force(np.column_stack([R_VC, 0 * R_VC, 0 * R_VC]))[:, 0], 0))

E_S = np.concatenate([[0.0], np.geomspace(0.1, 15.0, 30)])
E_A = np.arange(0.0, 8.01, 0.25)
Rn, phn = np.hypot(x, y), np.arctan2(y, x)


def annuli(E):
    i = np.digitize(Rn, E) - 1
    ok = (i >= 0) & (i < len(E) - 1)
    M = np.bincount(i[ok], weights=ms[ok], minlength=len(E) - 1)
    c = np.bincount(i[ok], weights=ms[ok] * np.cos(2 * phn[ok]), minlength=len(E) - 1)
    s = np.bincount(i[ok], weights=ms[ok] * np.sin(2 * phn[ok]), minlength=len(E) - 1)
    return M / (np.pi * (E[1:] ** 2 - E[:-1] ** 2)), np.hypot(c, s) / np.maximum(M, 1e-30)


SIG_T, _ = annuli(E_S)
_, A2_T = annuli(E_A)
A2_T, E_A = A2_T[1:], E_A[1:]                    # drop R < 0.25: A2 of the centre is noise
RA = 0.5 * (E_A[1:] + E_A[:-1])                  # 31 annuli, 0.25-8 kpc
assert A2_T.size == RA.size
del w0
print('targets: v_c,*(R) at %d radii, Sigma in %d annuli, A2/A0 in %d annuli; peak RSS %.2f GB'
      % (R_VC.size, SIG_T.size, A2_T.size, _rss_gb()))

# ---------------------------------------------------------------- the model (JAX + AGAMA)
NSUB, NPHI = 3, 8                                 # phi in [0, pi): Sigma is bisymmetric
PHI = np.arange(NPHI) * np.pi / NPHI
_a = 7.0
ZP = 30.0 * np.sinh(np.linspace(0, _a, 121)) / np.sinh(_a)          # z >= 0, doubled


def polar(E):
    u_ = (np.arange(NSUB) + 0.5) / NSUB
    Rs_ = np.sqrt(E[:-1, None] ** 2 + u_[None, :] * (E[1:, None] ** 2 - E[:-1, None] ** 2))
    RR, PP, ZZ = np.meshgrid(Rs_.ravel(), PHI, ZP, indexing='ij')
    return [jnp.asarray(v.ravel()) for v in (RR * np.cos(PP), RR * np.sin(PP), ZZ)], Rs_.size


GS, NS = polar(E_S)
GA, NA = polar(E_A)
PV = np.arange(16) * 2 * np.pi / 16
XV = jnp.asarray((R_VC[:, None] * np.cos(PV)[None, :]).ravel())
YV = jnp.asarray((R_VC[:, None] * np.sin(PV)[None, :]).ravel())


def _mn(Md, Rs, Hs):
    return {'logM_disc': jnp.log10(Md), 'Rs_disc': Rs, 'Hs_disc': Hs, 'x_origin': 0.0,
            'y_origin': 0.0, 'z_origin': 0.0, 'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}


def rho_disc(kind, xx, yy, zz, Md, R1, h):
    if kind == 'mn':
        return MiyamotoNagai_density(xx, yy, zz, _mn(Md, R1, h))
    return Md / (4 * jnp.pi * R1 ** 2 * h) * jnp.exp(-jnp.hypot(xx, yy) / R1) * jnp.exp(-jnp.abs(zz) / h)


def rho_bb(xx, yy, zz, Mb, Lb, ab, bb, Mu, au, M2, L2, gu, gb=GAMMA_BAR, btype='T3'):
    return (getattr(DB, btype + '_density')(xx, yy, zz, Mb, ab, bb, Lb, gb)
            + T3_density(xx, yy, zz, M2, L2 / 5.0, bb, L2, GAMMA_BAR)
            + V4_density(xx, yy, zz, Mu, au, au, V4_L, gu))


def phi_bb(xx, yy, zz, Mb, Lb, ab, bb, Mu, au, M2, L2, gu, gb=GAMMA_BAR, btype='T3'):
    return (getattr(DB, btype + '_potential')(xx, yy, zz, Mb, ab, bb, Lb, gb)
            + T3_potential(xx, yy, zz, M2, L2 / 5.0, bb, L2, GAMMA_BAR)
            + V4_potential(xx, yy, zz, Mu, au, au, V4_L, gu))


def _vc2_of(phifun):
    """azimuthally averaged R dPhi/dR at z = 0, in (km/s)^2"""
    gx = jax.vmap(jax.grad(lambda q: phifun(q[0], q[1], 0.0)))(jnp.stack([XV, YV], 1))
    dPdR = (gx[:, 0] * XV + gx[:, 1] * YV) / jnp.hypot(XV, YV)
    return jnp.asarray(R_VC) * dPdR.reshape(R_VC.size, -1).mean(1) * KPCGYR_TO_KMS ** 2


def _make_profiles(kind, btype):
    @jax.jit
    def prof(Md, R1, h, Mb, Lb, ab, bb, Mu, au, M2, L2, gu, gb):
        def sig_phi(G_, n):
            r = (rho_disc(kind, *G_, Md, R1, h)
                 + rho_bb(*G_, Mb, Lb, ab, bb, Mu, au, M2, L2, gu, gb, btype)).reshape(n, NPHI, ZP.size)
            return 2.0 * jnp.trapezoid(r, jnp.asarray(ZP), axis=2)
        sig = sig_phi(GS, NS).reshape(len(E_S) - 1, NSUB, NPHI).mean(1).mean(1)
        a = sig_phi(GA, NA).reshape(RA.size, NSUB, NPHI).mean(1)
        a2 = jnp.abs(jnp.mean(a * jnp.exp(2j * jnp.asarray(PHI))[None, :], axis=1)) / a.mean(1)
        vc2 = _vc2_of(lambda X, Y, Z: phi_bb(X, Y, Z, Mb, Lb, ab, bb, Mu, au, M2, L2, gu, gb, btype))
        if kind == 'mn':
            vc2 = vc2 + _vc2_of(lambda X, Y, Z: MiyamotoNagai_potential(X, Y, Z, _mn(Md, R1, h)))
        return sig, a2, vc2
    return prof


_PROF = {}


def PROF(kind, btype):
    if (kind, btype) not in _PROF:
        _PROF[kind, btype] = _make_profiles(kind, btype)
    return _PROF[kind, btype]


def _vals(lp):
    """11 log10 values, gamma_bul, and (optionally) gamma_bar -> 13 physical values."""
    v = [10.0 ** q for q in lp[:11]] + [float(q) for q in lp[11:]]
    return v + [GAMMA_BAR] if len(v) == 12 else v

# AGAMA's 'Disk': Sigma0 exp(-R/Rd) x vertical profile; scaleHeight > 0 is EXPONENTIAL.  Checked
# here against our own rho_disc rather than assumed.
_d = agama.Density(type='Disk', surfaceDensity=1e9, scaleRadius=3.0, scaleHeight=0.4)
_p = np.array([[2.0, 0.0, 0.3], [5.0, 1.0, 0.0], [1.0, 0.0, 1.2]])
_ours = np.asarray(rho_disc('exp', *[jnp.asarray(_p[:, i]) for i in range(3)], 1e9 * 2 * np.pi * 9.0, 3.0, 0.4))
assert np.allclose(_d.density(_p), _ours, rtol=1e-6), (_d.density(_p), _ours)


def vc2_expdisc(Md, Rd, hz):
    pot = agama.Potential(type='Disk', surfaceDensity=Md / (2 * np.pi * Rd ** 2), scaleRadius=Rd,
                          scaleHeight=hz)
    return np.maximum(-R_VC * pot.force(np.column_stack([R_VC, 0 * R_VC, 0 * R_VC]))[:, 0], 0)


# family -> (disc kind, unpack p -> (Md, R1, h, Mb, Lb, ab, bb, Mu, au) as log10, then gamma_bul)
_L5, _A5, _NO = np.log10(5.0), np.log10(0.5), (-10.0, 0.0)      # _NO: no second bar
FAM = {
    'tied':         ('mn',  lambda p: (p[0], p[2], p[3], p[1], p[4], p[4] - _L5, p[3], p[1], _A5, *_NO, 0.0)),
    'free_mass':    ('mn',  lambda p: (p[0], p[2], p[3], p[1], p[4], p[4] - _L5, p[3], p[5], _A5, *_NO, 0.0)),
    'free_bulge':   ('mn',  lambda p: (p[0], p[2], p[3], p[1], p[4], p[4] - _L5, p[3], p[5], p[6], *_NO, 0.0)),
    'exp_tied':     ('exp', lambda p: (p[0], p[2], p[3], p[1], p[4], p[4] - _L5, p[3], p[1], _A5, *_NO, 0.0)),
    'exp_bbar':     ('exp', lambda p: (p[0], p[2], p[3], p[1], p[4], p[4] - _L5, p[5], p[1], _A5, *_NO, 0.0)),
    'exp_bbar_bul': ('exp', lambda p: (p[0], p[2], p[3], p[1], p[4], p[4] - _L5, p[5], p[6], _A5, *_NO, 0.0)),
    'mn_inner_free':  ('mn',  lambda p: (*p[[0, 2, 3, 1, 4, 5, 6, 7, 8]], *_NO, p[9])),
    'exp_inner_free': ('exp', lambda p: (*p[[0, 2, 3, 1, 4, 5, 6, 7, 8]], *_NO, p[9])),
    # p = [logMd, logMb1, logR1, logh, logL1, logMbul, logMb2, logL2]
    'mn_2bar':  ('mn',  lambda p: (p[0], p[2], p[3], p[1], p[4], p[4] - _L5, p[3], p[5], _A5, p[6], p[7], 0.0)),
    'exp_2bar': ('exp', lambda p: (p[0], p[2], p[3], p[1], p[4], p[4] - _L5, p[3], p[5], _A5, p[6], p[7], 0.0)),
    # p = [logMd, logMb, logR1, logh, logL, log a_bar, log b_bar, (gamma_bar)]; bulge tied as in the notebook
    'mn_ab':      ('mn',  lambda p: (p[0], p[2], p[3], p[1], p[4], p[5], p[6], p[1], _A5, *_NO, 0.0)),
    'exp_ab':     ('exp', lambda p: (p[0], p[2], p[3], p[1], p[4], p[5], p[6], p[1], _A5, *_NO, 0.0)),
    'mn_abg':     ('mn',  lambda p: (p[0], p[2], p[3], p[1], p[4], p[5], p[6], p[1], _A5, *_NO, 0.0, p[7])),
    'exp_abg':    ('exp', lambda p: (p[0], p[2], p[3], p[1], p[4], p[5], p[6], p[1], _A5, *_NO, 0.0, p[7])),
    'mn_abg_T1':  ('mn',  lambda p: (p[0], p[2], p[3], p[1], p[4], p[5], p[6], p[1], _A5, *_NO, 0.0, p[7]), 'T1'),
    'mn_abg_T2':  ('mn',  lambda p: (p[0], p[2], p[3], p[1], p[4], p[5], p[6], p[1], _A5, *_NO, 0.0, p[7]), 'T2'),
    'mn_abg_T4':  ('mn',  lambda p: (p[0], p[2], p[3], p[1], p[4], p[5], p[6], p[1], _A5, *_NO, 0.0, p[7]), 'T4'),
}


def fam_spec(fam):
    f = FAM[fam]
    return f[0], f[1], (f[2] if len(f) > 2 else 'T3')
EV, ES_, EA = [float(v) for v in A.sig.split(',')]
SUFFIX = '' if A.sig == '0.01,0.05,0.02' else '_sig' + A.sig.replace(',', '_')


def evaluate(kind, lp, btype='T3'):
    """lp = log10 of (Md, R1, h, Mb, Lb, ab, bb, Mu, au, M2, L2), gamma_bul[, gamma_bar] -> (sig, a2, vc, parts)."""
    v = _vals(lp)
    sig, a2, vc2 = [np.asarray(t) for t in PROF(kind, btype)(*[jnp.asarray(q) for q in v])]
    if kind == 'exp':
        vc2 = vc2 + vc2_expdisc(v[0], v[1], v[2])
    vc = np.sqrt(np.maximum(vc2, 0))
    parts = (float(np.sum(((vc - VC_T) / (EV * VC_T)) ** 2)),
             float(np.sum(((np.log10(sig) - np.log10(SIG_T)) / ES_) ** 2)),
             float(np.sum(((a2 - A2_T) / EA) ** 2)))
    return sig, a2, vc, parts


# ---------------------------------------------------------------- the box mass (the tie's M/L)
def _nodes(half, a, n):
    u_ = (np.arange(n) + 0.5) / n * 2 - 1
    return half * np.sinh(a * u_) / np.sinh(a), half * a * np.cosh(a * u_) / np.sinh(a) * (2 / n)


_BX, _BW = _nodes(BOX_HALF, 4.0, 128)
_BS, _BSW = _nodes(200.0, 7.0, 256)
_BY = jnp.asarray(np.broadcast_to(_BX[:, None], (128, 256)).ravel())
_BSS = jnp.asarray(np.broadcast_to(_BS[None, :], (128, 256)).ravel())
_WT = jnp.asarray((_BW[:, None] * _BSW[None, :]).ravel())
_RT = jnp.asarray(np.asarray(makeRotationMatrix(25.0, 65.0, 45.0)))


def box_mass(kind, lp, btype='T3'):
    v = _vals(lp)

    def col(X):
        xx, yy, zz = _RT.T @ jnp.stack([jnp.full_like(_BY, X), _BY, _BSS])
        return jnp.sum((rho_disc(kind, xx, yy, zz, *v[:3]) + rho_bb(xx, yy, zz, *v[3:], btype)) * _WT)
    return float(jnp.sum(jax.lax.map(col, jnp.asarray(_BX)) * jnp.asarray(_BW)))


def report(lab, kind, lp, btype='T3'):
    sig, a2, vc, parts = evaluate(kind, lp, btype)
    v = _vals(lp)
    k = R_VC <= 10
    ja, jt = int(np.argmax(a2)), int(np.argmax(A2_T))
    Mstar = v[0] + v[3] + v[7] + v[9]
    Mb_ = box_mass(kind, lp, btype)
    print('%-13s chi2 %7.1f = v_c %7.1f + Sigma %6.1f + A2 %6.1f | v_c rms %4.1f%% max %4.1f%% (R<10) | '
          'Sigma rms %.3f dex | A2 rms %.3f, peak %.3f @ %.2f (N-body %.3f @ %.2f)'
          % (lab, sum(parts), *parts, 100 * np.sqrt(np.mean((vc[k] / VC_T[k] - 1) ** 2)),
             100 * np.max(np.abs(vc[k] / VC_T[k] - 1)), np.sqrt(np.mean(np.log10(sig / SIG_T) ** 2)),
             np.sqrt(np.mean((a2 - A2_T) ** 2)), a2[ja], RA[ja], A2_T[jt], RA[jt]))
    print('%-13s   %s disc M %.3e R %.2f h %.3f | %s bar M %.3e L %.2f a %.2f b %.3f | bulge M %.3e '
          'a %.2f | bar2 M %.3e L %.2f | gamma_bul %.2f gamma_bar %.2f | M_* %.3e (N-body %.3e) | '
          'box mass %.3e -> log M/L %+.3f'
          % ('', 'MN ' if kind == 'mn' else 'exp', *v[:3], btype, *v[3:], Mstar, ms.sum(), Mb_,
             np.log10(Mb_ / L_BOX)))
    return dict(sig=sig, a2=a2, vc=vc, chi2=parts, lp=np.asarray(lp), Mstar=Mstar, box=Mb_)


out = {'cell7': report('cell 7', 'mn', np.r_[np.log10([10 ** 10.83, 12.0, 0.9, 10 ** 10.2, 4.87,
                                                       4.87 / 5, 0.9, 10 ** 10.2, 0.5]), _NO, 0.0])}

# ---------------------------------------------------------------- the fits
B_MN = [(9.5, 11.5), (9.0, 11.0), (-0.5, 1.5), (-1.5, 0.5), (-0.3, 1.0)]
B_EX = [(9.5, 11.5), (9.0, 11.0), (-0.3, 1.2), (-1.3, 0.3), (-0.3, 1.0)]
_INNER = [(-1.5, 0.5), (-1.5, 0.3), (8.5, 11.0), (-1.3, 0.5), (0.0, 1.5)]   # a_bar b_bar M_bul a_bul gamma
BND = {'tied': B_MN, 'free_mass': B_MN + [(8.5, 11.0)], 'free_bulge': B_MN + [(8.5, 11.0), (-1.3, 0.5)],
       'exp_tied': B_EX, 'exp_bbar': B_EX + [(-1.5, 0.3)], 'exp_bbar_bul': B_EX + [(-1.5, 0.3), (8.5, 11.0)],
       'mn_inner_free': B_MN + _INNER, 'exp_inner_free': B_EX + _INNER,
       'mn_2bar': B_MN + [(8.5, 11.0), (8.0, 11.0), (-0.3, 1.0)],
       'exp_2bar': B_EX + [(8.5, 11.0), (8.0, 11.0), (-0.3, 1.0)]}
_AB, _G = [(-1.5, 0.5), (-1.5, 0.3)], [(-1.0, 1.0)]
for _k in FAM:
    if '_ab' in _k:
        BND[_k] = (B_MN if _k.startswith('mn') else B_EX) + _AB + (_G if 'abg' in _k else [])
STARTS = {'mn': [[10.83, 10.2, np.log10(12.0), np.log10(0.9), np.log10(4.87)],
                 [10.732, 10.193, np.log10(5.28), np.log10(0.895), np.log10(3.23)],
                 [10.5, 10.0, 0.5, -0.3, 0.6]],
          'exp': [[10.6, 10.1, np.log10(2.7), np.log10(0.5), np.log10(4.87)],
                  [10.7, 10.0, np.log10(3.5), np.log10(0.3), np.log10(4.0)],
                  [10.5, 10.2, np.log10(2.2), np.log10(0.7), np.log10(5.5)]]}
EXTRA = {'tied': [], 'free_mass': [10.2], 'free_bulge': [10.2, np.log10(0.5)], 'exp_tied': [],
         'exp_bbar': [np.log10(0.5)], 'exp_bbar_bul': [np.log10(0.5), 10.1],
         'mn_inner_free': [np.log10(1.0), np.log10(0.8), 10.1, np.log10(0.5), 0.3],
         'exp_inner_free': [np.log10(1.0), np.log10(0.5), 10.1, np.log10(0.5), 0.3],
         'mn_2bar': [10.1, 9.8, np.log10(5.5)], 'exp_2bar': [10.1, 9.8, np.log10(5.5)]}
for _k in FAM:
    if '_ab' in _k:
        EXTRA[_k] = [np.log10(1.0), np.log10(0.8)] + ([0.8] if 'abg' in _k else [])


def objective(p, fam):
    kind, unpack, btype = fam_spec(fam)
    return sum(evaluate(kind, unpack(p), btype)[3])


for fam in (A.families.split(',') if A.families else FAM):
    kind, unpack, btype = fam_spec(fam)
    t0, tb = time.time(), None
    for s0 in (STARTS[kind] if not fam.endswith('2bar') else
               [np.r_[q[:4], np.log10(2.5)] for q in STARTS[kind]]):
        r = minimize(objective, np.r_[s0, EXTRA[fam]], args=(fam,), method='L-BFGS-B',
                     bounds=BND[fam], options={'maxiter': 300, 'eps': 1e-5})
        if tb is None or r.fun < tb.fun:
            tb = r
    print('\n[%s] best of %d starts, %.0f s' % (fam, len(STARTS[kind]), time.time() - t0))
    out[fam] = report(fam, kind, unpack(tb.x), btype)
print('\npeak RSS %.2f GB' % _rss_gb())

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cols = {'cell7': 'royalblue', 'tied': 'tomato', 'free_mass': 'darkorange', 'free_bulge': 'goldenrod',
        'exp_tied': 'seagreen', 'exp_bbar': 'teal', 'exp_bbar_bul': 'purple',
        'mn_inner_free': 'saddlebrown', 'exp_inner_free': 'magenta', 'mn_2bar': 'olive', 'exp_2bar': 'navy'}
labs = {'cell7': 'cell 7', 'tied': 'MN, bulge tied (notebook)', 'free_mass': 'MN, bulge mass free',
        'free_bulge': 'MN, bulge mass + size free', 'exp_tied': 'exp, b_bar = hz',
        'exp_bbar': 'exp, b_bar free', 'exp_bbar_bul': 'exp, b_bar + bulge mass free',
        'mn_inner_free': 'MN, bar a,b + bulge M,a,gamma free', 'exp_inner_free': 'exp, bar a,b + bulge M,a,gamma free',
        'mn_2bar': 'MN, two bars + bulge', 'exp_2bar': 'exp, two bars + bulge'}
fig, ax = plt.subplots(1, 4, figsize=(21, 4.8), constrained_layout=True)
RS = 0.5 * (E_S[1:] + E_S[:-1])
ax[0].plot(R_VC, VC_T, 'k', lw=2.5, label='N-body stars')
ax[1].plot(RS, SIG_T, 'k', lw=2.5)
ax[2].plot(RS, np.zeros_like(RS), 'k', lw=1)
ax[3].plot(RA, A2_T, 'k', lw=2.5)
for k, o in out.items():
    ls = '--' if k in ('tied', 'free_mass', 'free_bulge', 'mn_inner_free', 'mn_2bar') else '-'
    c = cols.get(k, None)
    ax[0].plot(R_VC, o['vc'], color=c, lw=1.6, ls=ls,
               label='%s  (chi2 %.0f, log M/L %+.2f)' % (labs.get(k, k), sum(o['chi2']), np.log10(o['box'] / L_BOX)))
    ax[1].plot(RS, o['sig'], color=c, lw=1.6, ls=ls)
    ax[2].plot(RS, np.log10(o['sig'] / SIG_T), color=c, lw=1.6, ls=ls)
    ax[3].plot(RA, o['a2'], color=c, lw=1.6, ls=ls)
ax[0].set(xlabel='R [kpc]', ylabel=r'$v_{c,\star}$ [km/s] (stars only)', title='stellar rotation curve')
ax[0].legend(fontsize=7.5, frameon=False, loc='upper right')
ax[1].set(xscale='log', yscale='log', xlabel='R [kpc]', ylabel=r'face-on $\Sigma_\star$ [M$_\odot$ kpc$^{-2}$]',
          title='surface density')
ax[2].set(xscale='log', ylim=(-0.6, 0.6), xlabel='R [kpc]', ylabel=r'$\log\Sigma/\Sigma_{\rm N-body}$',
          title='surface-density residual')
ax[3].set(xlabel='R [kpc]', ylabel=r'$A_2/A_0$', title='bar strength')
fig.suptitle('Direct fits of the stellar family to the N-body star particles (no orbits): sigma = 1%% in '
             'v_c, 0.05 dex in Sigma, 0.02 in A2/A0.  Dashed: Miyamoto-Nagai disc; solid: double exponential',
             fontsize=11)
os.makedirs(A.outdir, exist_ok=True)
outf = os.path.join(A.outdir, 'family_fit_to_nbody%s.png' % SUFFIX)
fig.savefig(outf, dpi=110)
np.savez(os.path.join(HERE, 'family_fit_to_nbody%s.npz' % SUFFIX), R_VC=R_VC, VC_T=VC_T, E_S=E_S,
         SIG_T=SIG_T, E_A=E_A, A2_T=A2_T, **{'lp_' + k: o['lp'] for k, o in out.items()})
print('wrote %s' % os.path.relpath(outf, HERE))
