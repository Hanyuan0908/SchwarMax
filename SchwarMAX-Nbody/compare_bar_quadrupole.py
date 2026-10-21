#!/usr/bin/env python
"""Does the model's bar act on the stars like the N-body's bar?  Gravity and streaming, by m=2.

  1. GRAVITY.  The m=2 part of the in-plane STELLAR potential, Phi_2(R) (amplitude, km^2/s^2), of
     the N-body (an AGAMA CylSpline on its star particles, bar along x) and of the model (the
     notebook's own potential with the halo switched off).
  2. STREAMING.  The m=2 part of the face-on mean radial velocity, <v_R>(R, phi) ~ V_2 cos 2(phi -
     phi_2), in the plane (|z| < 0.8 kpc): from the N-body particles, and from the model's ORBITS --
     the package's get_model_with_orbit_acc_fourier at the same settings as run_model, each stored
     point weighted by its time step and its orbit's NNLS weight.  v_R does not depend on the
     frame's rotation, and its m=2 is unchanged by the bar's symmetry images.  The phase is quoted
     in the sense of rotation (positive = ahead of the bar).

Points: the field chain's best fit, and the cell-10 truth.

    python compare_bar_quadrupole.py [--chain ../example/emcee_aic_exp_field.h5]
"""
import argparse, json, os, resource, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P = argparse.ArgumentParser()
P.add_argument('--chain', default=os.path.join(ROOT, 'example', 'emcee_aic_exp_field.h5'))
P.add_argument('--nb', default=os.path.join(HERE, 'SchwarMAX_Nbody_AIC_expdisc_field.ipynb'))
P.add_argument('--zmax', type=float, default=0.8, help='|z| cut for the in-plane streaming [kpc]')
P.add_argument('--cap', type=float, default=15.0)
P.add_argument('--plot-only', action='store_true',
               help='no computation: draw figures/bar_quadrupole_compare.png from the saved npz')
A = P.parse_args()
NPZ = os.path.join(ROOT, 'example', 'bar_quadrupole_compare.npz')


def plot(npz=NPZ):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = np.load(npz)
    R = d['R']
    sty = [('N-body', 'N-body', 'k', '-', 2.5), ('best_fit', 'best fit (L_bar 3.23 kpc)', 'tomato', '--', 2),
           ('truth', 'truth (L_bar 4.87 kpc)', 'royalblue', '-', 1.8)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for key, lab, c, ls, lw in sty:
        ax[0].plot(R, d[key + '_Phi2'], color=c, ls=ls, lw=lw, label=lab)
        ax[1].plot(R, d[key + '_V2'], color=c, ls=ls, lw=lw, label=lab)
        ax[2].plot(R, np.where(d[key + '_V2'] >= 1.5, d[key + '_phV2'], np.nan),   # no phase without amplitude
                   color=c, ls=ls, lw=lw, label=lab)
    ax[0].set(xlabel='R [kpc]', ylabel=r'$|\Phi_2|$ [km$^2$ s$^{-2}$]', title='m=2 of the in-plane stellar potential')
    ax[1].set(xlabel='R [kpc]', ylabel=r'$V_2$ [km/s]', title=r'm=2 of the face-on $\langle v_R \rangle$, $|z|<0.8$ kpc')
    ax[2].set(xlabel='R [kpc]', ylabel='phase of $V_2$ [deg]', ylim=(-90, 90),
              title='phase of $V_2$ vs the bar (+ = ahead)')
    ax[2].axhline(0, color='0.6', lw=0.8)
    for a in ax:
        a.set_xlim(0, 8); a.grid(alpha=0.3)
    ax[0].set_ylim(0, None); ax[1].set_ylim(0, None)
    ax[0].legend(frameon=False)
    fig.suptitle('Bar quadrupole: N-body vs the field-tied exponential-disc model (orbits from '
                 'get_model_with_orbit_acc_fourier, 7500 orbits); phase hidden where V_2 < 1.5 km/s', fontsize=11)
    os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    out = os.path.join(HERE, 'figures', 'bar_quadrupole_compare.png')
    fig.savefig(out, dpi=120)
    print('wrote %s' % os.path.relpath(out, HERE))


if A.plot_only:
    plot()
    sys.exit(0)


def _rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3


def _watch():
    while True:
        time.sleep(0.25)
        if _rss() > A.cap:
            sys.stderr.write('WATCHDOG: RSS %.1f GB > %.1f\n' % (_rss(), A.cap)); sys.stderr.flush(); os._exit(42)


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
from model import build_model_acc_meff_fourier, get_model_with_orbit_acc_fourier
from potentials import NFW_potential, T3_density, T3_potential, V4_density, V4_potential
from CylindricalSpline import get_phi_m, evaluate_phi, evaluate_phi_axisymmetric, get_acc
path = %r + "/"
''' % (ROOT, ROOT), G)
nb = json.load(open(A.nb))
for i in (4, 6, 8, 10):
    s = ''.join(nb['cells'][i]['source'])
    if i == 10:
        s = s[:s.index('t0 = time.time()')]
    exec(compile(s, '<cell %d>' % i, 'exec'), G)
jax, jnp = G['jax'], G['jnp']
K2 = float(G['KPCGYR_TO_KMS']) ** 2
KV = float(G['KPCGYR_TO_KMS'])
names = list(G['param_names'])
b = emcee.backends.HDFBackend(A.chain, read_only=True)
burn = int(3 * np.nanmax(b.get_autocorr_time(tol=0)))
S, L = b.get_chain(discard=burn, flat=True), b.get_log_prob(discard=burn, flat=True)
POINTS = {'best fit': S[int(np.argmax(L))].astype(float), 'truth': np.asarray(G['theta_true'], float)}

R_ED = np.arange(0.5, 8.01, 0.5)                    # rings for both comparisons [kpc]
R_C = 0.5 * (R_ED[1:] + R_ED[:-1])
NPHI = 64
PH = (np.arange(NPHI) + 0.5) * 2 * np.pi / NPHI


def m2(vals_phi):
    """Amplitude and phase of the cos/sin(2 phi) term of a function sampled at PH."""
    c = 2 * np.mean(vals_phi * np.cos(2 * PH), axis=-1)
    s = 2 * np.mean(vals_phi * np.sin(2 * PH), axis=-1)
    return np.hypot(c, s), 0.5 * np.degrees(np.arctan2(s, c))


def vr_quadrupole(x, y, z, vx, vy, w, vphi_sign):
    """V_2 amplitude and phase (in the sense of rotation) of <v_R> per ring, in the plane."""
    R = np.hypot(x, y)
    sel = np.abs(z) < A.zmax
    vR = (x * vx + y * vy) / np.maximum(R, 1e-9)
    ph = np.arctan2(y, x)
    k = np.digitize(R[sel], R_ED) - 1
    ok = (k >= 0) & (k < len(R_C))
    k, ws, vRs, phs = k[ok], w[sel][ok], vR[sel][ok], ph[sel][ok]
    W = np.bincount(k, ws, len(R_C))
    c = 2 * np.bincount(k, ws * vRs * np.cos(2 * phs), len(R_C)) / W
    s = 2 * np.bincount(k, ws * vRs * np.sin(2 * phs), len(R_C)) / W
    return np.hypot(c, s), vphi_sign * 0.5 * np.degrees(np.arctan2(s, c))


out = {}
# ---------------------------------------------------------------- the N-body
import agama
from astropy.constants import G as Gc
import astropy.units as u
agama.setUnits(length=1, velocity=1, mass=1)
w0, m = agama.readSnapshot(os.path.join(ROOT, 'example', 'Bar_model_TG21', 'model', 't_t0_7'))
m = m * (1 / ((Gc * u.Msun).to(u.kpc * (u.km / u.s) ** 2))).value
st = m != np.unique(m)[-1]
for r_ap in (10.0, 5.0, 3.0, 2.0):                 # make_mock_box_light.py's centring, positions AND velocities
    mc = st & (np.hypot(w0[:, 0], w0[:, 1]) < r_ap)
    for col in range(6):
        w0[:, col] -= np.sum(w0[mc, col] * m[mc]) / np.sum(m[mc])
w, ms = w0[st], m[st]
del w0
Rc, ph = np.hypot(w[:, 0], w[:, 1]), np.arctan2(w[:, 1], w[:, 0])
ann = np.arange(1, 5, 0.25)
bar = float(np.mean([0.5 * np.arctan2(np.mean(np.sin(2 * ph[s])), np.mean(np.cos(2 * ph[s])))
                     for s in ((Rc > ann[i]) & (Rc < ann[i + 1]) for i in range(len(ann) - 1))]))
c, s_ = np.cos(-bar), np.sin(-bar)
xb, yb = w[:, 0] * c - w[:, 1] * s_, w[:, 0] * s_ + w[:, 1] * c
vxb, vyb = w[:, 3] * c - w[:, 4] * s_, w[:, 3] * s_ + w[:, 4] * c
Lz = np.sum(ms * (xb * vyb - yb * vxb))
sgn_nb = np.sign(Lz)
print('N-body: %d stars; sense of rotation in the bar frame: %s' % (len(ms), 'anticlockwise' if sgn_nb > 0 else 'clockwise'))
V2, ph2 = vr_quadrupole(xb, yb, w[:, 2], vxb, vyb, ms, sgn_nb)
pot = agama.Potential(type='CylSpline', particles=(np.column_stack([xb, yb, w[:, 2]]), ms),
                      symmetry='bisymmetric', mmax=6, gridSizeR=60, gridSizez=40,
                      Rmin=0.05, Rmax=80.0, zmin=0.02, zmax=40.0)
RR, PP = np.meshgrid(R_C, PH, indexing='ij')
Phi = pot.potential(np.column_stack([(RR * np.cos(PP)).ravel(), (RR * np.sin(PP)).ravel(), 0 * RR.ravel()])).reshape(RR.shape)
P2, pph = m2(Phi)
out['N-body'] = dict(Phi2=P2, V2=V2, phV2=ph2)
del w, xb, yb, vxb, vyb, pot
print('   done (peak RSS %.1f GB)' % _rss(), flush=True)


# ---------------------------------------------------------------- the models
def setup(th):
    """run_model's own parameter handling (cell 8)."""
    lMh, lRh = G['logMenc_logc_to_logM_logRs'](th[0], th[2], r_enc=10.0, Delta=200., rho_crit=277.54)
    f_bar, Rd, hz, Lb = th[1], 10 ** th[3], 10 ** th[4], 10 ** th[5]
    al, be, ga = (np.degrees(th[k]) for k in (6, 7, 8))
    lMd, lMb, Ms, pb = [float(v) for v in G['stellar_masses'](f_bar, th[9], Rd, hz, Lb, Lb / 5.0, hz, al, be, ga)]
    ph_ = {'logM': float(lMh), 'Rs': float(10 ** lRh), 'a': 1.0, 'b': 1.0, 'c': 1.0,
           'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0, 'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}
    pb_ = {'logM_disc': lMd, 'Rd_disc': Rd, 'hz_disc': hz, 'logM_bar': lMb, 'L_bar': Lb,
           'a_bar': Lb / 5.0, 'b_bar': hz, 'mass_to_light_ratio': 10 ** th[9], 'Omega_bar': 10 ** th[10],
           'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0, 'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
           'alpha': al, 'beta': be, 'gamma': ga}
    return ph_, pb_, Ms


dd = G['dict_data']
nX, nY = [int(v) for v in np.asarray(dd['nX_nY'])]
for lab, th in POINTS.items():
    t0 = time.time()
    ph_, pb_, Ms = setup(th)
    dens, potf, acc, pax, dax = G['make_model_funcs'](pb_, ph_)
    # 1. gravity: the stellar potential's m=2 (halo off)
    nohalo = dict(ph_, logM=-30.0)
    Phi = np.asarray(potf(jnp.asarray(RR * np.cos(PP)), jnp.asarray(RR * np.sin(PP)), jnp.zeros_like(jnp.asarray(RR)),
                          pb_, nohalo)) * K2
    P2m, _ = m2(Phi)
    # 2. streaming: the orbit library, weighted
    res = G['get_model_with_orbit_acc_fourier'](
        dens, potf, acc, pax, dax, ph_, pb_, dd, G['NUM_VBIN'],
        mmax=G['F3D_MMAX'], nR_f=G['F3D_R'][2], nZ_f=G['F3D_Z'][2], log_R=G['F3D_LOG'], log_z=G['F3D_LOG'],
        R_lo_f=G['F3D_R'][0], R_hi_f=G['F3D_R'][1], z_lo_f=G['F3D_Z'][0], z_hi_f=G['F3D_Z'][1],
        n_phi_t=G['F3D_NPHI_T'], n_sub_t=G['F3D_NSUB_T'], sigma_frac_3d=G['SIG_FRAC_An'],
        floor_frac_3d=G['F3D_FLOOR'], v_sys=0.0, pow_R=G['F3D_POW_R'], pow_z=G['F3D_POW_Z'],
        sigma_scheme=G['F3D_SCHEME'], sigma_frac_m0=G['SIG_FRAC_A0'], close_inner=G['F3D_CLOSE_INNER'],
        Rzphi_n_tot=G['RZ_TOT'], Rzphi_n_grid=dd['Rzphi_n_grid'],
        Rzphi_lim_grid=jnp.array([dd['R_minmax'], dd['z_minmax'], dd['phi_minmax']]),
        xy_lim_grid=jnp.array([dd['X_minmax'], dd['Y_minmax']]), xy_n_grid=jnp.array([nX, nY]),
        nnls_maxiter=200, regularization=10.0 ** th[G['J_LAM']])
    wts = np.asarray(res['weights'], float)
    y = np.asarray(res['y_traj'], np.float32)[:, 0]                      # (n_orb, N_max, 6)
    t = np.asarray(res['t_traj'], float)[:, 0]
    del res
    dt = np.diff(t, axis=1, prepend=0.0)
    dt = np.where(dt > 0, dt, 0.0)
    Wpt = (wts[:, None] * dt / np.maximum(dt.sum(axis=1, keepdims=True), 1e-30)).ravel()
    keep = Wpt > 0
    x_, y_, z_ = (y[..., k].ravel()[keep] for k in (0, 1, 2))
    vx_, vy_ = (y[..., k].ravel()[keep] * KV for k in (3, 4))
    del y
    Lzm = np.sum(Wpt[keep] * (x_ * vy_ - y_ * vx_))
    V2m, ph2m = vr_quadrupole(x_, y_, z_, vx_, vy_, Wpt[keep], np.sign(Lzm))
    out[lab] = dict(Phi2=P2m, V2=V2m, phV2=ph2m)
    print('%s: L_bar %.2f, model M_* %.3e, rotation %s, %d of %d orbits carry weight (%.0f s, peak RSS %.1f GB)'
          % (lab, 10 ** th[5], Ms, 'anticlockwise' if Lzm > 0 else 'clockwise', int((wts > 0).sum()),
             len(wts), time.time() - t0, _rss()), flush=True)
    del x_, y_, z_, vx_, vy_, Wpt

print('\nm=2 of the in-plane STELLAR potential |Phi_2| [km^2/s^2], and of <v_R> in |z| < %.1f kpc:'
      ' V_2 [km/s] (phase, deg, + = ahead of the bar in the sense of rotation)' % A.zmax)
print('%9s | %26s | %33s' % ('R [kpc]', '|Phi_2|: N-body  best  truth', 'V_2 (phase): N-body  best  truth'))
for i, R in enumerate(R_C):
    print('%9.2f | %8.0f %8.0f %8.0f | %6.1f (%+4.0f) %6.1f (%+4.0f) %6.1f (%+4.0f)'
          % (R, out['N-body']['Phi2'][i], out['best fit']['Phi2'][i], out['truth']['Phi2'][i],
             out['N-body']['V2'][i], out['N-body']['phV2'][i], out['best fit']['V2'][i], out['best fit']['phV2'][i],
             out['truth']['V2'][i], out['truth']['phV2'][i]))
np.savez(NPZ, R=R_C,
         **{'%s_%s' % (k.replace(' ', '_'), q): v for k, d in out.items() for q, v in d.items()})
print('\npeak RSS %.1f GB' % _rss())
