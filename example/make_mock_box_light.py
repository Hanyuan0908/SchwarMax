#!/usr/bin/env python
"""Measure the mock's stellar light inside a sky box -- the L_BOX constant of the AIC notebook.

For a real galaxy L_BOX is summed from the full image.  The mock's IFU field covers only 83% of
the light, so the "full image" here is the N-body itself, projected exactly as the mock was:
centred on the stellar centre (example_analysis' prepare_nbody), the bar rotated onto x, then
X = R @ x with R = makeRotationMatrix(25, 65, 45) (alpha, beta, gamma in degrees), sky = (X0, X1).
The mock's surface density is M/L = 1 by construction, so stellar mass = light.

The projection is VALIDATED before it is used: the particles are binned onto the mock's own
pixels and Voronoi bins and compared with its `surface_density` (measured: median ratio 1.000,
0.03 dex rms, field total within 0.1%).  The x-mirrored variant prepare_nbody applies for the
plots matches worse (0.058 dex rms), so it is not used.  A box that holds ~all the light makes
the number insensitive to this choice anyway.

    python make_mock_box_light.py [--half 20]
"""
import argparse, os, pickle, resource, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # the package (model.py, utils.py, ...); notebooks, data and the N-body snapshot sit here in example/
P = argparse.ArgumentParser()
P.add_argument('--half', type=float, default=20.0, help='box half-width on the sky [kpc]')
P.add_argument('--nbody', default=os.path.join(HERE,
                                                'Bar_model_TG21', 'model', 't_t0_7'))
P.add_argument('--mock', default=os.path.join(HERE,
                                              'mock_Nbody_galaxy_beta65_gamma45_D50.pkl'))
P.add_argument('--cap', type=float, default=4.0, help='RSS cap [GB]; the process exits above it')
A = P.parse_args()


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == 'darwin' else 1e6)


def _watchdog():
    while True:
        if _rss_gb() > A.cap:
            print('RSS %.2f GB above the %.1f GB cap -- exiting' % (_rss_gb(), A.cap), flush=True)
            os._exit(42)
        time.sleep(0.5)


threading.Thread(target=_watchdog, daemon=True).start()
sys.path.insert(0, ROOT)
import agama
from astropy.constants import G
import astropy.units as u
from utils import makeRotationMatrix

ALPHA, BETA, GAMMA = 25.0, 65.0, 45.0          # the mock's generating angles [deg]

# ---- the N-body stars, centred and bar-aligned as in example_analysis' prepare_nbody
agama.setUnits(length=1, velocity=1, mass=1)
w0, m = agama.readSnapshot(A.nbody)
m = m * (1 / ((G * u.Msun).to(u.kpc * (u.km / u.s) ** 2))).value
st = m != np.unique(m)[-1]                       # the heaviest particles are the halo
for r_ap in (10.0, 5.0, 3.0, 2.0):
    mc = st & (np.hypot(w0[:, 0], w0[:, 1]) < r_ap)
    for col in range(6):
        w0[:, col] -= np.sum(w0[mc, col] * m[mc]) / np.sum(m[mc])
x, y, z, m = w0[st, 0], w0[st, 1], w0[st, 2], m[st]
del w0
Rc, ph = np.hypot(x, y), np.arctan2(y, x)
ann = np.arange(1, 5, 0.25)
bar = float(np.mean([0.5 * np.arctan2(np.mean(np.sin(2 * ph[s])), np.mean(np.cos(2 * ph[s])))
                     for s in ((Rc > ann[i]) & (Rc < ann[i + 1]) for i in range(len(ann) - 1))]))
c, s = np.cos(-bar), np.sin(-bar)
X, Y, _ = np.asarray(makeRotationMatrix(ALPHA, BETA, GAMMA)) @ np.stack([x * c - y * s, x * s + y * c, z])

# ---- validation: rebuild the mock's surface density from the particles
d = pickle.load(open(A.mock, 'rb'))
Xg = np.asarray(d['X_regular_grid'], float).ravel()
Yg = np.asarray(d['Y_regular_grid'], float).ravel()
nX, nY = [int(v) for v in d['nX_nY']]
(x0, x1), (y0, y1) = d['X_minmax'], d['Y_minmax']
dX, dY = (x1 - x0) / nX, (y1 - y0) / nY
bm = np.asarray(d['bin_mapping'])[:Xg.size]
O, npb = np.asarray(d['surface_density'], float), np.asarray(d['num_per_bin'], float)
lut = -np.ones((nX, nY), int)
lut[np.floor((Xg - x0) / dX).astype(int), np.floor((Yg - y0) / dY).astype(int)] = np.arange(Xg.size)
ix, iy = np.floor((X - x0) / dX).astype(int), np.floor((Y - y0) / dY).astype(int)
ok = (ix >= 0) & (ix < nX) & (iy >= 0) & (iy < nY)
b = bm[lut[ix[ok], iy[ok]]]
k = b >= 0
Sig = np.bincount(b[k], weights=m[ok][k], minlength=O.size) / (npb * dX * dY * 1e6)
lr = np.log10(np.maximum(Sig, 1e-30) / O)
L_field_data = float(np.sum(O * npb) * dX * dY * 1e6)
print('validation against the mock surface density (%d bins):' % O.size)
print('   log10(Sigma_particles / Sigma_mock): median %+.4f, rms about it %.4f dex'
      % (np.median(lr), np.sqrt(np.mean((lr - np.median(lr)) ** 2))))
print('   field light: particles %.4e, mock %.4e  (ratio %.4f)'
      % (m[ok][k].sum(), L_field_data, m[ok][k].sum() / L_field_data))
assert abs(np.median(lr)) < 0.01 and abs(m[ok][k].sum() / L_field_data - 1) < 0.01, \
    'the projection does not reproduce the mock -- do not trust L_BOX'

# ---- the number
inb = (np.abs(X) < A.half) & (np.abs(Y) < A.half)
print('\nN-body stellar total            %.4e Msun' % m.sum())
print('light in the IFU field          %.4e  (%.1f%% of the total)' % (L_field_data, 100 * L_field_data / m.sum()))
print('light in the sky box |X|,|Y| < %.1f kpc:' % A.half)
print('   L_BOX = %.4e   (%.2f%% of the total)' % (m[inb].sum(), 100 * m[inb].sum() / m.sum()))
print('peak RSS %.2f GB' % _rss_gb())
