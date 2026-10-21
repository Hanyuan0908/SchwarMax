#!/usr/bin/env python
"""Where do two points of the same chain differ?  chi2 per channel, split by deprojected radius.

Reads the caches plot_datavsmodel_aic.py writes (--stage model), so it runs no model.  For every
Voronoi bin it rebuilds the likelihood's own chi2 terms -- the surface density in log space at
ERR_DEN_LOGL = 1%, and h1..h4 against their errors -- and sums the difference B - A in rings of
deprojected radius R = sqrt(x'^2 + (y'/cos i)^2), where (x', y') are the sky coordinates turned by
alpha (as plot_datavsmodel_aic.py does) and i is the inclination of point A.

    python compare_dvm_points.py ../example/datavsmodel_bestfit_aic_exp_field.npz \\
                                 ../example/datavsmodel_bestfit_logL_barge0.672_aic_exp_field.npz
"""
import argparse, os
import numpy as np

P = argparse.ArgumentParser()
P.add_argument('a', help='cache of point A (the reference)')
P.add_argument('b', nargs='+', help='cache(s) of the point(s) to compare with A')
P.add_argument('--rings', default='0,1,2,3,4,5,6,8,12', help='deprojected ring edges [kpc]')
P.add_argument('--err-den', type=float, default=0.01, help="the likelihood's ERR_DEN_LOGL")
A = P.parse_args()
EDGES = np.array([float(v) for v in A.rings.split(',')])


def per_bin(C):
    """(5, n_bins) chi2 per bin: logSig, h1..h4 -- the terms model_likelihood sums."""
    out = [((np.log10(C['m_Sigma']) - np.log10(C['d_Sigma'])) * np.log(10) / A.err_den) ** 2]
    for k in (1, 2, 3, 4):
        out.append(((C['m_h%d' % k] - C['d_h%d' % k]) / C['e_h%d' % k]) ** 2)
    return np.array(out)


def describe(C, f):
    pn = [str(n) for n in C['param_names']]
    th = C['theta']
    req = ' within ' + ' & '.join(str(r) for r in C['require']) if 'require' in C.files and len(C['require']) else ''
    return ('%s%s: L_bar %.2f, R_d %.2f, f_bar %.3f, gamma %.1f deg | logL %.2f'
            % (str(C['point']), req, 10 ** th[pn.index('logL_bar')], 10 ** th[pn.index('logRd_disk')],
               th[pn.index('f_bar')], np.degrees(th[pn.index('gamma')]), float(C['logL'])))


Ca = np.load(A.a)
pn = [str(n) for n in Ca['param_names']]
al, inc = Ca['theta'][pn.index('alpha')], Ca['theta'][pn.index('beta')]
bm = np.asarray(Ca['bin_mapping'])[:-1]
nb = int(Ca['total_bins'])
X, Y = np.asarray(Ca['X_regular_grid']), np.asarray(Ca['Y_regular_grid'])
cnt = np.bincount(bm, minlength=nb)
Xb, Yb = np.bincount(bm, X, nb) / cnt, np.bincount(bm, Y, nb) / cnt          # bin centroids [kpc]
c, s = np.cos(al), np.sin(al)
xp, yp = c * Xb - s * Yb, s * Xb + c * Yb
Rdep = np.hypot(xp, yp / np.cos(inc))
ring = np.digitize(Rdep, EDGES) - 1
chA = per_bin(Ca)
S2 = float(Ca['S2'])
print('A = %s' % describe(Ca, A.a))
for f in A.b:
    Cb = np.load(f)
    assert np.array_equal(np.asarray(Cb['d_Sigma']), np.asarray(Ca['d_Sigma'])), 'different data'
    d = per_bin(Cb) - chA
    print('\nB = %s' % describe(Cb, f))
    print('B - A: delta chi2 (positive = B fits worse); delta logL = -0.5 * delta / S2, S2 = %.2f' % S2)
    print('%14s %6s %9s %8s %8s %8s %8s %9s' % ('R_dep [kpc]', 'bins', 'logSig', 'h1', 'h2', 'h3', 'h4', 'sum'))
    for r in range(len(EDGES) - 1):
        m = ring == r
        if not m.any():
            continue
        row = d[:, m].sum(axis=1)
        print('%6.1f - %-5.1f %6d %9.1f %8.1f %8.1f %8.1f %8.1f %9.1f'
              % (EDGES[r], EDGES[r + 1], m.sum(), *row, row.sum()))
    tot = d.sum(axis=1)
    print('%14s %6d %9.1f %8.1f %8.1f %8.1f %8.1f %9.1f' % ('all', nb, *tot, tot.sum()))
    if 'chi2_3d' in Cb.files and 'chi2_3d' in Ca.files:
        print('   2 m_eff: %+.1f  |  orbits vs the model\'s own 3-D density (not in logL): %.0f -> %.0f'
              % (2 * (float(Cb['m_eff_kin']) - float(Ca['m_eff_kin'])), float(Ca['chi2_3d']), float(Cb['chi2_3d'])))
