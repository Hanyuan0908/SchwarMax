#!/usr/bin/env python
"""Data vs model for the AIC fit of the mock -- the Section-2 figure of `example_analysis.ipynb`,
made with the NEW pipeline (`build_model_acc_meff_fourier`) instead of `build_model`.

The forward model is NOT re-implemented.  This script exec's the AIC notebook's OWN cells
4 (density, acceleration, m=0 callables), 6 (load + handedness flip) and 8 (configuration +
`run_model`), exactly as `test_example_aic_nb.py` does, so `run_model` is literally the
function the sampler called and cannot drift from the chain.  Only cell 2 (Colab mount +
pip pin) is replaced by a local header.

The best fit is picked as in `example_analysis.ipynb` cell 11: the maximum-logP sample after
discarding the first DISCARD_FIT = 700 steps.  The figure is cell 15's 5x3 layout -- rows
(Sigma, V, sigma, h3, h4) x columns (Model, Data, Residuals), disc-aligned frame, N-body
contours -- with the same colour scales, error conventions and chi2 stamps.

HANDEDNESS.  Cell 6 negates v0, V, h1, h3 so this pipeline's anticlockwise library can fit the
mock.  By default the figure is drawn in the mock's DELIVERED convention (the model's odd
moments are negated back as well), so it reads directly against the upstream figure.
`--frame fitted` draws what the NNLS actually saw.  The residual magnitudes are identical.

EXPONENTIAL DISC.  An emcee backend (*.h5) is read directly; a chain whose name carries
`aic_exp` runs on SchwarMAX_Nbody_AIC_expdisc.ipynb (its default discard is that notebook's
3 x max(tau) burn-in).  `--point truth` evaluates the notebook's cell-10 ground truth instead of
the best fit, through the same run_model, so the two figures are directly comparable.

LAYOUTS.  The 13-parameter chain (logM_disk, logM_bar and M/L all free; files *_aic.*) runs on
the archived archive/SchwarMAX_Nbody_AIC_13param.ipynb; the 12-parameter chain (stellar mass
derived from M/L; files *_aic_tied.*) on the live notebook.  The notebook is picked from the
chain's dimension, and the cache and figure names from the chain's file name, so the two runs
never overwrite each other.

MEMORY.  A watchdog aborts the process if RSS exceeds --cap GB.  The two stages are separate:
`--stage model` runs JAX once and caches everything the figure needs; `--stage plot` needs no
JAX and only reads that cache plus the N-body snapshot for the contours.

    python plot_datavsmodel_aic.py                  # both stages (model skipped if cached)
    python plot_datavsmodel_aic.py --stage model    # forward model -> example/datavsmodel_bestfit_aic.npz
    python plot_datavsmodel_aic.py --stage plot     # figure -> figures/
    python plot_datavsmodel_aic.py --chain example/emcee_aic_exp.h5 [--point truth]
"""
import argparse, json, os, pickle, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # the package (model.py, utils.py, ...); notebooks, data and the N-body snapshot sit here in example/

P = argparse.ArgumentParser()
P.add_argument('--stage', default='all', choices=('all', 'model', 'plot'))
P.add_argument('--chain', default=None,
               help='MCMC output; default the newest that exists of mcmc_results_aic_tied.pkl, '
                    'mcmc_checkpoint_aic_tied.pkl, mcmc_results_aic.pkl, mcmc_checkpoint_aic.pkl')
P.add_argument('--notebook', default=None,
               help='default: picked from the chain dimension (12 -> live, 13 -> archive/)')
P.add_argument('--discard', type=int, default=None,
               help="default example_analysis's DISCARD_FIT = 700 for a BlackJAX pickle, the "
                    "emcee cell's 3 x max(tau) for an .h5 backend")
P.add_argument('--point', default='bestfit', choices=('bestfit', 'truth'),
               help="bestfit: the max-logP sample after the discard; truth: the notebook's "
                    "cell-10 ground_truth with log10(lambda) = 0")
P.add_argument('--nboot', type=int, default=None,
               help="n_boot_meff; default the notebook's N_BOOT_MEFF, so logL is the chain's "
                    "objective.  It does not change the fit, only the m_eff estimate.")
P.add_argument('--frame', default='delivered', choices=('delivered', 'fitted'))
P.add_argument('--mock', default='mock_Nbody_galaxy_beta65_gamma45_D50_GH.pkl',
               help="bin dict in example/, substituted into cell 6.  The default _GH file has "
                    "V = v0 and sigma = s (NGC 1512's GH basis, see make_mock_gh_basis.py), "
                    "which is what the new pipeline's model V and sigma are.  The original "
                    "file's V/sigma are plain moments.  The FIT is the same for both: neither "
                    "V nor sigma enters the likelihood.")
P.add_argument('--nbody', default=os.path.join(HERE,
                                               'Bar_model_TG21', 'model', 't_t0_7'),
               help='N-body snapshot for the contour overlay (example_analysis uses t_t0_7)')
P.add_argument('--cache', default=None, help='default example/datavsmodel_bestfit_<run>.npz')
P.add_argument('--recompute', action='store_true', help='rerun the model even if cached')
P.add_argument('--outdir', default=os.path.join(HERE, 'figures'))
P.add_argument('--tag', default=None, help='default <run>_bestfit')
P.add_argument('--cap', type=float, default=8.0, help='abort if RSS exceeds this many GB')
A = P.parse_args()

# ---------------------------------------------------------------- memory watchdog
PEAK = [0.0]


def _rss_gb():
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024.0 ** 3 if sys.platform == 'darwin' else r / 1024.0 ** 2


def _watch():
    while True:
        time.sleep(0.25)
        g = _rss_gb()
        PEAK[0] = max(PEAK[0], g)
        if g > A.cap:
            sys.stderr.write('\n*** WATCHDOG: RSS %.2f GB > cap %.1f GB -- ABORTING\n'
                             % (g, A.cap))
            sys.stderr.flush()
            os._exit(42)


threading.Thread(target=_watch, daemon=True).start()
mark = lambda t: print('   [mem] %-28s peak RSS %5.2f GB' % (t, _rss_gb()), flush=True)

import numpy as np

NB_BY_NDIM = {12: os.path.join(HERE, 'SchwarMAX_Nbody_AIC.ipynb'),
              13: os.path.join(HERE, 'archive', 'SchwarMAX_Nbody_AIC_13param.ipynb')}


def chain_file():
    if A.chain is not None:
        return A.chain
    for run in ('aic_tied', 'aic'):                    # the newest layout first
        for kind in ('results', 'checkpoint'):
            f = os.path.join(HERE, 'mcmc_%s_%s.pkl' % (kind, run))
            if os.path.isfile(f):
                return f
    raise FileNotFoundError('no MCMC output in example/')


# the run name ('aic', 'aic_tied', 'aic_exp') keys the cache and the figure, so runs never collide
_base = os.path.basename(chain_file())
for _ext in ('.pkl', '.h5'):
    _base = _base[:-len(_ext)] if _base.endswith(_ext) else _base
_parts = _base.split('_', 2)
if _base.startswith('emcee_'):                         # emcee_<run>.h5 -- no results/checkpoint
    RUN = _base[len('emcee_'):]
else:                                                  # mcmc_<results|checkpoint>_<run>.pkl
    RUN = _parts[2] if len(_parts) == 3 and _parts[0] == 'mcmc' else '_'.join(_parts)
EXP = 'aic_exp' in RUN
A.cache = A.cache or os.path.join(HERE, 'datavsmodel_%s_%s.npz' % (A.point, RUN))
A.tag = A.tag or '%s_%s' % (RUN, A.point)


def load_chain():
    """(samples, logprob, label) -- the finished run if it exists, else the checkpoint; or an
    emcee HDF5 backend.  Also sets A.discard if it was not given."""
    f = chain_file()
    if f.endswith('.h5'):
        import emcee
        b = emcee.backends.HDFBackend(f, read_only=True)
        S, L = b.get_chain(), b.get_log_prob()
        if A.discard is None:
            A.discard = int(3 * np.nanmax(b.get_autocorr_time(tol=0)))
        return S, L, os.path.relpath(f, HERE)
    with open(f, 'rb') as fh:
        d = pickle.load(fh)
    if 'chain' in d:                                   # cell 17's final results
        S, L = np.asarray(d['chain']), np.asarray(d['logprob'])
    else:                                              # cell 17's checkpoint
        S, L = np.stack(d['all_samples'], 0), np.stack(d['all_logprob'], 0)
    if A.discard is None:
        A.discard = 700
    return S, L, os.path.relpath(f, HERE)


# ================================================================= STAGE 1: the forward model
def stage_model():
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    S, L, chain_label = load_chain()
    n_steps = S.shape[0]
    assert A.discard < n_steps, 'discard %d >= %d steps in %s' % (A.discard, n_steps, chain_label)
    flat, flat_lp = S[A.discard:].reshape(-1, S.shape[-1]), L[A.discard:].reshape(-1)
    ib = int(np.argmax(flat_lp))
    theta, lp_chain = flat[ib].astype(float), float(flat_lp[ib])
    print('chain %s: %d steps x %d chains x %d dims' % ((chain_label,) + S.shape))
    print('best logP after discarding %d steps: %.3f' % (A.discard, lp_chain))

    # The notebook's cell 2, minus Drive and pip.  Same imports, same `path` contract.
    header = '''
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
from CylindricalSpline import get_phi_m, evaluate_phi, evaluate_phi_axisymmetric, get_acc
path = %r + "/"
''' % (ROOT, ROOT)
    G = {'__name__': '__main__'}
    exec(compile(header, '<header>', 'exec'), G)
    nb_file = A.notebook or (os.path.join(HERE, 'SchwarMAX_Nbody_AIC_expdisc.ipynb') if EXP
                             else NB_BY_NDIM[S.shape[-1]])
    print('   notebook %s' % os.path.relpath(nb_file, HERE))
    nb = json.load(open(nb_file))
    for i in (4, 6, 8):
        src = ''.join(nb['cells'][i]['source'])
        if i == 6:
            orig = "filename = '/example/mock_Nbody_galaxy_beta65_gamma45_D50.pkl'"
            assert orig in src, 'cell 6 no longer names the mock the way this script expects'
            assert os.path.isfile(os.path.join(HERE, A.mock)), A.mock
            src = src.replace(orig, "filename = '/example/%s'" % A.mock)
            print('   cell 6 loads example/%s' % A.mock)
        t0 = time.time()
        exec(compile(src, '<notebook cell %d>' % i, 'exec'), G)
        print('   notebook cell %d OK (%.1f s)' % (i, time.time() - t0))
    mark('after cells 4/6/8')
    if A.point == 'truth':
        c10 = ''.join(nb['cells'][10]['source'])
        exec(compile(c10[:c10.index('theta_true')], '<notebook cell 10, truth only>', 'exec'), G)
        theta, lp_chain = np.asarray(list(G['ground_truth']) + [0.0], float), float('nan')
        print('   point = the cell-10 GROUND TRUTH (log10 lambda = 0), not a chain sample')

    jnp = G['jnp']
    assert theta.size == G['NDIM'], 'chain has %d dims, the notebook %d' % (theta.size, G['NDIM'])
    nboot = G['N_BOOT_MEFF'] if A.nboot is None else A.nboot
    dd = G['dict_data']

    t0 = time.time()
    res = G['run_model'](jnp.asarray(theta), dd, G['NUM_VBIN'], G['RZ_TOT'], n_boot=nboot)
    res['surface_density'].block_until_ready()
    t_model = time.time() - t0
    print('   run_model: %.1f s (%d orbits, n_boot_meff %d)' % (t_model, G['N_ORBIT'], nboot))
    mark('after run_model')

    # The chain's objective, from the SAME res (model_likelihood would integrate again).
    O = np.asarray(dd['XY_density_data'])
    d0 = np.asarray(res['surface_density'])[0]
    sig_l = (G['ERR_DEN_LOGL'] * O) / (O * np.log(10) + float(G['EPSILON']))
    chi2_den = float(np.nansum(((np.log10(d0 + G['EPSILON']) - np.log10(O + G['EPSILON']))
                                / sig_l) ** 2))
    chi2_kin = 0.0
    for k in (1, 2, 3, 4):
        chi2_kin += float(np.nansum(((np.asarray(res['h%d' % k])[0] - np.asarray(dd['h%d_data' % k]))
                                     / (np.asarray(dd['h%d_data_err' % k]) + G['EPSILON'])) ** 2))
    m_eff_kin = float(res['m_eff_kin'])
    # Cell 15's default ADOPT = 'gaussian' -> S2 = sqrt(2 N_kin).  The prior box of cell 17 has
    # widths that do not depend on the minimiser result (logL_bar is clipped to [0, 1]).
    logL = -0.5 * (chi2_kin + 2.0 * m_eff_kin + chi2_den) / G['S2']
    widths = {13: [6, 6, 6, 2, 2, 2, 1, np.pi, np.pi / 2, np.pi, 4, 2, 5],
              12: [6, 0.98, 2, 2, 2, 1, np.pi, np.pi / 2, np.pi, 4, 2, 5]}[theta.size]
    widths = np.asarray(widths, float)
    logprior = -float(np.sum(np.log(widths)))
    print('\nlogL here %.3f + log prior %.3f = %.3f   vs the chain\'s logP %.3f  (diff %+.3f)'
          % (logL, logprior, logL + logprior, lp_chain, logL + logprior - lp_chain))
    print('   chi2_den %.1f + chi2_kin %.1f + 2 m_eff_kin %.1f = %.1f'
          % (chi2_den, chi2_kin, 2 * m_eff_kin, chi2_den + chi2_kin + 2 * m_eff_kin))
    print('   (not bit-identical by construction: CPU vs GPU, and serial vs vmapped float32 land '
          'on different points of the NNLS ridge -- see the README)')
    print('per-bin chi2: logSig %.3f | h1 %.3f h2 %.3f h3 %.3f h4 %.3f | m_eff_kin %.2f'
          % ((chi2_den / O.size,) + tuple(
              float(np.nanmean(((np.asarray(res['h%d' % k])[0] - np.asarray(dd['h%d_data' % k]))
                                / np.asarray(dd['h%d_data_err' % k])) ** 2)) for k in (1, 2, 3, 4))
             + (m_eff_kin,)))
    print('flux level data/model %.4f' % float(O.sum() / d0.sum()))

    os.makedirs(os.path.dirname(A.cache), exist_ok=True)
    g = lambda k: np.asarray(dd[k])
    np.savez(A.cache,
             theta=theta, lp_chain=lp_chain, logL=logL, logprior=logprior,
             chi2_den=chi2_den, chi2_kin=chi2_kin, m_eff_kin=m_eff_kin, S2=G['S2'],
             n_orbit=G['N_ORBIT'], n_boot=nboot, discard=A.discard, n_steps=n_steps,
             chain_label=chain_label, flipped=bool(G['FLIP_HANDEDNESS']), mock=A.mock,
             point=A.point,
             param_names=np.asarray(G['param_names']),
             # model, unperturbed fit (row 0), in the FITTED handedness
             m_Sigma=d0, m_V=np.asarray(res['V'])[0], m_sigma=np.asarray(res['sigma'])[0],
             m_h1=np.asarray(res['h1'])[0], m_h2=np.asarray(res['h2'])[0],
             m_h3=np.asarray(res['h3'])[0], m_h4=np.asarray(res['h4'])[0],
             # data, as cell 6 left it (FITTED handedness)
             d_Sigma=O, d_V=g('V_data'), d_sigma=g('sigma_data'),
             d_h1=g('h1_data'), d_h2=g('h2_data'), d_h3=g('h3_data'), d_h4=g('h4_data'),
             e_h1=g('h1_data_err'), e_h2=g('h2_data_err'),
             e_h3=g('h3_data_err'), e_h4=g('h4_data_err'),
             # geometry for the mock-IFU maps
             bin_mapping=g('bin_mapping'), X_regular_grid=g('X_regular_grid'),
             Y_regular_grid=g('Y_regular_grid'), X_minmax=g('X_minmax'),
             Y_minmax=g('Y_minmax'), nX_nY=g('nX_nY'), total_bins=g('total_bins'))
    print('cached -> %s' % os.path.relpath(A.cache, HERE))
    mark('end of model stage')


# ================================================================= STAGE 2: the figure
def stage_plot():
    import scipy.ndimage
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm

    C = np.load(A.cache, allow_pickle=False)
    theta = C['theta']
    print('cache: %s | %d orbits | best logP %.3f after discarding %d of %d steps of %s | data %s'
          % (os.path.relpath(A.cache, HERE), int(C['n_orbit']), float(C['lp_chain']),
             int(C['discard']), int(C['n_steps']), str(C['chain_label']),
             str(C['mock']) if 'mock' in C.files else '(original mock)'))

    # ---- handedness of the DISPLAY: only odd-in-v quantities change sign ------------------
    sgn = -1.0 if (bool(C['flipped']) and A.frame == 'delivered') else 1.0
    y_xy, density_model = C['d_Sigma'], C['m_Sigma']
    y_V, V_model = sgn * C['d_V'], sgn * C['m_V']
    y_sig, sigma_model = C['d_sigma'], C['m_sigma']
    y_h3, h3_model = sgn * C['d_h3'], sgn * C['m_h3']
    y_h4, h4_model = C['d_h4'], C['m_h4']

    # ---- example_analysis cell 12: errors and per-channel chi2 ----------------------------
    sig_xy = y_xy * 0.01
    sig_V, sig_s = 5.0, 5.0
    sig_h3, sig_h4 = C['e_h3'], C['e_h4']

    def _chi2(d, m, e): return float(np.sum(((d - m) / (e + 1e-12)) ** 2))

    chi2_dict = {
        'Sigma': _chi2(y_xy, density_model, sig_xy),
        'V':     _chi2(y_V, V_model, sig_V),
        'sigma': _chi2(y_sig, sigma_model, sig_s),
        'h1':    _chi2(sgn * C['d_h1'], sgn * C['m_h1'], C['e_h1']),
        'h2':    _chi2(C['d_h2'], C['m_h2'], C['e_h2']),
        'h3':    _chi2(y_h3, h3_model, sig_h3),
        'h4':    _chi2(y_h4, h4_model, sig_h4),
    }
    print('\nPer-channel chi^2 (sum over %d bins), example_analysis definitions:' % y_xy.size)
    for k, v in chi2_dict.items():
        print('  %-6s: %10.2f' % (k, v))

    # ---- example_analysis cell 13: helpers --------------------------------------------------
    try:
        import cmasher as cmr
        cmap_Sigma, cmap_V, cmap_sig = cmr.sepia, cmr.iceburn, cmr.amber
    except ImportError:
        cmap_Sigma, cmap_V, cmap_sig = 'magma', 'RdBu_r', 'inferno'

    ALPHA_SIGN = +1.0
    CONTOUR_HALF_X_KPC = 14.0
    CONTOUR_HALF_Y_KPC = 14.0
    CONTOUR_NBIN = 200
    CONTOUR_SIGMA = 1.0

    def _Rz(t):
        c, s = np.cos(t), np.sin(t); return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    def _Rx(t):
        c, s = np.cos(t), np.sin(t); return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])

    def make_rotation_matrix_deg(alpha, beta, gamma):
        a, b, g = np.radians(alpha), np.radians(beta), np.radians(gamma)
        return (_Rz(g) @ _Rx(b) @ _Rz(a)).T

    def rot2d(angle_deg):
        a = np.radians(angle_deg); c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s], [s, c]])

    def bar_angle_bar_strength(x, y, R_annulus=np.arange(1, 5, 0.25)):
        R = np.sqrt(x * x + y * y); phi = np.arctan2(y, x)
        angles = []
        for i in range(len(R_annulus) - 1):
            sel = (R > R_annulus[i]) & (R < R_annulus[i + 1])
            if sel.sum() == 0: angles.append(0.0); continue
            A2 = np.sum(np.cos(2 * phi[sel])) / sel.sum()
            B2 = np.sum(np.sin(2 * phi[sel])) / sel.sum()
            angles.append(0.5 * np.arctan2(B2, A2))
        return 0.5 * (R_annulus[:-1] + R_annulus[1:]), np.array(angles)

    def prepare_nbody(snap):
        """Load N-body snapshot, centre on disc COM, flip x convention, bar-align."""
        import agama
        from astropy.constants import G
        import astropy.units as u
        agama.setUnits(length=1, velocity=1, mass=1)
        mass_unit = 1 / ((G * u.Msun).to(u.kpc * (u.km / u.s) ** 2))
        w0, mass = agama.readSnapshot(snap)
        mass = mass * mass_unit.value
        mask = (mass != np.unique(mass)[-1])
        for r_ap in [10.0, 5.0, 3.0, 2.0]:
            R = np.sqrt(w0[:, 0] ** 2 + w0[:, 1] ** 2)
            mc = mask & (R < r_ap)
            m_c = mass[mc]
            for col in range(6):
                w0[:, col] -= np.sum(w0[mc, col] * m_c) / np.sum(m_c)
        w0 = w0[mask]; mass = mass[mask]
        w0[:, 0] = -w0[:, 0]; w0[:, 3] = -w0[:, 3]
        R_mid, bar_angles = bar_angle_bar_strength(w0[:, 0], w0[:, 1])
        bar_angle = float(np.mean(bar_angles[R_mid < 4]))
        c, s = np.cos(-bar_angle), np.sin(-bar_angle)
        x, y, z, vx, vy, vz = w0.T
        w0 = np.array([x * c - y * s, x * s + y * c, z,
                       vx * c - vy * s, vx * s + vy * c, vz]).T
        return w0, mass

    def build_pixel_polygons(X_pix, Y_pix, dX_sky, dY_sky, R2):
        corner_offsets_sky = 0.5 * np.array([[-dX_sky, -dY_sky], [+dX_sky, -dY_sky],
                                             [+dX_sky, +dY_sky], [-dX_sky, +dY_sky]])
        corner_offsets_fig = (R2 @ corner_offsets_sky.T).T
        polys = np.empty((X_pix.size, 4, 2))
        polys[:, :, 0] = X_pix[:, None] + corner_offsets_fig[None, :, 0]
        polys[:, :, 1] = Y_pix[:, None] + corner_offsets_fig[None, :, 1]
        return polys

    def build_contour(w0_intrinsic, mass, alpha, beta, gamma, R2):
        Rproj = make_rotation_matrix_deg(alpha, beta, gamma)
        xyz_sky = (Rproj @ w0_intrinsic[:, :3].T).T
        xy_fig = (R2 @ xyz_sky[:, :2].T).T
        H, xe, ye = np.histogram2d(
            xy_fig[:, 0], xy_fig[:, 1], bins=CONTOUR_NBIN,
            range=[(-CONTOUR_HALF_X_KPC, CONTOUR_HALF_X_KPC),
                   (-CONTOUR_HALF_Y_KPC, CONTOUR_HALF_Y_KPC)],
            weights=mass,
        )
        H = scipy.ndimage.gaussian_filter(H, sigma=CONTOUR_SIGMA)
        H = np.maximum(H, 1e-30)
        log10H = np.log10(H)
        Hmax = H.max()
        levels = np.linspace(np.log10(Hmax) - 2.6, np.log10(Hmax) - 0.2, 6).tolist()
        return 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), log10H, levels

    # ---- example_analysis cell 14: disc-aligned frame from the best-fit angles -----------
    pn = [str(n) for n in C['param_names']]          # by NAME: the two layouts differ
    alpha_deg_fit = float(theta[pn.index('alpha')]) * 180 / np.pi
    beta_deg_fit = float(theta[pn.index('beta')]) * 180 / np.pi
    gamma_deg_fit = float(theta[pn.index('gamma')]) * 180 / np.pi

    angle_align = ALPHA_SIGN * alpha_deg_fit
    R2 = rot2d(angle_align)
    print('In-plane rotation applied: %.2f deg' % angle_align)

    bin_mapping = np.asarray(C['bin_mapping'])[:-1]   # drop sentinel
    X_reg, Y_reg = C['X_regular_grid'], C['Y_regular_grid']

    def to_pix(per_bin):
        return per_bin[bin_mapping]

    xy_fig = R2 @ np.stack([X_reg, Y_reg], axis=0)
    X_pix, Y_pix = xy_fig[0], xy_fig[1]
    X_min, X_max = float(C['X_minmax'][0]), float(C['X_minmax'][1])
    Y_min, Y_max = float(C['Y_minmax'][0]), float(C['Y_minmax'][1])
    nX, nY = int(C['nX_nY'][0]), int(C['nX_nY'][1])
    pix_polys = build_pixel_polygons(X_pix, Y_pix, (X_max - X_min) / nX, (Y_max - Y_min) / nY, R2)
    corners_fig = R2 @ np.array([[X_min, Y_min], [X_max, Y_min],
                                 [X_max, Y_max], [X_min, Y_max], [X_min, Y_min]]).T

    contour = None
    if A.nbody and os.path.isfile(A.nbody):
        print('Loading N-body snapshot for contour overlay...')
        w0_nb, mass_nb = prepare_nbody(A.nbody)
        contour = build_contour(w0_nb, mass_nb, alpha_deg_fit, beta_deg_fit, gamma_deg_fit, R2)
        print('  N-body disc particles: %d' % w0_nb.shape[0])
        del w0_nb, mass_nb
        mark('after N-body contours')
    else:
        print('No N-body snapshot at %s -- drawing without contours' % A.nbody)

    # ---- example_analysis cell 15: the 5x3 panel ------------------------------------------
    xlim = (-7.5, 7.5); ylim = (-5.5, 5.5)
    panels = [
        (r'$\Sigma_*$ [M$_\odot$/pc$^2$]', y_xy, density_model, cmap_Sigma, None, 1e1, 1e4, 0.30, chi2_dict['Sigma']),
        (r'$V_{\rm los}$ [km/s]', y_V, V_model, cmap_V, 5.0, -200, 200, 5.0, chi2_dict['V']),
        (r'$\sigma_v$ [km/s]', y_sig, sigma_model, cmap_sig, 5.0, 20, 200, 5.0, chi2_dict['sigma']),
        (r'$h_3$', y_h3, h3_model, cmap_V, sig_h3, -0.2, 0.2, 5.0, chi2_dict['h3']),
        (r'$h_4$', y_h4, h4_model, cmap_V, sig_h4, -0.2, 0.2, 5.0, chi2_dict['h4']),
    ]
    n_rows = len(panels)
    asp = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    fig_w = 12.0
    panel_axis_w = (fig_w * 0.88) / (3.0 + 2 * 0.05 + 0.18)
    fig_h = panel_axis_w * asp * n_rows / 0.92
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(n_rows, 6, width_ratios=[1, 1, 0.05, 0.32, 1, 0.05],
                           hspace=0.0, wspace=0.05, left=0.06, right=0.96, top=0.96, bottom=0.05)

    def _make_pc(vals_pix, cmap, vmin, vmax, log_norm=False):
        if log_norm:
            pc = PolyCollection(pix_polys, array=vals_pix, cmap=cmap, edgecolors='face',
                                linewidths=0.5, antialiased=True, rasterized=True,
                                norm=LogNorm(vmin=vmin, vmax=vmax))
        else:
            pc = PolyCollection(pix_polys, array=vals_pix, cmap=cmap, edgecolors='face',
                                linewidths=0.5, antialiased=True, rasterized=True)
            pc.set_clim(vmin, vmax)
        return pc

    def _draw_contour(ax):
        if contour is None:
            return
        xm, ym, log10H, levels = contour
        ax.contour(xm, ym, log10H.T, levels=levels, colors='dimgrey', linewidths=1, alpha=0.75)

    def _style_axes(ax, is_last_row, is_first_col):
        ax.set_xlim(xlim[1], xlim[0]); ax.set_ylim(ylim); ax.set_aspect('equal')
        if is_last_row:
            ax.set_xlabel('X [kpc]', fontsize=14)
        else:
            ax.set_xticklabels([])
        if is_first_col:
            ax.set_ylabel('Y [kpc]', fontsize=14)
        else:
            ax.set_yticklabels([])

    for i, (label, data_b, model_b, cmap, err_b, vmin, vmax, res_scale, chi2_val) in enumerate(panels):
        data_pix = to_pix(np.asarray(data_b))
        model_pix = to_pix(np.asarray(model_b))
        if err_b is None:
            res = (data_pix - model_pix) / np.maximum(np.abs(data_pix), 1e-12)
            res_label = r'$\frac{\Sigma_{\rm data} - \Sigma_{\rm model}}{\Sigma_{\rm data}}$'
        elif np.isscalar(err_b):
            res = (data_pix - model_pix) / err_b
            res_label = 'difference / error'
        else:
            res = (data_pix - model_pix) / (to_pix(np.asarray(err_b)) + 1e-12)
            res_label = 'difference / error'

        is_log = (i == 0); is_last = (i == n_rows - 1)
        ax_M = fig.add_subplot(gs[i, 0]); ax_D = fig.add_subplot(gs[i, 1])
        cax_M = fig.add_subplot(gs[i, 2])
        ax_R = fig.add_subplot(gs[i, 4]); cax_R = fig.add_subplot(gs[i, 5])

        pc_M = _make_pc(model_pix, cmap, vmin, vmax, log_norm=is_log)
        pc_D = _make_pc(data_pix, cmap, vmin, vmax, log_norm=is_log)
        ax_M.add_collection(pc_M); ax_D.add_collection(pc_D)
        cb = fig.colorbar(pc_D, cax=cax_M); cb.set_label(label, fontsize=14)
        cb.ax.tick_params(labelsize=10)

        pc_R = _make_pc(res, 'coolwarm', -res_scale, res_scale, log_norm=False)
        ax_R.add_collection(pc_R)
        cb = fig.colorbar(pc_R, cax=cax_R)
        cb.set_label(res_label, fontsize=(16 if i == 0 else 11))
        cb.ax.tick_params(labelsize=10)

        for ax in (ax_M, ax_D, ax_R):
            _draw_contour(ax)
            ax.plot(corners_fig[0], corners_fig[1], color='k', lw=0.4, alpha=0.25)
        if i == 0:
            ax_M.set_title('Model', fontsize=14)
            ax_D.set_title('Data', fontsize=14)
            ax_R.set_title('Residuals', fontsize=14)

        _style_axes(ax_M, is_last, is_first_col=True)
        _style_axes(ax_D, is_last, is_first_col=False)
        _style_axes(ax_R, is_last, is_first_col=False)

        if True:                                     # Sigma too: it dominates this likelihood
            ax_R.text(0.03, 0.95, rf'$\chi^2 = {chi2_val:.0f}$', transform=ax_R.transAxes,
                      fontsize=12, va='top', ha='left',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black',
                                alpha=0.7))

    _pt = str(C['point']) if 'point' in C.files else 'bestfit'
    _shape = ''
    if 'logRd_disk' in pn:
        _shape = (r'$R_d$ %.2f kpc, $h_z$ %.2f kpc, $L_{\rm bar}$ %.2f kpc, $f_{\rm bar}$ %.3f, '
                  % (10 ** theta[pn.index('logRd_disk')], 10 ** theta[pn.index('loghz_disk')],
                     10 ** theta[pn.index('logL_bar')], theta[pn.index('f_bar')]))
    fig.suptitle('%s -- %s%s'
                 % ('GROUND TRUTH (cell 10)' if _pt == 'truth' else
                    'BEST FIT (max logP after %d steps)' % int(C['discard']),
                    _shape,
                    r'logL %.2f  ($\chi^2_{\rm den}$ %.0f, $\chi^2_{\rm kin}$ %.0f, 2$m_{\rm eff}$ %.0f)'
                    % (float(C['logL']), float(C['chi2_den']), float(C['chi2_kin']),
                       2 * float(C['m_eff_kin']))), fontsize=12, y=0.995)
    os.makedirs(A.outdir, exist_ok=True)
    out = os.path.join(A.outdir, 'datavsmodel_%s_%s.png' % (A.tag, A.frame))
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('figure -> %s' % os.path.relpath(out, HERE))
    mark('end of plot stage')


def cache_is_stale():
    """True if there is no cache, or it was built from a different chain or chain length --
    e.g. from the checkpoint, before mcmc_results_aic.pkl existed."""
    if not os.path.isfile(A.cache):
        return True
    C = np.load(A.cache, allow_pickle=False)
    S, _, label = load_chain()
    mock = str(C['mock']) if 'mock' in C.files else 'mock_Nbody_galaxy_beta65_gamma45_D50.pkl'
    point = str(C['point']) if 'point' in C.files else 'bestfit'
    stale = (str(C['chain_label']) != label or int(C['n_steps']) != S.shape[0]
             or int(C['discard']) != A.discard or mock != A.mock or point != A.point)
    if stale:
        print('cache was built from %s (%d steps, discard %d, %s); now %s (%d steps, '
              'discard %d, %s) -> recomputing'
              % (C['chain_label'], int(C['n_steps']), int(C['discard']), mock,
                 label, S.shape[0], A.discard, A.mock))
    return stale


if A.stage == 'model' or (A.stage == 'all' and (A.recompute or cache_is_stale())):
    stage_model()
if A.stage in ('all', 'plot'):
    stage_plot()
print('peak RSS %.2f GB (cap %.1f)' % (_rss_gb(), A.cap))
