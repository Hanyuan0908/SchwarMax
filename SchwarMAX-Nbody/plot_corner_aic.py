#!/usr/bin/env python
"""Corner plot of an AIC chain against the generating truth -- the notebook's cell 18, run on a
downloaded checkpoint or results file, plus split R-hat and the pull table.

The truth is NOT retyped here: `ground_truth` is exec'd out of the notebook's own cell 10 (only
the lines that define it, so no model is run), and the parameter names out of cell 8.  The
notebook is picked from the chain's dimension: 12 -> the live 12-parameter notebook (stellar mass
derived from M/L), 13 -> archive/, whose cell 10 holds the OLD MINIMISER, not the truth, so the
13-parameter case is refused rather than plotted against the wrong values.

An emcee HDF5 backend (*.h5) is read directly.  A chain whose name carries `aic_exp` takes its
names and truth from the exponential-disc notebook instead, and its default burn-in is that
notebook's own rule, 3 x the longest autocorrelation time.

    python plot_corner_aic.py                                  # newest *_aic_tied chain
    python plot_corner_aic.py --chain example/mcmc_results_aic_tied.pkl --burnin 300
    python plot_corner_aic.py --chain example/emcee_aic_exp.h5             # exponential disc
"""
import argparse, json, os, pickle, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # the repository: the shared package (model.py, ...) and example/ (mock data, N-body snapshot, run outputs)
P = argparse.ArgumentParser()
P.add_argument('--chain', default=None,
               help='default example/mcmc_results_aic_tied.pkl, else mcmc_checkpoint_aic_tied.pkl')
P.add_argument('--burnin', type=int, default=None,
               help="the notebook's BURNIN: 300 for a BlackJAX pickle, 3 tau for an emcee .h5")
P.add_argument('--outdir', default=os.path.join(HERE, 'figures'))
A = P.parse_args()

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import corner

# ---------------------------------------------------------------- the chain
if A.chain is None:
    for k in ('results', 'checkpoint'):
        A.chain = os.path.join(ROOT, 'example', 'mcmc_%s_aic_tied.pkl' % k)
        if os.path.isfile(A.chain):
            break
EXP = 'aic_exp' in os.path.basename(A.chain)
if A.chain.endswith('.h5'):                          # emcee's live HDF5 backend
    import emcee
    _b = emcee.backends.HDFBackend(A.chain, read_only=True)
    S, L = _b.get_chain(), _b.get_log_prob()
    _tau = _b.get_autocorr_time(tol=0)
    if A.burnin is None:                             # the emcee cell's BURNIN rule
        A.burnin = int(3 * np.nanmax(_tau))
    print('emcee autocorrelation time %.0f - %.0f steps' % (np.nanmin(_tau), np.nanmax(_tau)))
else:
    d = pickle.load(open(A.chain, 'rb'))
    if 'chain' in d:                                 # cell 17's final results
        S, L = np.asarray(d['chain']), np.asarray(d['logprob'])
    else:                                            # cell 17's checkpoint
        S, L = np.stack(d['all_samples'], 0), np.stack(d['all_logprob'], 0)
    if A.burnin is None:
        A.burnin = 300
n_step, n_chain, ndim = S.shape
assert ndim == 12, ('%d-D chain: only the 12-parameter layout has the real truth in its cell 10 '
                    '(the archived 13-parameter notebook lists the old minimiser there)' % ndim)
assert A.burnin < n_step, 'burn-in %d >= %d steps' % (A.burnin, n_step)

# ---------------------------------------------------------------- names + truth, from the notebook
FIELD = 'aic_exp_field' in os.path.basename(A.chain)
nb = json.load(open(os.path.join(HERE, 'SchwarMAX_Nbody_AIC_expdisc_field.ipynb' if FIELD else
                                 'SchwarMAX_Nbody_AIC_expdisc.ipynb' if EXP
                                 else 'SchwarMAX_Nbody_AIC.ipynb')))
c8, c10 = (''.join(nb['cells'][i]['source']) for i in (8, 10))
m = re.search(r"param_names = (\[.*?\])", c8, re.S)
names = eval(m.group(1))
truth_src = c10[:c10.index('theta_true')]            # _Md, _Mb, ground_truth -- nothing else
assert ('logRd_disk' in names) == EXP, 'the chain name and the notebook disagree on the disc family'
g = {'np': np}
exec(truth_src, g)
truth = list(g['ground_truth']) + [None]             # log10(lambda) has no truth
J_LAM = names.index('log10_lambda')
assert len(names) == ndim and len(truth) == ndim and truth[J_LAM] is None

# ---------------------------------------------------------------- diagnostics
post = S[A.burnin:]                                  # (n, chains, ndim)
flat = post.reshape(-1, ndim)
acc = float(np.mean(np.any(np.diff(S[A.burnin:], axis=0) != 0, axis=2)))


def split_rhat(x):
    """Gelman-Rubin on split chains; x is (n, chains)."""
    n = x.shape[0] // 2
    c = np.concatenate([x[:n], x[n:2 * n]], axis=1)  # (n, 2*chains)
    W = np.mean(np.var(c, axis=0, ddof=1))
    B = n * np.var(np.mean(c, axis=0), ddof=1)
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


print('chain %s: %d steps x %d chains x %d dims; burn-in %d -> %d samples'
      % (os.path.relpath(A.chain, HERE), n_step, n_chain, ndim, A.burnin, flat.shape[0]))
print('acceptance after burn-in %.3f | logP median %.2f, max %.2f'
      % (acc, np.median(L[A.burnin:]), L[A.burnin:].max()))
print('\n%-26s %10s %10s %10s %8s %7s' % ('parameter', 'truth', 'median', '+/-', 'pull', 'R-hat'))
rows = []
for j, n in enumerate(names):
    q16, q50, q84 = np.percentile(flat[:, j], [16, 50, 84])
    s = 0.5 * (q84 - q16)
    rh = split_rhat(post[:, :, j])
    t = truth[j]
    pull = (q50 - t) / s if (t is not None and s > 0) else np.nan
    rows.append((n, t, q50, s, pull, rh))
    print('%-26s %10s %10.4f %10.4f %8s %7.3f'
          % (n, '--' if t is None else '%10.4f' % t, q50, s,
             '--' if t is None else '%8.2f' % pull, rh))
for n, conv in (('alpha', np.degrees), ('beta', np.degrees), ('gamma', np.degrees)):
    j = names.index(n)
    q16, q50, q84 = np.percentile(conv(flat[:, j]), [16, 50, 84])
    print('   %-6s %.2f +/- %.2f deg (truth %.1f)' % (n, q50, 0.5 * (q84 - q16), np.degrees(truth[j])))
jO = names.index('log_Omega')
KPCGYR_TO_KMS = 0.9777922216807892     # constants.py: 10**log_Omega is in 1/Gyr, not km/s/kpc
q16, q50, q84 = np.percentile(10 ** flat[:, jO] * KPCGYR_TO_KMS, [16, 50, 84])
print('   Omega  %.2f +/- %.2f km/s/kpc (truth %.2f)'
      % (q50, 0.5 * (q84 - q16), 10 ** truth[jO] * KPCGYR_TO_KMS))

# ---------------------------------------------------------------- the figure (cell 18)
# Axis ranges that always CONTAIN the truth.  corner's default range is the samples' span, and a
# truth outside it is silently not drawn -- so the parameter that missed most would be the one
# whose truth line is hidden.
rng_ = []
for j in range(ndim):
    lo, hi = np.percentile(flat[:, j], [0.5, 99.5])
    if truth[j] is not None:
        lo, hi = min(lo, truth[j]), max(hi, truth[j])
    rng_.append((lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)))
fig = corner.corner(flat, labels=names, truths=truth, truth_color='tab:red', range=rng_,
                    quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.3f',
                    title_kwargs={'fontsize': 8}, label_kwargs={'fontsize': 8})
fig.suptitle('AIC pipeline, %sstellar mass tied to the light -- red = %s\n'
             % ('exponential disc, IFU-field M/L tie, ' if FIELD else 'exponential disc, ' if EXP else '',
                'exp-family fit to the N-body (halo, angles, Omega, M/L: generating)' if EXP
                else 'generating truth') +
             '%s: %d steps x %d chains, burn-in %d, acceptance %.2f'
             % (os.path.basename(A.chain), n_step, n_chain, A.burnin, acc), fontsize=13)
os.makedirs(A.outdir, exist_ok=True)
tag = re.sub(r'\.(pkl|h5)$', '', os.path.basename(A.chain)).replace('mcmc_', '').replace('emcee_', '')
out = os.path.join(A.outdir, 'corner_%s_%dsteps.png' % (tag, n_step))
fig.savefig(out, dpi=110, bbox_inches='tight')
print('\nwrote %s' % os.path.relpath(out, HERE))
