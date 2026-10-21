#!/usr/bin/env python
"""Build a CHEAP version of an AIC notebook: the dAIC calibration and the MCMC only.

Input:  an exponential-disc AIC notebook (default the IFU-field, 3%-logL one)
Output: <input>_mcmc_only.ipynb

What is dropped, and what replaces it:
  cell 10   the truth is DEFINED but no longer evaluated (chi2_report + logL at the truth: one
            JIT compile and one model evaluation)
  cell 11   the three likelihood scans (27 evaluations)   -> a note
  cell 13   the minimiser (968 evaluations, ~1 h)          -> a LOADER for the minimiser ensemble of
            an earlier run, RUN_TAG PREV_TAG.  Cell 15 needs that ensemble's spread (pos) and best
            point (best); cell 17 starts its walkers around MINIMISER_RESULT.
Everything else is the input notebook's own, byte for byte -- in particular the likelihood, the
dAIC calibration (cell 15), the emcee cell (17), the corner (18) and RUN_TAG, so the chain is the
same likelihood's chain as the full notebook would write, and the analysis scripts map it to the
full notebook.  The cell count stays 19, so test_example_aic_expdisc_nb.py addresses the same cells.

The loaded ensemble's logL values were computed under the PREVIOUS run's likelihood; they only pick
the best point and set the spread, which is all cells 15 and 17 use them for.

    python build_example_mcmc_only_nb.py [--base SchwarMAX_Nbody_AIC_expdisc_field_den3.ipynb]
                                         [--prev-tag aic_exp_field]
"""
import argparse, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
A = argparse.ArgumentParser()
A.add_argument('--base', default=os.path.join(HERE, 'SchwarMAX_Nbody_AIC_expdisc_field_den3.ipynb'))
A.add_argument('--prev-tag', dest='prev_tag', default='aic_exp_field',
               help='RUN_TAG of the run whose minimiser ensemble is loaded')
A.add_argument('--out', default=None)
A = A.parse_args()
A.out = A.out or A.base.replace('.ipynb', '_mcmc_only.ipynb')

nb = json.load(open(A.base))
CELLS = [[c['cell_type'], ''.join(c['source'])] for c in nb['cells']]
assert len(CELLS) == 19, 'the base notebook changed its cell count (%d)' % len(CELLS)
for i, head in ((10, '# The disc and bar shapes and f_bar'), (11, '# The old notebook scans M/L and Omega'),
                (12, '# Optimisation pipeline'), (13, '# Multi-start optimisation'),
                (15, 'N_CAL  = 24'), (17, '# ======')):
    assert CELLS[i][1].startswith(head), 'cell %d is not what this builder expects: %r' % (i, CELLS[i][1][:60])
TAG = re.search(r"RUN_TAG = '([^']+)'", CELLS[8][1]).group(1)

# ---------------------------------------------------------------- cell 0: say what this is
CELLS[0][1] = ('''# CHEAP RUN: the $\\Delta$AIC calibration and the MCMC only

**This is `%s` with the expensive set-up removed.** Nothing in the model or the likelihood
changes; it is the same likelihood, so the chain is written with the same `RUN_TAG = '%s'`.

* **No likelihood scans** (cell 11) and **no evaluation at the truth** (cell 10 only defines it).
* **No minimiser** (cell 13): it **loads the minimiser ensemble of an earlier run**,
  `PREV_TAG = '%s'` -- the files `example/minimiser_pos_%s.npy`, `minimiser_val_%s.npy` and
  `minimisation_result_%s.npy`. Their logL values belong to that run's likelihood; here they only
  pick the best point and set the ensemble spread, which is all cells 15 and 17 use.
* Then the $\\Delta$AIC calibration (cell 15) and the MCMC (cell 17), unchanged.

---

''' % (os.path.basename(A.base), TAG, A.prev_tag, A.prev_tag, A.prev_tag, A.prev_tag)) + CELLS[0][1]

# ---------------------------------------------------------------- cell 10: define, do not evaluate
c10 = CELLS[10][1]
CELLS[10][1] = c10[:c10.index('t0 = time.time()')].rstrip() + '''

# (the full notebook evaluates chi2_report and logL here; this cheap run skips that evaluation)
print('ground truth defined:', ', '.join('%s %.4f' % (n, float(v)) for n, v in zip(param_names, theta_true)))'''

# ---------------------------------------------------------------- cell 11: the scans -> a note
CELLS[11] = ['markdown', '''*The likelihood scans of the full notebook (M/L, $\\Omega$, inclination; 27 evaluations) are
skipped in this cheap run.*''']

# ---------------------------------------------------------------- cells 12-13: load the minimiser
CELLS[12][1] = '''# The minimiser ensemble of an earlier run

The minimiser is not run here. Its saved ensemble is loaded instead: every particle's final
position (`pos`), its logL (`val`) and the best one (`best`). The $\\Delta$AIC calibration below
steps away from `best` in units of the ensemble spread, and the MCMC starts its walkers in a ball
around `best`.'''
CELLS[13][1] = '''# Load the minimiser ensemble of an EARLIER run (no minimisation in this notebook).
PREV_TAG = %r      # <<< the RUN_TAG of the run whose minimiser ensemble to use
MINIMISER_RESULT = os.path.join(path, 'example/minimisation_result_%%s.npy' %% PREV_TAG)
MINIMISER_POS    = os.path.join(path, 'example/minimiser_pos_%%s.npy' %% PREV_TAG)
MINIMISER_VAL    = os.path.join(path, 'example/minimiser_val_%%s.npy' %% PREV_TAG)
for _f in (MINIMISER_RESULT, MINIMISER_POS, MINIMISER_VAL):
    assert os.path.isfile(_f), ('%%s is missing -- run the full notebook with RUN_TAG %%r first, or set '
                                'PREV_TAG to a run whose minimiser files are in example/' %% (_f, PREV_TAG))

pos = np.load(MINIMISER_POS)
val = np.load(MINIMISER_VAL)
best = pos[int(np.argmax(val))]
assert pos.shape[1] == NDIM, 'the loaded ensemble has %%d parameters, this notebook %%d' %% (pos.shape[1], NDIM)
assert np.allclose(best, np.load(MINIMISER_RESULT)), 'minimisation_result does not match the best of the ensemble'
print('loaded the minimiser ensemble of run %%r: %%d particles; best logL %%.4f (under THAT run\\'s likelihood),'
      ' spread %%.4f' %% (PREV_TAG, len(val), val.max(), float(np.std(val))))
print('%%-26s %%11s %%11s %%11s' %% ('parameter', 'truth', 'best fit', 'offset'))
for j, n in enumerate(param_names):
    t = float(np.asarray(theta_true)[j])
    print('%%-26s %%11.4f %%11.4f %%+11.4f' %% (n, t, best[j], best[j] - t))''' % A.prev_tag

# ================================================================= write + guards
cells = []
for t, s in CELLS:
    lines = s.split('\n')
    c = {'cell_type': t, 'metadata': {}, 'source': [l + '\n' for l in lines[:-1]] + [lines[-1]]}
    if t == 'code':
        c['execution_count'] = None
        c['outputs'] = []
    cells.append(c)
out = {'cells': cells, 'metadata': nb.get('metadata', {}), 'nbformat': 4, 'nbformat_minor': 5}
code = '\n'.join(s for t, s in CELLS if t == 'code')
assert 'parallel_minimize(' not in code and 'chi2_report(theta_true)' not in code
assert 'best = pos[int(np.argmax(val))]' in code and 'res = np.load(MINIMISER_RESULT)' in code
assert ("RUN_TAG = '%s'" % TAG) in code
assert len(cells) == 19
json.dump(out, open(A.out, 'w'), indent=1)
print('wrote %s  (19 cells; RUN_TAG %r; loads the minimiser ensemble of %r)' % (A.out, TAG, A.prev_tag))
