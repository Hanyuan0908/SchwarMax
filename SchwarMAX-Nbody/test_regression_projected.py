#!/usr/bin/env python
"""Regression gate for the PROJECTED (line-of-sight) Schwarzschild scheme.

Runs test_example_aic_expdisc_nb.py on both exponential-disc notebooks and requires control 5 --
the vmapped likelihood at the truth and at beta + 0.03 rad -- to reproduce the values pinned below.
Every commit to the shared package must pass this, from either workstream (docs/DEVELOPMENT.md).
A change that is MEANT to move these numbers must say so in its commit message and re-pin them here.

Pinned 2026-10-20 at commit 9e629d1: CPU, float32, jax 0.7.2, numpy 2.4.3, the test's defaults
(600 orbits, n_realizations 2, n_boot_meff 3).  Three independent runs gave identical values.
Takes ~1 min and ~4 GB per notebook.

    python test_regression_projected.py [--tol 0.01]
"""
import argparse, os, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
P = argparse.ArgumentParser()
P.add_argument('--tol', type=float, default=0.01, help='absolute logL tolerance')
A = P.parse_args()

PINNED = {
    'SchwarMAX_Nbody_AIC_expdisc.ipynb':       (-272.49628, -279.69803),
    'SchwarMAX_Nbody_AIC_expdisc_field.ipynb': (-192.75204, -204.71582),
}

failed = []
for nb, vals in PINNED.items():
    t0 = time.time()
    with tempfile.TemporaryDirectory() as scratch:
        r = subprocess.run([sys.executable, os.path.join(HERE, 'test_example_aic_expdisc_nb.py'),
                            '--nb', os.path.join(HERE, nb), '--scratch', os.path.join(scratch, 's'),
                            # '=' form: the values are negative and would parse as options
                            '--expect=%r,%r' % vals, '--tol', str(A.tol)],
                           capture_output=True, text=True)
    ok = r.returncode == 0
    line = [l for l in r.stdout.splitlines() if 'expected' in l or 'vmapped logL' in l]
    print('%-42s %s  (%.0f s)' % (nb, 'PASS' if ok else 'FAIL', time.time() - t0))
    for l in line:
        print('   ' + l.strip())
    if not ok:
        failed.append(nb)
        print('\n'.join(r.stdout.splitlines()[-15:]))
        print('\n'.join(r.stderr.splitlines()[-15:]))

if failed:
    sys.exit('REGRESSION GATE FAILED: %s' % ', '.join(failed))
print('\nREGRESSION GATE PASSED: the projected scheme reproduces its pinned likelihood values.')
