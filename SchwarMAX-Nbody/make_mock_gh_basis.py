#!/usr/bin/env python
"""Rewrite the mock's V and sigma in the Gauss-Hermite basis that NGC 1512 uses.

NGC 1512 (`ngc1512_snr230_recon_rc_v1.pkl`, meta `gh_basis: v0=V_ppxf,s=sigma_ppxf,h1=h2=0`)
defines V and sigma as the GH REFERENCE GAUSSIAN -- pPXF's V and sigma:
    V_mean == v0,  V_sigma == s  (exactly, every bin),  h1 = h2 = 0,
    V_mean_err == sqrt2*s*h1_err,  V_sigma_err == sqrt2*s*h2_err.
The mock satisfies everything except the first line: its V_mean / V_sigma are NOT v0 / s
(rms 4.5 / 5.4 km/s apart, i.e. about one 5 km/s error bar), and they track sqrt3*s*h3 /
sqrt6*s*h4 (corr 0.975 / 0.943), as plain moments of a skewed LOSVD would.

The new pipeline's model V and sigma are the GH reference Gaussian (`h_to_V_sigma_GH`,
V = v0 + sqrt2*s*h1, sigma = s*(1 + sqrt2*h2)), so against V_mean / V_sigma the V and sigma
residuals mix two definitions; against v0 / s they reduce EXACTLY to the h1 / h2 residuals.

Only V_mean and V_sigma change.  Neither enters any likelihood (old or new: `V_data` and
`sigma_data` are loaded by `load_data_bootstrap` and read by nothing in model.py), so a fit is
unaffected.  The original values are kept under `V_mean_moment` / `V_sigma_moment`, which
`load_data_bootstrap` ignores.  The ORIGINAL file is left untouched: the old `build_model`
reports V and sigma with `h_to_V_sigma` (plain moments) and is consistent with it.

    python make_mock_gh_basis.py
"""
import hashlib, os, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # the repository: the shared package (model.py, ...) and example/ (mock data, N-body snapshot, run outputs)
SRC = os.path.join(ROOT, 'example', 'mock_Nbody_galaxy_beta65_gamma45_D50.pkl')
DST = os.path.join(ROOT, 'example', 'mock_Nbody_galaxy_beta65_gamma45_D50_GH.pkl')

with open(SRC, 'rb') as f:
    raw = f.read()
d = pickle.loads(raw)
a = lambda k: np.asarray(d[k], float)
v0, s = a('v0'), a('s')

# The GH basis is only consistent if the data are already expanded about (v0, s).
assert np.abs(a('h1')).max() < 1e-5 and np.abs(a('h2')).max() < 1e-5, 'h1/h2 are not zero'
assert np.allclose(a('V_mean_err'), np.sqrt(2) * s * a('h1_err'), rtol=1e-5), 'V_err != sqrt2 s h1_err'
assert np.allclose(a('V_sigma_err'), np.sqrt(2) * s * a('h2_err'), rtol=1e-5), 'sig_err != sqrt2 s h2_err'
assert 'V_mean_moment' not in d, 'input already converted'

dV, dS = a('V_mean') - v0, a('V_sigma') - s
print('before:  V_mean - v0   rms %.2f  max %.2f km/s' % (np.sqrt(np.mean(dV ** 2)), np.abs(dV).max()))
print('         V_sigma - s   rms %.2f  max %.2f km/s' % (np.sqrt(np.mean(dS ** 2)), np.abs(dS).max()))

out = dict(d)
out['V_mean_moment'] = d['V_mean']
out['V_sigma_moment'] = d['V_sigma']
out['V_mean'] = np.asarray(d['v0']).astype(np.asarray(d['V_mean']).dtype).copy()
out['V_sigma'] = np.asarray(d['s']).astype(np.asarray(d['V_sigma']).dtype).copy()
out['meta'] = {
    'gh_basis': 'v0=V_ppxf,s=sigma_ppxf,h1=h2=0',      # NGC 1512's own string
    'derived_from': os.path.basename(SRC),
    'derived_from_md5': hashlib.md5(raw).hexdigest(),
    'changed': 'V_mean := v0, V_sigma := s; originals kept as V_mean_moment, V_sigma_moment',
}

assert np.array_equal(out['V_mean'], d['v0']) and np.array_equal(out['V_sigma'], d['s'])
for k in d:                                            # nothing else moved
    if k not in ('V_mean', 'V_sigma'):
        assert out[k] is d[k], k
with open(DST, 'wb') as f:
    pickle.dump(out, f)
print('after:   V_mean == v0, V_sigma == s in all %d bins' % v0.size)
print('wrote %s' % os.path.relpath(DST, HERE))
