#!/usr/bin/env python
"""Build the IFU-FIELD-NORMALISED variant of the exponential-disc mock notebook.

Input:  SchwarMAX_Nbody_AIC_expdisc.ipynb (as the user last ran it -- their emcee settings are kept)
Output: SchwarMAX_Nbody_AIC_expdisc_field.ipynb

ONE thing changes: the stellar mass is tied to the light inside the IFU FIELD (the data's own
footprint, sky |X| < 7.27, |Y| < 4.85 kpc) instead of a 20 kpc sky box:

    M_star * p_box(theta) = M/L * L_BOX,   "box" = the IFU field,  L_BOX = sum of the data's light

Why (COLLAB_PROGRESS_LOG 2026-10-20(b)): with the 20 kpc box, the model's LIGHT in the box is
L_BOX for any M/L, so the share inside the field is set by the shapes.  The exponential disc
over-fills 10-20 kpc where the N-body is truncated; at the truth's shapes only 0.676 of the box
light lands in the field against the N-body's 0.835, and the chain shrank Rd / L_bar to put it back.

The code names (L_BOX, p_box, box_fractions) are kept so every analysis script still runs; only
the region changes.  The region and L_BOX are read from dict_data, not typed in.  Quadrature:
128 x 96 sky nodes (sinh a = 2), measured within 6e-5 of a uniform 1000 x 667 reference over
hz 0.1-2, Rd 2-12, L 2-8, i 40-80 deg.

Also: EMCEE_WALKERS 24 -> 32 (what the user ran on Colab; 24 with chunk 8 pads a quarter of every
batch), and the EMCEE_CHUNK comment now says WALKERS // 4 (the snooker move uses 4 groups).
Every output file carries RUN_TAG = 'aic_exp_field'.  Outputs are cleared.

    python build_example_aic_expdisc_field_nb.py
"""
import argparse, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
A = argparse.ArgumentParser()
A.add_argument('--base', default=os.path.join(HERE, 'SchwarMAX_Nbody_AIC_expdisc.ipynb'))
A.add_argument('--out', default=None, help='default SchwarMAX_Nbody_AIC_expdisc_field[_den<N>].ipynb')
A.add_argument('--den-logl', dest='den_logl', type=float, default=0.01,
               help="ERR_DEN_LOGL, the surface-density error in logL only (the NNLS keeps its 1%%). "
                    "Anything but 0.01 builds a VARIANT with its own RUN_TAG, aic_exp_field_den<percent>, "
                    "so a resumed run never continues a chain made with another likelihood.")
A = A.parse_args()
VARIANT = abs(A.den_logl - 0.01) > 1e-12
TAG = 'aic_exp_field' + ('_den%g' % (100 * A.den_logl) if VARIANT else '')
A.out = A.out or os.path.join(HERE, 'SchwarMAX_Nbody_AIC_expdisc_%s.ipynb' % TAG[len('aic_exp_'):])

nb = json.load(open(A.base))
CELLS = [[c['cell_type'], ''.join(c['source'])] for c in nb['cells']]
assert len(CELLS) == 19, 'the expdisc notebook changed its cell count (%d)' % len(CELLS)
for i, head in ((0, '# SchwarMAX — the AIC pipeline on the mock N-body galaxy: double-exponential disc'),
                (7, '# The AIC likelihood'),
                (8, '# ================================================================= CONFIGURATION'),
                (9, '# Example code for calling'), (17, '# ======')):
    assert CELLS[i][1].startswith(head), 'cell %d is not what this builder expects: %r' % (i, CELLS[i][1][:60])


def edit(i, old, new):
    assert CELLS[i][1].count(old) == 1, 'cell %d: anchor not found exactly once: %r' % (i, old[:70])
    CELLS[i][1] = CELLS[i][1].replace(old, new)


# ---------------------------------------------------------------- cell 0: what this variant is
edit(0, '# SchwarMAX — the AIC pipeline on the mock N-body galaxy: double-exponential disc\n',
     '''# SchwarMAX — the AIC pipeline on the mock N-body galaxy: double-exponential disc, IFU-field normalisation

**This is the IFU-field-normalised version of `SchwarMAX_Nbody_AIC_expdisc.ipynb`.** One thing
differs: the stellar mass is tied to the light **inside the IFU field** ($|X|<7.27$,
$|Y|<4.85$ kpc on the sky — the data's own footprint) instead of a 20 kpc sky box. With the box,
the model's light in the box is $L_{\\rm box}$ whatever $M/L$ is, so the share of it inside the
field is set by the shapes alone. The exponential disc over-fills 10–20 kpc, where the N-body is
truncated: at the truth's shapes only 68% of the box light falls in the field against the
N-body's 83.5%, and the box-tied chain shrank $R_d$ and $L_{\\rm bar}$ to put the light back
(`COLLAB_PROGRESS_LOG.md`, 2026-10-20(b)). Every file this notebook writes carries
`RUN_TAG = '__TAG__'`.

---

''')
edit(0, "1512's form. Every file it writes carries `RUN_TAG = 'aic_exp'`.",
     "1512's form.")
edit(0, '**derived**: $M/L\\times L_{\\rm box}/p_{\\rm box}(\\theta)$, split by $f_{\\rm bar}$',
     '**derived**: $M/L\\times L_{\\rm IFU}/p_{\\rm IFU}(\\theta)$, split by $f_{\\rm bar}$')

# ---------------------------------------------------------------- cell 7: the normalisation, in words
edit(7, '''$$M_\\star\\,p_{\\rm box}(\\theta)=(M/L)\\,L_{\\rm box},\\qquad''',
     '''$$M_\\star\\,p_{\\rm IFU}(\\theta)=(M/L)\\,L_{\\rm IFU},\\qquad''')
_a = CELLS[7][1].index('* $L_{\\rm box}$ is the light inside a sky box')
_b = CELLS[7][1].index('**Why.**')
CELLS[7][1] = CELLS[7][1][:_a] + '''* $L_{\\rm IFU}$ is the light **inside the IFU field**, summed from the data themselves: each
  Voronoi bin's surface density times its pixel count times the pixel area ($5.349\\times10^{10}$
  for the mock, every pixel binned). For a real galaxy, the same sum over the image inside the IFU
  footprint — no light from outside the data is needed.
* $p_{\\rm IFU}(\\theta)$ is the fraction of the **model's** stellar mass that projects into the
  field at the current shapes and viewing angles, recomputed at every $\\theta$ (at the generating
  parameters: 44% of $M_{\\rm disc}$, 99% of the bar + bulge).
* So the model's light inside the field equals the data's **at every** $\\theta$, and the shapes
  are no longer asked to move light into the field. What happens outside it — where the
  exponential and the truncated N-body disagree — no longer sets the normalisation.
* In the code the names stay `L_BOX`, `p_box` and `box_fractions`: the "box" is now the field.

''' + CELLS[7][1][_b:]

# ---------------------------------------------------------------- cell 8: the code
_a = CELLS[8][1].index('# ================================================================= ONE stellar normalisation')
_b = CELLS[8][1].index('def box_fractions(')
CELLS[8][1] = CELLS[8][1][:_a] + '''# ================================================================= ONE stellar normalisation
# The stellar mass is DERIVED (markdown above):  M_star * p_box(theta) = M/L * L_BOX.
# THE "BOX" IS THE IFU FIELD: the data's own footprint, sky |X| < BOX_HALF_X, |Y| < BOX_HALF_Y.
# Both the region and L_BOX are read from dict_data, so nothing here is typed in.
_Xmm = np.asarray(dict_data['X_minmax'], float)
_Ymm = np.asarray(dict_data['Y_minmax'], float)
assert abs(_Xmm.sum()) < 1e-3 and abs(_Ymm.sum()) < 1e-3, 'the IFU field must be centred on the galaxy'
BOX_HALF_X, BOX_HALF_Y = 0.5 * (_Xmm[1] - _Xmm[0]), 0.5 * (_Ymm[1] - _Ymm[0])   # 7.272, 4.848 kpc
_nXY  = [int(v) for v in np.asarray(dict_data['nX_nY'])]
_bmap = np.asarray(dict_data['bin_mapping'])[:_nXY[0] * _nXY[1]]
assert np.all((_bmap >= 0) & (_bmap < NUM_VBIN)), 'an unbinned pixel: the field is not the full rectangle'
# L_BOX = the light in the field: each bin's surface density [Lsun/pc^2] x its pixel count x the
# pixel area.  (For a real galaxy: the same sum over the image inside the IFU footprint.)
_PIX_PC2 = (2 * BOX_HALF_X / _nXY[0]) * (2 * BOX_HALF_Y / _nXY[1]) * 1e6
L_BOX = float(np.sum(np.asarray(dict_data['XY_density_data'])
                     * np.asarray(dict_data['num_per_bin'])) * _PIX_PC2)        # 5.349e10 (mock)
BOX_NX, BOX_NY, BOX_NLOS, BOX_SMAX = 128, 96, 256, 200.0   # within 6e-5 of a uniform 1000 x 667,
                          # line-of-sight 1024-to-400 kpc reference for hz 0.1-2, Rd 2-12, L 2-8 kpc,
                          # i 40-80 deg (measured)
print('normalisation region = the IFU field |X| < %.3f, |Y| < %.3f kpc: L_BOX = %.4e Lsun'
      % (BOX_HALF_X, BOX_HALF_Y, L_BOX))


def _sinh_nodes(half, a, n):
    """Midpoint nodes of x = half*sinh(a*u)/sinh(a), u uniform in (-1, 1), and their dx: dense
    at the centre (the V4 bulge has a = 0.5 kpc), coarse in the smooth outskirts."""
    u = (jnp.arange(n) + 0.5) / n * 2.0 - 1.0
    return (half * jnp.sinh(a * u) / jnp.sinh(a),
            half * a * jnp.cosh(a * u) / jnp.sinh(a) * (2.0 / n))


_BX, _BW   = _sinh_nodes(BOX_HALF_X, 2.0, BOX_NX)    # sky X
_BYn, _BYW = _sinh_nodes(BOX_HALF_Y, 2.0, BOX_NY)    # sky Y
_BS, _BSW  = _sinh_nodes(BOX_SMAX, 7.0, BOX_NLOS)    # line of sight
_BY  = jnp.broadcast_to(_BYn[:, None], (BOX_NY, BOX_NLOS)).ravel()
_BSS = jnp.broadcast_to(_BS[None, :], (BOX_NY, BOX_NLOS)).ravel()
_BWT = (_BYW[:, None] * _BSW[None, :]).ravel()


''' + CELLS[8][1][_b:]
edit(8, '''    pair, that projects inside the sky box.  Angles in DEGREES, the convention of model.py:''',
     '''    pair, that projects inside the IFU field.  Angles in DEGREES, the convention of model.py:''')
edit(8, '''    One sky column per lax.map step (~33k points), so it adds ~0.1 GB, not ~4 GB."""''',
     '''    One sky column per lax.map step (~25k points), so it adds ~0.1 GB, not ~3 GB."""''')
edit(8, "RUN_TAG = 'aic_exp'", "RUN_TAG = '%s'" % TAG)
CELLS[0][1] = CELLS[0][1].replace('__TAG__', TAG)
if VARIANT:
    # the likelihood's surface-density error: logL only.  The NNLS still solves at 1% (utils.py:619),
    # so the orbits keep reproducing the light; logL stops charging the parametric family's own
    # density misfit (the orbit solutions miss the map by ~3.5% rms per bin) at 1%.
    edit(8, "ERR_DEN_LOGL = 0.01       # <<< TUNE ME.  The mock carries 1%; NGC 1512 production uses 5%.",
         "ERR_DEN_LOGL = %g       # <<< TUNE ME.  The NNLS still solves at 1%%; only logL is relaxed (see cell 0)."
         % A.den_logl)
    edit(0, '''Every file this notebook writes carries
`RUN_TAG = '%s'`.''' % TAG,
         '''Every file this notebook writes carries
`RUN_TAG = '%s'`.

**And one more change: the surface density is scored at %g%% in logL** (`ERR_DEN_LOGL`), while the
NNLS still solves at 1%%. The orbits therefore still have to reproduce the light -- without that
the kinematics lose their meaning -- but the likelihood no longer charges the parametric family's
own density misfit at 1%%: the orbit solutions miss the map by ~3.5%% rms per bin, which no member
of this family can beat. At 1%% that misfit, not the kinematics, picked the bar length
(`docs/devlog_projected.md`, 2026-10-21).''' % (TAG, 100 * A.den_logl))
edit(8, '''# Every file this notebook writes carries RUN_TAG, so neither the Miyamoto-Nagai chain
# (*_aic_tied*) nor the 13-parameter one (*_aic*) is ever resumed into, or overwritten by, this one.''',
     '''# Every file this notebook writes carries RUN_TAG, so neither the box-normalised exponential-disc
# chain (*_aic_exp.*), the Miyamoto-Nagai chain (*_aic_tied*) nor the 13-parameter one (*_aic*) is
# ever resumed into, or overwritten by, this one.''')

# ---------------------------------------------------------------- cell 9: the truth's M/L
edit(9, '''At these shapes the
model needs 28% more mass in the box than the light ($\\log M/L=+0.108$) to match the N-body's
stellar potential. `chi2_report` prints the derived $M_\\star$ and $p_{\\rm box}$.''',
     '''Normalised on the IFU
field, these shapes need only 3.8% more mass than the field's light ($\\log M/L=+0.016$) to match
the N-body's stellar potential; normalised on the 20 kpc box they needed 28% ($+0.108$), because
the exponential puts too much of its light outside the field. `chi2_report` prints the derived
$M_\\star$ and $p_{\\rm box}$, which is now the fraction inside the IFU field.''')

# ---------------------------------------------------------------- cell 17: walkers + the chunk rule
# (the expdisc builder now emits both already; this only acts on an older expdisc notebook)
for _old, _new in (('EMCEE_WALKERS = 24        # emcee refuses red-blue moves with fewer than 2*NDIM = 24 walkers',
                    'EMCEE_WALKERS = 32        # >= 2*NDIM = 24 (emcee refuses fewer); a multiple of 4*EMCEE_CHUNK'),
                   ('''                          # it a divisor of EMCEE_WALKERS // 2.  Lower it on an out-of-memory error.''',
                    '''                          # it a divisor of EMCEE_WALKERS // 4: the snooker move updates the
                          # walkers in 4 groups, and a short group is padded to a full chunk.
                          # Lower it on an out-of-memory error.''')):
    if _new not in CELLS[17][1]:
        edit(17, _old, _new)

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

flat = '\n'.join(s for t, s in CELLS if t == 'code')
assert ("RUN_TAG = '%s'" % TAG) in flat and "RUN_TAG = 'aic_exp'\n" not in flat
assert ('ERR_DEN_LOGL = %g ' % A.den_logl) in flat
assert 'BOX_HALF = 20.0' not in flat and '6.4078e10' not in flat, 'the 20 kpc box survived'
assert "L_BOX = float(np.sum(np.asarray(dict_data['XY_density_data'])" in flat
assert '_BWT = (_BYW[:, None] * _BSW[None, :]).ravel()' in flat
assert 'M_star = 10.0 ** log_mass_to_light_ratio * L_BOX / p_box' in flat
assert re.search(r'EMCEE_WALKERS = (\d+)', flat).group(1) == '32'
json.dump(out, open(A.out, 'w'), indent=1)
print('wrote %s  (%d cells, outputs cleared)' % (A.out, len(cells)))
