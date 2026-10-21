# Dev log — the projected (line-of-sight) scheme, `SchwarMAX-Nbody/`

Append-only. History before 2026-10-20 is in the PHANGS workspace's `COLLAB_PROGRESS_LOG.md`
(entries 2026-10-15 to 2026-10-20(f)).

### 2026-10-20 — starting state

- **Source:** commit 90750f1 contains `build_model_acc_meff[_fourier]`, the Fourier 3-D block,
  `density_func_axi`, `get_model_with_orbit_acc_fourier`, and the fix to the realisation-average
  guard.
- **Mock tests (`SchwarMAX-Nbody/`):** three AIC notebooks, all with the stellar mass tied to M/L:
  - Miyamoto–Nagai disc + BlackJAX, tied to the 20 kpc box
  - exponential disc + emcee, tied to the 20 kpc box
  - exponential disc + emcee, tied to the IFU-field light
  - The builders reproduce the committed notebooks cell for cell.
- **Results so far:**
  - The box-tied exponential-disc chain (643 steps × 32 walkers) recovers M/L, Ω, h_z and the
    viewing angles.
  - It finds R_d 3.9 kpc (truth 6.5) and L_bar 3.7 kpc (truth 5.3).
  - Cause: the box tie fixes the light in a 20 kpc box. The exponential over-fills the truncated
    outer N-body disc, so the truth's shapes put 19% too little light in the IFU field.
  - With the field tie the truth's logL deficit falls from 59 to 10.
  - Next: run the field-tied chain (user, on Colab).
- **Regression gate pinned:** expdisc −272.49628 / −279.69803, field −192.75204 / −204.71582
  (CPU, jax 0.7.2).

### 2026-10-21 — the IFU-field-tied chain (`example/emcee_aic_exp_field.h5`)

- **Chain:** 700 steps × 32 walkers. τ = 31–45 (15.6 τ), burn-in 134, acceptance 0.21, R-hat
  1.10–1.22. log R_d was still creeping up over the last 300 steps.
- **Recovered well** (pull in parentheses):
  - M/L: +0.002 ± 0.043 (0.0)
  - log M_10kpc: 10.73 ± 0.08 (−0.3)
  - Ω: 24.5 ± 1.5 km/s/kpc, truth 24.6
  - inclination: 65.9 ± 1.4°
  - α: 25.1°
  - f_bar: 0.33 ± 0.09 (−0.6)
  - h_z: 0.82 kpc (+1.0)
- **R_d** is 8.3 kpc (+0.17 / −0.16 dex), truth 6.5 (+0.7). It is no longer biased low, but it sits on
  a tight R_d–f_bar ridge, and the truth lies on that ridge. With the field tie, the disc beyond the
  field is constrained only through that trade-off: total M★ is 9.9e10 (8.1–15.2) against the
  N-body's 6.4.
- **L_bar** is unchanged at 3.72 kpc, truth 5.34 (−2.4); the box-tied chain gave 3.68. **So the
  short bar does not come from the M/L tie.**
- **Structure:**
  - Inside the field, face-on Σ is within ±0.07 dex of the N-body at 2–7 kpc, better than the
    truth's own +0.16/+0.21 at 3–4.8 kpc.
  - M★(<R) is within 1–4% of the N-body at 2–7.3 kpc.
  - A₂/A₀ peaks at 0.37 at 2.3 kpc and falls to half by 4.2 kpc. The N-body peaks at 0.40 at
    3.3 kpc and falls to half by 5.4 kpc; the box chain gave 0.30 / 2.0 / 3.7. The amplitude is now
    right, but the profile is still about 1 kpc too far in, as L_bar says.
- **Script changes:**
  - `plot_corner_aic.py` and `compare_structure_aic_exp.py` now take the field notebook for
    `*_aic_exp_field*` chains. Their stellar-mass ties differ, so the derived masses would otherwise
    be wrong.
  - The structure script gains `--overlay NPZ=LABEL` and names its outputs after the chain. Re-run
    on the box chain, it reproduces the previous numbers exactly.
