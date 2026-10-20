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
