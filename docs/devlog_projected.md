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

### 2026-10-21 — the field chain against the ORIGINAL L_bar truth (4.87 kpc)

- `--truth-lbar KPC` in `plot_corner_aic.py` / `compare_structure_aic_exp.py` replaces only the truth's
  L_bar; the outputs get a `_Ltruth<KPC>` suffix.
- With 4.87 kpc (published cell 7) the L_bar pull is −1.8 (vs −2.4 against 5.34).
- That hybrid truth's A₂/A₀ peaks at 0.37 at 2.6 kpc and falls to half by 4.8 kpc. The genuine
  original truth (the MN cell-7 model) falls to half by 4.8 kpc too, with a peak of 0.42 at 2.9 kpc.
- The chain: 0.37 / 2.3 / 4.2 kpc. The N-body: 0.40 / 3.3 / 5.4 kpc.

### 2026-10-21 — the exp notebooks' truth L_bar is now 4.87 kpc (user decision)

- Cell 10 of both exponential-disc notebooks now uses the published example's L_bar, 4.87 kpc,
  shared with the Miyamoto–Nagai notebook. The exp-family fit's other values are kept. Changed in
  `build_example_aic_expdisc_nb.py`, and both notebooks were rebuilt.
  - Only cells 9–10 changed. Rebuilding also drops 42 KB of Colab widget state and Colab cell
    metadata that the previously committed copies still carried.
  - The derived masses barely move: p_box and p_field change by under 1e-3, so the direct fit's
    M/L stays at +0.108 (box tie) and +0.016 (field tie).
- **Regression gate re-pinned:** the truth is its evaluation point, and the package is unchanged.
  - expdisc: −257.30835 / −282.28104 (was −272.49628 / −279.69803)
  - field: −180.9246 / −190.58885 (was −192.75204 / −204.71582)
  - logL at the truth itself rises by 15.2 (box tie) and 11.8 (field tie): the shorter-bar truth
    fits the mock better.
- **Where the 5.34 came from:** `fit_family_to_nbody.py` `exp_tied` uses run_model's ties
  (a_bar = L/5, b_bar = h_z, V4 mass = M_bar) and fits v_c,*(R) at 1%, face-on Σ(R) at 0.05 dex
  and A₂/A₀(R) at 0.02 jointly. A fit to the stellar v_c ALONE, with the same ties, gives
  L ≈ 3.25 kpc (2026-10-19 test). So the A₂/A₀ term is what lengthens the bar. The chains' 3.7 kpc
  sits between the potential-only and the A₂-matching answers.

### 2026-10-21 — why the field chain prefers a short bar: logL and data-vs-model at fixed points

- **Profile from the chain:** max logP per L_bar bin is nearly flat from 3.0 to 4.8 kpc (−78.4
  to −79.2), −80.8 at 4.8–5.5. The 99th percentile is 5.1 kpc. L_bar is weakly constrained; 4.87
  costs ~1.5–2.4 logP.
- **BUG (mine), fixed:** `plot_datavsmodel_aic.py` scored `*_aic_exp_field*` chains with the
  BOX-tied notebook (a different likelihood): its best fit came out 25 logL off. It now picks the
  field notebook, records the notebook in its cache, and treats a cache made with another
  notebook as stale. Also new: `--require NAME>=VALUE` (best fit within a range) and a full
  channel χ² including the orbits' 3-D mismatch. `compare_dvm_points.py` splits Δχ² by channel
  and deprojected radius.
- **Evaluation-path noise:** at 600 orbits, logL at a fixed θ differs by ~10 between serial and
  vmapped (batch 1/2/3) evaluation. It is not the NNLS: it is unchanged from 1000 to 3000
  iterations. Likely chaotic orbits under float32 program differences. At the production 7500
  orbits it is 0.1–0.7, within ~1 of the GPU chain. So compare points at 7500 orbits, never at 600.
- **Points on the field likelihood** (7500 orbits, CPU; each within 1.0 of the chain's logP):
  - A, best fit (L_bar 3.23): logL −69.05, 3-D mismatch 1813
  - C, best with L_bar ≥ 4.7 (L_bar 4.72, R_d 8.4): −70.62, 3-D mismatch 3673
  - D, truth (L_bar 4.87): −78.11, 3-D mismatch 7448
  - C − A by radius: Δχ² +272, almost all at deprojected R 1–4 kpc, in the surface density
    (+185) and h3 (+55); h1 improves (−27).
- **The decisive scan,** A with ONLY L_bar changed:

  | L_bar (kpc) | logL | χ²_den | 3-D mismatch | kinematics |
  |---|---|---|---|---|
  | 3.23 | −69.05 | 7676 | 1813 | |
  | 4.0 | −69.57 | 7840 | 2900 | |
  | 4.87 | −71.59 | 8067 | 4918 | flat (h1 −19, h2 +35, h3 +44, h4 −1) |

  The penalty comes from the orbits failing to build the long T3 bar's own 3-D density, which
  the NNLS pays for in the 2-D surface density. The ties are NOT shown to be the cause (the
  claim is withdrawn).
- **Next test:** can the library build a long bar at all? Fit the 3-D block alone at L_bar 3.23
  vs 4.87, with more orbits or wider seeding.

### 2026-10-21 — the 2-D vs 3-D density conflict is in A₀, at high |z| (user's hypothesis, confirmed)

- **Settings (field notebook):**
  - 2-D: the OBSERVED map in 609 bins, 1% in both the NNLS (`utils.py:619`) and logL; no m_eff
    charge.
  - 3-D: the PARAMETRIC density in 15 R rings (0–10 kpc, uniform in √R) × 6 |z| layers
    (0–5 kpc, uniform in √|z|) × 7 modes (A₀, c/s of m = 2, 4, 6) = 630 constraints.
    σ(A₀) = 2% A₀, σ(c_m, s_m) = 4% A₀ (scheme 'a0'; the 6% floor is unused).
- **New `compare_target_vs_nbody.py`** bins the N-body stars into the same cells with the package's
  binner and scores the target with the likelihood's errors. No orbits.
- **Result at the chain's best fit** (L_bar 3.23):
  - target vs N-body: χ² A₀ 64,312; m=2 742, m=4 224, m=6 54. The mismatch is the ring MASS,
    not the bar harmonics.
  - Cylindrical M(<R) agrees to 0.5–3% beyond 1 kpc.
  - 86% of the A₀ χ² is at |z| > 2.45 kpc. There the model holds 1.5e9 M☉ (2.7% of its box
    mass) against the N-body's 2.9e8 (0.5%): 4× at 2.45–3.6 kpc, 29× at 3.6–5 kpc. Two-thirds
    comes from the exponential disc's vertical tail, one-third from the T3 bar (b_bar = h_z tie);
    the V4 bulge adds ~nothing. The N-body's vertical profile drops ~10× per layer above 1.5 kpc,
    much steeper than exponential.
- **Longer bar** (L_bar 4.87, the rest fixed): A₀ χ² rises to 70,886, in-plane.
  - Ring mass at 0.65–1.83 kpc falls to 0.66–0.82 of the N-body's, and rises to 1.26–1.48 at
    2.35–5.1 kpc.
  - So the long T3 bar puts its mass too far out. That is the link between L_bar and the conflict.
  - The truth (L_bar 4.87, R_d 6.47) shows the same: 71k, rings 0.64–1.47.
- **On the sky:** the parametric projection misses the map by 0.085 dex rms, split as 0.055 dex
  axisymmetric (+5% at R<1, −6 to −7% at 1–2.5 kpc, +12% at 10–13, +34% beyond, i.e. the field
  corners) and 0.065 dex non-axisymmetric.
- **Options (not yet tested):**
  - a vertical profile that falls faster (sech² or Gaussian)
  - b_bar decoupled from h_z
  - 3-D errors that don't give near-empty high-|z| cells 2%-of-themselves weight (e.g. relative to
    the ring's total A₀, or a lower |z| limit)
  - the model's own projection as the 2-D target, as Jin+2025 / FORSTAND do

### 2026-10-21 — the bar the SKY MAP wants (no orbits): `fit_density_to_sky.py`

- **Setup:** the field notebook's exp family, projected and normalised as the likelihood does, fitted
  to the 609-bin map in log Σ at 1%. Angles fixed at the truth; b_bar = h_z and bulge = bar mass
  kept. Both starts (chain best fit, truth) converge to the same minimum.
- **Fits:**

  | fit | L_bar | a_bar | R_d | h_z | f_bar | rms |
  |---|---|---|---|---|---|---|
  | a_bar = L/5 | 3.05 | 0.61 | 5.06 | 0.83 | 0.425 | 0.0759 dex |
  | a_bar free | 2.93 | 0.91 | 5.15 | 0.79 | 0.422 | 0.0753 dex |

  For reference: the chain's best fit gives 0.0835 dex, the truth (L_bar 4.87) 0.1003.
- **Reading:**
  - The sky map by itself wants L_bar ≈ 3 kpc, which is what the chain found (3.2–3.7). The short
    bar is driven by the projected light, not by the orbits or the kinematics.
  - Freeing a_bar barely helps, and makes the bar SHORTER and FATTER, not long and concentrated.
    So the a_bar = L/5 tie is not the cause either.
- **In 3-D against the N-body** (|z| < 2.45 kpc), the sky-fit bars:
  - ring mass 0.81 at 1–2 kpc and 1.14 at 2.4–4.3 kpc
  - |A₂|/A₀ peak 0.28–0.31 at 2.1 kpc, half by 3.6 kpc; the N-body peaks at 0.40 at 3.3 kpc and
    is half by 5.1
- **Interpretation (hypothesis):** the N-body's long m=2 carries little MASS at 4–5 kpc (an
  elongated, low-density bar end), while a T3 bar can only make m=2 by putting mass there. The
  family cannot produce the N-body's A₂/A₀ extent while matching its light, so "L_bar truth" is
  ill-defined in this family.

### 2026-10-21 — CORRECTION: why the sky map prefers a short bar

- The previous entry's "the N-body's long m=2 carries little mass" was wrong; the user rightly
  rejected it. Measured in the plane (|z| < 0.8 kpc) from the Fourier coefficients, with
  ρ(φ) ∝ A₀ + 2Σ c_m cos mφ: along the bar (φ=0) vs across it (φ=90°).
  - **N-body:** along the bar the density is nearly FLAT from 2.4 to 5 kpc (2.7–3.2e9 per ring
    unit). Across it there is a DEEP gap: along/across = 4.2, 4.7, 3.7 at 2.35–4.31 kpc.
  - **Truth model (L_bar 4.87):** along 1.8–2.1× the N-body at 2.4–4.3 kpc, across 1.2–1.6×,
    ring mass 1.5–1.8×.
  - **Sky-fit model (L_bar 3.05):** along 0.6–0.9× beyond 2.9 kpc, across 1.4–2.0×, ring mass
    1.0–1.3×.
- **Reading:**
  - The N-body's m=2 comes largely from a DEFICIT across the bar. The exponential disc is
    axisymmetric and fills that gap.
  - A T3 bar can only build the contrast by ADDING mass along the bar, which overshoots the ring's
    light by 1.5–1.8×.
  - The sky map rejects that and settles for a short bar.
- **Matches Dattathri+2024's R_hole term** (a disc depleted perpendicular to the bar, inside
  corotation).
- **To test (no orbits):** a sky fit with γ_bar free (a flatter needle), and with a disc
  depleted across the bar.

### 2026-10-21 — does the short bar act like the N-body bar? No (`compare_bar_quadrupole.py`)

- **Measured** (user's question): the m=2 of the in-plane STELLAR potential |Φ₂|(R), and the m=2 of
  the face-on ⟨v_R⟩ in |z| < 0.8 kpc. The model's streaming comes from its orbits:
  get_model_with_orbit_acc_fourier at run_model's settings, points weighted by dt × NNLS weight.
  All three systems rotate anticlockwise in the bar frame, and all V₂ phases are 45–50° behind
  the bar.

  | | best fit (L_bar 3.23) | truth (L_bar 4.87) |
  |---|---|---|
  | Φ₂ vs N-body | 1.3–1.4× at 1.25–2.25 kpc; 0.70, 0.56, 0.48, 0.41× at 3.75, 4.25, 4.75, 5.75 kpc | 1.3–1.5× at 1.75–3.75 kpc; matches beyond 5 kpc |
  | V₂ | ~12 km/s at 1.75–2.75 kpc; 7.2, 2.6, 1.2 at 3.75, 4.25, 4.75 kpc | 12–13 km/s inside 3.75 kpc; 9.1, 6.0 at 4.25, 4.75 kpc |

  The N-body's V₂ is 15–18 km/s from 1.25 to 4.25 kpc and 9.8 km/s at 4.75 kpc.
- **Reading:**
  - The gap ACROSS the N-body bar is itself m=2 density, and it gravitates, so the N-body's bar
    potential extends further out.
  - The short best-fit bar is not an equivalent redrawing: its Φ₂ is too concentrated, and its
    orbits barely stream beyond 3.5 kpc.
  - The truth model's Φ₂ reaches the right distance but is too strong inside 4 kpc, yet its
    orbits stream LESS than the N-body there. The model orbits respond more weakly than the
    N-body's stars even to a stronger bar.
  - The LOS kinematics only weakly constrain the face-on v_R. The h1 χ² improved slightly with a
    longer bar (−19 to −27).
- Face-on ⟨v_R⟩ maps (`figures/vR_faceon_maps.png`, `compare_bar_quadrupole.py --plot-only`): all three
  show the same four-lobed bar quadrupole with the same orientation.
  - N-body: broad lobes centred ~3 kpc from the centre, reaching ~5 kpc, ±20–25 km/s.
  - Best fit: the same pattern SHRUNK, lobes centred ~2 kpc, confined inside ~3.5 kpc, ±15–20 km/s.
  - Truth: lobes reach about the N-body's size but are narrower and elongated along the bar.
