# SchwarMAX

**A GPU-friendly Schwarzschild orbit-superposition modelling framework for barred galaxies, implemented in [JAX](https://github.com/google/jax).**

SchwarMAX reconstructs the mass distribution and orbital structure of a galaxy
from integral-field-unit (IFU) stellar kinematics. Given Voronoi-binned surface
density, mean line-of-sight velocity, velocity dispersion, and Gauss–Hermite
moments (h₁–h₄), it jointly constrains a dark-matter halo, a stellar disc, a
rotating stellar bar, the viewing geometry, the stellar mass-to-light ratio, and
the bar pattern speed.

The heavy inner loop — building an orbital library, integrating orbits in the
bar's rotating frame, projecting them into observables, and solving a
non-negative least-squares (NNLS) problem for the orbital weights — is written to
run entirely on the GPU. A full model evaluation takes **~1 second on an NVIDIA
A100**, roughly an order of magnitude faster than traditional CPU
implementations, which makes it practical to explore a 10–20 dimensional
parameter space with MCMC.

This code accompanies the paper:

> **SchwarMAX: a GPU-friendly Schwarzschild orbit-superposition modelling framework**
> H. Zhang, D. Chemaly, E. Vasiliev, V. Belokurov, N. W. Evans, J. Shen (2026), submitted to MNRAS.

> **You are on the `developing-code` branch.** `main` is the published code. This branch adds,
> on top of it: the AIC likelihood with the effective number of orbital parameters `m_eff`
> (Thomas & Lipka 2022), a Fourier-ring 3-D density constraint, an explicit m=0 density for the
> Jeans seeding (`density_func_axi`, now a **required** argument of every `*_acc` builder), digital-twin
> trajectories on the Fourier path, and the N-body AIC examples. It also fixes a bug that
> under-scaled the whole design matrix of the orbit-library average by up to 10%, so `build_model_acc`
> results differ slightly from `main`. See [New on this branch](#new-on-this-branch), and read
> [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) before contributing.

---

## What the method does

SchwarMAX follows the classic Schwarzschild recipe (see §2 of Vasiliev & Valluri
2020 for a general overview), with each stage re-implemented for accelerators:

1. **Potential construction** — analytic density–potential pairs for an NFW
   halo, a Miyamoto–Nagai disc, and a Dehnen & Aly (2023) T3 bar + V4 boxy/peanut
   bulge. (Cylindrical-spline expansions of arbitrary densities are also
   supported.)
2. **Orbit library** — initial conditions are sampled from a fiducial
   double-exponential disc, given velocities from the axisymmetrised Jeans
   equations, and integrated in the bar's co-rotating frame with an adaptive
   third-order (Bogacki–Shampine) integrator. Fourfold bar symmetry is applied.
3. **Projection to observables** — orbits are rotated to the sky frame via three
   Euler angles (α, β, γ) and binned into the Voronoi apertures and a 3-D
   (R, φ, z) grid; per-orbit Gauss–Hermite moments are accumulated on the fly.
4. **Orbital-weight optimisation** — an ADMM non-negative least-squares solver
   finds the weights that best reproduce the density and LOSVD, with a
   bootstrap-marginalised, jackknife-rescaled likelihood used to drive the MCMC.

Validation on mock IFU data from a barred *N*-body simulation recovers the
density profiles, rotation curve, enclosed-mass profile, orbital structure, and
bar pattern speed (to ≲10%) across a range of viewing angles.

---

## Repository layout

The package modules live at the repository root. Put the repo root on your
`sys.path` and import directly (e.g. `from model import build_model`).

| File | Purpose |
|------|---------|
| `model.py` | **Main entry points.** AIC (this branch): `build_model_acc_meff_fourier`, `build_model_acc_meff`, `build_model_acc_fourier`, `get_model_with_orbit_acc_fourier`. v2: `build_model_acc`, `get_model_with_orbit_acc`, `model_deltaChi2_jackknife_acc`. v1 (analytic disc+bar only): `build_model`, `get_model_with_orbit`, `model_deltaChi2_jackknife` |
| `fourier3d.py` | The Fourier-ring 3-D density block: ring edges, the target's cos/sin(m phi) coefficients, their errors |
| `likelihood.py` | `make_mge_logL` (v2 worked example), `model_likelihood` (v1 MCMC log-likelihood), `calculate_delta_chi2` |
| `mge_model.py` | MGE density and the exact 1-D-quadrature MGE potential builder (`get_phi_m_exact`) |
| `deprojection.py` | MGE deprojection for a barred galaxy: triaxial bar axis ratios, inclination floor |
| `utils.py` | Data loader (`load_data_bootstrap`), parameter converters, rotation curves, rotation matrices |
| `potentials.py` | NFW halo + Miyamoto–Nagai disc potential/density |
| `dehnen_bar.py` | Dehnen & Aly (2023) T3/T4/V4 bar + bulge potential/density pairs |
| `jeans.py` | Jeans-equation initial-condition sampler |
| `integrates.py` | Adaptive leapfrog orbit integrator (vmapped, chunked) |
| `nnls.py` | ADMM non-negative least-squares solver (single + bootstrap) |
| `ghmoments.py` | Gauss–Hermite basis and `h → (V, σ)` conversion |
| `CylindricalSpline.py`, `CubicSpline.py` | Cylindrical-spline potential expansion for non-analytic densities |
| `constants.py` | Physical constants and lookup tables |
| `example/` | End-to-end notebooks, example data, a finished MCMC checkpoint, and the NGC 4371 acceptance test |
| `SchwarMAX-Nbody/` | The AIC pipeline tested on the N-body mock: notebooks, analysis scripts, regression gate (this branch) |
| `docs/DEVELOPMENT.md` | **How to develop on this branch**: workstream folders, the add-only rule for the package, the regression gate |
| `docs/LLM_GUIDE.md` | In-depth developer/agent guide (function signatures, workflows, pitfalls) |

---

## Installation

```bash
git clone https://github.com/Hanyuan0908/SchwarMax.git
cd SchwarMax

# The N-body snapshot (~126 MB) is stored with Git LFS.
git lfs install
git lfs pull
```

If you skip the LFS step, `example/Bar_model_TG21/model/t_t0_7` will be a small
text pointer and the ground-truth cells in the analysis notebook will fail.

### Dependencies

A GPU is strongly recommended for MCMC (each model evaluation is ~1 s on an A100
but ~60–100 s on CPU). Single-model evaluation and analysis are fine on CPU.

```bash
pip install "jax[cuda]" jaxopt numpy scipy matplotlib corner astropy pandas tqdm
pip install blackjax          # only needed to run an MCMC
pip install agama             # only needed for N-body ground-truth comparisons
```

The example MCMC notebook was validated with `jax==0.7.2 jaxlib==0.7.2`.

---

## Getting started

The `example/` folder is the canonical, end-to-end demonstration. Start here.

### 1. Run a model / an MCMC — [`example/example_schwarmax.ipynb`](example/example_schwarmax.ipynb)

A Colab-friendly notebook that:
- loads the mock IFU data via `load_data_bootstrap`,
- defines the `density_func` and `potential_func` closures,
- builds the SchwarMAX likelihood, and
- runs a BlackJAX adaptive random-walk Metropolis MCMC, checkpointing as it goes.

> The first cell assumes Google Colab (`from google.colab import drive` + pip
> installs). To run locally, replace it with `sys.path.insert(0, "..")` and set
> `path`/`filename` to point at `example/`.

### 2. Analyse a finished run — [`example/example_analysis.ipynb`](example/example_analysis.ipynb)

A four-part tutorial that works from the shipped `mcmc_checkpoint.pkl`:
1. **Corner plot** of the posterior against the ground truth.
2. **Data vs. best-fit model** maps (Σ, V, σ, h₃, h₄) using `build_model`.
3. **Rotation curve & enclosed-mass** recovery vs. the N-body truth.
4. **Orbital structure** — a circularity-vs-radius histogram from
   `get_model_with_orbit`.

This notebook runs locally out of the box (it adds the repo root to `sys.path`
automatically).

### 3. Model a real galaxy from an external potential — v2

`build_model_acc` is the recommended entry point. It takes four callables and never
branches on what kind of potential it holds, so the same code serves an MGE, a
CylindricalSpline, an analytic profile, or any sum of these:

```python
from model import build_model_acc
res = build_model_acc(density_func, potential_func, acceleration_func, potential_func_axi,
                      density_func_axi,                       # required on this branch
                      params_halo_pot, params_disk_rho, dict_data, num_Vbin, ...)
```

* `density_func_axi` is the **m=0 density** that matches `potential_func_axi`; the Jeans seeding
  needs both, and on this branch they are passed rather than derived.

* `acceleration_func` is **required, with no fallback** — splines and MGEs supply their own
  analytic gradient, and the orbit integrator calls it ~1e9 times per model evaluation.
* `potential_func_axi` is the **m=0 (azimuthally averaged)** potential, used only by the Jeans
  velocity seeding, which assumes Φ = Φ(R, z). The integrator still receives the FULL barred
  potential.

`likelihood.make_mge_logL` is a complete worked example, and
[`example/test_acceptance_ngc4371.py`](example/test_acceptance_ngc4371.py) reproduces the
published NGC 4371 best fit with it to 3.5% on every channel.

### 4. Go deeper — [`docs/LLM_GUIDE.md`](docs/LLM_GUIDE.md)

A detailed reference covering every public function signature, the `dict_data`
schema, how to build your own forward model from the low-level pieces (custom
potentials, the integrator, the NNLS solver), and a list of real-world pitfalls
(unit and angle conventions, bootstrap-stacked outputs, jackknife masking, JIT
recompilation). Read this before writing new code against the package.

### 5. The AIC pipeline on the N-body mock — `SchwarMAX-Nbody/`

The same mock as `example_schwarmax.ipynb`, modelled with `build_model_acc_meff_fourier` and the
likelihood `logL = -0.5 (chi2_kin + 2 m_eff_kin + chi2_den) / S2`. The stellar mass is not sampled:
it is derived from M/L and the observed light, `M_star * p(theta) = M/L * L`, where `p` is the
fraction of the model's own stellar mass that projects into the chosen region. 12 parameters.

| notebook | stellar disc | sampler | `L` = the light inside | outputs in `example/` |
|---|---|---|---|---|
| [`SchwarMAX_Nbody_AIC.ipynb`](SchwarMAX-Nbody/SchwarMAX_Nbody_AIC.ipynb) | Miyamoto–Nagai | BlackJAX RMH | a 20 kpc sky box | `*_aic_tied.*` |
| [`SchwarMAX_Nbody_AIC_expdisc.ipynb`](SchwarMAX-Nbody/SchwarMAX_Nbody_AIC_expdisc.ipynb) | double exponential, CylSpline potential | emcee | a 20 kpc sky box | `*_aic_exp.*` |
| [`SchwarMAX_Nbody_AIC_expdisc_field.ipynb`](SchwarMAX-Nbody/SchwarMAX_Nbody_AIC_expdisc_field.ipynb) | double exponential, CylSpline potential | emcee | the IFU field (summed from the data) | `*_aic_exp_field.*` |

**Running them.**
- **On Colab:** upload the repository folder to `My Drive` as `SchwarMAX`, open a notebook on a GPU
  runtime and run the cells in order. Cell 2 finds the folder by itself (edit `DRIVE_FOLDER` if
  you used another name). It pins `jax[cuda12]==0.7.2` as a matched set and removes any other
  `jax-cuda*` plugin, because a stray plugin fails with `Unexpected PJRT_ExecuteOptions size`.
- **Locally:** run the notebook from `SchwarMAX-Nbody/`, and it finds the repository root one
  level up. Its data come from `example/`, and its outputs are written there too.
- **Read cell 6 first.** It flips the mock's handedness, because every `*_acc` builder seeds the
  library anticlockwise and the mock was made for the clockwise `build_model`.

**emcee batching.** Each emcee step evaluates the walkers in groups: 2 for the DE move, 4 for the
snooker move. Each group is padded to a whole `EMCEE_CHUNK`, so **`EMCEE_CHUNK` must divide
`EMCEE_WALKERS / 4`**. 32 walkers with a chunk of 8 cost the same per step as 24 walkers.

**What these runs showed.** Tied to the 20 kpc box, the exponential-disc chain recovers M/L, Ω,
h_z and the viewing angles, but finds R_d ≈ 3.9 kpc against 6.5 and L_bar ≈ 3.7 kpc against 5.3.
The cause is the tie. It fixes the model's light inside the box, an exponential over-fills
10–20 kpc where the N-body is truncated, and so the truth's shapes put 19% too little light inside
the IFU field. Tying the mass to the field light instead cuts the truth's logL deficit from 59 to
10. The field-tied chain has not been run yet.

### 6. Analysis scripts (in `SchwarMAX-Nbody/`)

All of them `exec` the notebooks' own cells, so they cannot drift from the model the chain sampled.
Pass `--help` for the options.

| task | script |
|---|---|
| **Data vs model**: Σ, V, σ, h₃, h₄ maps, as model / data / residual, with χ² per channel. `--point bestfit` (the max-logP sample) or `--point truth`. Reads BlackJAX pickles or emcee `.h5` | [`SchwarMAX-Nbody/plot_datavsmodel_aic.py`](SchwarMAX-Nbody/plot_datavsmodel_aic.py) |
| Corner plot against the truth, with pulls and split R-hat | [`SchwarMAX-Nbody/plot_corner_aic.py`](SchwarMAX-Nbody/plot_corner_aic.py) |
| Recovered structure vs the N-body: Σ(R), M★(<R), A₂/A₀, v_c, M_DM(<r) | [`SchwarMAX-Nbody/compare_structure_aic.py`](SchwarMAX-Nbody/compare_structure_aic.py) (MN disc), [`SchwarMAX-Nbody/compare_structure_aic_exp.py`](SchwarMAX-Nbody/compare_structure_aic_exp.py) (exponential disc) |
| The likelihood split by channel at the truth, the median and shape swaps (`--nb` picks the notebook; `--param-only` projects the parametric model without orbits) | [`SchwarMAX-Nbody/diagnose_expdisc_bias.py`](SchwarMAX-Nbody/diagnose_expdisc_bias.py) |
| Smoke-test a notebook locally (600 orbits, CPU, ~4 GB) before spending GPU time | [`SchwarMAX-Nbody/test_example_aic_expdisc_nb.py`](SchwarMAX-Nbody/test_example_aic_expdisc_nb.py) |
| **Regression gate**: run before any commit that touches the package ([rules](docs/DEVELOPMENT.md)) | [`SchwarMAX-Nbody/test_regression_projected.py`](SchwarMAX-Nbody/test_regression_projected.py) |
| Rebuild the exponential-disc notebooks from the Miyamoto–Nagai one | [`SchwarMAX-Nbody/build_example_aic_expdisc_nb.py`](SchwarMAX-Nbody/build_example_aic_expdisc_nb.py), [`SchwarMAX-Nbody/build_example_aic_expdisc_field_nb.py`](SchwarMAX-Nbody/build_example_aic_expdisc_field_nb.py) |
| Rebuild `L_BOX` (the light in the 20 kpc box) from the N-body | [`SchwarMAX-Nbody/make_mock_box_light.py`](SchwarMAX-Nbody/make_mock_box_light.py) |
| Rebuild the `_GH` mock (V = v₀, σ = s, the pipeline's GH basis) | [`SchwarMAX-Nbody/make_mock_gh_basis.py`](SchwarMAX-Nbody/make_mock_gh_basis.py) |
| Rebuild the exponential-disc ground truth (a direct fit of the family to the N-body) | [`SchwarMAX-Nbody/fit_family_to_nbody.py`](SchwarMAX-Nbody/fit_family_to_nbody.py) |

For the v1 pipeline, data vs model is part 2 of
[`example/example_analysis.ipynb`](example/example_analysis.ipynb). The structure, L_BOX and
ground-truth scripts need `agama` and the Git-LFS snapshot.

### 7. Digital twins — an N-body realisation of a fitted model

A twin draws particles from the **fitted orbit library itself**, so it is a realisation of the
Schwarzschild solution, not of a smoothed distribution function. The package side is
[`get_model_with_orbit_acc_fourier`](model.py) (same arguments as `build_model_acc_fourier`).
It returns the unperturbed NNLS `weights` and every orbit's trajectory, `y_traj` with shape
`(n_orbits, 1, N_max, 6)`, and `t_traj`.

**There is no stand-alone twin script in this repository yet.** The driver we use is specific to
NGC 1512 and lives outside the package. The recipe it implements, each step found necessary on
real builds:

1. **Use the Fourier path.** The regular-grid `get_model_with_orbit_acc` solves its NNLS without
   the fourfold bar symmetry that the fit applies, so its weights belong to a different problem.
   Measured h₁ χ² per bin: 23.7 for that twin path against 1.36 for the fit.
   `get_model_with_orbit_acc_fourier` symmetrises every block.
2. **Particles per orbit:** `n_i ~ Multinomial(N, w / sum(w))`.
3. **One of the four bar images per particle.** The returned `y_traj` is the identity image only.
   Multiply each drawn (x, y, z, vₓ, v_y, v_z) by one of these sign patterns, chosen uniformly:
   (+,+,+,+,+,+), (+,+,−,+,+,−), (−,−,+,−,−,+), (−,−,−,−,−,−). These are the symmetry images the
   NNLS was solved under.
4. **Draw in continuous time.** For each particle draw `t ~ U(t_first, t_last)` over that orbit's
   own span, and evaluate `scipy.interpolate.CubicSpline(t_traj, y_traj)(t)`. Never pick stored
   points: the integrator is adaptive, so they are not uniform in time. Also, only about 700
   orbits carry effective weight, so 1e6 draws produced 56,611 bit-identical particles, which a
   tree code cannot split.
5. **Do not add Ω × r.** The stored velocities are already inertial components in bar-frame axes.
6. **Use a wider 3-D block for the twin than for the chain,** with the orbit seeds widened to
   match. The NNLS only constrains weights where the 3-D block has cells, so anything outside it
   is arbitrary in the twin.
7. **Acceptance test:** project the particles, re-measure Σ, V, σ and h₁–h₄, and require the
   **fit's own χ²** at the same orbit count. A twin that misses it is a twin of a model that was
   never fitted.
8. **Dark matter:** sample a DF (e.g. AGAMA `QuasiSpherical`) in the potential of **stars + halo**,
   not the halo alone. Halo-only is about 2× too cold in the centre, and the halo contracts at once.
9. **NEMO snapshots** (`agama.writeSnapshot`): kpc, km/s, mass column = G·M with
   G = 4.300917e-6 kpc (km/s)² / M☉.

---

## New on this branch

- **`build_model_acc_meff`, `build_model_acc_meff_fourier`.** The generalized AIC of Thomas & Lipka
  (2022): `m_eff`, the effective number of free orbital parameters, measured from bootstrapped
  refits at fixed λ. `regularization` stays traced, so λ can be sampled (the notebooks do).
  - Per-channel `m_eff_bins` and paired `chi2_boot` are returned for calibrating ΔAIC.
  - `err_den_meff=0.0` makes the surface density a hard constraint in the flexibility count.
- **`build_model_acc_fourier`, `fourier3d.py`.** The 3-D constraint is the cos/sin(mφ) content of
  rings in (R, |z|), even m ≤ `mmax`, instead of regular (R, z, φ) cells.
  - The errors are tied to the ring mass (`sigma_scheme='a0'`), not to the mode amplitudes, which
    would reward a stronger bar.
  - Ring spacing is uniform in R^`pow_R` and |z|^`pow_z`; `close_inner=True` closes the midplane slab.
- **`density_func_axi`**, the m=0 density for the axisymmetrised Jeans seeding. It is now a required
  positional argument of every `*_acc` builder, after `potential_func_axi`.
- **`get_model_with_orbit_acc_fourier`**, trajectories for twins, symmetrised and carrying `v_sys`.
- **`CylindricalSpline.get_phi_m_split`.**
- **Fix: the orbit-library realisation average divided by `valid.sum() + 0.1`.** That under-scaled
  the whole design matrix, by 10% on twins and 2.4% in the MCMC. It now divides by
  `max(valid.sum(), 1)`.
- `model.SCHWARMAX_FEATURES` lists these, and the notebooks assert on it, so an older copy of the
  package fails at import instead of hours into a chain.

---

## Parameter convention

The public model has **12 free parameters**, sampled in this order:

| # | Parameter | Meaning | Unit |
|--:|-----------|---------|------|
| 0 | `logM_DM(<10kpc)` | enclosed dark-matter mass within 10 kpc | log₁₀ M☉ |
| 1 | `logM_disc` | stellar disc mass | log₁₀ M☉ |
| 2 | `logM_bar` | stellar bar mass (T3 = V4) | log₁₀ M☉ |
| 3 | `logc` | NFW halo concentration | log₁₀ |
| 4 | `logR_disc` | disc radial scale length | log₁₀ kpc |
| 5 | `logH_disc` | disc vertical scale height | log₁₀ kpc |
| 6 | `logL_bar` | bar half-length | log₁₀ kpc |
| 7 | `alpha` | Euler angle α (bar in-plane angle) | radians |
| 8 | `beta` | Euler angle β (inclination) | radians |
| 9 | `gamma` | Euler angle γ (position angle) | radians |
| 10 | `logΥ` | stellar mass-to-light ratio | log₁₀ M/L |
| 11 | `logΩ_p` | bar pattern speed | log₁₀ kpc/Gyr |

Angles are in **radians** inside the parameter vector but in **degrees** in the
baryon dictionary passed to `build_model` — the `unpack_params` helper in the
analysis notebook handles the conversion. See §4 and §14 of the LLM guide for the
full convention and the common footguns.

---

## Data availability

- `example/mock_Nbody_galaxy_beta65_gamma45_D50.pkl` — Voronoi-binned mock IFU
  data (Σ, V, σ, h₁–h₄ with errors), at α = 25°, β = 65°, γ = 45°, D = 50 Mpc.
- `example/mock_Nbody_galaxy_beta65_gamma45_D50_GH.pkl` — the same mock with V and σ in the
  pipeline's Gauss–Hermite basis (V = v₀, σ = s); read by `plot_datavsmodel_aic.py`.
- `example/mcmc_checkpoint.pkl` — a finished 12-D chain (1000 steps × 32 chains).
- `example/Bar_model_TG21/model/t_t0_7` — the source *N*-body snapshot
  (Tepper-Garcia et al. 2021 Milky-Way analogue; Git LFS) and its precomputed AGAMA
  CylSpline potential (`t_t0_7.ini`).

---

## Citation

If you use SchwarMAX in your work, please cite the accompanying paper:

```bibtex
@article{schwarmax2026,
  author  = {Zhang, HanYuan and Chemaly, David and Vasiliev, Eugene and
             Belokurov, Vasily and Evans, N. Wyn and Shen, Juntai},
  title   = {{SchwarMAX}: a GPU-friendly Schwarzschild orbit-superposition
             modelling framework},
  journal = {Monthly Notices of the Royal Astronomical Society},
  year    = {2026},
  note    = {submitted}
}
```

## License

Released under the [MIT License](LICENSE). Correspondence: HanYuan Zhang
(hz420@cam.ac.uk), David Chemaly (dc824@cam.ac.uk).
