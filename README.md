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
| `model.py` | **Main entry points**: `build_model`, `get_model_with_orbit`, `model_deltaChi2_jackknife` |
| `likelihood.py` | `model_likelihood` (MCMC log-likelihood), `calculate_delta_chi2` |
| `utils.py` | Data loader (`load_data_bootstrap`), parameter converters, rotation curves, rotation matrices |
| `potentials.py` | NFW halo + Miyamoto–Nagai disc potential/density |
| `dehnen_bar.py` | Dehnen & Aly (2023) T3/T4/V4 bar + bulge potential/density pairs |
| `jeans.py` | Jeans-equation initial-condition sampler |
| `integrates.py` | Adaptive leapfrog orbit integrator (vmapped, chunked) |
| `nnls.py` | ADMM non-negative least-squares solver (single + bootstrap) |
| `ghmoments.py` | Gauss–Hermite basis and `h → (V, σ)` conversion |
| `CylindricalSpline.py`, `CubicSpline.py` | Cylindrical-spline potential expansion for non-analytic densities |
| `constants.py` | Physical constants and lookup tables |
| `example/` | End-to-end notebooks, example data, and a finished MCMC checkpoint |
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

### 3. Go deeper — [`docs/LLM_GUIDE.md`](docs/LLM_GUIDE.md)

A detailed reference covering every public function signature, the `dict_data`
schema, how to build your own forward model from the low-level pieces (custom
potentials, the integrator, the NNLS solver), and a list of real-world pitfalls
(unit and angle conventions, bootstrap-stacked outputs, jackknife masking, JIT
recompilation). Read this before writing new code against the package.

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
