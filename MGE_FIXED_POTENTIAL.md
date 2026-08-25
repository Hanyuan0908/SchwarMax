# MGE / fixed-potential path

Adds support for modelling a real galaxy whose baryonic potential is supplied externally --
as an MGE, a CylindricalSpline, an analytic profile, or any sum of these -- rather than being
generated from the package's analytic disc + bar.

## What was added

| file | purpose |
|---|---|
| `model.py` | `build_model_acc` and two siblings: copies of `build_model` that take an explicit `acceleration_func` and `potential_func_axi`. The originals are untouched. |
| `jeans.py` | `get_jeans_moments_improved` / `get_w0_new_improved`: Jeans solver taking `density_func` and `potential_func` as plain callables. Gauss-Legendre quadrature and autodiff derivatives. |
| `deprojection.py` | MGE deprojection with a triaxial bar; `solve_bar_pq`, `deproject_components`, `inclination_floor`. |
| `mge_model.py` | MGE density and the exact CylindricalSpline potential builder. |
| `utils.py` | `axisymmetrize_potential`. |
| `ghmoments.py` | `h_to_V_sigma_GH` -- the Gauss-Hermite reference-Gaussian conversion (pPXF's `v0`/`s`). |
| `potentials.py` | analytic `NFW_acceleration`. |
| `likelihood.py` | `make_mge_logL` -- worked v2 example, validated by `example/test_acceptance_ngc4371.py`. |

## Design

`build_model_acc` takes four callables and never branches on what kind of potential it holds:

```python
build_model_acc(density_func, potential_func, acceleration_func, potential_func_axi,
                params_halo_pot, params_disk_rho, dict_data, num_Vbin, ...)
```

* **`acceleration_func` is required, with no fallback.** Splines and MGEs supply their own analytic
  gradient (`get_acc`); differentiating the potential instead would be far too slow, since the
  orbit integrator calls this ~1e9 times per model evaluation.
* **`potential_func_axi` is the m=0 (azimuthally averaged) potential**, used only by the Jeans
  velocity seeding, which assumes Phi = Phi(R, z). The orbit integrator must still receive the FULL
  potential -- the bar has to be present for the orbits. Feeding the Jeans a barred potential
  samples the bar's long axis and biases the circular speed.
* **Model parameters travel through closures, not through `params_baryon`.** Define the four
  callables inside the jitted likelihood so they close over the traced parameters of the current
  sample. `params_disk_rho` then carries only what `build_model_acc` itself reads:
  `mass_to_light_ratio`, `Omega_bar`, `alpha`, `beta`, `gamma`. There is no `if` anywhere, at trace
  or run time, so the likelihood still jits and vmaps.

## Choosing `potential_func_axi`

Three strategies, all with the same signature; pick on cost and accuracy, not on potential type:

| strategy | cost | error |
|---|---|---|
| the object's own m=0 (`evaluate_phi_axisymmetric`) | free | exact |
| `axisymmetrize_potential(pot, n_phi)` | n_phi x per-call | exact only if band-limited below n_phi |
| expand once: `get_phi_m(rho, ..., Mmax=0)` then `evaluate_phi_axisymmetric` | one build, then a lookup | spline error, no aliasing |

Prefer the stored m=0 when the potential has one. Since Phi is additive, axisymmetrise only the
components that need it -- an axisymmetric disc and a spherical halo are already their own m=0, so a
mixed model should average the bar alone. Check `n_phi` convergence against 2x: azimuthal averaging
annihilates every harmonic except multiples of `n_phi`, and a cusped analytic bar has real power at
m >= 8.

## Conventions

* Sky = (x, y), line of sight = z. With `R = Rz(-alpha) Rx(-beta) Rz(-gamma)`: **alpha** is the
  on-sky position angle, **beta** the inclination, **gamma** the bar azimuth.
* Anticlockwise seeding (Lz > 0) pairs with `+Omega_bar`. The mirrored convention gives an identical
  static projection but is **not** equivalent through the rotating-frame integration.
* pPXF `v0`/`s` are the Gauss-Hermite reference Gaussian, not classical moments, so use
  `h_to_V_sigma_GH` when converting back to V and sigma for reporting.
* float64 is fine on CPU. Do not enable it for GPU runs -- it is both much slower and much more
  memory-hungry there.


## Validation

`example/test_acceptance_ngc4371.py` reproduces the published NGC 4371 best fit through
`build_model_acc` to within 3.5% on every channel:

```
channel      new    published    ratio
logSig     0.0724     0.0700     1.035
h1         0.9168     0.9100     1.007
h2         0.6168     0.6200     0.995
h3         1.1985     1.1900     1.007
h4         0.8229     0.8000     1.029
581 bins, 10 000 orbits, 181 s, float64 CPU
```

Run `example/prepare_ngc4371_acceptance.py` first to cache the binned data and potential.

## `density_func_axi` — the m=0 density for the Jeans

`build_model_acc` and its two siblings take **`density_func_axi`** alongside
`potential_func_axi`. The Jeans equations are axisymmetric, so they are given the m=0
density and the m=0 potential; `density_func` keeps the full (possibly barred) density for
the 3-D density target.

For an MGE or any axisymmetric model the two coincide — pass `density_func` twice.

Why it matters for a barred parametric density: `jax.grad(vertical_pressure)` differentiates
through the density, so handing the Jeans the barred version is both physically inconsistent
and the dominant memory cost of the model. Measured on a Dehnen T3 bar at 1000 orbits:

```
full barred density passed to the Jeans   0.906 GB   (pre-merge baseline 0.640 GB)
density_func_axi, bar m=0 tabulated       0.481 GB
```

Two traps, both measured, both to be avoided:

* **Do not azimuthally average inside the Jeans.** A `vmap` over n_phi inside
  `jax.grad(vertical_pressure)` multiplies the reverse-mode tape: 0.91 -> 5.87 GB.
  Tabulate the m=0 component once on an (R,|z|) grid and interpolate.
* **Do not fold a compact bulge into a wide m=0 spline.** An NR=30/NZ=20 grid spanning
  1e-3..50 kpc cannot resolve a 0.3 kpc bulge; doing so drove logSig chi2 from 13.3 to 140.8.

### Stability, NGC 4371 acceptance test, same session

```
                  time     logSig    h1      h2      h3      h4
main              188 s    0.0742  0.9110  0.6196  1.1958  0.8213
this branch       180 s    0.0742  0.9110  0.6196  1.1958  0.8213
```
chi2 bit-identical, 4% faster. Always A/B in ONE session: machine state drifts enough that a
cross-session comparison is meaningless.
