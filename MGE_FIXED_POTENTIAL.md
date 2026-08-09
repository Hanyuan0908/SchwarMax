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
| `utils.py` | `axisymmetrize_potential`, `tabulate_axisymmetric_potential`. |
| `ghmoments.py` | `h_to_V_sigma_GH` -- the Gauss-Hermite reference-Gaussian conversion (pPXF's `v0`/`s`). |
| `potentials.py` | analytic `NFW_acceleration`. |
| `example_disc_bar_halo.py` | worked example: NFW halo + CylSpline axisymmetric disc + analytic triaxial bar. |

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
| `tabulate_axisymmetric_potential(pot, ...)` | decoupled from source | interpolation error |

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
