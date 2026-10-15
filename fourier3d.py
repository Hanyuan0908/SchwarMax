"""Fourier-mode 3-D density constraint: rings x vertical floors x azimuthal harmonics.

PURELY ADDITIVE.  Nothing in integrates.py, model.py, nnls.py or utils.py is modified, so every
existing result stays reproducible.  The output is a flat (n_ring*n_mode,) vector in exactly the
layout `solve_nnls_admm_bootstrap` already consumes, so the NNLS needs no change at all -- only
the 3-D block's contents and its error vector differ.

WHY
---
The regular grid spends ~12 azimuthal cells per (R,z) ring.  Those 12 numbers ARE a Fourier basis
already: 12 samples of a pi-periodic function carry exactly m = 0,2,...,12.  But they are a BAD
basis for this problem:

  * every orbit that misses a phi cell contributes nothing to it, so the outer cells are
    shot-noise dominated -- at R = 9 kpc a cell is 2.4 kpc wide azimuthally yet holds 1/12 of the
    ring's points;
  * the resolution is spent where it is least informative.  MEASURED on NGC1512_twin_moregas,
    |A_m|/A_0 at R = 4-6 kpc: m=2 0.241, m=4 0.101, m=6 0.055, m=8 0.032, m=10 0.025, against an
    odd-m noise floor of 0.002-0.003 (odd m are exactly zero under the 4-fold symmetry, so they
    measure the noise).  So the power is concentrated in the lowest few even modes and the tail
    is small but real.

Storing the harmonics directly lets the same budget buy (R,z) resolution instead, and every orbit
contributes to every mode because each coefficient integrates over ALL azimuths.

WHAT IS STORED
--------------
Per ring, `n_mode = 1 + 2*(mmax//2)` numbers:

    A_0,  c_2, s_2,  c_4, s_4,  ...,  c_mmax, s_mmax          with
    c_m = sum_i w_i cos(m phi_i),   s_m = sum_i w_i sin(m phi_i),   A_0 = sum_i w_i

ONLY EVEN m.  integrates.py:73 expands every trajectory chunk by the 4-fold bar symmetry
    (x,y,z), (x,y,-z), (-x,-y,z), (-x,-y,-z)
whose in-plane content is {(x,y), (-x,-y)}, making the density exactly pi-periodic in phi.  Odd
harmonics are therefore zero BY CONSTRUCTION and storing them would only fit shot noise.

The SIN terms are kept.  The target's are zero for a bar built along +x, so constraining them
explicitly forbids the orbit solution from twisting the bar -- something no amplitude-only
comparison would catch.

NO TRIGONOMETRY
---------------
cos(2phi) = (x^2-y^2)/R^2 and sin(2phi) = 2xy/R^2 are rational in coordinates we already have, and
the higher even harmonics follow by the Chebyshev recursion

    c_{k+1} = 2*c_2*c_k - c_{k-1},      s_{k+1} = 2*c_2*s_k - s_{k-1}       (c_k = cos(2k phi))

so the arctan2 the regular binner needs is never computed.  (It does NOT make this cheaper overall
-- the 7-wide scatter costs more than the arctan2 saved, measured ~2x -- but it is exact and it
removes a transcendental from the inner loop.)

EDGES, LINEAR OR LOG
--------------------
Both binners take EXPLICIT EDGE ARRAYS, so linear, log and arbitrary spacing are the same code
path with no flags and no branching -- `np.linspace` or `np.geomspace`, the caller's choice.
Two implementations, benchmarked against each other:

  bin_fourier_rings_edges   general, O(n_bins) branch-free comparisons per point
  bin_fourier_rings_uniform O(1), for edges uniform in a MONOTONIC TRANSFORM u (u = R for linear,
                            u = log R for log).  Pass the transform's parameters, not the edges.

TRAP, and it is a silent one
----------------------------
The regular grid's error model is `sig = sigma_frac_3d * y_Rzphi`, i.e. FRACTIONAL on the cell
value.  Fourier coefficients can be zero or negative, so that gives sigma <= 0 and the NNLS goes
NaN -- the same family as the `s` must-be-non-zero bug that NaN'd the outer-disc dataset.  Use
`fourier_sigma()` below, which scales every mode of a ring by that ring's A_0.
"""
import jax
import jax.numpy as jnp
from functools import partial


# ----------------------------------------------------------------------------------------------
#  harmonics without trigonometry
# ----------------------------------------------------------------------------------------------
@partial(jax.jit, static_argnames=('mmax',))
def even_harmonics(x, y, mmax):
    """(cos 2phi, sin 2phi, cos 4phi, sin 4phi, ...) up to mmax, from x and y alone.

    Returns (N, 2*(mmax//2)) ordered c_2, s_2, c_4, s_4, ...  Exact, not an approximation:
    c_2 = (x^2-y^2)/R^2 and s_2 = 2xy/R^2 are the double-angle identities, and the Chebyshev
    recursion is exact in floating point up to rounding.
    """
    R2 = x*x + y*y
    inv = 1.0/jnp.maximum(R2, 1e-30)
    c_prev, s_prev = jnp.ones_like(x), jnp.zeros_like(x)          # m = 0
    c_cur, s_cur = (x*x - y*y)*inv, (2.0*x*y)*inv                 # m = 2
    out = []
    for _ in range(mmax//2):
        out += [c_cur, s_cur]
        c_next = 2.0*((x*x - y*y)*inv)*c_cur - c_prev
        s_next = 2.0*((x*x - y*y)*inv)*s_cur - s_prev
        c_prev, s_prev, c_cur, s_cur = c_cur, s_cur, c_next, s_next
    return jnp.stack(out, axis=-1) if out else jnp.zeros((x.shape[0], 0))


def _weights(x, y, w, mmax):
    """(N, n_mode) accumulator rows: [w, w*c2, w*s2, w*c4, w*s4, ...]"""
    H = even_harmonics(x, y, mmax)
    return jnp.concatenate([w[:, None], w[:, None]*H], axis=-1)


# ----------------------------------------------------------------------------------------------
#  binners
# ----------------------------------------------------------------------------------------------
@partial(jax.jit, static_argnames=('mmax', 'n_ring'))
def bin_fourier_rings_edges(x, y, z, w, R_edges, z_edges, mmax, n_ring):
    """General edges (linear, log or arbitrary).  Branch-free: index = sum(v >= interior edges).

    `jnp.searchsorted` would also work but compiles to a binary search with data-dependent
    gathers and measured ~5x slower; the comparison sum is O(n_bins) but vectorises cleanly.

    Points OUTSIDE the edges are DROPPED, not clipped.  That is what assign_regular_grid does
    (it sends them to the sentinel index n_bins.prod(), which segment_sum then discards), and the
    target integrates only over the box, so clipping them into the end rings would pile every
    orbit's excursion beyond R_hi or above z_hi onto the boundary ring and make the orbit solution
    look far heavier than the target there.  MEASURED, log grid R 0.1-10 / |z| 0.05-3: clipping
    gave A_0(model)/A_0(target) = 1.276 overall and 1.751 in the outermost ring; dropping gives
    1.001 / 1.003.  Do not "simplify" this away.
    """
    nR = R_edges.shape[0] - 1
    R = jnp.hypot(x, y)
    az = jnp.abs(z)
    iR = jnp.sum(R.reshape(-1, 1) >= R_edges[None, 1:-1], axis=-1)
    iZ = jnp.sum(az.reshape(-1, 1) >= z_edges[None, 1:-1], axis=-1)
    ring = iR + nR*iZ
    in_box = ((R >= R_edges[0]) & (R < R_edges[-1]) &
              (az >= z_edges[0]) & (az < z_edges[-1]))
    return jax.ops.segment_sum(_weights(x, y, w*in_box, mmax), ring, n_ring)


@partial(jax.jit, static_argnames=('mmax', 'n_ring', 'nR', 'nZ', 'log_R', 'log_z',
                                   'pow_R', 'pow_z', 'close_inner'))
def bin_fourier_rings_uniform(x, y, z, w, R_lo, R_hi, nR, z_lo, z_hi, nZ,
                              mmax, n_ring, log_R=False, log_z=False,
                              pow_R=None, pow_z=None, close_inner=False):
    """O(1) indexing for edges uniform in u, where u = v (linear) or u = log v (log).

    Same result as bin_fourier_rings_edges with np.linspace / np.geomspace edges, without the
    per-bin comparisons.  log_R / log_z are STATIC so the transform is compiled away.
    """
    # Rings are uniform in a monotonic transform u of the coordinate:
    #   pow_R = 1    -> u = v            LINEAR
    #   pow_R = 0.5  -> u = sqrt(v)      "mildly log": widths stretch by ~6x over 0.2-10 kpc
    #   log_R = True -> u = log v        LOG, widths stretch by ~38x
    # pow_* takes precedence over log_* when given; both are STATIC so the branch compiles away.
    # WHY IT MATTERS: with sigma proportional to A_0 every ring carries the SAME weight, so the
    # spacing sets how the constraint budget is divided between components.  MEASURED on NGC 1512
    # (15 rings, 0.2-10 kpc): pure log gives the NSD 53% of the budget for 25% of the mass and the
    # disc 13% for 35%; sqrt(R) gives 27% vs 25% and 20% vs 35%; linear gives 13% vs 25%.
    R = jnp.hypot(x, y)
    az = jnp.abs(z)
    _uR = (lambda v: v**pow_R) if pow_R is not None else ((lambda v: jnp.log(v)) if log_R else (lambda v: v))
    _uZ = (lambda v: v**pow_z) if pow_z is not None else ((lambda v: jnp.log(v)) if log_z else (lambda v: v))
    uR = _uR(jnp.maximum(R, R_lo)); aR = _uR(R_lo); bR = _uR(R_hi)
    uZ = _uZ(jnp.maximum(az, z_lo)); aZ = _uZ(z_lo); bZ = _uZ(z_hi)
    iR = jnp.clip(jnp.floor((uR - aR)/(bR - aR)*nR).astype(jnp.int32), 0, nR - 1)
    iZ = jnp.clip(jnp.floor((uZ - aZ)/(bZ - aZ)*nZ).astype(jnp.int32), 0, nZ - 1)
    # DROP out-of-box points (see bin_fourier_rings_edges): the clip above would otherwise pile
    # everything beyond R_hi / above z_hi -- and everything inside R_lo / below z_lo -- onto the
    # boundary rings, which the target does not contain.
    # close_inner=True DELETES the two lower-bound terms, so bin 0 runs 0 -> e_1: jnp.maximum
    # above and jnp.clip already map every sub-lo point onto index 0, and these terms were the
    # only thing throwing them away again.  Strictly less arithmetic; see fourier_edges.  The
    # UPPER bounds always stay -- beyond R_hi / above z_hi the target genuinely has no cell.
    if close_inner:
        in_box = (R < R_hi) & (az < z_hi)
    else:
        in_box = (R >= R_lo) & (R < R_hi) & (az >= z_lo) & (az < z_hi)
    return jax.ops.segment_sum(_weights(x, y, w*in_box, mmax), iR + nR*iZ, n_ring)


# ----------------------------------------------------------------------------------------------
#  the target: Fourier modes of the parametric density
# ----------------------------------------------------------------------------------------------
@partial(jax.jit, static_argnames=('rho_fct', 'mmax', 'n_phi', 'n_sub'))
def target_fourier_modes(R_edges, z_edges, rho_fct, dict_params, mmax,
                         phi_lo=-jnp.pi, phi_hi=jnp.pi, n_phi=32, n_sub=8):
    """Fourier modes of the ANALYTIC density over each (R,z) ring.

        A_m(ring) = int rho(R,z,phi) cos(m phi) R dR dz dphi        (and the sin counterpart)

    The phi quadrature is EQUISPACED, which for a smooth periodic integrand converges
    exponentially -- far better here than the Sobol sampling get_mass uses, and the reason this
    can be accurate with n_phi = 32.  (R,z) use an n_sub x n_sub midpoint rule per cell, which is
    what makes it correct on a LOG grid: get_mass samples uniformly inside the cell and would
    under-weight the inner edge, where the density is highest, by exactly the amount the log
    spacing was introduced to resolve.

    PHI MUST SPAN THE FULL 2*pi.  The binners accept every phi -- they never restrict the
    azimuth, because the 4-fold symmetry has already put all four images into the point set.  So
    the target has to integrate over 2*pi to match.  VERIFIED against a 400x360x120 brute-force
    quadrature fed through the binner: with (-pi, pi) the ratio target/binner is 0.9994 on A_0 and
    0.9996 on c_2; with the half domain (-pi/2, pi/2) it is exactly 0.4997 -- a silent factor of
    two in every constraint.  That is why the defaults are +-pi and not the half domain the
    regular grid uses.

    ACCURACY.  For A_0 the phi rule is not the bottleneck at all (16 and 32 agree to every digit),
    but for the HIGH HARMONICS it is: see the Nyquist guard below.  n_phi = 32 is converged to
    3 digits through m = 10; n_phi = 16 silently zeroes m = 8 and biases m = 6 by 12%.
    The (R,z) midpoint rule converges as n_sub^-2:  n_sub = 2 -> 5.0e-2,  4 -> 1.7e-2,
    8 -> 3.9e-3 max relative error on A_0.  n_sub = 4 is WORSE than the Sobol-1024 rule in
    get_mass; 8 is the default for that reason.

    The harmonics come from the SAME even_harmonics() the binner uses, evaluated on the unit
    circle, so the two sides cannot drift apart in convention.

    Returns (n_ring, n_mode) in the same column order as the binners.
    """
    # NYQUIST GUARD.  n_phi equispaced MIDPOINTS resolve m only up to n_phi/2 - 1, and at exactly
    # m = n_phi/2 the rule is DEGENERATE: cos(m*phi_k) = cos((k+1/2)*pi) = 0 for every k, so the
    # cosine coefficient is identically zero and the sine one is aliased.  MEASURED at n_phi = 16
    # on the NGC 1512 moregas density, ring 3.98-6.31 kpc, |A_m|/A_0:
    #     m=6  4.67e-02 (converged 5.32e-02, 12% LOW -- m=10 folded onto it)
    #     m=8  3.01e-07 (converged 1.91e-02, i.e. exactly ZERO)
    # A zero target is not a harmless approximation: it makes the NNLS force the orbit solution's
    # m=8 to zero, a constraint the density never asked for.  n_phi = 32 is converged to 3 digits
    # for m <= 10 and is the default.  Raise rather than silently mis-measure.
    if n_phi < 2*mmax + 4:
        raise ValueError('target_fourier_modes: n_phi=%d is too small for mmax=%d. The midpoint '
                         'rule is degenerate at m = n_phi/2 and aliased above it; use n_phi >= '
                         '%d (32 is the tested default).' % (n_phi, mmax, 2*mmax + 4))
    nR = R_edges.shape[0] - 1
    nZ = z_edges.shape[0] - 1
    # midpoints of n_sub sub-cells in each of R and z, and their widths
    f = (jnp.arange(n_sub) + 0.5)/n_sub
    Rl, Rr = R_edges[:-1], R_edges[1:]
    zl, zr = z_edges[:-1], z_edges[1:]
    Rs = Rl[:, None] + f[None, :]*(Rr - Rl)[:, None]          # (nR, n_sub)
    zs = zl[:, None] + f[None, :]*(zr - zl)[:, None]          # (nZ, n_sub)
    dR = (Rr - Rl)/n_sub
    dz = (zr - zl)/n_sub
    ph = phi_lo + (jnp.arange(n_phi) + 0.5)/n_phi*(phi_hi - phi_lo)
    dphi = (phi_hi - phi_lo)/n_phi

    def one_ring(iR, iZ):
        RR = Rs[iR][:, None, None]                             # (n_sub,1,1)
        ZZ = zs[iZ][None, :, None]
        PP = ph[None, None, :]
        rho = rho_fct(RR, ZZ, PP, dict_params)                 # broadcast to (n_sub,n_sub,n_phi)
        # volume element R dR dz dphi; the factor 2 is the z < 0 half, which the 4-fold
        # symmetry makes identical and which the binner also folds in via |z|.
        base = rho*RR*dR[iR]*dz[iZ]*dphi*2.0
        A0 = jnp.sum(base)
        H = even_harmonics(jnp.cos(PP.ravel()), jnp.sin(PP.ravel()), mmax)   # unit circle -> phi
        cols = [A0]
        for k in range(H.shape[1]):
            cols.append(jnp.sum(base*H[:, k].reshape(1, 1, -1)))
        return jnp.stack(cols)

    iR = jnp.arange(nR*nZ) % nR
    iZ = jnp.arange(nR*nZ) // nR
    return jax.vmap(one_ring)(iR, iZ)


def fourier_sigma(target_modes, sigma_frac, n_phi_equiv=12):
    """Error vector for the 3-D block.  Pass the TARGET modes, never the model's.

    TWO things must be right here.

    1.  Scale by the ring's A_0, never by the mode's own value.  `sigma_frac*A_m` is <= 0 wherever
        a coefficient is small or negative and the NNLS divides by it -- a silent NaN, the same
        family as the `s`-must-be-non-zero bug that killed the outer-disc dataset.

    2.  Use the TARGET's A_0.  It is a constant, so the constraint stays LINEAR in the orbit
        weights.  Using the model's A_0 would make sigma a function of w and the NNLS would no
        longer be solving a least-squares problem at all.

    NORMALISATION.  The regular grid applies sigma_j = sigma_frac * (cell mass) over n_phi_equiv
    azimuthal cells.  Writing that as an iid error sigma_frac*ybar on each cell and transforming
    with the (orthogonal) DFT gives

        sigma(A_0) = sigma_frac * A_0 / sqrt(n_phi)
        sigma(A_m) = sigma_frac * A_0 / sqrt(2*n_phi)        for m > 0

    so at n_phi_equiv = 12 and sigma_frac = 0.04 the old constraint is worth 1.2% on the ring mass
    and 0.82% of A_0 on each harmonic -- i.e. 3.4% on m=2 (A_2/A_0 = 0.24) and 26% on m=8
    (0.032), which is the right prior: weak modes constrained loosely.

    n_phi_equiv only sets the STRENGTH relative to the old grid; it is not a binning.  Leave it at
    12 to reproduce the current constraint, raise it to tighten.
    """
    A0 = jnp.abs(target_modes[:, :1])
    s0 = sigma_frac*A0/jnp.sqrt(n_phi_equiv)
    sm = sigma_frac*A0/jnp.sqrt(2.0*n_phi_equiv)
    out = jnp.concatenate([s0, jnp.broadcast_to(sm, (target_modes.shape[0],
                                                     target_modes.shape[1]-1))], axis=1)
    return out + 1e-10


def fourier_sigma_a0(target_modes, sigma_frac, sigma_frac_m0=None):
    """sigma = sigma_frac * A_0 for the harmonics, sigma_frac_m0 * A_0 for the m=0 column.

    sigma_frac_m0 defaults to sigma_frac (one tolerance for everything).  Setting it SMALLER
    tightens the RING MASS without touching the angular tolerance, which is the right lever when
    the mass is what the orbit solution is failing: MEASURED on the rank-3 draw, chi2/constraint
    was 44.4 for the A_0 column against 3.35 for the harmonics.  Going to 0.5%/2% takes the
    enclosed-mass error over the box from +5.8% to +1.4%, for chi2_den 1.44 -> 2.18.
    Tightening the harmonics as well buys nothing: at 0.5%/1% and 0.5%/0.5% the mass ratios are
    unchanged (1.014 -> 1.015 -> 1.016) and chi2_den only drifts up.

    The point of this scheme is what it does NOT depend on: the mode amplitudes.  With
    fourier_sigma_amplitude (sigma = f*|A_m|) the tolerance shrinks wherever the model is more
    axisymmetric, so the likelihood systematically rewards a STRONGER bar -- the error model is
    a function of the parameters being fitted.  MEASURED on NGC 1512: across the bar the weaker
    -barred fit was held to a tolerance up to 2.3x tighter than the stronger-barred one purely
    because its own |A_2| was smaller, and the sampler followed, driving R_d to 12.1 kpc (41%
    beyond the photometric grid) and the disc down to 22% of the stellar mass.

    A_0 is the ring MASS, which the projected light pins, so it barely moves between candidate
    models (23% between those two fits, against 230% for |A_2|).  Setting one absolute tolerance
    per ring from it leaves the harmonics constrained in ABSOLUTE terms and removes the incentive.

    Strength, for orientation: this is sqrt(2*n_phi) = 4.9x LOOSER on the harmonics than
    fourier_sigma(..., n_phi_equiv=12), which is the DFT image of the regular grid's own error.
    It constrains the ring mass to sigma_frac and leaves |A_2|/A_0 free at roughly
    sigma_frac/(|A_2|/A_0) -- about 13% where the bar is strong and effectively unconstrained
    where it is not.  That is a deliberate weakening: it buys freedom from the bias above.
    """
    f0 = sigma_frac if sigma_frac_m0 is None else sigma_frac_m0
    A0 = jnp.abs(target_modes[:, :1])
    return jnp.concatenate([f0*A0,
                            jnp.broadcast_to(sigma_frac*A0,
                                             (target_modes.shape[0], target_modes.shape[1]-1))],
                           axis=1) + 1e-30


def fourier_sigma_amplitude(target_modes, sigma_frac=0.05, floor_frac=0.06):
    """sigma = sigma_frac * |A_n|, with |A_n| = sqrt(c_n^2 + s_n^2) of the TARGET.

    This is the "5% on each harmonic" error model: every mode is reproduced to the same RELATIVE
    precision, rather than to the same absolute fraction of A_0.  Both the cos and the sin
    component of a mode share that mode's |A_n|, which is the right thing -- they are two
    components of one physical amplitude and splitting them would make sigma depend on the bar's
    orientation.

    THE FLOOR IS NOT OPTIONAL.  |A_n| goes to zero wherever a harmonic vanishes -- outside the
    bar, or for high m almost everywhere -- and sigma = 0 makes the NNLS divide by zero.  Worse
    than the NaN: even a small-but-finite sigma there would demand the orbits reproduce an exactly
    zero mode, which they cannot.  The orbit library's own shot noise sets the floor: MEASURED on
    NGC1512_twin_moregas, the odd harmonics (exactly zero by the 4-fold symmetry) come out at
    0.002-0.005 of A_0, so nothing can be reproduced better than ~0.003*A_0.  floor_frac = 0.06
    with sigma_frac = 0.05 puts the floor at 0.003*A_0, right at that noise level.

        sigma_n = sigma_frac * max(|A_n|, floor_frac*A_0)

    A_0 itself always uses |A_0| = A_0, so the ring mass is constrained to sigma_frac exactly.

    Compare `fourier_sigma()`, which instead reproduces the REGULAR GRID's constraint strength
    (sigma_frac*A_0/sqrt(2*n_phi), the same absolute error on every mode).  The two differ most at
    high m: at m=8 (|A_8|/A_0 = 0.032) this gives 0.05*0.032 = 0.0016*A_0 against the regular
    grid's 0.0082*A_0, i.e. 5x tighter -- which is why the floor matters.
    """
    A0 = jnp.abs(target_modes[:, :1])                       # (n_ring, 1)
    rest = target_modes[:, 1:]
    n_pair = rest.shape[1]//2
    cs = rest.reshape(rest.shape[0], n_pair, 2)
    amp = jnp.sqrt(jnp.sum(cs**2, axis=-1))                 # |A_n| per mode, (n_ring, n_pair)
    amp = jnp.maximum(amp, floor_frac*A0)
    sig_m = sigma_frac*jnp.repeat(amp, 2, axis=1)           # same sigma for cos and sin
    return jnp.concatenate([sigma_frac*A0, sig_m], axis=1) + 1e-30


def fourier_edges(lo, hi, n, log=False, p=None, close_inner=False):
    """The ring edges the binners use, so the target cannot disagree with the model.

    p=1 linear, p=0.5 sqrt ("mildly log"), log=True logarithmic.  p takes precedence over log.

    close_inner=True moves the FIRST edge to 0, so bin 0 runs 0 -> e_1 instead of lo -> e_1.
    `lo` then stops being a physical boundary and becomes only the reference point of the index
    formula (the `log` and `sqrt` transforms still need lo > 0, which jnp.maximum(v, lo) in the
    binner supplies).  Bins 1..n and the transform are untouched.

    WHY.  `lo` exists because the log/sqrt spacing needs a positive inner edge, and everything
    below it was being DROPPED from the 3-D block -- an unconstrained hole in the one place the
    cold orbits live.  MEASURED on NGC1512_twin_rank3 (z_lo = 0.05 kpc): the analytic target puts
    6.15% of M_star below z_lo, the orbit solution's particles put 12.86% -- a factor 2.09, and
    2.28 once normalised to the constrained box.  Every region the constraint DOES cover sits at
    1.0-1.1.  So the slab was not a passive gap; the NNLS was moving mass into it.

    COST: none.  On the binner side this is strictly LESS arithmetic (two comparisons deleted from
    the in-box mask): measured 697.1 us against 748.5, x0.937, and bit-identical (max|diff|
    0.000e+00) to bin_fourier_rings_edges with e_0 = 0.  On the target side the midpoint rule gets
    MORE accurate, not less, even though ring 0 in R widens x2.03: the integrand is rho*R, so
    extending to R -> 0 adds a region the Jacobian suppresses, growing the ring's A_0 faster than
    the quadrature error.  Max relative error on A_0 at the production n_sub = 8, against an
    independent Gauss-Legendre 96x96 reference: 4.707e-03 open -> 3.917e-03 closed.
    """
    # jnp, not numpy: lo/hi are TRACED inside build_model_acc_fourier (only n, log and p are
    # static), so a numpy linspace raises TracerArrayConversionError under jit.
    if p is not None:
        u = jnp.linspace(lo**p, hi**p, n+1)
        e = u**(1.0/p)
    elif log:
        e = jnp.exp(jnp.linspace(jnp.log(lo), jnp.log(hi), n+1))
    else:
        e = jnp.linspace(lo, hi, n+1)
    return e.at[0].set(0.0) if close_inner else e
