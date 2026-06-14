import jax
import jax.numpy as jnp
from functools import partial
from constants import *
from utils import assign_regular_grid
from ghmoments import gauss, H0, H1, H2, H3, H4

def _deriv_barred(state, acc_fn, Omega):
    """RHS of the co-rotating frame ODE.

    State = [x, y, z, vx, vy, vz] where velocities are inertial
    components rotated into bar-frame axes.

    ODE:  dx/dt  = vx + Omega*y      dvx/dt = ax + Omega*vy
          dy/dt  = vy - Omega*x      dvy/dt = ay - Omega*vx
          dz/dt  = vz                dvz/dt = az
    """
    x, y_pos, z, vx, vy, vz = state[0], state[1], state[2], state[3], state[4], state[5]
    ax, ay, az = acc_fn(x, y_pos, z)
    return jnp.array([
        vx + Omega * y_pos,
        vy - Omega * x,
        vz,
        ax + Omega * vy,
        ay - Omega * vx,
        az,
    ])

def _compute_EJ(y, pot_fn, Omega):
    """Jacobi energy: E_J = 0.5*v^2 + Phi - Omega*(x*vy - y*vx)."""
    x, y_pos, z, vx, vy, vz = y[0], y[1], y[2], y[3], y[4], y[5]
    E_kin = 0.5 * (vx**2 + vy**2 + vz**2)
    E_pot = pot_fn(x[jnp.newaxis], y_pos[jnp.newaxis], z[jnp.newaxis])[0]
    Lz = x * vy - y_pos * vx
    return E_kin + E_pot - Omega * Lz

@partial(jax.jit, static_argnames=('acc_fn', 'pot_fn', 'N_max', 'chunk_size', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_adaptive_barred_chunked(
    w0, acc_fn, pot_fn, N_max, T_total,
    dt_init=0.010, Omega=0.0,
    atol=1e-8, rtol=1e-6,
    dt_min=1e-5, dt_max=0.1,
    num_Vbin=1028, bin_mapping=jnp.zeros(2400, dtype=jnp.int32),
    num_per_bin=jnp.zeros(1028, dtype=jnp.int32),
    Rzphi_minmax=jnp.array([[0, 10.], [-3, 3], [-jnp.pi, jnp.pi]]),
    XY_minmax=jnp.array([[-10., 10.], [-2., 2.]]),
    nRzphi=jnp.array([10, 6, 6]), nXY=jnp.array([40, 30]),
    num_segments_Rzphi=360,
    v0=jnp.zeros(1028), s=jnp.ones(1028) * 5.0,
    rotation_matrix=jnp.eye(3),
    chunk_size=500,
):
    """Adaptive BS2(3) integrator with chunked binning.

    Same physics as integrate_adaptive_barred, but instead of storing the full
    (N_max, 6) trajectory and binning at the end, it processes the orbit in
    chunks of `chunk_size` steps: each chunk stores a small (chunk_size, 6)
    trajectory buffer, bins it vectorially, and accumulates into running sums.

    Memory per orbit: O(chunk_size * 6 + num_Vbin + num_segments_Rzphi)
    instead of O(N_max * 6 + num_Vbin + num_segments_Rzphi).

    N_max must be divisible by chunk_size.
    """
    n_chunks = N_max // chunk_size

    # Precompute grid constants
    Rzphi_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nRzphi[:-1])])
    XY_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nXY[:-1])])
    area_pixel = ((XY_minmax[0, 1] - XY_minmax[0, 0]) / nXY[0]) * \
                 ((XY_minmax[1, 1] - XY_minmax[1, 0]) / nXY[1]) * 1e6

    # 4-fold bar symmetry signs
    sign_sym = jnp.array([
        [ 1,  1,  1,  1,  1,  1],
        [ 1,  1, -1,  1,  1, -1],
        [-1, -1,  1, -1, -1,  1],
        [-1, -1, -1, -1, -1, -1],
    ])

    # Initial derivative and E_J
    k1_init = _deriv_barred(w0, acc_fn, Omega)
    EJ_0 = _compute_EJ(w0, pot_fn, Omega)

    # BS23 error coefficients
    e1, e2, e3, e4 = -5.0/72.0, 1.0/12.0, 1.0/9.0, -1.0/8.0
    safety = 0.9

    # ── Inner scan: one RK step ──
    def rk_step(carry, _):
        t, y, dt, k1 = carry

        k2 = _deriv_barred(y + 0.5 * dt * k1, acc_fn, Omega)
        k3 = _deriv_barred(y + 0.75 * dt * k2, acc_fn, Omega)
        y_new = y + dt * (2.0/9.0 * k1 + 1.0/3.0 * k2 + 4.0/9.0 * k3)
        k4 = _deriv_barred(y_new, acc_fn, Omega)

        err_vec = dt * (e1 * k1 + e2 * k2 + e3 * k3 + e4 * k4)
        scale = atol + rtol * jnp.maximum(jnp.fabs(y), jnp.fabs(y_new))
        err_norm = jnp.sqrt(jnp.mean((err_vec / scale)**2))
        accept = err_norm <= 1.0

        y_out = jnp.where(accept, y_new, y)
        t_new = jnp.where(accept, t + dt, t)
        k1_new = jnp.where(accept, k4, k1)

        scale_factor = safety * jnp.power(jnp.maximum(err_norm, 1e-10), -1.0/3.0)
        scale_factor = jnp.clip(scale_factor, 0.2, 5.0)
        dt_new = jnp.clip(dt * scale_factor, dt_min, dt_max)

        dt_out = jnp.where(accept, dt, 0.0)
        return (t_new, y_out, dt_new, k1_new), (y_out, dt_out)

    # ── Bin a chunk of trajectory points (vectorized) ──
    def bin_chunk(y_chunk, dt_chunk, Rzphi_acc, counts_acc, sw0_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc, T_acc):
        # Apply 4-fold symmetry: (chunk_size, 6) -> (4*chunk_size, 6)
        y_sym = (y_chunk[None, :, :] * sign_sym[:, None, :]).reshape(-1, 6)
        dt_sym = jnp.tile(dt_chunk, 4)

        # Rzphi binning (pre-rotation frame)
        R_vals = jnp.sqrt(y_sym[:, 0]**2 + y_sym[:, 1]**2)
        phi_vals = jnp.arctan2(y_sym[:, 1], y_sym[:, 0])
        Rzphi = jnp.stack([R_vals, y_sym[:, 2], phi_vals], axis=-1)

        Rzphi_indices = assign_regular_grid(Rzphi,
                                            grid_min=Rzphi_minmax[:, 0],
                                            grid_max=Rzphi_minmax[:, 1],
                                            n_bins=nRzphi,
                                            strides=Rzphi_strides)

        # XY binning (post-rotation)
        x_pos = y_sym[:, :3]
        v_vel = y_sym[:, 3:]
        x_rot = (rotation_matrix @ x_pos.T).T
        v_rot = (rotation_matrix @ v_vel.T).T
        XY = jnp.stack([x_rot[:, 0], x_rot[:, 1]], axis=-1)

        XY_indices = assign_regular_grid(XY,
                                         grid_min=XY_minmax[:, 0],
                                         grid_max=XY_minmax[:, 1],
                                         n_bins=nXY,
                                         strides=XY_strides)
        Vbin_indices = bin_mapping[XY_indices]

        # GH moment accumulation (unnormalized — we normalize at the end)
        vz = v_rot[:, 2] * KPCGYR_TO_KMS
        v0_cell = v0[Vbin_indices]
        s_cell = s[Vbin_indices]
        Rzphi_acc = Rzphi_acc + jax.ops.segment_sum(dt_sym, Rzphi_indices, num_segments=num_segments_Rzphi)
        counts_acc = counts_acc + jax.ops.segment_sum(dt_sym, Vbin_indices, num_segments=num_Vbin)

        _y, _gau = gauss(vz, v0_cell, s_cell)
        _H0, _H1, _H2, _H3, _H4 = H0(_y), H1(_y), H2(_y), H3(_y), H4(_y)

        h0 = jax.ops.segment_sum(_gau * dt_sym * _H0, Vbin_indices, num_segments=num_Vbin)
        h1 = jax.ops.segment_sum(_gau * dt_sym * _H1, Vbin_indices, num_segments=num_Vbin)
        h2 = jax.ops.segment_sum(_gau * dt_sym * _H2, Vbin_indices, num_segments=num_Vbin)
        h3 = jax.ops.segment_sum(_gau * dt_sym * _H3, Vbin_indices, num_segments=num_Vbin)
        h4 = jax.ops.segment_sum(_gau * dt_sym * _H4, Vbin_indices, num_segments=num_Vbin)

        # h1 = h1 / (h0 + EPSILON)
        # h2 = h2 / (h0 + EPSILON)
        # h3 = h3 / (h0 + EPSILON)
        # h4 = h4 / (h0 + EPSILON)

        sw0_acc = sw0_acc + h0
        sw1_acc = sw1_acc + h1
        sw2_acc = sw2_acc + h2
        sw3_acc = sw3_acc + h3
        sw4_acc = sw4_acc + h4

        T_acc = T_acc + jnp.sum(dt_sym)

        return Rzphi_acc, counts_acc, sw0_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc, T_acc

    # ── Outer scan: process one chunk per iteration ──
    def chunk_body(carry, _):
        t, y, dt, k1, Rzphi_acc, counts_acc, sw0_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc, T_acc = carry

        # Inner scan: chunk_size RK steps
        (t_new, y_new, dt_new, k1_new), (y_chunk, dt_chunk) = jax.lax.scan(
            rk_step, (t, y, dt, k1), xs=None, length=chunk_size
        )

        # Bin the chunk
        Rzphi_acc, counts_acc, sw0_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc, T_acc = \
            bin_chunk(y_chunk, dt_chunk, Rzphi_acc, counts_acc, sw0_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc, T_acc)

        return (t_new, y_new, dt_new, k1_new, Rzphi_acc, counts_acc, sw0_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc, T_acc), None

    # Initialize accumulators
    init_carry = (
        0.0, w0, dt_init, k1_init,
        jnp.zeros(num_segments_Rzphi),  # Rzphi_acc
        jnp.zeros(num_Vbin),            # counts_acc
        jnp.zeros(num_Vbin),            # sw0_acc
        jnp.zeros(num_Vbin),            # sw1_acc
        jnp.zeros(num_Vbin),            # sw2_acc
        jnp.zeros(num_Vbin),            # sw3_acc
        jnp.zeros(num_Vbin),            # sw4_acc
        0.0,                            # T_acc
    )

    (t_final, y_final, _, _, Rzphi_acc, counts_acc, sw0_acc, sw1_acc, sw2_acc, sw3_acc, sw4_acc, T_acc), _ = \
        jax.lax.scan(chunk_body, init_carry, xs=None, length=n_chunks)

    # ── Orbit-level validity ──
    EJ_final = _compute_EJ(y_final, pot_fn, Omega)
    delta_EJ = jnp.fabs(EJ_final / EJ_0 - 1.0)
    valid = jnp.where((delta_EJ < 0.1) & (t_final > T_total / 10), 1.0, 0.0)

    # ── Normalize bin-occupancy quantities by T_integrated ──
    # We deliberately keep the GH moments un-normalised by sw0 (Agama "option b"):
    # each orbit's contribution per bin is sw_k / T_integrated for k = 0..4.
    # Dividing by T (rather than dropping the division entirely) keeps the per-orbit
    # entries O(1) when an orbit weight is O(1), regardless of integration length.
    T_integrated = T_acc + EPSILON
    Rzphi_bin_counts = Rzphi_acc / T_integrated

    h0 = sw0_acc / T_integrated
    h1 = sw1_acc / T_integrated
    h2 = sw2_acc / T_integrated
    h3 = sw3_acc / T_integrated
    h4 = sw4_acc / T_integrated
    surface_density = (counts_acc / T_integrated) / (num_per_bin * area_pixel + EPSILON)

    n_accepted = T_acc / (dt_init + EPSILON)  # approximate

    # Zero out bad orbits
    Rzphi_bin_counts = jnp.where(valid > 0.5, Rzphi_bin_counts, jnp.zeros_like(Rzphi_bin_counts))
    surface_density = jnp.where(valid > 0.5, surface_density, jnp.zeros_like(surface_density))
    h0 = jnp.where(valid > 0.5, h0, jnp.zeros_like(h0))
    h1 = jnp.where(valid > 0.5, h1, jnp.zeros_like(h1))
    h2 = jnp.where(valid > 0.5, h2, jnp.zeros_like(h2))
    h3 = jnp.where(valid > 0.5, h3, jnp.zeros_like(h3))
    h4 = jnp.where(valid > 0.5, h4, jnp.zeros_like(h4))

    return Rzphi_bin_counts, surface_density, h0, h1, h2, h3, h4, valid, n_accepted, T_integrated


_integrate_adaptive_chunked_vmap = jax.vmap(integrate_adaptive_barred_chunked,
                    in_axes=(
                            0, None, None, None, 0,
                            0, None,
                            None, None,
                            None, None,
                            None, None, None,
                            None, None,
                            None, None, None,
                            None, None, None,
                            None))


@partial(jax.jit, static_argnames=('acc_fn', 'pot_fn', 'N_max', 'chunk_size', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_adaptive_batch_chunked(w0, acc_fn, pot_fn, N_max, T_total,
                             dt_init=0.010, Omega=0.0,
                             atol=1e-8, rtol=1e-6,
                             dt_min=1e-5, dt_max=0.1,
                             num_Vbin=1028, bin_mapping=jnp.zeros(2400, dtype=jnp.int32),
                             num_per_bin=jnp.zeros(1028, dtype=jnp.int32),
                             Rzphi_minmax=jnp.array([[0, 10.], [-3, 3], [-jnp.pi, jnp.pi]]),
                             XY_minmax=jnp.array([[-10., 10.], [-2., 2.]]),
                             nRzphi=jnp.array([10, 6, 6]), nXY=jnp.array([40, 30]),
                             num_segments_Rzphi=360,
                             v0=jnp.zeros(1028), s=jnp.ones(1028) * 5.0,
                             rotation_matrix=jnp.eye(3),
                             chunk_size=500):
    """Batch adaptive integration with chunked binning, analogous to integrate_adaptive_batch."""

    Rzphi_bin_counts, surface_density, h0, h1, h2, h3, h4, valid, n_accepted, T_integrated = _integrate_adaptive_chunked_vmap(
        w0, acc_fn, pot_fn, N_max, T_total,
        dt_init, Omega, atol, rtol, dt_min, dt_max,
        num_Vbin, bin_mapping, num_per_bin,
        Rzphi_minmax, XY_minmax,
        nRzphi, nXY, num_segments_Rzphi,
        v0, s, rotation_matrix,
        chunk_size)

    A_Rzphi = Rzphi_bin_counts.T
    A_xy = surface_density.T
    A_h0 = h0.T
    A_h1 = h1.T
    A_h2 = h2.T
    A_h3 = h3.T
    A_h4 = h4.T

    # Uniform average across realisations for the per-particle library entry.
    # h_k_T = sw_k / T_integrated is linear in the orbit phase-space density,
    # so a linear (un-weighted) average across jitter realisations is correct.
    weights = jnp.ones(A_Rzphi.shape[1]) / (valid.sum() + 0.1)

    Rzphi_bin_counts_out = A_Rzphi @ weights
    surface_density_out = A_xy @ weights
    h0_out = A_h0 @ weights
    h1_out = A_h1 @ weights
    h2_out = A_h2 @ weights
    h3_out = A_h3 @ weights
    h4_out = A_h4 @ weights

    return Rzphi_bin_counts_out, surface_density_out, h0_out, h1_out, h2_out, h3_out, h4_out, valid.sum()


_integrate_adaptive_batch_chunked_vmap = jax.vmap(integrate_adaptive_batch_chunked,
                    in_axes=(
                            0, None, None, None, 0,
                            0, None,
                            None, None,
                            None, None,
                            None, None, None,
                            None, None,
                            None, None, None,
                            None, None, None,
                            None))



# ===========================================================================================================

@partial(jax.jit, static_argnames=('acc_fn', 'pot_fn', 'N_max', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_adaptive_barred_withtraj(
    w0, acc_fn, pot_fn, N_max, T_total,
    dt_init=0.010, Omega=0.0,
    atol=1e-8, rtol=1e-6,
    dt_min=1e-5, dt_max=0.1,
    num_Vbin=1028, bin_mapping=jnp.zeros(2400, dtype=jnp.int32),
    num_per_bin=jnp.zeros(1028, dtype=jnp.int32),
    Rzphi_minmax=jnp.array([[0, 10.], [-3, 3], [-jnp.pi, jnp.pi]]),
    XY_minmax=jnp.array([[-10., 10.], [-2., 2.]]),
    nRzphi=jnp.array([10, 6, 6]), nXY=jnp.array([40, 30]),
    num_segments_Rzphi=360,
    v0=jnp.zeros(1028), s=jnp.ones(1028) * 5.0,
    rotation_matrix=jnp.eye(3),
):
    """Bogacki-Shampine RK2(3) integrator with embedded error control.

    Uses 3 acc_fn calls per step (FSAL). No pot_fn calls during integration.
    Step size controlled by local truncation error, not energy conservation.
    E_J checked only at orbit level for validity.

    Parameters
    ----------
    w0 : array (6,)
        Initial phase-space state [x, y, z, vx, vy, vz] in bar frame.
    acc_fn : callable
        Acceleration function acc_fn(x, y, z) -> (ax, ay, az) in bar frame.
    pot_fn : callable
        Potential function pot_fn(x, y, z) -> Phi (only for orbit-level E_J check).
    N_max : int
        Maximum number of scan iterations (static for jit).
    T_total : float
        Total integration time.
    dt_init : float
        Initial step size.
    Omega : float
        Bar pattern speed.
    atol : float
        Absolute error tolerance (per component).
    rtol : float
        Relative error tolerance (per component).
    dt_min, dt_max : float
        Step size bounds.

    Returns
    -------
    8-tuple: (Rzphi_bin_counts, surface_density, h1, h2, h3, h4, valid, n_accepted)
    """
    # Compute initial derivative (FSAL: reused as k1 in first step)
    k1_init = _deriv_barred(w0, acc_fn, Omega)

    # Compute initial E_J for orbit-level validity check
    EJ_0 = _compute_EJ(w0, pot_fn, Omega)

    # BS23 error coefficients: err = y3 - y2
    # err = dt * (-5/72*k1 + 1/12*k2 + 1/9*k3 - 1/8*k4)
    e1, e2, e3, e4 = -5.0/72.0, 1.0/12.0, 1.0/9.0, -1.0/8.0

    safety = 0.9

    def scan_body(carry, _):
        t, y, dt, k1 = carry

        # No early stopping — all N_max iterations do useful work.
        # Orbits past T_total keep integrating; extra points improve
        # the time-averaged density/kinematics at no extra cost.

        # ── Bogacki-Shampine RK2(3) stages ──
        # k1 is carried from previous step (FSAL)
        k2 = _deriv_barred(y + 0.5 * dt * k1, acc_fn, Omega)          # stage 2
        k3 = _deriv_barred(y + 0.75 * dt * k2, acc_fn, Omega)         # stage 3
        y_new = y + dt * (2.0/9.0 * k1 + 1.0/3.0 * k2 + 4.0/9.0 * k3)  # 3rd order
        k4 = _deriv_barred(y_new, acc_fn, Omega)                       # stage 4 (FSAL)

        # ── Embedded error estimate ──
        err_vec = dt * (e1 * k1 + e2 * k2 + e3 * k3 + e4 * k4)

        # Per-component scaling (AGAMA-style)
        scale = atol + rtol * jnp.maximum(jnp.fabs(y), jnp.fabs(y_new))
        err_scaled = err_vec / scale
        err_norm = jnp.sqrt(jnp.mean(err_scaled**2))  # RMS norm

        # Accept if err_norm <= 1
        accept = err_norm <= 1.0

        # Update state
        y_out = jnp.where(accept, y_new, y)
        t_new = jnp.where(accept, t + dt, t)
        # FSAL: if accepted, next k1 = k4; if rejected, keep old k1
        k1_new = jnp.where(accept, k4, k1)

        # Step size adjustment: dt_new = dt * safety * err^(-1/3) for 3rd order method
        scale_factor = safety * jnp.power(jnp.maximum(err_norm, 1e-10), -1.0/3.0)
        scale_factor = jnp.clip(scale_factor, 0.2, 5.0)
        dt_new = dt * scale_factor
        dt_new = jnp.clip(dt_new, dt_min, dt_max)

        valid_flag = accept.astype(jnp.float32)
        # Output dt_used for time-weighting (0 if rejected)
        dt_out = jnp.where(accept, dt, 0.0)

        return (t_new, y_out, dt_new, k1_new), (y_out, valid_flag, dt_out)

    init_carry = (0.0, w0, dt_init, k1_init)
    (t_final, y_final, dt_final, _), (y_traj, valid_mask, dt_traj) = jax.lax.scan(
        scan_body, init_carry, xs=None, length=N_max
    )
    t_traj = jnp.cumsum(dt_traj)
    x_pos = y_traj[:, :3]
    v_vel = y_traj[:, 3:] * KPCGYR_TO_KMS
    o_traj = jnp.concatenate([x_pos, v_vel], axis=-1)

    n_accepted = jnp.sum(valid_mask)

    # Orbit-level validity: check total E_J drift (same threshold as leapfrog)
    EJ_final = _compute_EJ(y_final, pot_fn, Omega)
    delta_EJ = jnp.fabs(EJ_final / EJ_0 - 1.0)
    valid = jnp.where((delta_EJ < 0.1) & (t_final > T_total / 10), 1.0, 0.0)#

    # ── Post-scan binning with dt-weighting ──
    # Rzphi binning (pre-rotation frame)
    R_vals = jnp.sqrt(y_traj[:, 0]**2 + y_traj[:, 1]**2)
    phi_vals = jnp.arctan2(y_traj[:, 1], y_traj[:, 0])
    Rzphi = jnp.stack([R_vals, y_traj[:, 2], phi_vals], axis=-1)

    # Rotation for XY projection (new convention: project onto (x_rot, y_rot),
    # line-of-sight = z_rot, matching generate_mock_correctGH.py).
    x_pos = y_traj[:, :3]
    v_vel = y_traj[:, 3:]
    x_rot = (rotation_matrix @ x_pos.T).T
    v_rot = (rotation_matrix @ v_vel.T).T
    wN_rot = jnp.concatenate([x_rot, v_rot], axis=-1)
    XY = jnp.stack([wN_rot[:, 0], wN_rot[:, 1]], axis=-1)

    Rzphi_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nRzphi[:-1])])
    Rzphi_indices = assign_regular_grid(Rzphi,
                                        grid_min=Rzphi_minmax[:, 0],
                                        grid_max=Rzphi_minmax[:, 1],
                                        n_bins=nRzphi,
                                        strides=Rzphi_strides)

    XY_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nXY[:-1])])
    XY_indices = assign_regular_grid(XY,
                                     grid_min=XY_minmax[:, 0],
                                     grid_max=XY_minmax[:, 1],
                                     n_bins=nXY,
                                     strides=XY_strides)

    area_pixel = ((XY_minmax[0, 1] - XY_minmax[0, 0]) / nXY[0]) * \
                 ((XY_minmax[1, 1] - XY_minmax[1, 0]) / nXY[1]) * 1e6

    Vbin_indices = bin_mapping[XY_indices]

    # Time-weight: divide by T_integrated so per-orbit contributions are O(1) at unit weight.
    T_integrated = jnp.sum(dt_traj) + EPSILON
    dt_norm = dt_traj / T_integrated

    Rzphi_bin_counts = jax.ops.segment_sum(dt_norm,
                                           Rzphi_indices,
                                           num_segments=num_segments_Rzphi)

    v0_cell = v0[Vbin_indices]
    s_cell  = s[Vbin_indices]

    # vdMF / option-b GH accumulation: project on z_rot (= LOS), build sw_k = sum dt · α(y)/s · H_k(y).
    vz = wN_rot[:, 5] * KPCGYR_TO_KMS
    _y, _gau = gauss(vz, v0_cell, s_cell)
    _H0, _H1, _H2, _H3, _H4 = H0(_y), H1(_y), H2(_y), H3(_y), H4(_y)

    counts = jax.ops.segment_sum(dt_norm, Vbin_indices, num_segments=num_Vbin)
    sw0    = jax.ops.segment_sum(dt_norm * _gau * _H0, Vbin_indices, num_segments=num_Vbin)
    sw1    = jax.ops.segment_sum(dt_norm * _gau * _H1, Vbin_indices, num_segments=num_Vbin)
    sw2    = jax.ops.segment_sum(dt_norm * _gau * _H2, Vbin_indices, num_segments=num_Vbin)
    sw3    = jax.ops.segment_sum(dt_norm * _gau * _H3, Vbin_indices, num_segments=num_Vbin)
    sw4    = jax.ops.segment_sum(dt_norm * _gau * _H4, Vbin_indices, num_segments=num_Vbin)

    # Un-normalised moments (dt already normalised by T_integrated above).
    h0 = sw0
    h1 = sw1
    h2 = sw2
    h3 = sw3
    h4 = sw4
    surface_density = counts / (num_per_bin * area_pixel + EPSILON)

    # Zero out bad orbits
    Rzphi_bin_counts = jnp.where(valid > 0.5, Rzphi_bin_counts, jnp.zeros_like(Rzphi_bin_counts))
    surface_density = jnp.where(valid > 0.5, surface_density, jnp.zeros_like(surface_density))
    h0 = jnp.where(valid > 0.5, h0, jnp.zeros_like(h0))
    h1 = jnp.where(valid > 0.5, h1, jnp.zeros_like(h1))
    h2 = jnp.where(valid > 0.5, h2, jnp.zeros_like(h2))
    h3 = jnp.where(valid > 0.5, h3, jnp.zeros_like(h3))
    h4 = jnp.where(valid > 0.5, h4, jnp.zeros_like(h4))

    return Rzphi_bin_counts, surface_density, h0, h1, h2, h3, h4, valid, n_accepted, T_integrated, o_traj, t_traj


_integrate_adaptive_withtraj_vmap = jax.vmap(integrate_adaptive_barred_withtraj,
                    in_axes=(
                            0, None, None, None, 0,    # w0 per-orbit, T_total per-orbit
                            0, None,       # dt_init per-orbit, Omega broadcast
                            None, None,    # atol, rtol broadcast
                            None, None,    # dt_min, dt_max
                            None, None, None,
                            None, None,
                            None, None, None,
                            None, None, None))

@partial(jax.jit, static_argnames=('acc_fn', 'pot_fn', 'N_max', 'num_Vbin', 'num_segments_Rzphi'))
def integrate_adaptive_withtraj_batch(w0, acc_fn, pot_fn, N_max, T_total,
                             dt_init=0.010, Omega=0.0,
                             atol=1e-8, rtol=1e-6,
                             dt_min=1e-5, dt_max=0.1,
                             num_Vbin=1028, bin_mapping=jnp.zeros(2400, dtype=jnp.int32),
                             num_per_bin=jnp.zeros(1028, dtype=jnp.int32),
                             Rzphi_minmax=jnp.array([[0, 10.], [-3, 3], [-jnp.pi, jnp.pi]]),
                             XY_minmax=jnp.array([[-10., 10.], [-2., 2.]]),
                             nRzphi=jnp.array([10, 6, 6]), nXY=jnp.array([40, 30]),
                             num_segments_Rzphi=360,
                             v0=jnp.zeros(1028), s=jnp.ones(1028) * 5.0,
                             rotation_matrix=jnp.eye(3)):
    """Batch adaptive integration over multiple orbits, analogous to integrate_batch."""

    Rzphi_bin_counts, surface_density, h0, h1, h2, h3, h4, valid, _, _, y_traj, t_traj = _integrate_adaptive_withtraj_vmap(
        w0, acc_fn, pot_fn, N_max, T_total,
        dt_init, Omega, atol, rtol, dt_min, dt_max,
        num_Vbin, bin_mapping, num_per_bin,
        Rzphi_minmax, XY_minmax,
        nRzphi, nXY, num_segments_Rzphi,
        v0, s, rotation_matrix)

    # New (option-b) convention: h_k = sw_k / T_integrated, un-normalised; A_h0 exposed.
    A_Rzphi = Rzphi_bin_counts.T
    A_xy = surface_density.T
    A_h0 = h0.T
    A_h1 = h1.T
    A_h2 = h2.T
    A_h3 = h3.T
    A_h4 = h4.T

    # Uniform-average across realisations: sw_k is linear in orbit density,
    # so a plain mean is correct (no A_xy weighting needed under option-b).
    weights = jnp.ones(A_Rzphi.shape[1]) / (valid.sum() + 0.1)

    Rzphi_bin_counts_out = A_Rzphi @ weights
    surface_density_out  = A_xy @ weights
    h0_out = A_h0 @ weights
    h1_out = A_h1 @ weights
    h2_out = A_h2 @ weights
    h3_out = A_h3 @ weights
    h4_out = A_h4 @ weights

    return Rzphi_bin_counts_out, surface_density_out, h0_out, h1_out, h2_out, h3_out, h4_out, valid.sum(), y_traj, t_traj


_integrate_adaptive_withtraj_batch_vmap = jax.vmap(integrate_adaptive_withtraj_batch,
                    in_axes=(
                            0, None, None, None, 0,
                            0, None,
                            None, None,
                            None, None,
                            None, None, None,
                            None, None,
                            None, None, None,
                            None, None, None))