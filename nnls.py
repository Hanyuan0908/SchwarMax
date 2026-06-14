import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=("maxiter",))
def solve_nnls_admm(
    A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    y_Rzphi, y_xy, gamma_kin, y_h1, y_h2, y_h3, y_h4,
    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    lambda_reg=1, maxiter=200,
    w_rzphi = jnp.sqrt(1.0), w_xy = jnp.sqrt(1.0), w_h = jnp.sqrt(1.0)
):
    eps = 1e-8
    gamma_kin_safe = jnp.where(jnp.abs(gamma_kin) > eps, gamma_kin, eps)

    # vdMF / Agama option-b construction: kinematic rows weight orbits by A_h0
    # (carried in A_h_k via the gamma_kin denominator), not by A_xy.
    U_rz  = w_rzphi * A_Rzphi / (sig_Rzphi[:, None] + eps)
    y_rz  = w_rzphi * y_Rzphi / (sig_Rzphi + eps)
    U_xy_ = w_xy * A_xy / (sig_xy[:, None] + eps)
    y_xy_ = w_xy * y_xy / (sig_xy + eps)
    U_h1_ = w_h * A_h1 / (gamma_kin_safe[:, None] * (sig_A1[:, None] + eps))
    U_h2_ = w_h * A_h2 / (gamma_kin_safe[:, None] * (sig_A2[:, None] + eps))
    U_h3_ = w_h * A_h3 / (gamma_kin_safe[:, None] * (sig_A3[:, None] + eps))
    U_h4_ = w_h * A_h4 / (gamma_kin_safe[:, None] * (sig_A4[:, None] + eps))
    y_h1_ = w_h * y_h1 / (sig_A1 + eps)
    y_h2_ = w_h * y_h2 / (sig_A2 + eps)
    y_h3_ = w_h * y_h3 / (sig_A3 + eps)
    y_h4_ = w_h * y_h4 / (sig_A4 + eps)

    U = jnp.vstack([U_rz, U_xy_, U_h1_, U_h2_, U_h3_, U_h4_])
    y = jnp.concatenate([y_rz, y_xy_, y_h1_, y_h2_, y_h3_, y_h4_])

    n_orb = U.shape[1]
    reg   = lambda_reg / n_orb

    Q = U.T @ U + reg * jnp.eye(n_orb, dtype=U.dtype)
    c = -(U.T @ y)

    rho = jnp.trace(Q) / n_orb

    L_chol = jnp.linalg.cholesky(Q + rho * jnp.eye(n_orb, dtype=U.dtype))

    w_init = jnp.ones(n_orb, dtype=U.dtype) * (jnp.sum(y_Rzphi) / n_orb)
    z_init = w_init.copy()
    u_init = jnp.zeros(n_orb, dtype=U.dtype)

    alpha = 1.6

    def admm_step(carry, _):
        w, z, u = carry
        rhs = rho * (z - u) - c
        w_new = jax.scipy.linalg.cho_solve((L_chol, True), rhs)
        w_hat = alpha * w_new + (1.0 - alpha) * z
        z_new = jnp.maximum(0.0, w_hat + u)
        u_new = u + w_hat - z_new
        return (w_new, z_new, u_new), None

    (_, z_final, _), _ = jax.lax.scan(
        admm_step,
        (w_init, z_init, u_init),
        xs=None,
        length=maxiter,
    )
    return jax.lax.stop_gradient(z_final)

@partial(jax.jit, static_argnames=("maxiter",))
def solve_nnls_admm_bootstrap(
    A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    y_Rzphi, y_xy, gamma_kin,
    y_xy_boot, y_h1_boot, y_h2_boot, y_h3_boot, y_h4_boot,
    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    lambda_reg=1, maxiter=200,
    w_rzphi = jnp.sqrt(1.0), w_xy = jnp.sqrt(1.0), w_h = jnp.sqrt(1.0)
):
    """
    Bootstrap ADMM solver: solve NNLS for multiple observation realizations.

    y_Rzphi is model-computed (3D density grid) and shared across all bootstraps.
    Only the observational data (y_xy, h1-h4) are bootstrapped.

    Builds U, Q, L_chol once. Vmaps only the ADMM scan loop over
    the N_boot different RHS vectors c_i = -U^T y_i.

    Args:
        A_*: orbital library matrices (n_bins, n_orb) — shared.
            A_h_k (k=1..4) are the *un-normalised* per-orbit moments sw_k/T_integrated
            (vdMF / Agama option-b convention).
        y_Rzphi: model-computed 3D density (n_Rzphi_bins,) — shared, NOT bootstrapped
        y_xy: original surface density (n_xy_bins,) — for U matrix construction
        gamma_kin: per-bin base-Gaussian amplitude (n_xy_bins,), used as kinematic
            normaliser in place of y_xy. = M_b / (norm_b · 2√π · s_b).
        y_xy_boot: bootstrapped surface density (N_boot, n_xy_bins)
        y_h1_boot..y_h4_boot: bootstrapped kinematics (N_boot, n_xy_bins)
        sig_*: error vectors (n_bins,) — shared

    Returns:
        weights_all: (N_boot, n_orb) — NNLS weights for each bootstrap
    """
    eps = 1e-8
    gamma_kin_safe = jnp.where(jnp.abs(gamma_kin) > eps, gamma_kin, eps)

    # ---- Build design matrix U (shared across all bootstraps) ----
    # Kinematic rows: residual = (Σᵢ wᵢ · A_h_k[i,b]) / gamma_kin_b - h_k_obs[b],
    # weighted by 1/sigma_hk. Mass rows: A_xy / sigma_xy (unchanged, photometric).
    U_rz  = w_rzphi * A_Rzphi / (sig_Rzphi[:, None] + eps)
    U_xy_ = w_xy * A_xy / (sig_xy[:, None] + eps)
    U_h1_ = w_h * A_h1 / (gamma_kin_safe[:, None] * (sig_A1[:, None] + eps))
    U_h2_ = w_h * A_h2 / (gamma_kin_safe[:, None] * (sig_A2[:, None] + eps))
    U_h3_ = w_h * A_h3 / (gamma_kin_safe[:, None] * (sig_A3[:, None] + eps))
    U_h4_ = w_h * A_h4 / (gamma_kin_safe[:, None] * (sig_A4[:, None] + eps))
    U = jnp.vstack([U_rz, U_xy_, U_h1_, U_h2_, U_h3_, U_h4_])

    n_orb = U.shape[1]
    reg = lambda_reg / n_orb

    # ---- Shared Cholesky (computed once) ----
    Q = U.T @ U + reg * jnp.eye(n_orb, dtype=U.dtype)
    rho = jnp.trace(Q) / n_orb
    L_chol = jnp.linalg.cholesky(Q + rho * jnp.eye(n_orb, dtype=U.dtype))
    w_init_val = jnp.sum(y_Rzphi) / n_orb
    alpha = 1.6

    # ---- Normalized y_Rzphi (shared, not bootstrapped) ----
    y_rz_shared = w_rzphi * y_Rzphi / (sig_Rzphi + eps)  # (n_Rzphi_bins,)

    # ---- Build normalized y vectors for each bootstrap ----
    # y_Rzphi part is the same for all bootstraps; only xy and h1-h4 vary
    def _build_y_vec(y_xy_i, y_h1_i, y_h2_i, y_h3_i, y_h4_i):
        y_xy_ = w_xy * y_xy_i / (sig_xy + eps)
        y_h1_ = w_h * y_h1_i / (sig_A1 + eps)
        y_h2_ = w_h * y_h2_i / (sig_A2 + eps)
        y_h3_ = w_h * y_h3_i / (sig_A3 + eps)
        y_h4_ = w_h * y_h4_i / (sig_A4 + eps)
        return jnp.concatenate([y_rz_shared, y_xy_, y_h1_, y_h2_, y_h3_, y_h4_])

    y_vecs = jax.vmap(_build_y_vec)(
        y_xy_boot, y_h1_boot, y_h2_boot, y_h3_boot, y_h4_boot
    )  # (N_boot, n_data)

    # ---- c_i = -U^T y_i for all bootstraps ----
    c_all = -(y_vecs @ U)  # (N_boot, n_orb)

    # ---- Vmapped ADMM scan ----
    def admm_scan(c_vec):
        w_init = jnp.ones(n_orb, dtype=U.dtype) * w_init_val
        z_init = w_init.copy()
        u_init = jnp.zeros(n_orb, dtype=U.dtype)

        def admm_step(carry, _):
            w, z, u = carry
            rhs = rho * (z - u) - c_vec
            w_new = jax.scipy.linalg.cho_solve((L_chol, True), rhs)
            w_hat = alpha * w_new + (1.0 - alpha) * z
            z_new = jnp.maximum(0.0, w_hat + u)
            u_new = u + w_hat - z_new
            return (w_new, z_new, u_new), None

        (_, z_final, _), _ = jax.lax.scan(
            admm_step, (w_init, z_init, u_init), xs=None, length=maxiter,
        )
        return z_final

    weights_all = jax.lax.stop_gradient(jax.vmap(admm_scan)(c_all))  # (N_boot, n_orb)
    return weights_all
