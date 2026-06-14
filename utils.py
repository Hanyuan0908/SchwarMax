import numpy as np
import jax
import jax.numpy as jnp
import pickle
from scipy.stats import qmc
from functools import partial

from constants import G, EPSILON, TWOPI

@jax.jit
def assign_regular_grid(positions, grid_min, grid_max, n_bins, strides):
    """
    Assign positions to regular grid cells.
    
    Parameters
    ----------
    positions : array (N, D) - particle positions
    grid_min : array (D,) - minimum bounds for each dimension
    grid_max : array (D,) - maximum bounds for each dimension
    n_bins : array (D,) - number of bins in each dimension (int)
    
    Returns
    -------
    bin_indices : array (N,) - flattened bin index for each particle
    """
    # Normalize to [0, 1]
    normalized = (positions - grid_min) / (grid_max - grid_min)
    # jax.debug.print("normalized: {normalized}", normalized=normalized)
    
    # Convert to bin indices
    bin_coords = jnp.floor(normalized * n_bins).astype(jnp.int32)
    
    # Clip to valid range [0, n_bins-1]
    bin_coords = jnp.clip(bin_coords, 0, n_bins - 1)
    
    # Flatten to 1D index: idx = i + j*nx + k*nx*ny
    bin_indices = jnp.sum(bin_coords * strides, axis=-1)

    # Check if any normalized coordinate is >= 1.0 or < 0.0 for each particle
    out_of_bounds = jnp.any((normalized >= 1.0) | (normalized < 0.0), axis=-1)

    return jnp.where(out_of_bounds, n_bins.prod(), bin_indices)


@jax.jit
def assign_regular_grid1d(positions, grid_min, grid_max, n_bins):
    """
    Assign positions to regular grid cells.
    
    Parameters
    ----------
    positions : array (N) - particle positions
    grid_min : float - minimum bounds for each dimension
    grid_max : float - maximum bounds for each dimension
    n_bins : int - number of bins in each dimension
    
    Returns
    -------
    bin_indices : array (N,) - flattened bin index for each particle
    """

    dx = (grid_max - grid_min)/n_bins
    index = (positions - grid_min) // dx

    out_of_bounds = (index >= n_bins) | (index < 0.0)
    return jnp.where(out_of_bounds, n_bins, index).astype(jnp.int32)



@jax.jit
def get_mat(x, y, z):
    v1 = jnp.array([0.0, 0.0, 1.0])
    I3 = jnp.eye(3)

    # Create a fixed-shape vector from inputs
    v2 = jnp.array([x, y, z])
    # Normalize v2 in one step
    v2 = v2 / (jnp.linalg.norm(v2) + EPSILON)

    # Compute the angle using a fused dot and clip operation
    angle = jnp.arccos(jnp.clip(jnp.dot(v1, v2), -1.0, 1.0))

    # Compute normalized rotation axis
    v3 = jnp.cross(v1, v2)
    v3 = v3 / (jnp.linalg.norm(v3) + EPSILON)

    # Build the skew-symmetric matrix K for Rodrigues' formula
    K = jnp.array([
        [0, -v3[2], v3[1]],
        [v3[2], 0, -v3[0]],
        [-v3[1], v3[0], 0]
    ])

    sin_angle = jnp.sin(angle)
    cos_angle = jnp.cos(angle)

    # Compute rotation matrix using Rodrigues' formula
    rot_mat = I3 + sin_angle * K + (1 - cos_angle) * jnp.dot(K, K)
    return rot_mat

@jax.jit
def go_to_bar_ref(xv, angle):
    # Rotate contourclockwise with positive angle
    sina, cosa = jnp.sin(angle), jnp.cos(angle)
    x, y, z, vx, vy, vz = xv
    x_new  = x * cosa - y * sina
    y_new  = x * sina + y * cosa
    vx_new = vx * cosa - vy * sina
    vy_new = vx * sina + vy * cosa

    return xv.at[0].set(x_new).at[1].set(y_new).at[3].set(vx_new).at[4].set(vy_new)

@partial(jax.jit, static_argnames=('xlim', 'ylim', 'zlim', 'dx', 'dy', 'dz'))
def histogram3d(x, weights, xlim=(-10, 10), ylim=(-10, 10), zlim=(-3, 3), dx=1.0, dy=1.0, dz=1.0):
    # Define bin edges for each dimension
    x_bins = jnp.arange(xlim[0], xlim[1] + dx, dx)
    y_bins = jnp.arange(ylim[0], ylim[1] + dy, dy)
    z_bins = jnp.arange(zlim[0], zlim[1] + dz, dz)

    bins, _ = jnp.histogramdd(x, bins=[x_bins, y_bins, z_bins], weights=weights)
    return bins

# ---------- helpers ----------
def shift_origin(x, y, z, p):
    # Convert scalar params to arrays matching x,y,z shape
    x0 = jnp.asarray(p["x_origin"])
    y0 = jnp.asarray(p["y_origin"])
    z0 = jnp.asarray(p["z_origin"])

    # Broadcast to match shapes of inputs
    x0 = jnp.broadcast_to(x0, x.shape)
    y0 = jnp.broadcast_to(y0, y.shape)
    z0 = jnp.broadcast_to(z0, z.shape)

    # Stack as a 3-vector field
    return jnp.stack([x - x0, y - y0, z - z0], axis=0)

def rotate_zaxis(vec, p):
    # vec: (3, ...)
    R = get_mat(p["dirx"], p["diry"], p["dirz"])  # (3,3)
    # Tensordot over axis: (i,a) * (a,...) -> (i,...)
    return jnp.tensordot(R, vec, axes=[[1],[0]])

@jax.jit
def getCartesianFromCylindrical_clockwise(R, phi, vR, vphi):
    """
    Reverts R, phi, vR, vphi back to Cartesian x, y, vx, vy.
    Consistent with the 'clockwise' vphi convention provided.
    """
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    
    # 1. Positions
    x = R * cos_phi
    y = R * sin_phi
    
    # 2. Velocities
    # Derived by inverting the linear system from your input function
    vx = vR * cos_phi + vphi * sin_phi
    vy = vR * sin_phi - vphi * cos_phi
    
    return x, y, vx, vy

def Rz(t):
    ct, st = jnp.cos(t), jnp.sin(t)
    return jnp.array([[ct, -st, 0.0],
                     [st,  ct, 0.0],
                     [0.0, 0.0, 1.0]])

def Rx(t):
    ct, st = jnp.cos(t), jnp.sin(t)
    return jnp.array([[1.0, 0.0, 0.0],
                     [0.0,  ct, -st],
                     [0.0,  st,  ct]])

def makeRotationMatrix(alpha, beta, gamma):

    alpha, beta, gamma = jnp.radians(alpha), jnp.radians(beta), jnp.radians(gamma)
    return (Rz(gamma) @ Rx(beta) @ Rz(alpha)).T   # X = R @ x


@partial(jax.jit, static_argnames=("potential_fn",))
def estimate_orbital_timescale(R, potential_fn, potential_args=(), z=0.0, dR=1e-3):
    """
    Order-of-magnitude orbital timescale from a gravitational potential.

    Uses a local circular-orbit estimate:
        Omega^2(R) = (1 / R) * dPhi/dR
        T_orb(R)   = 2*pi / Omega

    Parameters
    ----------
    R : float or array-like
        Cylindrical radius (kpc).
    potential_fn : callable
        Function with signature:
            potential_fn(x, y, z, *potential_args) -> Phi
        and Phi in units of kpc^2 / Gyr^2.
    potential_args : tuple, optional
        Extra arguments forwarded to potential_fn.
    z : float, optional
        Height where dPhi/dR is evaluated (default 0.0).
    dR : float, optional
        Finite-difference step in kpc.

    Returns
    -------
    T_orb : float or jnp.ndarray
        Estimated orbital timescale in Gyr.
    """
    R = jnp.asarray(R)
    R_shape = R.shape
    R_flat = jnp.ravel(R)
    R_safe = jnp.maximum(jnp.abs(R_flat), 2 * dR)
    min_val = 1e-20

    def phi_of_R_scalar(r):
        return potential_fn(r, 0.0, z, *potential_args)

    def dphi_dr_scalar(r):
        return (phi_of_R_scalar(r + dR) - phi_of_R_scalar(r - dR)) / (2.0 * dR)

    dPhi_dR = jax.vmap(dphi_dr_scalar)(R_safe)
    omega2 = jnp.maximum(jnp.abs(dPhi_dR) / R_safe, min_val)
    omega = jnp.sqrt(omega2)

    T_orb = 2 * jnp.pi / omega
    return jnp.reshape(T_orb, R_shape)


@partial(jax.jit, static_argnames=("potential_fn",))
def get_rotation_curve(R, potential_fn, potential_args=(), z=0.0, dR=1e-3):
    """
    Circular speed curve for an axisymmetric potential.

    Uses:
        v_c^2(R, z) = R * dPhi/dR

    Parameters
    ----------
    R : float or array-like
        Cylindrical radius (kpc).
    potential_fn : callable
        Function with signature:
            potential_fn(x, y, z, *potential_args) -> Phi
        and Phi in units of kpc^2 / Gyr^2.
    potential_args : tuple, optional
        Extra arguments forwarded to potential_fn.
    z : float, optional
        Height where dPhi/dR is evaluated (default 0.0).
    dR : float, optional
        Finite-difference step in kpc.

    Returns
    -------
    v_c : float or jnp.ndarray
        Circular speed in kpc / Gyr, with the same shape as R.
    """
    R = jnp.asarray(R)
    R_shape = R.shape
    R_flat = jnp.ravel(R)
    R_safe = jnp.maximum(jnp.abs(R_flat), 2 * dR)

    def phi_of_R_scalar(r):
        return potential_fn(r, 0.0, z, *potential_args)

    def dphi_dr_scalar(r):
        return (phi_of_R_scalar(r + dR) - phi_of_R_scalar(r - dR)) / (2.0 * dR)

    dPhi_dR = jax.vmap(dphi_dr_scalar)(R_safe)
    vc2 = jnp.maximum(R_safe * dPhi_dR, 0.0)
    v_c = jnp.sqrt(vc2)
    return jnp.reshape(v_c, R_shape)

def build_Lcirc_of_E(potential_fn, potential_args=(), R_min=1e-2, R_max=1e2, n_grid=2000, dR=1e-3):
    """
    Build an interpolation function Lcirc(E) from a gravitational potential.

    For a grid of radii R, computes the energy and angular momentum of circular
    orbits in the midplane (z=0):
        Vc(R) = sqrt(R * dPhi/dR)
        E_circ(R) = Phi(R, 0) + 0.5 * Vc(R)^2
        L_circ(R) = R * Vc(R)

    Then returns an interpolator that maps any energy E to Lcirc(E).

    Parameters
    ----------
    potential_fn : callable
        Potential function with signature potential_fn(x, y, z, *potential_args) -> Phi.
        Phi in units of kpc^2 / Gyr^2.
    potential_args : tuple, optional
        Extra arguments forwarded to potential_fn.
    R_min, R_max : float
        Range of radii for the lookup table (kpc).
    n_grid : int
        Number of grid points (log-spaced).
    dR : float
        Finite-difference step for dPhi/dR.

    Returns
    -------
    Lcirc_interp : callable
        Function E -> Lcirc(E). Input and output are numpy arrays.
        E in kpc^2/Gyr^2, Lcirc in kpc^2/Gyr.
    E_circ_grid : ndarray
        The energy grid (for reference / plotting).
    L_circ_grid : ndarray
        The Lcirc grid (for reference / plotting).
    """
    from scipy.interpolate import interp1d

    R_grid = np.logspace(np.log10(R_min), np.log10(R_max), n_grid)

    # Evaluate potential and its radial derivative on the midplane
    Phi_grid = np.array([float(potential_fn(r, 0.0, 0.0, *potential_args)) for r in R_grid])
    Phi_plus = np.array([float(potential_fn(r + dR, 0.0, 0.0, *potential_args)) for r in R_grid])
    Phi_minus = np.array([float(potential_fn(r - dR, 0.0, 0.0, *potential_args)) for r in R_grid])
    dPhi_dR = (Phi_plus - Phi_minus) / (2.0 * dR)

    Vc2 = np.maximum(R_grid * dPhi_dR, 0.0)
    Vc = np.sqrt(Vc2)

    E_circ = Phi_grid + 0.5 * Vc2
    L_circ = R_grid * Vc

    # E_circ should be monotonically increasing with R for bound orbits.
    # Sort to ensure monotonicity for interpolation.
    sort_idx = np.argsort(E_circ)
    E_circ = E_circ[sort_idx]
    L_circ = L_circ[sort_idx]

    # Remove duplicates
    mask = np.diff(E_circ, prepend=-np.inf) > 0
    E_circ = E_circ[mask]
    L_circ = L_circ[mask]

    Lcirc_interp = interp1d(E_circ, L_circ, kind='cubic',
                            bounds_error=False, fill_value=(L_circ[0], L_circ[-1]))

    return Lcirc_interp, E_circ, L_circ


def halo_mass_from_stellar_mass(M_star,
    N=0.0351, log10_M1=11.59, beta=1.376, gamma=0.608,
    mmin=1e9, mmax=3e16, tol=1e-6, max_iter=200):
    """
    Return halo mass M_h [Msun] for a given stellar mass M_star [Msun]
    using the Moster+2013 z=0 SHMR (median relation).
    """
    def mstar_from_mh(Mh):
        x = Mh / (10**log10_M1)
        return 2*N*Mh / (x**(-beta) + x**gamma)

    a, b = mmin, mmax
    for _ in range(max_iter):
        mid = 10**((jnp.log10(a)+jnp.log10(b))/2)
        if mstar_from_mh(mid) > M_star:
            b = mid
        else:
            a = mid
        if abs(jnp.log10(b) - jnp.log10(a)) < tol:
            return 10**((jnp.log10(a)+jnp.log10(b))/2)

    return 10**((jnp.log10(a)+jnp.log10(b))/2)



def XexpX_pdf_log(x, a):
    """
    Probability density function of the distribution proportional to x * exp(-x/a).
    
    Parameters
    ----------
    x : array_like
        Points at which to evaluate the PDF. Can be scalar or array.
    a : float
        Scale parameter > 0.
    
    Returns
    -------
    pdf : array_like
        The PDF values at x.
    """
    # Ensure a > 0
    a = jnp.asarray(a)
    # PDF formula: (1/a^2) * x * exp(-x/a)
    pdf = jnp.log(x) - jnp.log(a**2) - (x / a)
    return jnp.where(x >= 0, pdf, -jnp.inf)

def expX_pdf_log(x, a):
    """
    Probability density function of the distribution proportional to exp(-x/a).
    
    Parameters
    ----------
    x : array_like
        Points at which to evaluate the PDF. Can be scalar or array.
    a : float
        Scale parameter > 0.
    
    Returns
    -------
    pdf : array_like
        The PDF values at x.
    """
    # Ensure a > 0
    a = jnp.asarray(a)
    # PDF formula: (1/a^2) * x * exp(-x/a)
    pdf = jnp.log(a) - (x / a)
    return jnp.where(x >= 0, pdf, -jnp.inf)

def compute_transfer_matrix(sigma_psf, nX, nY, X_minmax, Y_minmax,
                             bin_mapping, total_bins, grid_res=500):
    """
    Compute PSF transfer matrix P using grid-based convolution (Option 2).

    P[i,j] = fraction of flux from Voronoi bin j observed in bin i after
    Gaussian PSF convolution. Columns sum to 1 (flux conservation).

    Parameters
    ----------
    sigma_psf : float
        PSF standard deviation in kpc. If 0, returns identity matrix.
    nX, nY : int
        Number of pixels in the regular grid (X and Y directions).
    X_minmax, Y_minmax : tuple of (float, float)
        FOV limits (min, max) in kpc for X and Y.
    bin_mapping : array of int, shape (nX*nY,) or (nX*nY+1,)
        Maps each regular grid pixel to a Voronoi bin ID.
        If length nX*nY+1, the last entry is treated as a sentinel and dropped.
    total_bins : int
        Number of Voronoi bins.
    grid_res : int, optional
        Resolution of the fine grid used for convolution (default 500).

    Returns
    -------
    P : ndarray, shape (total_bins, total_bins)
        PSF transfer matrix.
    """
    from scipy.ndimage import gaussian_filter

    bin_mapping = np.asarray(bin_mapping)
    if len(bin_mapping) == nX * nY + 1:
        bin_mapping = bin_mapping[:-1]  # drop sentinel

    X_min, X_max = X_minmax
    Y_min, Y_max = Y_minmax

    if sigma_psf == 0:
        return np.eye(total_bins)

    # Fine regular grid covering the FOV
    x_edges = np.linspace(X_min, X_max, grid_res + 1)
    y_edges = np.linspace(Y_min, Y_max, grid_res + 1)
    x_c = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_c = 0.5 * (y_edges[:-1] + y_edges[1:])
    XX, YY = np.meshgrid(x_c, y_c)  # (grid_res, grid_res)

    # Assign fine grid points to Voronoi bins via the regular grid
    nx = (XX.ravel() - X_min) / (X_max - X_min)
    ny = (YY.ravel() - Y_min) / (Y_max - Y_min)
    ix = np.clip(np.floor(nx * nX).astype(int), 0, nX - 1)
    iy = np.clip(np.floor(ny * nY).astype(int), 0, nY - 1)
    fine_bin_ids = bin_mapping[ix + iy * nX].reshape(grid_res, grid_res)

    # PSF sigma in fine-grid pixel units
    pixel_size = x_edges[1] - x_edges[0]
    sigma_pix = sigma_psf / pixel_size

    # Build P by convolving indicator images
    P = np.zeros((total_bins, total_bins))
    for j in range(total_bins):
        indicator_j = (fine_bin_ids == j).astype(float)
        convolved_j = gaussian_filter(indicator_j, sigma=sigma_pix, mode='constant')
        for i in range(total_bins):
            P[i, j] = convolved_j[fine_bin_ids == i].sum()

    # Column-normalise (flux conservation)
    col_sums = P.sum(axis=0)
    col_sums = np.where(col_sums > 0, col_sums, 1.0)
    P = P / col_sums[None, :]

    return P


@partial(jax.jit,static_argnums=(2))
def sample_from_logP(x_grid, logP, N, key):
    """
    Draw N samples from the distribution defined by logP on the grid x_grid
    using the inverse‐CDF method.
    """
    # 1) Shift & exponentiate for numerical stability
    logP = jnp.asarray(logP)
    logP = logP - jnp.max(logP)
    P = jnp.exp(logP)

    # 2) Normalize to get a proper probability mass on the grid
    P /= P.sum()

    # 3) Build the CDF
    cdf = jnp.cumsum(P)

    # 4) Sample uniforms and invert the CDF via linear interpolation
    # jax_random_key2 = jax.random.PRNGKey(random_seed)
    u = jax.random.uniform(key, shape=(N,))
    samples = jnp.interp(u, cdf, x_grid)
    return samples


# ── NFW parameter transforms (JAX-compatible) ──────────────────────

@jax.jit
def logM_logRs_to_logMenc_logc(logM_halo, logRs_halo, r_enc=10.0, Delta=200., rho_crit=277.54):
    """NFW (logM, logRs) -> (logM_enc(<r_enc), log_concentration)."""
    M = 10.0 ** logM_halo
    Rs = 10.0 ** logRs_halo
    x = r_enc / Rs
    M_enc = M * (jnp.log(1.0 + x) - x / (1.0 + x))
    R_vir = (3.0 * M / (4.0 * jnp.pi * Delta * rho_crit)) ** (1.0 / 3.0)
    c = R_vir / Rs
    return jnp.log10(M_enc), jnp.log10(c)


@jax.jit
def logMenc_logc_to_logM_logRs(logM_enc, log_c, r_enc=10.0, Delta=200., rho_crit=277.54):
    """Inverse: (logM_enc(<r_enc), log_concentration) -> (logM, logRs).

    Solves M_enc = M * [ln(1+x) - x/(1+x)] where x = r_enc/Rs and
    Rs = R_vir/c = (3M/(4 pi Delta rho_crit))^(1/3) / c.

    Uses Newton's method (6 iterations, quadratic convergence).
    """
    M_enc = 10.0 ** logM_enc
    c = 10.0 ** log_c
    coeff = 4.0 * jnp.pi * Delta * rho_crit / 3.0
    ln10 = jnp.log(10.0)

    def _residual_and_deriv(logM_trial):
        M = 10.0 ** logM_trial
        Rs = (M / coeff) ** (1.0 / 3.0) / c
        x = r_enc / Rs
        f_nfw = jnp.log(1.0 + x) - x / (1.0 + x)
        f = M * f_nfw - M_enc

        # df/d(logM) via chain rule:
        #   d/d(logM) = d/dM * dM/d(logM) = d/dM * M * ln(10)
        #   dRs/dM = Rs / (3M)
        #   dx/dM = -x / (3M)
        #   d(f_nfw)/dx = x / (1+x)^2
        #   d(M*f_nfw)/dM = f_nfw + M * d(f_nfw)/dx * dx/dM
        #                 = f_nfw - x^2 / (3*(1+x)^2)
        df_dM = f_nfw - x * x / (3.0 * (1.0 + x) ** 2)
        df = df_dM * M * ln10
        return f, df

    # Initial guess: logM_enc + 0.5 (M is typically 1–10x M_enc)
    logM = logM_enc + 0.5

    def newton_step(logM, _):
        f, df = _residual_and_deriv(logM)
        return logM - f / df, None

    logM_sol, _ = jax.lax.scan(newton_step, logM, None, length=6)

    M_sol = 10.0 ** logM_sol
    Rs_sol = (M_sol / coeff) ** (1.0 / 3.0) / c
    return logM_sol, jnp.log10(Rs_sol)


def load_data_bootstrap(path, filename, N_BOOTSTRAP = 100, n_samples = 5_000):
    with open(path + filename, 'rb') as f:
        bin_dict = pickle.load(f)

    X_minmax = jnp.array(bin_dict['X_minmax'])
    Y_minmax = jnp.array(bin_dict['Y_minmax'])
    nX_nY = jnp.array(bin_dict['nX_nY'])

    # voronoi binning mapping and data
    num_per_bin = jnp.array(bin_dict['num_per_bin'])
    total_bins = jnp.array(bin_dict['total_bins'])
    bin_mapping = jnp.array(bin_dict['bin_mapping'])
    surface_density = jnp.array(bin_dict['surface_density'])
    V_data = jnp.array(bin_dict['V_mean'])
    sigma_data = jnp.array(bin_dict['V_sigma'])
    h1_data = jnp.array(bin_dict['h1'])
    h2_data = jnp.array(bin_dict['h2'])
    h3_data = jnp.array(bin_dict['h3'])
    h4_data = jnp.array(bin_dict['h4'])
    v0 = jnp.array(bin_dict['v0'])
    s = jnp.array(bin_dict['s'])
    alpha, beta, gamma = bin_dict['orientation']


    XY_density_data_err = 0.01 * surface_density + EPSILON
    V_data_err = jnp.array(bin_dict['V_mean_err'])
    sigma_data_err = jnp.array(bin_dict['V_sigma_err'])
    h1_data_err = jnp.array(bin_dict['h1_err'])
    h2_data_err = jnp.array(bin_dict['h2_err'])
    h3_data_err = jnp.array(bin_dict['h3_err'])
    h4_data_err = jnp.array(bin_dict['h4_err'])

    rng = np.random.default_rng(42)
    XY_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(surface_density)))
    h1_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h1_data)))
    h2_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h2_data)))
    h3_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h3_data)))
    h4_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(h4_data)))
    V_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(V_data)))
    sigma_standard_normal = rng.normal(size=(N_BOOTSTRAP, len(sigma_data)))
    XY_standard_normal[0, :] = 0.0
    h1_standard_normal[0, :] = 0.0
    h2_standard_normal[0, :] = 0.0
    h3_standard_normal[0, :] = 0.0
    h4_standard_normal[0, :] = 0.0
    V_standard_normal[0, :] = 0.0
    sigma_standard_normal[0, :] = 0.0
    XY_standard_normal = jnp.array(XY_standard_normal)
    h1_standard_normal = jnp.array(h1_standard_normal)
    h2_standard_normal = jnp.array(h2_standard_normal)
    h3_standard_normal = jnp.array(h3_standard_normal)
    h4_standard_normal = jnp.array(h4_standard_normal)
    V_standard_normal = jnp.array(V_standard_normal)
    sigma_standard_normal = jnp.array(sigma_standard_normal)

    R_min, R_max =0., 10.
    z_min, z_max = -3., 3.
    n_R, n_z, n_phi =10, 6, 10
    n_tot = int(n_R * n_z * n_phi)
    R_edge = jnp.linspace(R_min, R_max, n_R+1)
    z_edge = jnp.linspace(z_min, z_max, n_z+1)
    phi_edge = jnp.linspace(-jnp.pi, jnp.pi, n_phi+1)
    R_mids, z_mids, phi_mids = 0.5 * (R_edge[:-1] + R_edge[1:]), 0.5 * (z_edge[:-1] + z_edge[1:]), 0.5 * (phi_edge[:-1] + phi_edge[1:])
    dR, dz, dphi = R_edge[1]-R_edge[0], z_edge[1]-z_edge[0], phi_edge[1]-phi_edge[0]
    R_mids_mesh, z_mids_mesh, phi_mids_mesh = jnp.meshgrid(R_mids, z_mids, phi_mids, indexing='ij')
    Rzphi_mid_grid = jnp.stack([R_mids_mesh.ravel(), z_mids_mesh.ravel(), phi_mids_mesh.ravel()], axis=-1)  # (n_R*n_z*n_phi, 3)
    R_grid = Rzphi_mid_grid[:,0]
    z_grid = Rzphi_mid_grid[:,1]
    phi_grid = Rzphi_mid_grid[:,2]
    dR = np.unique(R_grid)[1] - np.unique(R_grid)[0]
    dz = np.unique(z_grid)[1] - np.unique(z_grid)[0]
    dphi = np.unique(phi_grid)[1] - np.unique(phi_grid)[0]
    Rzphi_minmax=jnp.array([[R_min, R_max],[z_min, z_max],[-jnp.pi, jnp.pi]])
    nRzphi=jnp.array([n_R,n_z,n_phi])
    num_segments_Rzphi=nRzphi.prod()
    Rzphi_strides = jnp.concatenate([jnp.array([1]), jnp.cumprod(nRzphi[:-1])])
    Rzphi_grid_indices = assign_regular_grid(Rzphi_mid_grid,
                                        grid_min=Rzphi_minmax[:,0],
                                        grid_max=Rzphi_minmax[:,1],
                                        n_bins=nRzphi,
                                        strides=Rzphi_strides)
    _, COUNTS = jnp.unique(Rzphi_grid_indices, return_counts=True)
    argsort = jnp.argsort(Rzphi_grid_indices)
    R_grid = R_grid[argsort]
    z_grid = z_grid[argsort]
    phi_grid = phi_grid[argsort]
    sampler = qmc.Sobol(d=3, scramble=False)
    sample_for_integration = sampler.random_base2(m=10)

    
    X_regular_grid, Y_regular_grid = bin_dict['X_regular_grid'], bin_dict['Y_regular_grid']
    dX = jnp.unique(X_regular_grid)[1] - jnp.unique(X_regular_grid)[0]
    dY = jnp.unique(Y_regular_grid)[1] - jnp.unique(Y_regular_grid)[0]
    sampler = qmc.Sobol(d=3, scramble=False)
    sample = sampler.random_base2(m=10)

    n_samples = n_samples  # Same number as original data
    x_grid = np.linspace(0., 12., 1000)
    logP_xexp = XexpX_pdf_log(x_grid, 3)   # radial scale length [kpc]; must match get_dict_data_bootstrap in SchwarMAX/likelihoods_bar.py
    key = jax.random.PRNGKey(10086)
    R_samples = sample_from_logP(x_grid, logP_xexp, n_samples, key)
    phi_samples = np.random.default_rng(42).uniform(0, 2*np.pi, size=n_samples)

    x_samples, y_samples = R_samples * np.cos(phi_samples), R_samples * np.sin(phi_samples)

    x_grid = np.linspace(0, 5, 1000)
    logP_exp = expX_pdf_log(x_grid, 1.2)   # vertical scale height [kpc]; must match get_dict_data_bootstrap in SchwarMAX/likelihoods_bar.py
    key = jax.random.PRNGKey(10010)
    z_samples = sample_from_logP(x_grid, logP_exp, n_samples, key)
    w0 = np.array([
        x_samples,
        y_samples,
        z_samples,
    ]).T


    dict_data = {
        'v0': v0,
        's': s,

        'XY_density_data': surface_density,
        'XY_density_data_err': XY_density_data_err,
        'V_data': V_data,
        'V_data_err': V_data_err,
        'sigma_data': sigma_data,
        'sigma_data_err': sigma_data_err,
        'h1_data': h1_data,
        'h1_data_err': h1_data_err,
        'h2_data': h2_data,
        'h2_data_err': h2_data_err,
        'h3_data': h3_data,
        'h3_data_err': h3_data_err,
        'h4_data': h4_data,
        'h4_data_err': h4_data_err,
        'num_per_bin': num_per_bin,
        'bin_mapping': bin_mapping,
        'total_bins': total_bins.item(),

        'XY_standard_normal': XY_standard_normal,
        'h1_standard_normal': h1_standard_normal,
        'h2_standard_normal': h2_standard_normal,
        'h3_standard_normal': h3_standard_normal,
        'h4_standard_normal': h4_standard_normal,
        'V_standard_normal': V_standard_normal,
        'sigma_standard_normal': sigma_standard_normal,

        'R_grid': R_grid,
        'z_grid': z_grid,
        'phi_grid': phi_grid,
        'R_minmax': [R_min, R_max],
        'z_minmax': [z_min, z_max],
        'phi_minmax': [-jnp.pi, jnp.pi],
        'Rzphi_n_tot': n_tot,
        'Rzphi_n_grid': jnp.array([n_R, n_z, n_phi]),
        'dR': dR,
        'dz': dz,
        'dphi': dphi,
        'sample_for_integration': sample_for_integration,

        'X_regular_grid': X_regular_grid,
        'Y_regular_grid': Y_regular_grid,
        'dX': dX,
        'dY': dY,
        'sample_for_integration_XY': sample,
        'X_minmax': X_minmax,
        'Y_minmax': Y_minmax,
        'nX_nY': nX_nY,

        'w0': w0
    }

    return dict_data