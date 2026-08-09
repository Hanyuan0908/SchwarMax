"""
MGE-deprojection model build for SchwarMAX.  build_model is NOT modified: this
module constructs the MGE density_func + 1-D-quadrature potential_func and hands
them to the existing build_model.

Pieces:
  deproject_mge(sig2d,q2d, incl, phi, psi_bar, R0)  -> intrinsic 3D Gaussians (M,sig,p,q)
      disk (sig'>R0): axisymmetric, Jin Eq.3   ;  bar (sig'<=R0): triaxial, Jin Eq.5-7
  mge_density(x,y,z, G3)      -> sum of aligned 3D Gaussians  [Msun/kpc^3]
  mge_potential(x,y,z, G3)    -> 1-D Gauss-Legendre quadrature triaxial potential (JAX)
  build_model_mge(...)        -> deproject -> closures -> call build_model
"""
import numpy as np, jax, jax.numpy as jnp
from constants import G as Gc   # schwarmax G (kpc^2/Gyr^2 units) -- MUST match build_model
_u,_w = np.polynomial.legendre.leggauss(64); U=jnp.asarray(0.5*(_u+1)); W=jnp.asarray(0.5*_w)

# ---------------- deprojection (numpy; runs once per parameter set) ----------------
def deproject_mge(L2d, sig2d, q2d, incl_deg, phi_deg, psibar_deg, R0, ML=1.0):
    """2D Gaussians (total lum L2d, major-axis sigma sig2d, projected axis ratio q2d)
    -> intrinsic 3D Gaussians. Returns arrays (M, sig, p, q). Bar (sig<=R0) triaxial."""
    th=np.radians(incl_deg); ph=np.radians(phi_deg); ps=np.radians(psibar_deg)
    ct,st=np.cos(th),np.sin(th)
    M=[]; S=[]; P=[]; Q=[]
    for L,sg,qp in zip(L2d,sig2d,q2d):
        if sg>R0:  # disk: axisymmetric oblate (Jin Eq.3), sigma unchanged
            q2=(qp**2-ct**2)/max(st**2,1e-6); q=np.sqrt(np.clip(q2,1e-4,1.0)); p=0.999; sig=sg
        else:      # bar: triaxial (Jin Eq.5-7)
            dp=1-qp**2; cp,sp=np.cos(ph),np.sin(ph); cs,ss=np.cos(ps),np.sin(ps)
            sec=1/ct; cot=cp/sp
            den=2*st**2*(dp*cs*(cs+cot*sec*ss)-1)
            oneq=dp*(2*np.cos(2*ps)+np.sin(2*ps)*(sec*cot-ct*np.tan(ph)))/den
            pmq =dp*(2*np.cos(2*ps)+np.sin(2*ps)*(ct*cot-sec*np.tan(ph)))/den
            q2=np.clip(1-oneq,1e-4,1.0); p2=np.clip(q2+pmq,q2,1.0)
            q=np.sqrt(q2); p=np.sqrt(p2)
            u2=(1/qp**2)*np.sqrt(p2*ct**2+q2*st**2*(p2*cp**2+sp**2)); u=np.sqrt(max(u2,1e-6))
            sig=sg/u
        # 3D mass of the gaussian: M = ML * L (light->mass); density norm handled in mge_density
        M.append(ML*L); S.append(sig); P.append(p); Q.append(q)
    return np.array(M),np.array(S),np.array(P),np.array(Q)

# ---------------- density + potential (JAX, aligned triaxial Gaussians) ----------------
@jax.jit
def mge_density(x,y,z, M,S,P,Q):
    x=jnp.asarray(x); sh=x.shape
    xf=x.ravel(); yf=jnp.asarray(y).ravel(); zf=jnp.asarray(z).ravel()   # (N,)
    rho0=M/((2*jnp.pi)**1.5*S**3*P*Q)                                     # (ng,)
    val=rho0[None,:]*jnp.exp(-0.5*(xf[:,None]**2+yf[:,None]**2/P[None,:]**2
                                   +zf[:,None]**2/Q[None,:]**2)/S[None,:]**2)   # (N,ng)
    return jnp.sum(val,axis=1).reshape(sh)

@jax.jit
def mge_potential(x,y,z, M,S,P,Q):
    x=jnp.asarray(x,float); sh=x.shape
    xf=x.ravel(); yf=jnp.asarray(y,float).ravel(); zf=jnp.asarray(z,float).ravel()  # (N,)
    eb2=1-P**2; ec2=1-Q**2                                               # (ng,)
    u=U[None,:,None]                                                     # (1,nu,1)
    denb=1-eb2[None,None,:]*u**2; denc=1-ec2[None,None,:]*u**2           # (1,nu,ng)
    arg=(u**2)/(2*S[None,None,:]**2)*(xf[:,None,None]**2
         +yf[:,None,None]**2/denb+zf[:,None,None]**2/denc)               # (N,nu,ng)
    integ=jnp.exp(-arg)/jnp.sqrt(denb*denc)                             # (N,nu,ng)
    quad=jnp.sum(W[None,:,None]*integ,axis=1)                           # (N,ng) 1-D integral
    pot=-jnp.sqrt(2/jnp.pi)*Gc*jnp.sum((M/S)[None,:]*quad,axis=1)       # (N,)
    return pot.reshape(sh)

# ---------------- EXACT-MGE potential on a harmonic (R,z,m) grid ----------------
# Drop-in replacement for CylindricalSpline.get_phi_m: same dphi format (so evaluate_phi/get_acc
# are reused unchanged), but the grid nodes are filled by the EXACT 1-D quadrature mge_potential
# instead of the Poisson solve.  ~5x faster to build and exact (fixes CylSpline's inner-region error).
def get_phi_m_exact(M, S, P, Q, NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax, Nphi):
    from CylindricalSpline import _Rz_rescale_forward, jax_precompute_splines
    Marr = jnp.arange(0, Mmax + 1)
    R = jnp.geomspace(jnp.maximum(Rmin, 1e-3), Rmax, NR)
    Zpos = jnp.geomspace(jnp.maximum(Zmin, 1e-3), Zmax, NZ)
    Z_nonneg = jnp.concatenate([jnp.array([0.0]), Zpos])
    R0 = R[NR // 2]
    phi = jnp.linspace(0.0, 2 * jnp.pi, Nphi, endpoint=False)
    _R, _z = _Rz_rescale_forward(R, Z_nonneg, R0)
    Rg, Zg = jnp.meshgrid(R, Z_nonneg, indexing="ij")            # (NR, NZ+1)

    # exact potential on the FULL (Nphi, NR, NZ+1) grid in ONE vectorized call -- mge_potential handles any
    # point shape. Replaces jax.lax.map over phi, which is *sequential* on GPU (32 serial kernel launches).
    PH = phi[:, None, None]                                      # (Nphi,1,1)
    Xg = Rg[None] * jnp.cos(PH); Yg = Rg[None] * jnp.sin(PH); Zt = jnp.broadcast_to(Zg[None], Xg.shape)
    Phi_grid = mge_potential(Xg, Yg, Zt, M, S, P, Q)            # (Nphi, NR, NZ+1)

    # azimuthal harmonics: phi_m[m] = (1/Nphi) sum_phi Phi * exp(-i m phi)  (matches compute_rho_m)
    exp_mphi = jnp.exp(-1j * Marr[:, None] * phi[None, :])       # (NM, Nphi)
    phi_m = jnp.einsum('kij,mk->mij', Phi_grid.astype(exp_mphi.dtype), exp_mphi) / Nphi

    # spline coefficients for ALL harmonics at once (vmap over m); was a sequential python loop before.
    pr = phi_m.real; pi = phi_m.imag                            # (NM, NR, NZ+1)
    _spl = jax.vmap(lambda g: jax_precompute_splines((_R, _z), g))
    Mxr, Myr = _spl(pr); Mxi, Myi = _spl(pi)
    return {'Rgrid': _R, 'Zgrid': _z, 'R0': jnp.array([R0]), 'm': Marr,
            'Phi_m_real': pr, 'Phi_m_img': pi, 'Mx_real': Mxr, 'My_real': Myr, 'Mx_img': Mxi, 'My_img': Myi}

# ---------------- wrapper: build the MGE model via the UNCHANGED build_model ----------------
def build_model_mge(build_model, NFW_potential, mge2d, params, dict_data, **bm_kwargs):
    """mge2d = (L2d, sig2d, q2d, R0).  params dict: incl,phi,psi_bar, ML, and halo (logM,Rs).
    Returns build_model(...) output with density=MGE, potential=MGE(1-D quad)+NFW."""
    L2d,sig2d,q2d,R0 = mge2d
    M,S,P,Q = deproject_mge(L2d,sig2d,q2d, params['incl'],params['phi'],params['psi_bar'], R0, ML=params['ML'])
    M,S,P,Q = jnp.array(M),jnp.array(S),jnp.array(P),jnp.array(Q)
    @jax.jit
    def density_func(x,y,z,p): return mge_density(x,y,z,M,S,P,Q)
    @jax.jit
    def potential_func(x,y,z,pb,ph): return mge_potential(x,y,z,M,S,P,Q)+NFW_potential(x,y,z,ph)
    return density_func, potential_func, (M,S,P,Q)


# ---------------------------------------------------------------------------------
# Factory: MGE parameters + M/L (+ halo)  ->  (density_func, potential_func, acceleration_func)
#
# These are exactly the three callables build_model_acc() / model_deltaChi2_jackknife_acc()
# / get_model_with_orbit_acc() expect, so NOTHING in those functions has to change to run
# an MGE ("fixed potential") model.
#
# The expensive part -- get_phi_m_exact, the harmonic expansion of the exact 1-D-quadrature
# MGE potential -- is done ONCE here and closed over.  The returned callables then only do
# cheap spline evaluations (evaluate_phi / get_acc), which is why the acceleration is handed
# in analytically rather than autodiffed.
# ---------------------------------------------------------------------------------
def make_mge_funcs(sigma_x, p, q, mass,
                   halo_potential=None, halo_acceleration=None,
                   NR=50, NZ=30, Rmin=1e-3, Rmax=50.0, Zmin=1e-3, Zmax=20.0,
                   Mmax=2, Nphi=32):
    """Build the three model callables for a deprojected 3-D MGE plus a halo.

    Parameters
    ----------
    sigma_x, p, q : (ng,) intrinsic major-axis sigma [kpc] and axis ratios
        sigma_y/sigma_x, sigma_z/sigma_x.  Get these from
        deprojection.deproject_components(inc, gamma, ...).
    mass : (ng,) MASS per Gaussian [Msun], i.e. luminosity * (M/L) -- M/L must be baked in
        HERE, because it is the potential that has to scale.  (The 2-D density data is
        separately converted with light_to_mass_ratio = 1/(M/L).)
    halo_potential, halo_acceleration : callables (x, y, z, params_halo).  Typically
        potentials.NFW_potential and potentials.NFW_acceleration.

    Returns
    -------
    density_func(x, y, z, params_baryon)                -> nu   [Msun/kpc^3]
    potential_func(x, y, z, params_baryon, params_halo) -> Phi  [kpc^2/Gyr^2]
    acceleration_func(x, y, z, params_baryon, params_halo) -> (3,) acceleration

    Notes
    -----
    * density_func is written scalar-in/scalar-out (summing over Gaussians) so it vmaps
      cleanly inside the Jeans solver.
    * Mmax=2 keeps the bar (m = 0, 1, 2).  Mmax=0 would make the model axisymmetric.
    * Build this ONCE per parameter set, outside the MCMC step: the callables are static
      arguments of build_model_acc, so a fresh closure per sample forces a recompile.
    """
    from CylindricalSpline import evaluate_phi, get_acc

    sigma_x = jnp.asarray(sigma_x); p = jnp.asarray(p)
    q = jnp.asarray(q);             mass = jnp.asarray(mass)

    # exact 1-D-quadrature MGE potential, expanded on the harmonic grid -- ONCE
    dphi = get_phi_m_exact(mass, sigma_x, p, q, NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax, Nphi)

    # central density of each Gaussian: M / ((2 pi)^{3/2} sx sy sz)
    rho0 = mass / ((2 * jnp.pi) ** 1.5 * sigma_x * (p * sigma_x) * (q * sigma_x))

    @jax.jit
    def density_func(x, y, z, params_baryon):
        r2 = (x**2 + y**2 / p**2 + z**2 / q**2) / sigma_x**2
        return jnp.sum(rho0 * jnp.exp(-0.5 * r2))

    @jax.jit
    def potential_func(x, y, z, params_baryon, params_halo):
        return evaluate_phi(x, y, z, dphi) + halo_potential(x, y, z, params_halo)

    @jax.jit
    def acceleration_func(x, y, z, params_baryon, params_halo):
        return get_acc(x, y, z, dphi) + halo_acceleration(x, y, z, params_halo)

    return density_func, potential_func, acceleration_func
