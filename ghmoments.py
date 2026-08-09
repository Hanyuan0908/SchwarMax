import jax
import jax.numpy as jnp

# y = (x - m) / s   is the scaled variable

def gauss(x, v, s):
    y = (x-v) / s
    gaussian = (2*jnp.pi)**-0.5 / s * jnp.exp(-0.5 * y**2)
    return y, gaussian

def H0(y):
    return 1.0 + y*0

def H1(y):
    return 2**0.5 * y

def H2(y):
    m=2
    hp = H1(y)
    hpp = H0(y)
    h2 = (2**0.5 * y * hp - (m-1)**0.5 * hpp) / m**0.5
    return h2

def H3(y):
    m=3
    hp = H2(y)
    hpp = H1(y)
    h3 = (2**0.5 * y * hp - (m-1)**0.5 * hpp) / m**0.5
    return h3

def H4(y):
    m=4
    hp = H3(y)
    hpp = H2(y)
    h4 = (2**0.5 * y * hp - (m-1)**0.5 * hpp) / m**0.5
    return h4

def H5(y):
    m=5
    hp = H4(y)
    hpp = H3(y)
    h5 = (2**0.5 * y * hp - (m-1)**0.5 * hpp) / m**0.5
    return h5

def H6(y):
    m=6
    hp = H5(y)
    hpp = H4(y)
    h6 = (2**0.5 * y * hp - (m-1)**0.5 * hpp) / m**0.5
    return h6


@jax.jit
def h_to_V_sigma(h1, h2, v0, s, h3=0.0, h4=0.0):
    """
    Convert GH coefficients (h1..h4) to the actual mean and sigma of the
    LOSVD, using the van der Marel & Franx (1993) orthonormal Hermite basis
    where H_n(y) = He_n(sqrt(2) y) / sqrt(n!).

    To leading order in h_k (keeps both linear-in-h and the unavoidable
    quadratic cross terms from the second central moment):

        V_mean   = v0 + s * (sqrt(2)*h1 + sqrt(3)*h3)
        sigma^2  = s^2 * [1 + 2*sqrt(2)*h2 + 2*sqrt(6)*h4
                          - (sqrt(2)*h1 + sqrt(3)*h3)^2]

    h3 and h4 default to 0, so legacy callers that only pass (h1, h2, v0, s)
    still get the old (h3 = h4 = 0) form.  For the diagnostic V/sigma maps
    pass the model's fitted h3, h4 to avoid systematic offsets when the
    LOSVD is non-Gaussian.
    """
    sqrt2 = jnp.sqrt(2.0)
    sqrt3 = jnp.sqrt(3.0)
    sqrt6 = jnp.sqrt(6.0)
    delta_v = sqrt2 * h1 + sqrt3 * h3
    v_mean  = v0 + s * delta_v
    sigma2_over_s2 = 1.0 + 2.0*sqrt2*h2 + 2.0*sqrt6*h4 - delta_v**2
    sigma = s * jnp.sqrt(jnp.clip(sigma2_over_s2, 1e-10))
    return v_mean, sigma

@jax.jit
def h_to_V_sigma_GH(h1, h2, v0, s):
    """
    pPXF / paper convention: the Gauss-Hermite REFERENCE-GAUSSIAN parameters
    (V_GH, sigma_GH) -- the centre and width of the Gaussian for which the model
    LOSVD would have h1 = h2 = 0.

    THIS is what pPXF (and hence TIMER / PHANGS-MUSE) report as V and sigma.  They
    are NOT the classical mean and dispersion, which is what h_to_V_sigma() above
    returns.  Use this one whenever the data's (v0, s) came from a pPXF Gauss-Hermite
    fit, i.e. whenever the data satisfies h1 = h2 = 0 by construction.

    To FIRST ORDER in (h1, h2):

        V_GH     = v0 + sqrt(2) * s * h1
        sigma_GH = s * (1 + sqrt(2) * h2)

    Derivation: Taylor-expand a Gaussian shifted by (dV, ds) about (v0, s) to first
    order and project onto the orthonormal Hermite basis H1 = sqrt2 w,
    H2 = (2 w^2 - 1)/sqrt2, giving h1 = (dV/s)/sqrt2, h2 = (ds/s)/sqrt2.
    (van der Marel & Franx 1993 sec. 2; Rix & White 1992; Bender, Saglia & Gerhard
    1994; Cappellari 2017.)

    Difference from h_to_V_sigma(): that adds a sqrt(3)*s*h3 skewness term (and a
    sqrt(6) h4 term in sigma) which the data's v0, s do NOT contain.  Using it against
    pPXF v0/s therefore introduces a systematic offset wherever the LOSVD is skewed.
    With h_to_V_sigma_GH the V/sigma residual maps reduce exactly to the h1/h2
    residuals, which is the desired behaviour.

    ACCURACY CAVEAT (verified numerically): the neglected terms are not pure
    O(h1^2, h2^2) truncation -- for a realistic LOSVD they are dominated by the
    second-order cross terms V ~ s*h1*h3 and sigma ~ s*h2*h4.  With typical
    h3, h4 ~ 0.05-0.1 this biases V_GH by ~0.5-1 km/s (growing with h3) and sigma_GH
    similarly (growing with h4) -- small against a ~7 km/s error bar.  For an exact
    match, solve h1(V, sigma) = h2(V, sigma) = 0 by root-finding instead, which folds
    in the h3, h4 couplings automatically.
    """
    sqrt2 = jnp.sqrt(2.0)
    V_GH = v0 + sqrt2 * s * h1
    sigma_GH = s * (1.0 + sqrt2 * h2)
    return V_GH, sigma_GH
