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