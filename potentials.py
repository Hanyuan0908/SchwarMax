import jax
import jax.numpy as jnp
from utils import get_mat
from constants import G, EPSILON
from dehnen_bar import T0_potential, T0_density, T1_potential, T1_density, T2_potential, T2_density, T3_potential, T3_density, T4_potential, T4_density, V0_potential, V0_density, V1_potential, V1_density, V2_potential, V2_density, V3_potential, V3_density, V4_potential, V4_density

# ---------- helpers ----------
@jax.jit
def _shift(x, y, z, p):
    return jnp.array([x - p['x_origin'], y - p['y_origin'], z - p['z_origin']])

@jax.jit
def _rotate(vec, p):
    R = get_mat(p['dirx'], p['diry'], p['dirz'])
    return R @ vec  # matvec


@jax.jit
def NFW_potential(x, y, z, params):
    '''
    params: dict with keys 'logM', 'Rs', 'a', 'b', 'c', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    rin = _shift(x, y, z, params)
    rvec = _rotate(rin, params)  
    rx, ry, rz = rvec
    r = jnp.sqrt((rx/params['a'])**2 + (ry/params['b'])**2 + (rz/params['c'])**2 + EPSILON)
    return -G * 10**params['logM'] * jnp.log(1 + r / params['Rs']) / (r + EPSILON)  # kpc^2 / Gyr^2

@jax.jit
def MiyamotoNagai_density(x, y, z, params):
    '''
    params: dict with keys 'logM', 'Rs', 'Hs', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    # Shift and rotate coordinates
    rin = _shift(x, y, z, params)       # (3, ...)
    rvec = _rotate(rin, params)         # (3, ...)
    rx, ry, rz = rvec + EPSILON

    # Cylindrical R in rotated frame
    R = jnp.sqrt(rx**2 + ry**2)

    # Vertical scale height uses rz (IMPORTANT FIX)
    beta = jnp.sqrt(rz**2 + params["Hs_disc"]**2)

    D2 = R*R + (params["Rs_disc"] + beta)**2
    num = params["Rs_disc"] * R*R + (params["Rs_disc"] + 3.0*beta) * (params["Rs_disc"] + beta)**2
    den = beta**3 * D2**2.5

    return (params["Hs_disc"]**2 * 10.0**params["logM_disc"] / (4 * jnp.pi)) * (num / den)

@jax.jit
def MiyamotoNagai_potential(x, y, z, params):
    '''
    params: dict with keys 'logM_disc', 'Rs_disc', 'Hs_disc', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    rin  = _shift(x, y, z, params)
    rvec = _rotate(rin, params)  
    rx, ry, rz = rvec + EPSILON

    R = (rx**2 + ry**2)**0.5

    denom2 = (R**2 + (params['Rs_disc'] + (rz**2 + params['Hs_disc']**2)**0.5)**2)

    Phi = - G * 10**params['logM_disc'] / (denom2**0.5)

    return Phi