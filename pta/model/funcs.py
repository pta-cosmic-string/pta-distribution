import numpy as np
import scipy as sp
import healpy as hp

from .const import *

def mu_0(gamma):
    cos_gamma = np.cos(gamma)
    mu = 3 * (1/3 - 1/6 * (1 - cos_gamma)/2 + (1 - cos_gamma)/2 * np.log((1 - cos_gamma)/2))
    return mu

def K_0(Omega, p1, p2):
    a = np.einsum('l,l->', Omega, p1)
    b = np.einsum('l,l->', Omega, p2)
    c = np.einsum('l,l->', p1, p2)

    return 3 * (
        1/3 - 1/6 * (1 - c)/2 
        + (1 - c)/2 * np.log((1 - c)/2)
    ) 

def K_12(Omega, p1, p2):
    a = np.einsum('l,l->', Omega, p1)
    b = np.einsum('l,l->', Omega, p2)
    c = np.einsum('l,l->', p1, p2)

    return 3/4 * (
        2*(c - a * b)**2/((1 + a)*(1 + b)) 
        - (1 - a)*(1 - b)
    ) 

def _scaled_expi(w, switch=100.0, max_terms=50):
    """
    Вычисляет exp(-w) * Ei(w) устойчиво.
    Для |w| < switch — напрямую.
    Для |w| >= switch — через асимптотический ряд.
    """
    w = np.asarray(w, dtype=np.complex128)

    if w.ndim == 0:
        if abs(w) < switch:
            return - np.exp(-w) * sp.special.exp1(-w)
        term = 1.0 / w
        s = term
        for n in range(1, max_terms):
            term *= n / w
            s_new = s + term
            if abs(term) <= np.finfo(float).eps * abs(s_new):
                return s_new
            s = s_new
        return s

    out = np.empty_like(w)
    small = np.abs(w) < switch
    out[small] = np.exp(-w[small]) * expi(w[small])

    big = ~small
    if np.any(big):
        wb = w[big]
        term = 1.0 / wb
        s = term.copy()
        for n in range(1, max_terms):
            term *= n / wb
            s_new = s + term
            if np.all(np.abs(term) <= np.finfo(float).eps * np.abs(s_new)):
                s = s_new
                break
            s = s_new
        out[big] = s

    return out

def expi_stable(x, s, a, k):
    z = x + 1j * s
    w = k * (z - a)
    norm = 2 * k * np.exp(-k * (a + 1.0))/(-np.expm1(-2.0 * k))
    return norm * np.real(_scaled_expi(w))

def K_exp(Omega, p1, p2, kappa):
    a = np.einsum('l,l->', Omega, p1)
    b = np.einsum('l,l->', Omega, p2)
    c = np.einsum('l,l->', p1, p2)
    V = np.sqrt(1 + 2*a*b*c - c**2 - a**2 - b**2)
    k = kappa
    t = (a + b)/(1 + c)
    s = V/(1 + c) 

    L_g = (
        + expi_stable(t, s, a, k)
        + expi_stable(t, s, b, k)
        - expi_stable(t, s, -1, k)
        - expi_stable(t, s, +1, k)
    )
        
    D_g = (
        + (b - c*a)/(1 - a**2) * (1/np.tanh(k) - a * (np.exp(-k*(1+a))/a * 2/(1 - np.exp(-2*k)) + 1))
        + (a - c*b)/(1 - b**2) * (1/np.tanh(k) - b * (np.exp(-k*(1+b))/b * 2/(1 - np.exp(-2*k)) + 1))
        + (c - 3*a*b)/6 * (1/np.tanh(k)*3/k - 3/k**2 - 1)
        - (a+b)/2 * (1/np.tanh(k) - 1/k)
    )

    return 3 * (
        + 1/3 - 1/6 * (1 - c)/2
        + (1 - c)/2 * L_g
        + 1/2 * D_g
    )


# Distributions
def isotropic(skymap, args=None):
    return np.ones(skymap.phi.shape)

def stochastic(skymap, args=43):
    seed_bg = args
    rng_bg = np.random.default_rng(seed=seed_bg)
    eta = (rng_bg.normal(size=skymap.npix) + 1j * rng_bg.normal(size=skymap.npix)) / np.sqrt(2)
    return np.abs(eta)**2

def delta_2d(skymap, args=(PI/2,0)):
    theta0, phi0 = args
    ipix = hp.ang2pix(skymap.nside, theta0, phi0)
    delta = np.zeros(skymap.npix, dtype=float)
    delta[ipix] = 4*PI / skymap.dOmega
    return delta

def gaussian(skymap, args=(PI/2,0,1)):
    theta0, phi0, kappa = args
    Omega_0 = hp.ang2vec(theta0, phi0)
    delta_Omega = np.einsum('li,l->i', skymap.Omega, Omega_0)
    gauss  = 2*kappa/(1 - np.exp(-2*kappa)) * np.exp(kappa * (delta_Omega-1))
    return gauss

def string_2d(skymap, args=(PI/2, 360)):
    theta0, lenght = args
    phi = np.linspace(-lenght/2, +lenght/2 , 1000) * DEG
    theta = np.full_like(phi, PI/2)
    ipixs = hp.ang2pix(skymap.nside, theta, phi)
    delta = np.zeros(skymap.npix, dtype=float)
    delta[ipixs] = 4*PI / skymap.dOmega / len(ipixs)
    return delta

def dipole(skymap, args=None):
    cond = skymap.phi <= PI
    ipix = hp.ang2pix(skymap.nside, skymap.theta[cond], skymap.phi[cond])
    dipole = np.zeros(skymap.npix)
    dipole[ipix] = 4*PI / skymap.dOmega / len(ipix)
    return dipole

def quadrupole(skymap, args=None):
    cond = np.logical_and(skymap.phi <= PI, skymap.theta <= PI/2)
    ipix = hp.ang2pix(skymap.nside, skymap.theta[cond], skymap.phi[cond])
    dipole = np.zeros(skymap.npix)
    dipole[ipix] = 4*PI / skymap.dOmega / len(ipix)
    return dipole

def point(skymap, args=(PI/2,0,100)):
    theta0, phi0, radius_deg = args
    vec_center = hp.ang2vec(theta0, phi0) 
    ipix_disc = hp.query_disc(skymap.nside, vec=vec_center, radius=np.radians(radius_deg))
    point = np.zeros(skymap.npix, dtype=float)
    if len(ipix_disc) == 0:
        ipix = hp.ang2pix(skymap.nside, theta0, phi0, nest=False)
        point[ipix] = 4*PI / skymap.dOmega
    else:
        point[ipix_disc] = 4*PI / skymap.dOmega / len(ipix_disc)
    
    return point

# Spectra
def delta_1d(timeline, args=(10)):
    f0 = args
    idx = np.argmin(np.abs(timeline.f - f0))
    delta = np.zeros_like(timeline.f)
    delta[idx] = 1.0/timeline.df
    return delta

def power(timeline, args=(-5)):
    alpha = args
    d = np.abs(timeline.f + EPS)**(alpha)
    d[timeline.f==0] = 0
    norm = 1 / np.sum(d * timeline.df)
    return norm * d
