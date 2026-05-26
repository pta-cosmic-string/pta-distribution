import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from .const import *
from .funcs import *

class SkyMap:
    def __init__(self, nside = 8):
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)

        self._init_angles()
        self._init_vectors()

    def _init_angles(self):        
        self.theta, self.phi = hp.pix2ang(self.nside, np.arange(self.npix))
        self.dOmega = hp.nside2pixarea(self.nside)
    
    def _init_vectors(self):
        self.vectors = np.array(hp.pix2vec(self.nside, np.arange(self.npix)))

        self.Omega = np.array([-np.sin(self.theta)*np.cos(self.phi), 
                               -np.sin(self.theta)*np.sin(self.phi), 
                               -np.cos(self.theta)])
        
        self.m = np.array([+np.sin(self.phi), 
                           -np.cos(self.phi),
                           +np.zeros(self.phi.shape)])
        self.n = np.array([-np.cos(self.theta)*np.cos(self.phi), 
                           -np.cos(self.theta)*np.sin(self.phi), 
                           +np.sin(self.theta)])
                
        self.e_p = np.einsum('ik,jk->ijk', self.m, self.m) - np.einsum('ik,jk->ijk', self.n, self.n) 
        self.e_c = np.einsum('ik,jk->ijk', self.m, self.n) + np.einsum('ik,jk->ijk', self.n, self.m) 
        
        self.e = self.e_p + 1j * self.e_c

    def get_data(self, func, **kwargs):
        return func(self.theta, self.phi, **kwargs)

    def plot(self, data=None, title="HEALPix сетка", unit="Amplitude", cmap="plasma", show=True, save=True):
        if data is None:
            data = np.arange(self.npix)
        hp.mollview(data, title=title, unit=unit, cmap=cmap)
        hp.graticule()

        if not os.path.exists('data'):
            os.mkdir('data')
        if save:
            plt.savefig(f'data/map.png')
        if show:
            plt.show()

class TimeLine:
    def __init__(self, n_times=10, T=1):
        self.n_times = n_times
        self.T = T
        self._init_times()

    def _init_times(self):
        self.dt = self.T / (self.n_times - 1)
        self.df = 1.0 / self.T
        self.t = np.linspace(0, self.T, self.n_times)
        self.f = np.fft.fftfreq(self.n_times, d=self.dt)

    def plot(self, data=None, key='t', title=None, show=True, save=True, log=True):
        if data is None:
            data = np.ones((self.n_times))
        name = None

        if key == 't':
            name = 'timeline'
            plt.plot(self.t, data)
            plt.xlabel("time")
            plt.ylabel("data")
            if title is None:
                plt.title(name)
            else:
                plt.title(title)
           

        elif key=='f':
            name = 'spectrum'
            if log:
                plt.loglog(self.f[self.f>=0], data[self.f>=0])
            else:
                plt.plot(self.f[self.f>=0], data[self.f>=0])
            plt.xlabel("frequency")
            plt.ylabel("data")
            if title is None:
                plt.title(name)
            else:
                plt.title(title)
            
        
        if not os.path.exists('data'):
            os.mkdir('data')
        if save:
            plt.savefig(f'data/{name}.png')
        if show:
            plt.show()

class PulsarArray:
    def __init__(self, n_array, radius=1, seed_psr=42):
        self.n_array = n_array
        self.radius = radius
        self.rng_psr = np.random.default_rng(seed=seed_psr)
        self._init_array()

    def _init_array(self):
        phi = self.rng_psr.uniform(0, 2*PI, self.n_array)
        cos_theta = self.rng_psr.uniform(-1, 1, self.n_array)
        theta = np.arccos(cos_theta)

        rho = self.radius * self.rng_psr.uniform(0, 1, self.n_array) ** (1/3)

        x = rho * np.sin(theta) * np.cos(phi)
        y = rho * np.sin(theta) * np.sin(phi)
        z = rho * np.cos(theta)

        self.pulsar_arr = np.array([x, y, z]).T
        self.pulsar_dist = rho.T
        self.pulsar_vec = self.pulsar_arr / np.tile(self.pulsar_dist[:,None], (1, 3))
        self.n_pulsars = self.pulsar_vec.shape[0]

    def get_pixs(self, skymap):
        vec = self.pulsar_vec
        pix_idx = hp.vec2pix(skymap.nside, vec[:,0], vec[:,1], vec[:,2], nest=False)
        data = np.zeros((skymap.npix))
        data[pix_idx] = 1.0
        return data
    
    def plot(self, show=True, save=True):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.pulsar_arr.T[0], self.pulsar_arr.T[1], self.pulsar_arr.T[2], c='blue', s=20, label='PTA')
        ax.scatter(0, 0, 0, c='red', s=100, label='Observer')

        ax.set_box_aspect([1, 1, 1])

        ax.set_xlabel("x, kpc")
        ax.set_ylabel("y, kpc")
        ax.set_zlabel("z, kpc")
        ax.set_title("Distribution of Pulsar Array")
        ax.legend()
        
        if not os.path.exists('data'):
            os.mkdir('data')
        if save:
            plt.savefig('data/PTA.png')
        if show:
            plt.show()

class GravitationalWave:
    def __init__(self, spectrum=delta_1d, distribution=isotropic):
        self.spectrum = spectrum
        self.distribution = distribution
    def generate_wave(self, timeline, skymap, spec_args=None, distr_args=None):
        if spec_args is None:
            H = self.spectrum(timeline)
        else:
            H =  self.spectrum(timeline, spec_args)

        if distr_args is None:
            G = self.distribution(skymap)
        else: 
            G = self.distribution(skymap, distr_args)

        return H, G

class Telescope:
    def __init__(self, skymap, timeline, pulsar_arr, seed_bg=43, norm=1):
        self.skymap = skymap
        self.timeline = timeline
        self.pulsar_arr = pulsar_arr
        self.rng_bg = np.random.default_rng(seed=seed_bg)
        self.norm = 3*norm

    def observe(self, grav_wave):
        p = self.pulsar_arr.pulsar_vec
        lp = self.pulsar_arr.pulsar_dist
        Np = self.pulsar_arr.n_pulsars
        Nt = self.timeline.n_times
        Ng = self.skymap.npix
        H, G = grav_wave.generate_wave(self.timeline, self.skymap)
        eta = (self.rng_bg.normal(size=Ng) + 1j * self.rng_bg.normal(size=Ng)) / np.sqrt(2 * 4 * PI)

        self.skymap.plot(G * np.abs(eta)**2 , title='Gravitational Wave Background')

        def z_calc(p, l):
            f = np.tile(self.timeline.f[:, None], (1, Ng))
            H_f = np.tile(H[:, None], (1, Ng))
            
            beta = (1 + np.einsum('li,l->i', self.skymap.Omega, p))
            
            F = 1/2 * np.einsum('lm,lmi->i',np.einsum('l,m->lm', p, p), self.skymap.e) / beta
            
            Phi = F * np.sqrt(G) * eta / np.sqrt(self.skymap.dOmega)

            beta = np.tile(beta[None,:], (Nt, 1))
            
            alpha = l * f * (KPC * F_YR)

            T =  1 - np.exp(-1j  * 2 * PI * alpha * beta)

            B = T * np.sqrt(H_f)

            Phi = np.tile(Phi[None, :], (Nt, 1))

            z = np.real(np.fft.ifft(Phi * B, axis=0) * self.timeline.df * Nt)
            Z = np.sum(z * self.skymap.dOmega, axis=1)

            return Z
        
        Z = np.zeros((Np, Nt))

        for k in tqdm(range(Np)):
            p_k = p[k]
            lp_k = lp[k]
            Z[k] = z_calc(p_k, lp_k)
        
        h2 = np.sum(H * self.timeline.df)
        g2 = np.sum(G * np.abs(eta)**2  * self.skymap.dOmega)
        
        i, j = np.arange(0, Np),  np.arange(0, Np)
        i, j = np.meshgrid(i, j, indexing='xy')
        mask = i>j
        i, j = i[mask], j[mask]

        z1, z2 = Z[i], Z[j]
        mu = np.zeros(i.shape)

        def mu_calc(z1, z2):
            R = 2 * np.sum(z1 * z2 * self.timeline.dt, axis=0)
            mu_k = self.norm * 1/(h2 * g2) * R
            return mu_k
        
        for k in tqdm(range(i.shape[0])):
            mu_k = mu_calc(z1[k], z2[k])
            mu[k] = mu_k

        p1, p2 = p[i], p[j]
        gamma = np.arccos(np.einsum('kl,kl->k',p1,p2))
        gamma_pta, mu_pta = gamma, mu

        self.gamma_pta, self.mu_pta = gamma_pta, mu_pta
        return gamma_pta, mu_pta

    def average(self, gamma, mu, n_average = 21):
        dgamma = PI / n_average
        gamma_mean = np.linspace(dgamma/2, PI - dgamma/2, n_average)

        mu_m = np.zeros(n_average)
        mu_m2 = np.zeros(n_average) 
        N_m = np.zeros(n_average)

        for k in tqdm(range(gamma.shape[0])):
            Gamma_k, gamma_k = mu[k], gamma[k]
            mask = (gamma_mean - dgamma <= gamma_k) & (gamma_k < gamma_mean + dgamma)
            mu_m[mask] += Gamma_k
            mu_m2[mask] += Gamma_k**2
            N_m[mask] += 1

        mu_mean = mu_m / N_m
        mu_std = np.sqrt(mu_m2 / N_m - mu_mean**2) 
        
        self.gamma_mean, self.mu_mean, self.mu_std = gamma_mean, mu_mean, mu_std
        return gamma_mean, mu_mean, mu_std
    
    def theory(self, n_theory = 1000):
        gamma_theory = np.linspace(0 + EPS, PI, n_theory)
        mu_theory = self.norm * mu_0(gamma_theory)
        self.gamma_theory, self.mu_theory = gamma_theory, mu_theory
        return gamma_theory, mu_theory

    def plot(self, key='hd', show=True, save=True):
        if key == 'hd':
            plt.grid(True)
            plt.scatter(self.gamma_pta * RAD, self.mu_pta, c='blue', linewidths=1, label='PTA')
            plt.plot(self.gamma_mean * RAD, self.mu_mean, color='red', label='obs HD')
            plt.plot(self.gamma_mean * RAD, self.mu_mean + self.mu_std, color='red', linestyle='--', label='obs Var[HD]')
            plt.plot(self.gamma_mean * RAD, self.mu_mean - self.mu_std, color='red', linestyle='--')
            plt.plot(self.gamma_theory * RAD, self.mu_theory, color='black', label='theory HD')

            plt.title("HD curve")
            plt.xlabel("$\\gamma$, deg")
            plt.ylabel("$\\Gamma(\\gamma)$")
            plt.legend()
            if not os.path.exists('data'):
                os.mkdir('data')
            if save:
                plt.savefig('data/HD.png')
            if show:
                plt.show()

        elif key == 'var-hd':
            plt.grid(True)
            plt.plot(self.gamma_mean * 180/PI, self.mu_std, color='blue', label='obs Var[HD]')
            plt.title("HD curve")
            plt.xlabel("$\\gamma$, deg")
            plt.ylabel("$\\Gamma(\\gamma)$")
            plt.legend()
            if not os.path.exists('data'):
                os.mkdir('data')
            plt.savefig('data/Var_HD.png')
            if show:
                plt.show()

        else:
            print("There is no such plot function.")

class IdealTelescope:
    def __init__(self, skymap, timeline, pulsar_arr, norm = 1):
        self.skymap = skymap
        self.timeline = timeline
        self.pulsar_arr = pulsar_arr
        self.norm = norm

    def analytics(self, args=(PI/2,0,1)):
        p = self.pulsar_arr.pulsar_vec
        lp = self.pulsar_arr.pulsar_dist
        Np = self.pulsar_arr.n_pulsars
        Ng = self.skymap.npix

        theta0, phi0, kappa = args
        Omega_0 = hp.ang2vec(theta0, phi0)
        P = gaussian(self.skymap, args=args)
        self.skymap.plot(P , title='Gravitational Wave Background')

        i, j = np.arange(0, Np), np.arange(0, Np)
        i, j = np.meshgrid(i, j, indexing='xy')
        mask = i>j
        i, j = i[mask], j[mask]

        p1, p2 = p[i], p[j]
        mu = np.zeros(i.shape)

        def mu_calc(p1, p2):
            return self.norm * K_exp(Omega_0, p1, p2, kappa)
        
        for k in tqdm(range(i.shape[0])):
            mu_k = mu_calc(p1[k], p2[k])
            mu[k] = mu_k

        gamma = np.arccos(np.einsum('kl,kl->k',p1,p2))
        gamma_pta, mu_pta = gamma, mu

        self.gamma_pta_true, self.mu_pta_true = gamma_pta, mu_pta
        return gamma_pta, mu_pta

    def observe(self, grav_wave, args=(PI/2,0,1)):
        p = self.pulsar_arr.pulsar_vec
        lp = self.pulsar_arr.pulsar_dist
        Np = self.pulsar_arr.n_pulsars
        Ng = self.skymap.npix
        theta0, phi0, kappa = args
        Omega_0 = hp.ang2vec(theta0, phi0)
        _, P = grav_wave.generate_wave(self.timeline, self.skymap, distr_args=args)
        self.skymap.plot(P , title='Gravitational Wave Background')

        i, j = np.arange(0, Np), np.arange(0, Np)
        i, j = np.meshgrid(i, j, indexing='xy')
        mask = i>j
        i, j = i[mask], j[mask]

        p1, p2 = p[i], p[j]
        mu = np.zeros(i.shape)

        def mu_calc(p1, p2):
            beta1 = (1 + np.einsum('li,l->i', self.skymap.Omega, p1))
            beta2 = (1 + np.einsum('li,l->i', self.skymap.Omega, p2))
            F1 = 1/2 * np.einsum('lm,lmi->i',np.einsum('l,m->lm', p1, p1), self.skymap.e) / beta1
            F2 = 1/2 * np.einsum('lm,lmi->i',np.einsum('l,m->lm', p2, p2), self.skymap.e) / beta2
            K12 = np.real(F1 * np.conjugate(F2))
            K12 = 1/4 * (2 * (np.einsum('l,l->', p1, p2) - np.einsum('li,l->i', self.skymap.Omega, p1) * np.einsum('li,l->i', self.skymap.Omega, p2))**2 / ((1 + np.einsum('li,l->i', self.skymap.Omega, p1)) * (1 + np.einsum('li,l->i', self.skymap.Omega, p2))) - ((1 - np.einsum('li,l->i', self.skymap.Omega, p1)) * (1 - np.einsum('li,l->i', self.skymap.Omega, p2)))) 
            Gamma_12 = self.norm * np.sum(K12 * P * self.skymap.dOmega) / (4 * PI)
            return Gamma_12
        
        for k in tqdm(range(i.shape[0])):
            mu_k = mu_calc(p1[k], p2[k])
            mu[k] = mu_k

        gamma = np.arccos(np.einsum('kl,kl->k',p1,p2))
        gamma_pta, mu_pta = gamma, mu

        self.gamma_pta_est, self.mu_pta_est = gamma_pta, mu_pta
        return gamma_pta, mu_pta
    
    def average(self, gamma, mu, n_average = 21):
        dgamma = PI / n_average
        gamma_mean = np.linspace(dgamma/2, PI - dgamma/2, n_average)

        mu_m = np.zeros(n_average)
        mu_m2 = np.zeros(n_average) 
        N_m = np.zeros(n_average)

        for k in tqdm(range(gamma.shape[0])):
            Gamma_k, gamma_k = mu[k], gamma[k]
            mask = (gamma_mean - dgamma <= gamma_k) & (gamma_k < gamma_mean + dgamma)
            mu_m[mask] += Gamma_k
            mu_m2[mask] += Gamma_k**2
            N_m[mask] += 1

        mu_mean = mu_m / N_m
        mu_std = np.sqrt(mu_m2 / N_m - mu_mean**2) 
        
        self.gamma_mean, self.mu_mean, self.mu_std = gamma_mean, mu_mean, mu_std
        return gamma_mean, mu_mean, mu_std
    
    def theory(self, n_theory = 1000):
        gamma_theory = np.linspace(0 + EPS, PI, n_theory)
        mu_theory = self.norm * mu_0(gamma_theory)
        self.gamma_theory, self.mu_theory = gamma_theory, mu_theory
        return gamma_theory, mu_theory


    def plot(self, key='hd', show=True, save=True):
        if key == 'hd':
            plt.grid(True)
            plt.scatter(self.gamma_pta_est * RAD, self.mu_pta_est, c='blue', linewidths=1, label='PTA-est')
            plt.plot(self.gamma_mean * RAD, self.mu_mean, color='red', label='obs HD')
            plt.plot(self.gamma_theory * RAD, self.mu_theory, color='black', label='theory HD')

            plt.title("HD curve")
            plt.xlabel("$\\gamma$, deg")
            plt.ylabel("$\\Gamma(\\gamma)$")
            plt.legend()
            if not os.path.exists('data'):
                os.mkdir('data')
            if save:
                plt.savefig('data/HD.png')
            if show:
                plt.show()

        elif key == 'check-hd':
            plt.grid(True)
            plt.scatter(self.gamma_pta_est * RAD, self.mu_pta_est, c='green', linewidths=1, label='PTA-est')
            plt.scatter(self.gamma_pta_true * RAD, self.mu_pta_true, c='blue', linewidths=1, label='PTA-true')
            plt.plot(self.gamma_mean * RAD, self.mu_mean, color='red', label='obs HD')
            plt.plot(self.gamma_theory * RAD, self.mu_theory, color='black', label='theory HD')

            plt.title("HD curve")
            plt.xlabel("$\\gamma$, deg")
            plt.ylabel("$\\Gamma(\\gamma)$")
            plt.legend()
            if not os.path.exists('data'):
                os.mkdir('data')
            if save:
                plt.savefig('data/HD.png')
            if show:
                plt.show()

        elif key == 'theory-hd':
            plt.grid(True)
            plt.scatter(self.gamma_pta_true * RAD, self.mu_pta_true, c='blue', linewidths=1, label='PTA-true')
            plt.plot(self.gamma_mean * RAD, self.mu_mean, color='red', label='obs HD')
            plt.plot(self.gamma_theory * RAD, self.mu_theory, color='black', label='theory HD')

            plt.title("HD curve")
            plt.xlabel("$\\gamma$, deg")
            plt.ylabel("$\\Gamma(\\gamma)$")
            plt.legend()
            if not os.path.exists('data'):
                os.mkdir('data')
            if save:
                plt.savefig('data/HD.png')
            if show:
                plt.show()

        elif key == 'var-hd':
            plt.grid(True)
            plt.plot(self.gamma_mean * 180/PI, self.mu_std, color='blue', label='obs Var[HD]')
            plt.title("HD curve")
            plt.xlabel("$\\gamma$, deg")
            plt.ylabel("$\\Gamma(\\gamma)$")
            plt.legend()
            if not os.path.exists('data'):
                os.mkdir('data')
            plt.savefig('data/Var_HD.png')
            if show:
                plt.show()

        elif key == 'delta-hd':
            plt.grid(True)
            plt.scatter(self.gamma_pta_est * RAD, np.abs((self.mu_pta_est - self.mu_pta_true)), c='orange', linewidths=1, label='delta-PTA')

            plt.title("HD curve")
            plt.xlabel("$\\gamma$, deg")
            plt.ylabel("$\\Gamma(\\gamma)$")
            plt.legend()
            if not os.path.exists('data'):
                os.mkdir('data')
            if save:
                plt.savefig('data/HD.png')
            if show:
                plt.show()

        else:
            print("There is no such plot function.")