import healpy as hp
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

PI = np.pi
year = 3.1536e7 
f_yr = 1 / year
pc = 3.26 * year
kpc = 1e3 * pc
EPS = 1e-5


def mu_0(gamma):
    cos_gamma = np.cos(gamma)
    mu = 1/3 - 1/6 * (1 - cos_gamma)/2 + (1 - cos_gamma)/2 * np.log((1 - cos_gamma)/2)
    return mu


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

    def draw(self, data=None, title="HEALPix сетка", unit="Amplitude", cmap="plasma"):
        if data is None:
            data = np.arange(self.npix)

        hp.mollview(data, title=title, unit=unit, cmap=cmap)
        hp.graticule()
        plt.show()

class Spectrum:
    def __init__(self, n_times=10, T=1):
        self.n_times = n_times
        self.T = 1
        self._init_times()

    def _init_times(self):
        self.dt = self.T / (self.n_times - 1)
        self.df = 1.0 / self.T
        self.t = np.linspace(0, self.T, self.n_times)
        self.f = np.fft.fftfreq(self.n_times, d=self.dt)

    def draw_f(self, data=None):
        if data is None:
            data = np.ones((self.n_times))
        plt.plot(self.f[self.f>=0], data[self.f>=0])
        plt.xlabel("frequency")
        plt.ylabel("data")
        plt.show()

    def draw_t(self, data=None):
        if data is None:
            data = np.arange(self.n_times)
        plt.plot(self.t, data)
        plt.xlabel("time")
        plt.ylabel("data")
        plt.show()

class PulsarArray:
    def __init__(self, n_array):
        self.n_array = n_array
        self._init_array()

    def _init_array(self):
        x = (2 * np.random.random((self.n_array,)) - 1)
        y = (2 * np.random.random((self.n_array,)) - 1)
        z = (2 * np.random.random((self.n_array,)) - 1)
        rho = np.sqrt(x**2 + y**2 + z**2)
        mask = (rho <= 1) 

        self.pulsar_arr = np.array([x[mask], y[mask], z[mask]]).T
        self.pulsar_dist = rho[mask].T
        self.pulsar_vec = self.pulsar_arr / np.tile(self.pulsar_dist[:,None], (1, 3))
        self.n_pulsars = self.pulsar_vec.shape[0]

    def pixs(self, skymap):
        vec = self.pulsar_vec
        pix_idx = hp.vec2pix(skymap.nside, vec[:,0], vec[:,1], vec[:,2], nest=False)
        data = np.zeros((skymap.npix))
        data[pix_idx] = 1.0
        return data

class GravitationalWave:
    def __init__(self, spec="delta", distr="iso", A=1, f0=1, alpha=-5, phi0=0, theta0=PI/2):
        self.spec = spec
        self.distr = distr
        self.A = A
        self.f0 = f0
        self.alpha = alpha
        self.phi0 = phi0
        self.theta0 = theta0

    def generate_wave(self, spectrum, skymap):
        if self.spec == 'delta':
            H = self.A * self._delta_1d(self.f0, spectrum)
        elif self.spec == 'pow':
            H = self.A * self._pow(self.alpha, spectrum)
        
        if self.distr == 'iso':
            G = self._isotropic(skymap)
        elif self.distr == 'point':
            G = self._point(self.phi0, self.theta0, skymap)
        elif self.distr == 'delta':
            G = self._delta_2d(self.phi0, self.theta0, skymap)
        elif self.distr == '2delta':
            G = self._delta_2d(self.phi0 - PI/2, self.theta0, skymap) + self._delta_2d(self.phi0 + PI/2, self.theta0, skymap)
        elif self.distr == 'dipole':
            G = self._dipole(skymap)
        elif self.distr == 'quadrupole':
            G = self._quadrupole(skymap)
        return H, G

    def _isotropic(self, skymap):
        g = np.ones((skymap.npix))
        return g
    
    def _delta_2d(self, phi0, theta0, skymap):
        ipix = hp.ang2pix(skymap.nside, theta0, phi0)
        delta = np.zeros(skymap.npix, dtype=float)
        delta[ipix] = 1
        return delta
    
    def _dipole(self, skymap):
        cond = skymap.phi <= PI
        ipix = hp.ang2pix(skymap.nside, skymap.theta[cond], skymap.phi[cond], nest=False)
        dipole = np.zeros(skymap.npix, dtype=float)
        dipole[ipix] = 4*PI / skymap.dOmega / len(ipix)
        return dipole

    def _quadrupole(self, skymap):
        cond = np.logical_and(skymap.phi <= PI, skymap.theta <= PI/2)
        ipix = hp.ang2pix(skymap.nside, skymap.theta[cond], skymap.phi[cond])
        dipole = np.zeros(skymap.npix, dtype=float)
        dipole[ipix] = 4*PI / skymap.dOmega / len(ipix)
        return dipole
    
    def _point(self, phi0, theta0, skymap):
        radius_deg = 10
        vec_center = hp.ang2vec(theta0, phi0) 
        ipix_disc = hp.query_disc(skymap.nside, vec=vec_center, radius=np.radians(radius_deg))
        point = np.zeros(skymap.npix, dtype=float)
        if len(ipix_disc) == 0:
            ipix = hp.ang2pix(skymap.nside, theta0, phi0, nest=False)
            point[ipix] = 4*PI / skymap.dOmega
        else:
            point[ipix_disc] = 4*PI / skymap.dOmega / len(ipix_disc)
        
        return point
    
    def _delta_1d(self, f0, spectrum):
        # t = spectrum.t
        # signal = np.exp(2j * PI * f0 * t)
        # delta = np.fft.fft(signal)

        idx = np.argmin(np.abs(spectrum.f - f0))
        delta = np.zeros_like(spectrum.f)
        delta[idx] = 1.0/spectrum.df
        return delta
    
    def _pow(self, alpha, spectrum):
        d = np.abs(spectrum.f + EPS)**(alpha)
        d[spectrum.f==0] = 0
        norm = 1 / np.sum(d * spectrum.df)
        return norm * d

class Telescope:
    def __init__(self, skymap, spectrum, pulsar_array):
        self.map = skymap
        self.spec = spectrum
        self.pa = pulsar_array

    def observate_HD(self, grav_wave):
        p = self.pa.pulsar_vec
        lp = self.pa.pulsar_dist
        Np = self.pa.n_pulsars
        Nt = self.spec.n_times
        Ng = self.map.npix
        H, G = grav_wave.generate_wave(self.spec, self.map)
        
        eta = (np.random.normal(size=Ng) + 1j * np.random.normal(size=Ng)) / np.sqrt(2)
        # self.map.draw(np.sqrt(G))

        # print(np.sum(np.abs(eta)**2 * self.map.dOmega))
        # self.map.draw(np.sqrt(G) * np.real(eta))
        # self.map.draw(np.sqrt(G) * np.imag(eta))

        def z_calc(p, l):
            f = np.tile(self.spec.f[:, None], (1, Ng))
            H_f = np.tile(H[:, None], (1, Ng))
            
            beta = (1 + np.einsum('li,l->i', self.map.Omega, p))
            
            F = 1/2 * np.einsum('lm,lmi->i',np.einsum('l,m->lm', p, p), self.map.e) / beta
            
            Phi = F * np.sqrt(G) * eta / np.sqrt(self.map.dOmega)

            beta = np.tile(beta[None,:], (Nt, 1))
            
            alpha = l * f * (kpc * f_yr)

            T =  1 - np.exp(-1j  * 2 * PI * alpha * beta)

            B = T * np.sqrt(H_f)

            Phi = np.tile(Phi[None, :], (Nt, 1))

            z = np.real(np.fft.ifft(Phi * B, axis=0) * self.spec.df * Nt)
            Z = np.sum(z * self.map.dOmega, axis=1)

            return Z
        
        Z = np.zeros((Np, Nt))

        for k in tqdm(range(Np)):
            p_k = p[k]
            lp_k = lp[k]
            Z[k] = z_calc(p_k, lp_k)
        
        h2 = np.sum(H * self.spec.df)
        g2 = np.sum(G * self.map.dOmega)
        i, j = np.arange(0, Np),  np.arange(0, Np)
        i, j = np.meshgrid(i, j, indexing='xy')
        mask = i>j
        i, j = i[mask], j[mask]

        z1, z2 = Z[i], Z[j]
        mu = np.zeros(i.shape)

        def mu_calc(z1, z2):
            R = 2 * np.sum(z1 * z2 * self.spec.dt, axis=0)
            R0 = np.sqrt(np.sum(z1 * z1 * self.spec.dt, axis=0)) * np.sqrt(np.sum(z2 * z2 * self.spec.dt, axis=0))
            mu_k = 1/(h2 * g2) * R / R0
            return mu_k
        
        for k in tqdm(range(i.shape[0])):
            mu_k = mu_calc(z1[k], z2[k])
            mu[k] = mu_k

        p1, p2 = p[i], p[j]
        gamma = np.arccos(np.einsum('kl,kl->k',p1,p2))

        return gamma, mu


    def theory_HD(self, grav_wave):
        p = self.pa.pulsar_vec
        lp = self.pa.pulsar_dist
        Np = self.pa.n_pulsars
        Nt = self.spec.n_times
        Ng = self.map.npix
        H, G = grav_wave.generate_wave(self.spec, self.map)

        eta = (np.random.normal(size=Ng) + 1j * np.random.normal(size=Ng)) / np.sqrt(2)
        
        h2 = np.sum(H * self.spec.df)
        g2 = np.sum(G * self.map.dOmega)
        print(g2)
        self.map.draw(G)
        
        i, j = np.arange(0, Np),  np.arange(0, Np)
        i, j = np.meshgrid(i, j, indexing='xy')
        mask = i>j
        i, j = i[mask], j[mask]

        p1, p2 = p[i], p[j]
        l1, l2 = lp[i], lp[j]
        mu = np.zeros(i.shape)

        def mu_calc(p1, p2, l1, l2):
            beta_1 = (1 + np.einsum('li,l->i', self.map.Omega, p1))
            beta_2 = (1 + np.einsum('li,l->i', self.map.Omega, p2))
            
            F_1 = 1/2 * np.einsum('lm,lmi->i',np.einsum('l,m->lm', p1, p1), self.map.e) / beta_1
            F_2 = 1/2 * np.einsum('lm,lmi->i',np.einsum('l,m->lm', p2, p2), self.map.e) / beta_2
            
            Psi_1 = 1 * (F_1 * np.sqrt(G) * eta)
            Psi_2 = np.conjugate(F_2 * np.sqrt(G) * eta)

            # self.map.draw(np.abs(eta)**2)
            self.map.draw(np.real(F_1) * np.real(F_2))

            R = np.real(np.sum(Psi_1 * Psi_2 * self.map.dOmega))
            # R = np.sum(Psi_1 * np.sqrt(self.map.dOmega)) * np.sum(Psi_2 * np.sqrt(self.map.dOmega))

            mu_k = 1/(g2) * R  

            # Phi_1 = F_1 * np.sqrt(G) * eta / np.sqrt(self.map.dOmega)
            # Phi_2 = F_2 * np.sqrt(G) * eta / np.sqrt(self.map.dOmega)

            # f = np.tile(self.spec.f[:, None], (1, Ng))
            # H_f = np.tile(H[:, None], (1, Ng))

            # beta_1 = np.tile(beta_1[None,:], (Nt, 1))
            # beta_2 = np.tile(beta_2[None,:], (Nt, 1))
            
            # alpha_1 = l1 * f * (kpc * f_yr)
            # alpha_2 = l2 * f * (kpc * f_yr)

            # T_1 =  1 - np.exp(-1j  * 2 * PI * alpha_1 * beta_1)
            # T_2 =  1 - np.exp(-1j  * 2 * PI * alpha_2 * beta_2)

            # B_1 = T_1 * np.sqrt(H_f)
            # B_2 = T_2 * np.sqrt(H_f)

            # Phi_1 = np.tile(Phi_1[None, :], (Nt, 1))
            # Phi_2 = np.tile(Phi_2[None, :], (Nt, 1))

            # z_1 = np.real(np.fft.ifft(Phi_1 * B_1, axis=0) * self.spec.df) 
            # z_2 = np.real(np.fft.ifft(Phi_2 * B_2, axis=0) * self.spec.df)

            # Z_1 = np.sum(z_1 * self.map.dOmega, axis=1)
            # Z_2 = np.sum(z_2 * self.map.dOmega, axis=1)
            
            # # Z = Z_1 * Z_2 / self.map.dOmega
            # Z = np.sum(z_1 * z_2 * self.map.dOmega, axis=1)

            # R = 2 * np.sum(Z * self.spec.dt, axis=0)
            
            # mu_k = 1/(h2 * g2) * R 

            return mu_k
        

        for k in tqdm(range(i.shape[0])):
            mu[k] = mu_calc(p1[k], p2[k], l1[k], l2[k])

        gamma = np.arccos(np.einsum('kl,kl->k',p1,p2))

        return gamma, mu

    def plot_HD_curve(self, gw=None, key='theory', show=True):
        gamma, Gamma = np.array([]), np.array([])

        if key=='theory':
            gamma, Gamma = self.theory_HD(gw)

        if key=='obs':
            gamma, Gamma = self.observate_HD(gw)

        N_g = 21
        dgamma = PI / N_g
        gamma_m = np.linspace(dgamma/2, PI - dgamma/2, N_g)

        Gamma_m = np.zeros(N_g)
        Gamma_m2 = np.zeros(N_g)   # сумма квадратов
        N_m = np.zeros(N_g)

        for k in tqdm(range(gamma.shape[0])):
            Gamma_k, gamma_k = Gamma[k], gamma[k]

            mask = (gamma_m - dgamma <= gamma_k) & (gamma_k < gamma_m + dgamma)

            Gamma_m[mask] += Gamma_k
            Gamma_m2[mask] += Gamma_k**2
            N_m[mask] += 1

        # Среднее
        Gamma_mean = Gamma_m / N_m

        # Дисперсия и стандартное отклонение
        Gamma_var = Gamma_m2 / N_m - Gamma_mean**2
        Gamma_std = np.sqrt(Gamma_var)

        Gamma_var_plus = Gamma_mean + Gamma_std
        Gamma_var_minus = Gamma_mean - Gamma_std

        gamma_0 = np.linspace(0 + EPS, PI, 1000)
        Gamma_0 = mu_0(gamma_0)

        plt.grid(True)
        plt.scatter(gamma * 180/PI, Gamma, c='blue', linewidths=1, label='PTA')
        plt.plot(gamma_m * 180/PI, Gamma_mean, color='red', label='obs HD')
        plt.plot(gamma_m * 180/PI, Gamma_var_plus, color='red', linestyle='--', label='obs Var[HD]')
        plt.plot(gamma_m * 180/PI, Gamma_var_minus, color='red', linestyle='--')
        plt.plot(gamma_0 * 180/PI, Gamma_0, color='black', label='theory HD')

        plt.title("HD curve")
        plt.xlabel("$\\gamma$, deg")
        plt.ylabel("$\\Gamma(\\gamma)$")
        plt.legend()
        if not os.path.exists('data'):
            os.mkdir('data')
        plt.savefig('data/HD_obs.png')
        if show:
            plt.show()

    def plot_HD_var(self, gw=None, key='theory', show=True):
        gamma, Gamma = np.array([]), np.array([])

        if key=='theory':
            gamma, Gamma = self.theory_HD(gw)

        if key=='obs':
            gamma, Gamma = self.observate_HD(gw)

        N_g = 21
        dgamma = PI / N_g
        gamma_m = np.linspace(dgamma/2, PI - dgamma/2, N_g)

        Gamma_m = np.zeros(N_g)
        Gamma_m2 = np.zeros(N_g)   # сумма квадратов
        N_m = np.zeros(N_g)

        for k in tqdm(range(gamma.shape[0])):
            Gamma_k, gamma_k = Gamma[k], gamma[k]

            mask = (gamma_m - dgamma <= gamma_k) & (gamma_k < gamma_m + dgamma)

            Gamma_m[mask] += Gamma_k
            Gamma_m2[mask] += Gamma_k**2
            N_m[mask] += 1

        # Среднее
        Gamma_mean = Gamma_m / N_m

        # Дисперсия и стандартное отклонение
        Gamma_var = Gamma_m2 / N_m - Gamma_mean**2
        Gamma_std = np.sqrt(Gamma_var)

        gamma_0 = np.linspace(0 + 1e-5, PI, 1000)
        Gamma_0 = mu_0(gamma_0)

        plt.grid(True)
        # plt.scatter(gamma * 180/PI, Gamma_std_pt, c='blue', linewidths=1, label='PTA')
        plt.plot(gamma_m * 180/PI, Gamma_std, color='blue', label='obs Var[HD]')
        # plt.plot(gamma_m * 180/PI, Gamma_mean - Gamma_std, color='red', label='obs HD var')
        # plt.plot(gamma_0 * 180/PI, Gamma_0, color='black', label='theory HD')

        plt.title("HD curve")
        plt.xlabel("$\\gamma$, deg")
        plt.ylabel("$\\Gamma(\\gamma)$")
        plt.legend()
        if not os.path.exists('data'):
            os.mkdir('data')
        plt.savefig('data/Var_HD_obs.png')
        if show:
            plt.show()

skymap = SkyMap(10)
pulsar_array = PulsarArray(1000)
grav_wave = GravitationalWave(spec='delta', distr='point')
spec = Spectrum(10)

telescope = Telescope(skymap, spec, pulsar_array)
telescope.plot_HD_curve(grav_wave, key='theory')
# telescope.plot_HD_var(grav_wave, key='obs')