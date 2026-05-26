import pta

class Test(pta.Experiment):
    def __init__(self, n_map=100, n_spec=100, n_arr=100, seed_psr=42, seed_bg=43, distribution=pta.dipole, spectrum=pta.delta_1d, distr_args=(1), spec_args=(10)):
        self.skymap = pta.SkyMap(n_map)
        self.timeline = pta.TimeLine(n_spec)
        self.pulsar_arr = pta.PulsarArray(n_arr, seed_psr=seed_psr)
        self.telescope = pta.Telescope(self.skymap, self.timeline, self.pulsar_arr, seed_bg=seed_bg)
        self.grav_wave = pta.GravitationalWave(distribution=distribution, spectrum=spectrum)
        self.spec_args = spec_args
        self.distr_args = distr_args
    
    def pipeline(self, data):
        gamma_pta, mu_pta = self.telescope.observe(self.grav_wave)
        gamma_mean, mu_mean, mu_std = self.telescope.average(gamma_pta, mu_pta)
        gamma_theory, mu_theory = self.telescope.theory()

        # self.timeline.plot()
        # data = self.pulsar_arr.get_pixs(self.skymap)
        # self.skymap.plot(data, title='Distribution of Pulsar Array')
        # self.pulsar_arr.plot()
        # H, G = self.grav_wave.generate_wave(self.timeline, self.skymap, self.distr_args, self.spec_args)
        # self.skymap.plot(G, title='Gravitational Wave Background')
        # self.timeline.plot(H, key='f', title='Spectrum of GWB')
        return 0
    
    def postprocess(self, result):
        self.telescope.plot('hd')
        self.telescope.plot('var-hd')
        return 0

test = Test(n_map=100, 
            n_spec=10, 
            n_arr=1000, 
            seed_psr=42,
            seed_bg=43,
            distribution=pta.isotropic, 
            spectrum=pta.delta_1d,)
test.run()

class TestIdeal(pta.Experiment):
    def __init__(self, n_map=100, n_spec=100, n_arr=42, seed_psr=100, distribution=pta.dipole, spectrum=pta.delta_1d, distr_args=(1), spec_args=(10)):
        self.skymap = pta.SkyMap(n_map)
        self.timeline = pta.TimeLine(n_spec)
        self.pulsar_arr = pta.PulsarArray(n_arr, seed_psr=seed_psr)
        self.telescope = pta.IdealTelescope(self.skymap, self.timeline, self.pulsar_arr)
        self.grav_wave = pta.GravitationalWave(distribution=distribution, spectrum=spectrum)
        self.spec_args = spec_args
        self.distr_args = distr_args
    
    def pipeline(self, data):
        length = 360 # crit ~ 10.6
        theta, phi, kappa = pta.PI/2, pta.PI, 1/(length * pta.PI/180)**2
        print(kappa)
        args = theta, phi, kappa
        gamma_pta, mu_pta = self.telescope.analytics(args=args)
        # gamma_pta, mu_pta = self.telescope.observe(self.grav_wave, args=args)
        gamma_mean, mu_mean, mu_std = self.telescope.average(gamma_pta, mu_pta)
        gamma_theory, mu_theory = self.telescope.theory()
        return 0
    
    def postprocess(self, result):
        self.telescope.plot('theory-hd')
        # self.telescope.plot('var-hd')
        return 0
    
test = TestIdeal(n_map=100, 
                 n_spec=2, 
                 n_arr=100, 
                 seed_psr=42,
                 distribution=pta.gaussian, 
                 spectrum=pta.delta_1d,)
test.run()