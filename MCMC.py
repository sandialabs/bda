import numpy as np
from scipy.stats import uniform, norm, gamma, multivariate_t, t, weibull_min, multivariate_normal
import matplotlib.pyplot as plt
import matplotlib
import os
import time
import math

class Attribute(object):
    pass


class MCMC:

    def __init__(self, data, priors, likelihood_dist, par_names, figdir,
        verbose=True, display=1000):
        """ initialize MCMC

        Parameters:
        data (array): the data to be analyzed - array of (size nsamples x ndata_sets)
        priors (list of scipy distributions): a list of the priors over the distribution hyperparameters
        pars (list): list of names of the parameters being analysed (i.e. 'mu', 'delta')
        likelihood_dist (str): name of likelihood distribution

        """

        self.data = data
        self.priors = priors
        self.likelihood_dist = likelihood_dist
        self.pars = par_names
        self.par_dim = len(par_names)
        self._verbose = verbose
        self._display = 1000
        self._ndata = len(self.data)
        self._MCMC_dir = f"{figdir}/MCMC"
        if not os.path.exists(self._MCMC_dir):
            os.makedirs(self._MCMC_dir)
        self.get_par_dims()


    def simulate(self, nsamples=100_000, burnin=10_000, cov_check=2_000,
        prop_var=None, par_0=None, diagnostic_check=20_000,
        iter_start_diag=10_000, par_blocks=None):

        self._prop_var = prop_var
        self._nsamples = nsamples
        self._burnin = burnin
        self._cov_check = cov_check
        self._diagnostic_check = diagnostic_check
        self._iter_start_diag = iter_start_diag

        # set up blocks
        self._par_blocks = par_blocks
        if par_blocks is None:
            self._par_blocks = [[i] for i in np.arange(self.par_dim)]
        self._npar_blocks = len(self._par_blocks)

        self.initialize_chain(par_0)
        self.select_sampler()

        self._target_accept = np.array([0.2, 0.5])
        if self._iter_start_diag < self._burnin:
            self._iter_start_diag = self._burnin

        self._prop_var = prop_var
        if prop_var is None:
            prop_var_order = []
            for block in self._par_blocks:
                prop_var_order.append([self.get_order_of_number(i) for i in self._current_pars[0, block]])

            self._prop_var = [None] * self._npar_blocks
            for par_block in np.arange(self._npar_blocks):
                self._prop_var[par_block] = np.asarray(np.diag([10 ** order for order in prop_var_order[par_block]]))


        time_start = time.time()
        for sample_number in np.arange(self._nsamples):
            if self._verbose and np.mod(sample_number, self._display) == 0 :
                    print(f"MCMC sample = {sample_number}")

            if np.mod(sample_number, self._cov_check) == 0 and sample_number > 0 and sample_number < self._burnin:
               self.perform_cov_check(sample_number)

            for par_block, par_block_index in enumerate(self._par_blocks):
                self.sample_par(par_block, par_block_index, sample_number)
            self._mcmc_chain[sample_number, ...] = self._current_pars
            self._log_post[sample_number] = self._current_log_post

            if np.mod(sample_number+1, self._diagnostic_check) == 0 and sample_number+1 > self._iter_start_diag:
                time_end = time.time()
                elapsed_time = time_end - time_start
                self.perform_diagnostics(elapsed_time, sample_number)
                self.save_chain(sample_number)
                time_start = time.time()


        self.save_chain()


    def initialize_chain(self, par_0):
        self._mcmc_chain = np.zeros((self._nsamples, self.par_dim))
        if par_0 is None:
            par_0 = np.zeros((1, self.par_dim))
            par_idx = 0
            self._prior_names = list(self.priors.__dict__.keys())
            for idx, name in enumerate(self._prior_names):
                prior = getattr(self.priors, name)
                random_sample = np.atleast_1d(prior.rvs())
                npars = len(random_sample)
                par_0[0, par_idx:par_idx+npars] = random_sample
                par_idx += npars
        self._current_pars = np.asarray((par_0))
        self._mcmc_chain[0, :] = self._current_pars
        self._acceptance = np.zeros((self._nsamples, self._npar_blocks))
        self._log_post = np.zeros(self._nsamples)
        self._current_log_post = self.calculate_unnorm_log_post(self._current_pars)
        self._log_post[0] = self._current_log_post



    def sample_par(self, par_block, par_block_index, sample_number):
        sampler = self._sampler[par_block]
        sampler(par_block, par_block_index, sample_number)



    def metropolis_hastings(self, par_block, par_block_index, sample_number):
        self._proposed_pars = self._current_pars.copy()
        while True:
            self._proposed_pars[0, par_block_index] = multivariate_normal(self._current_pars[0, par_block_index], self._prop_var[par_block]).rvs()
            if (self._proposed_pars[0, par_block_index] > 0).all():
                break
        self._proposed_log_post = self.calculate_unnorm_log_post(self._proposed_pars)

        if np.log(uniform(0,1).rvs(1)) < self._proposed_log_post - self._current_log_post:
            self._current_pars[0, par_block_index] = self._proposed_pars[0, par_block_index].copy()
            self._current_log_post = self._proposed_log_post.copy()
            self._acceptance[sample_number, par_block] = 1
        else:
            self._acceptance[sample_number, par_block] = 0



    def select_sampler(self):
        self._sampler = [self.metropolis_hastings, self.metropolis_hastings]



    def perform_cov_check(self, sample_number):
        accept_rate = (1 + self._acceptance[sample_number - self._cov_check : sample_number - 1, ...].sum(axis=0)) / self._cov_check
        if self._verbose:
            print(f"Accept rate: {accept_rate}")
        for par_block, par_block_index in enumerate(self._par_blocks):
            if accept_rate[par_block] < self._target_accept[0] or accept_rate[par_block] > self._target_accept[1]:
                if len(par_block_index) > 1 and self._sampler[par_block] == self.metropolis_hastings:
                    temp_chain = self._mcmc_chain[sample_number - self._cov_check : sample_number, par_block_index]
                    C = np.cov(temp_chain.T)
                    cov_diag = np.diag(C) * accept_rate / \
                        (self._target_accept[0] + np.ptp(self._target_accept) / 2)
                    np.fill_diagonal(self._prop_var[par_block], cov_diag)
                    self._prop_var[par_block] = self.cov_mat_adj(temp_chain, self._prop_var[par_block])
                elif len(par_block_index) == 1 and self._sampler[par_block] == self.metropolis_hastings:
                    self._prop_var[par_block] = self._prop_var[par_block] * accept_rate[par_block] / \
                        (self._target_accept[0] + np.ptp(self._target_accept) / 2)



    def perform_diagnostics(self, elapsed_time, sample_number):
        print(f"time to perform last {self._diagnostic_check} iterations: {elapsed_time/60} minutes")
        print(f"acceptance rate:  {self._acceptance[:sample_number, ...].sum(axis=0)/sample_number}")

        par_dims = list(self._par_dims.values())
        plt.figure(figsize=(12,8))
        for dd in range(par_dims[0]):
            plt.subplot(1, par_dims[0], dd+1)
            plt.plot(self._mcmc_chain[self._iter_start_diag:sample_number, dd])
            plt.title(r'Trace $\sigma0_{dd}$')
        plt.savefig(f"{self._MCMC_dir}/sigma0_trace_iter_{sample_number+1}.png")
        plt.close()

        plt.figure(figsize=(12,8))
        for dd in range(par_dims[1]):
            plt.subplot(1, par_dims[1], dd+1)
            plt.plot(self._mcmc_chain[self._iter_start_diag:sample_number, par_dims[0] + dd])
            plt.title(r'Trace $\rho_{dd}$')
        plt.savefig(f"{self._MCMC_dir}/rho_trace_iter_{sample_number+1}.png")
        plt.close()

        plt.figure(figsize=(12,8))
        plt.plot(self._log_post[self._iter_start_diag:sample_number])
        plt.title(r'Log $\pi(\theta | y)$')
        plt.savefig(f"{self._MCMC_dir}/logpost_iter_{sample_number+1}.png")
        plt.close()



    def save_chain(self, sample_number=None):
        self.trace = {}
        for par_idx, par in enumerate(self.pars):
            if sample_number is None:
                self.trace[par] = self._mcmc_chain[:, par_idx]
            else:
                self.trace[par] = self._mcmc_chain[:sample_number, par_idx]
        if sample_number is None:
            self.trace['log_post'] = self._log_post
        else:
            self.trace['log_post'] = self._log_post[:sample_number]



    def calculate_unnorm_log_post(self, pars):
        # calculate log prior
        log_prior = self.calculate_log_prior(pars)
        log_likelihood = self.calculate_log_likelihood(pars)
        return log_prior + log_likelihood



    def calculate_log_prior(self, pars):
        log_prior = 0
        par_idx = 0
        for idx, par_name in enumerate(self.priors.__dict__):
            prior = self.priors.__dict__[par_name]
            npars = len(np.atleast_1d(prior.rvs()))
            curr_par = pars[0, par_idx:par_idx + npars]
            log_prior += prior.logpdf(curr_par)
            par_idx += npars
        return log_prior



    def calculate_log_likelihood(self, pars):

        if self.likelihood_dist == 'weibull':
            return self.weibull_log_likelihood(pars)


    def weibull_log_likelihood(self, pars):

        par_dim = list(self._par_dims.values())
        sig0 = pars[0, :par_dim[0]]
        rho = pars[0, par_dim[0]:]
        loglike = 0
        for data_idx, data_set in enumerate(self.data):
            loglike += weibull_min.logpdf(data_set, rho[data_idx], scale=sig0[data_idx]).sum()
        return loglike



    def cov_mat_adj(self, pars, cov_mat_orig):

        D = pars.shape[1]
        # adjust the ratio of the proposal variances based on the covariance of the parameters

        def_by = 1

        chaincorr = np.corrcoef(pars.T)
        chaincorr[np.isnan(chaincorr)] = 0
        new_stepcor_chain = np.zeros((D, D))
        temp_stepvar_chain = np.zeros((D, D))

        for rr in range(D):
            for cc in range(D):
                if rr == cc:
                    new_stepcor_chain[rr, cc] = 1
                elif np.abs(chaincorr[rr, cc]) < 0.6:
                    new_stepcor_chain[rr, cc] = 0
                else:
                    new_stepcor_chain[rr, cc] = np.sign(chaincorr[rr, cc]) *\
                        (np.abs(chaincorr[rr, cc]) - 0.4)


        pvars = np.diag(cov_mat_orig)/def_by

        for rr in range(D):
            for cc in range(D):
                temp_stepvar_chain[rr, cc] = new_stepcor_chain[rr, cc] * np.sqrt(pvars[rr] * pvars[cc])

        tempvar = np.diag(temp_stepvar_chain)

        for dd in range(D):
            if tempvar[dd] <= 0:
                temp_stepvar_chain[dd, dd] = cov_mat_orig[dd, dd]


        cov_mat_new = temp_stepvar_chain

        return cov_mat_new

    def get_order_of_number(self, number):
        if number == 0:
            return 0
        return float(math.floor(math.log10(abs(number))))


    def get_par_dims(self):
        self._par_dims = {}
        for idx, par_name in enumerate(self.priors.__dict__):
            prior = self.priors.__dict__[par_name]
            npars = len(np.atleast_1d(prior.rvs()))
            self._par_dims[par_name] = npars
