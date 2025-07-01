import numpy as np
import MCMC
from scipy.stats import weibull_min, chi2, gaussian_kde
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gamma
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

class BayesFactor:

    def __init__(self, analysis, estimator='harmonic', bounds=None, figdir=None, thin=10):
        """ initialize BayesFactor

        Parameters:
        analysis (list): list of bayesian_data_analysis objects

        Returns:
        bayes_factor (list of floats): list of floats containing the bayes' factor for each chain

        """

        self.nchains = len(analysis)
        self.mcmc_chains = [analysis[i].sim_post for i in np.arange(self.nchains)]
        self.likelihood_dists = [analysis[i].likelihood_dist for i in np.arange(self.nchains)]
        par_info = [analysis[i].mcmc._par_dims for i in np.arange(self.nchains)]
        self.par_dims = [list(par_info[i].values()) for i in np.arange(self.nchains)]
        self.par_names = [analysis[i].pars for i in np.arange(self.nchains)]
        self.data = [analysis[i].data for i in np.arange(self.nchains)]
        self.priors = [analysis[i].priors for i in np.arange(self.nchains)]
        self.estimator = estimator
        self.map_pars = [analysis[i].map_pars for i in np.arange(self.nchains)]
        self.bounds = bounds
        self.figdir = figdir
        self.thin = thin

    def calculate_marginal_likelihood(self):

        self.z_estimate = []
        for idx in range(self.nchains):
            par_chain = self.mcmc_chains[idx][self.par_names[idx]]
            nsamples = len(par_chain)
            data = self.data[idx]
            model_priors = self.priors[idx]
            likelihood_dist = self.likelihood_dists[idx]
            if self.estimator == 'harmonic':
                self.z_estimate.append(self.harmonic_mean_estimator(
                    idx, par_chain, model_priors))
            elif self.estimator == 'retargeted_harmonic':
                self.z_estimate.append(self.retargeted_harmonic_mean_estimator(idx, par_chain,
                    model_priors))
            elif self.estimator == 'truncated_harmonic':
                self.z_estimate.append(self.truncated_harmonic_mean_estimator(idx, par_chain,
                    model_priors))
            elif self.estimator == 'kde_harmonic':
                self.z_estimate.append(self.kde_harmonic_mean_estimator(idx, par_chain,
                    model_priors))
            elif self.estimator == 'monte_carlo_integration':
                self.z_estimate.append(self.monte_carlo_integration(model_priors,
                    data, idx, likelihood_dist))
            else:
                raise ValueError('marignal likelihood estimator not yet implemented.')

    def perform_likelihood_analysis(self):

        self.map_ll = []
        self.ll = []
        for idx in range(self.nchains):
            par_chain = self.mcmc_chains[idx][self.par_names[idx]]
            map_index = np.argmax(self.mcmc_chains[idx]['log_post'])
            map_pars = par_chain.iloc[[map_index]]

            nsamples = len(par_chain)
            data = self.data[idx]
            model_priors = self.priors[idx]
            likelihood_dist = self.likelihood_dists[idx]
            if likelihood_dist == 'weibull':
                self.map_ll.append(self.weibull_log_likelihood(map_pars, data, idx))
                self.ll.append(self.weibull_log_likelihood(par_chain.iloc[::self.thin], data, idx))
        self.analyse_ll()


    def harmonic_mean_estimator(self, model_idx, par_chain, model_priors):

        par_chain = par_chain.iloc[::self.thin]
        log_post = self.mcmc_chains[model_idx]['log_post'][::self.thin]
        nsamples = len(par_chain)
        log_prior = self.calculate_log_prior(model_priors, par_chain, model_idx)
        log_likelihood = log_post - log_prior.squeeze()
        p_hat = 1/nsamples * np.sum(1/np.exp(log_likelihood))
        return 1/p_hat

    def kde_harmonic_mean_estimator(self, model_idx, par_chain, model_priors, HPD=0.95):
        log_posterior = np.array(self.mcmc_chains[model_idx]['log_post'][::self.thin])
        thinned_chain = par_chain.iloc[::self.thin]
        nsamples = len(thinned_chain)

        sorted_indices = np.argsort(log_posterior)
        nkeep = int(np.floor(len(log_posterior) * HPD))
        hpd_indices = sorted_indices[-nkeep:]
        hpd_pars = np.array(thinned_chain.iloc[hpd_indices]) #self.mcmc_chains[model_idx][self.par_names[idx]][hpd_indices]
        kde = gaussian_kde(hpd_pars.T)

        indicator = np.zeros((nsamples))
        indicator[hpd_indices] = 1
        target_dist_prob =  kde.evaluate(thinned_chain.T) * indicator

        summants = target_dist_prob * (1/np.exp(log_posterior))
        p_hat = 1/nsamples * np.sum(summants)
        return 1/p_hat

    def truncated_harmonic_mean_estimator(self, model_idx, par_chain, model_priors, HPD=0.95):
        log_posterior = np.array(self.mcmc_chains[model_idx]['log_post'][::self.thin])
        thinned_chain = par_chain.iloc[::self.thin]
        nsamples = len(thinned_chain)

        sorted_indices = np.argsort(log_posterior)
        nkeep = int(np.floor(len(log_posterior) * HPD))
        hpd_indices = sorted_indices[-nkeep:]
        hpd_pars = np.array(thinned_chain.iloc[hpd_indices]) #self.mcmc_chains[model_idx][self.par_names[idx]][hpd_indices]
        hpd_volume = ConvexHull(hpd_pars).volume

        indicator = np.zeros((nsamples))
        indicator[hpd_indices] = 1
        target_dist_prob =  1/hpd_volume * indicator

        summants = target_dist_prob * (1/np.exp(log_posterior))
        p_hat = 1/nsamples * np.sum(summants)
        return 1/p_hat


    def retargeted_harmonic_mean_estimator(self, model_idx, par_chain, model_priors, thin=10):
        log_posterior = np.array(self.mcmc_chains[model_idx]['log_post'][::self.thin])
        posterior = np.exp(log_posterior)

        par_chain = np.array(par_chain)[::self.thin, ...]
        nsamples = len(par_chain)

        chain_mean = np.atleast_2d(par_chain.mean(axis=0))
        chain_cov = np.cov(par_chain.T)

        diff = np.expand_dims(par_chain - chain_mean, axis=-1)
        diff_T = np.moveaxis(diff.T, -1, 0)
        hypersphere = diff_T @ np.linalg.inv(chain_cov) @ diff

        dim = par_chain.shape[1]
        optimal_radius = self.learn_optimal_radius(np.exp(log_posterior), hypersphere, dim, initial_radius=1.)
        hypersphere_volume = self.calculate_hypersphere_volume(optimal_radius, dim, chain_cov)

        indicator = np.zeros((nsamples))
        indicator[hypersphere.squeeze() < optimal_radius ** 2] = 1
        target_dist_prob = 1/hypersphere_volume * indicator

        summants = target_dist_prob * (1/np.exp(log_posterior))
        p_hat = 1/nsamples * np.sum(summants)

        return 1/p_hat


    def hypersphere_target_dist(par_sample, hypersphere_volume, chain_mean, chain_cov, radius):
        return 1/hypersphere_volume * indicator_function(par_sample, chain_mean, chain_cov, radius)


    def indicator_function(par_sample, chain_mean, chain_cov, radius):
        diff = np.expand_dims(par_sample - chain_mean, axis=-1)
        diff_T = np.moveaxis(diff.T, -1, 0)
        hypersphere = diff_T @ np.linalg.inv(chain_cov) @ diff
        conditional = hypersphere < radius ** 2
        if conditional:
            return 1
        else:
            return 0

    def calculate_hypersphere_volume(self, radius, dim, chain_cov):
        num = np.pi ** (dim / 2) * (radius ** dim) * np.linalg.det(chain_cov) ** 0.5
        denom = gamma((dim / 2) + 1)
        return num / denom

    def learn_optimal_radius(self, posterior, hypersphere, dim, initial_radius=1.0):
        """ Learn the optimal radius form the hypersphere using optimization."""

        optimal_radius = minimize(self.cost_function, initial_radius, args=(posterior, hypersphere, dim), bounds=[(1e-5, None)])
        return optimal_radius.x[0]

    def cost_function(self, radius, posterior, hypersphere, dim):
        """
        Compute the cost function based on the current radius and posterior samples.
        """
        #if hypersphere < radius ** 2:
        logspace=True
        conditional = (hypersphere < (radius ** 2)).squeeze()
        if logspace:
            C_i = - np.log(posterior[conditional]) - dim *  np.log(radius)
            log_sum_of_squares = (2 * C_i).sum()
            #print(f"Radius: {radius}, Cost: {log_sum_of_squares}")
            return log_sum_of_squares
        else:
            C_i = 1 / (posterior[conditional] * (radius**dim))
            sum_of_squares = (C_i**2).sum()
            #print(f"Radius: {radius}, Cost: {sum_of_squares}")
            return sum_of_squares

        # Return the sum of squares of costs
        #sum_of_squares = np.sum(np.square(C_i))


    def monte_carlo_integration(self, model_priors, data, model_idx, likelihood_dist):
        nsamples = int(1e6)
        prior_samples = self.draw_prior_samples(model_priors, nsamples, model_idx)
        #prior_samples_df = pd.DataFrame(prior_samples, columns=self.par_names[model_idx])
        if likelihood_dist == 'weibull':
            log_likelihood = np.array(self.weibull_log_likelihood(prior_samples, data, model_idx))
        return 1/nsamples * np.sum(np.exp(log_likelihood))


    def weibull_log_likelihood(self, pars, data, model_idx, neg=False):

        pars = np.atleast_2d(np.asarray(pars))
        par_dim = self.par_dims[model_idx]
        nsamples = len(pars)
#        sig0 = pars[self.par_names[model_idx][:par_dim[0]]]
#        rho = pars[self.par_names[model_idx][par_dim[0]:]]
        sig0 = pars[..., :par_dim[0]]
        rho = pars[..., par_dim[0]:]
        loglike = [0] * nsamples
        for ii in np.arange(nsamples):
            #if np.mod(ii, 10000) == 0:
                #print(f"calculating log likelihood for sample: {ii}")
            for data_idx, data_set in enumerate(data):
                #loglike[ii] += weibull_min.logpdf(data_set, rho.iloc[ii, data_idx], scale=sig0.iloc[ii, data_idx]).sum()
                loglike[ii] += weibull_min.logpdf(data_set, rho[ii, data_idx], scale=sig0[ii, data_idx]).sum()
        if neg:
            return -1 * loglike[0]
        else:
            return loglike

    def calculate_log_prior(self, prior, samples, model_idx):
        log_prior = 0
        par_idx = 0
        npars = self.par_dims[model_idx]
        for idx, par_name in enumerate(prior.__dict__):
            curr_prior = prior.__dict__[par_name]
            curr_par = samples[self.par_names[model_idx][par_idx:par_idx + npars[idx]]]
            log_prior += curr_prior.logpdf(curr_par)
            par_idx += npars[idx]
        return log_prior

    def draw_prior_samples(self, prior, nsamples, model_idx, bounds=None):
        par_idx = 0
        npars = self.par_dims[model_idx]
        dim = np.array(npars).sum()

        if self.bounds is not None:
            # need to account for some bounds being None and some not
            mn, mx = np.array(self.bounds).T
        else:
            mn = -np.inf
            mx = np.inf

        samples = np.zeros((nsamples, dim))
        for idx, par_name in enumerate(prior.__dict__):
            curr_prior = prior.__dict__[par_name]
            par_samples = np.atleast_2d(curr_prior.rvs(nsamples))
            while True:
                out_of_bounds = (par_samples < mn[idx]) | (par_samples > mx[idx])
                if not np.any(out_of_bounds):
                    break
                i, j = np.where(out_of_bounds)
                replace_rows = np.unique(i)
                par_samples[replace_rows] = np.atleast_2d(curr_prior.rvs(len(replace_rows)))

            if par_samples.shape[0] != nsamples:
                par_samples = par_samples.T
                assert par_samples.shape[0] == nsamples
            samples[:, par_idx:par_idx + npars[idx]] = par_samples
            par_idx += npars[idx]
        return samples

    def plot_hist(self, hist_data, xlabel, figtitle, figname, MAP_val=None, credible_intervals=False, kde=False, nbins=30):

        plt.figure(figsize=(12,8))
        sns.histplot(hist_data, bins=nbins, kde=kde, color='white', edgecolor='black', stat='density')
        sns.kdeplot(hist_data, color='seagreen', linewidth=3)

        if credible_intervals:
            credible_intervals = [0.25, 0.75, 0.10, 0.90, 0.01, 0.99]
            CI = self.get_quantiles(hist_data, credible_intervals)
            colors = ['red', 'red', 'green', 'green', 'purple', 'purple']
            labels = ['50%CI', '80%CI', '98%CI']
            for idx, level in enumerate(CI):
                if np.mod(idx, 2) == 0:
                    plt.axvline(level, color=colors[idx], linestyle='--', linewidth=3,  label=labels[int(idx/2)])
                else:
                    plt.axvline(level, color=colors[idx], linestyle='--', linewidth=3)

        if MAP_val is not None:
            plt.axvline(MAP_val, color='k', linestyle='-', linewidth=3.5, label='MAP')

        plt.legend(fontsize=18)
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel('Density', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(figtitle, fontsize=20)
        plt.savefig(f"{self.figdir}/{figname}.png")
        plt.close("all")


    def analyse_ll(self):


        # log predictive density
        n_samples = len(self.ll[0])
        self.l = np.array([np.exp(self.ll[i]) for i in range(2)])
        self.lppd  = [np.log(1/n_samples * self.l[i].sum()) for i in range(2)]

        self.MLE = [None] * self.nchains
        MLE_ll = []
        post_mean = []
        post_mean_ll = []
        p_DIC = []
        p_WAIC_1 = []
        p_WAIC_2 = []
        for idx in range(self.nchains):
            par_chain = np.array(self.mcmc_chains[idx][self.par_names[idx]])
            map_pars = np.array(self.map_pars[idx])[:-1]
            data = self.data[idx]
            likelihood_dist = self.likelihood_dists[idx]
            post_mean.append(par_chain.mean(axis=0))
            if likelihood_dist == 'weibull':
                MLE = minimize(self.weibull_log_likelihood, map_pars, args=(data, idx, True), bounds=None) 
                self.MLE[idx] = MLE.x
                MLE_ll.append(-1 * MLE.fun)
                post_mean_ll.append(self.weibull_log_likelihood(post_mean[idx], data, idx)[0]) 
            p_DIC.append(2 * (post_mean_ll[idx] - 1/n_samples * sum(self.ll[idx])))
            p_WAIC_1.append(2 * (np.log(1/n_samples * sum(np.exp(self.ll[idx]))) - 1/n_samples * sum(self.ll[idx])))
            p_WAIC_2.append(1/n_samples * sum( (self.ll[idx] - post_mean_ll[idx])**2 ))
        print(f"MAP: {[np.array(self.map_pars[idx]) for idx in range(2)]}")
        print(f"MLE: {self.MLE}")
        print(f"Posterior Mean: {post_mean}")


        nmeas = len(self.data[1][0])
        model_dim = [sum(self.par_dims[i]) for i in range(2)]
        alpha = 0.05
        chi_dof = model_dim[0] - model_dim[1]

        ll_diff_map = self.map_ll[1][0] - self.map_ll[0][0]
        ll_diff_mle = MLE_ll[1] - MLE_ll[0]

        AIC_mle = [-2 * MLE_ll[i] + 2*model_dim[i] for i in range(2)]
        AIC_diff_mle = AIC_mle[0] - AIC_mle[1]

        AIC_map = [-2 * self.map_ll[i][0] + 2*model_dim[i] for i in range(2)]
        AIC_diff_map = AIC_map[0] - AIC_map[1]

        DIC_pm = [-2 * post_mean_ll[i] + 2 * p_DIC[i] for i in range(2)]
        DIC_diff_pm = DIC_pm[0] - DIC_pm[1]
        WAIC_1 = [-2 * self.lppd[i] + 2 * p_WAIC_1[i] for i in range(2)]
        WAIC_2 = [-2 * self.lppd[i] + 2 * p_WAIC_2[i] for i in range(2)]

        BIC_mle = [-2 * MLE_ll[i] + model_dim[i]*np.log(nmeas) for i in range(2)]
        BIC_diff_mle = BIC_mle[0] - BIC_mle[1]

        BIC_map = [-2 * self.map_ll[i][0] + model_dim[i]*np.log(nmeas) for i in range(2)]
        BIC_diff_map = BIC_map[0] - BIC_map[1]

        ll_ratio_test_statistic_map = -2 * ll_diff_map
        p_value_map = 1 - chi2.cdf(ll_ratio_test_statistic_map, chi_dof)
        ll_ratio_test_statistic_mle = -2 * ll_diff_mle
        p_value_mle = 1 - chi2.cdf(ll_ratio_test_statistic_mle, chi_dof)

        # histogram of likelihood difference
        ll_diff = np.array(self.ll[1]) - np.array(self.ll[0])
        xlabel = "{} - {}".format(r'$f(y | \theta, M_{2})$', r'$f(y | \theta, M_{1})$')
        figtitle = "Log-likelihood difference"
        figname = "ll_diff"
        self.plot_hist(ll_diff, xlabel, figtitle, figname, MAP_val=ll_diff_map, credible_intervals=True)

        # histogram of test statistic
        ll_ratio_test_statistic = -2 * ll_diff
        xlabel = "Log-likelihood test statistic"
        figtitle = "Log-likelihood test statistic"
        figname = "ll_test_statistic"
        self.plot_hist(ll_ratio_test_statistic, xlabel, figtitle, figname, MAP_val=ll_ratio_test_statistic_map, credible_intervals=True)

        # histogram of p-value
        p_value = 1 - chi2.cdf(ll_ratio_test_statistic, chi_dof)
        xlabel = "log-likelihood ratio p-value"
        figtitle = "Log-likelihood ratio p-value"
        figname = "ll_pvalue"
        self.plot_hist(p_value, xlabel, figtitle, figname, MAP_val=p_value_map, credible_intervals=True)


        # histogram of AIC/BIC
        if False:
            AIC_dist = [-2 * np.array(self.ll[i]) + 2*model_dim[i] for i in range(2)]
            BIC_dist = [-2 * np.array(self.ll[i]) + model_dim[i]*np.log(nmeas) for i in range(2)]
            for idx in range(2):
                xlabel = "AIC"
                figtitle = f"AIC Model {idx + 1}"
                figname = f"AIC_M{idx + 1}"
                self.plot_hist(AIC_dist[idx], xlabel, figtitle, figname, MAP_val=AIC_map[idx], credible_intervals=False)

                xlabel = "BIC"
                figtitle = f"BIC Model {idx + 1}"
                figname = f"BIC_M{idx + 1}"
                self.plot_hist(BIC_dist[idx], xlabel, figtitle, figname, MAP_val=BIC_map[idx], credible_intervals=False)

            xlabel = "AIC difference"
            figtitle = f"AIC Difference"
            figname = f"AIC_diff"
            AIC_diff = AIC_dist[0] - AIC_dist[1]
            AIC_diff_map = AIC_map[0] - AIC_map[1]
            self.plot_hist(AIC_diff, xlabel, figtitle, figname, MAP_val=AIC_diff_map, credible_intervals=True)

            xlabel = "BIC difference"
            figtitle = f"BIC Differnce"
            figname = f"BIC_diff"
            BIC_diff = BIC_dist[0] - BIC_dist[1]
            BIC_diff_map = BIC_map[0] - BIC_map[1]
            self.plot_hist(BIC_diff, xlabel, figtitle, figname, MAP_val=BIC_diff_map, credible_intervals=True)

            AIC_diff_CI = self.get_quantiles(AIC_diff, credible_intervals)
            BIC_diff_CI = self.get_quantiles(BIC_diff, credible_intervals)
            AIC_diff_prob = sum(AIC_diff > 1) / n_samples
            BIC_diff_prob = sum(BIC_diff > 1) / n_samples

        credible_intervals = [0.01, 0.10, 0.25, 0.75, 0.90, 0.99]
        ll_diff_CI = self.get_quantiles(ll_diff, credible_intervals)
        p_value_CI = self.get_quantiles(p_value, credible_intervals)
        p_value_prob = sum(p_value > alpha) / n_samples
        ll_diff_prob = sum(ll_diff > 0) / n_samples


        print(f"MAP p-value {p_value_map}: There is a {p_value_map} percecnt chance of observing this test-statistic")
        print(f"MLE p-value {p_value_mle}: There is a {p_value_mle} percecnt chance of observing this test-statistic")
        print("Log-likelihood difference quantiles: {}".format(ll_diff_CI))
        #print("Probability {} / {} > 1: {}".format(r'AIC $M_{1}$', r'AIC$M_{2})$', AIC_diff_prob))
        #print("AIC difference quantiles: {}".format(AIC_diff_CI))
        #print("Probability {} / {} > 1: {}".format(r'BIC $M_{1}$', r'BIC$M_{2})$', BIC_diff_prob))
        #print("BIC difference quantiles: {}".format(BIC_diff_CI))
        print("Probability p-value > {}: {}".format(alpha, p_value_prob))
        print("p-value quantiles: {}".format(p_value_CI))

        print("######################")
        print("##### MAP Values #####")
        print("######################")
        print(f"MAP log likelihood difference: {ll_diff_map}")
        print(f"MAP log likelihood test statistic: {ll_ratio_test_statistic_map}")
        print(f"MAP log likelihood p-value: {p_value_map}")
        print(f"MAP AIC: {AIC_map}")
        print(f"MLE AIC: {AIC_mle}")
        print(f"MAP BIC: {BIC_map}")
        print(f"MLE BIC: {BIC_mle}")
        print(f"DIC: {DIC_pm}")
        print(f"WAIC 1: {WAIC_1}")
        print(f"WAIC 2: {WAIC_2}")
        print(f"MAP log likelihood: {self.map_ll}")
        print(f"MLE log likelihood: {MLE_ll}")
        print(f"Post Mean log likelihood: {post_mean_ll}")
        print(f"LPPD: {self.lppd}")
        print(f"pDIC: {p_DIC}")
        print(f"pWAIC 1: {p_WAIC_1}")
        print(f"pWAIC 2: {p_WAIC_2}")
        print("######################")
        print("######################")

    def get_quantiles(self, samples, cred_int):
        """calculate quantile for given credible level """

        return np.quantile(samples, cred_int)


