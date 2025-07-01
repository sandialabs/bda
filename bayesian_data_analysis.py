import numpy as np

from scipy.stats import beta, norm, gamma, multivariate_normal, chi2, t, weibull_min
from scipy.special import gammaln, gammainc
import scipy.special
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator, LogLocator, LinearLocator
from matplotlib import ticker, colormaps

import pandas as pd
import seaborn as sns
from MCMC import MCMC
import plotly.graph_objects as go

class Attribute(object):
    pass

@ticker.FuncFormatter
def major_formatter(x, pos):
    return f'{x: .2f}'

class BDA:

    def __init__(self, data, priors, likelihood_dist, data_labels,
        pars, figdir):
        """ initialize the bayesian data anlaysis (BDA) study for
        given data and  priors over the distribution hyperparameters
        ---> for normal distribution, the hyperparameters are the mean and variance
        ---> for student-t, the hyperparameters are the mean, variance and d.o.f
        ---> for Weibull, the hyperparameters are sig0 (characteristic strenght) and rho (Weibull modulus)

        Parameters:
        data (array): the data to be analyzed - array of (size nsamples x ndata_sets)
        priors (list of scipy distributions): a list of the priors over the distribution hyperparameters
        pars (list): list of names of the distribution parameters being analysed (i.e. 'mu', 'delta')
        """

        self.data = data
        self.ndatasets = len(data)
        self.ndata_samples = [len(data[i]) for i in np.arange(self.ndatasets)]
        self.priors = Attribute()
        if likelihood_dist == 'normal' or likelihood_dist == 't':
            self.priors.mu = priors[0]
            self.priors.delta2 = priors[1]
            self.ndelta = len(self.priors.delta2)
            if likelihood_dist == 't':
                self.priors.tdf = priors[2]
        elif likelihood_dist == 'weibull':
            self.priors.sig0 = priors[0]
            self.priors.rho = priors[1]

        self.ndim = len(pars)
        self.pars = pars
        self.figdir = figdir
        self.data_labels = data_labels
        self.likelihood_dist = likelihood_dist

    def get_fig_strings(self):
        if self.likelihood_dist == 'normal':
            labels = [r'$\mu_{1}$', r'$\mu_{2}$', r'$\delta^{2}_{1}$', r'$\delta^{2}_{2}$']
            title_id = [r'$\mu$', r'$\delta^{2}$']
            fig_id = ['mu', 'delta2']
        elif self.likelihood_dist == 't':
            labels = [r'$\mu_{1}$', r'$\mu_{2}$', r'$\delta^{2}_{1}$', r'$\delta^{2}_{2}$', r'$tdf_{1}$', r'$tdf_{2}$']
            title_id = [r'$\mu$', r'$\delta^{2}$', r'$tdf$']
            fig_id = ['mu', 'delta2', 'd.o.f.']
        elif self.likelihood_dist == 'weibull':
            labels = [r'$\sigma_{0,1}$', r'$\sigma_{0,2}$', r'$\rho_{1}$', r'$\rho_{2}$']
            title_id = ['$\sigma_{0}$', '$\rho$']
            fig_id = ['sigma0', 'rho']

        return labels, title_id, fig_id


    def get_chi2_cl(self, cl, dof):
        return chi2.ppf(cl, dof)


    def get_CI_ellipse(self, ellipse_data, data_cov=None):

        # ellipse for 98% CI of data
        if data_cov is None:
            data_cov = np.cov(ellipse_data[0], ellipse_data[1])
        lambda_, v = np.linalg.eig(data_cov)
        lambda_ = np.sqrt(lambda_)
        ell_radius_x = 2*np.sqrt(5.991) * lambda_[0] # 5.991 is the chi-square value for 98% CI
        ell_radius_y = 2 * np.sqrt(5.991) * lambda_[1]
        ell_angle = np.rad2deg(np.arccos(v[0, 0]))
        ellipse = Ellipse((ellipse_data[0].mean(), ellipse_data[1].mean()),
            width=ell_radius_x, height=ell_radius_y, angle=ell_angle,
            edgecolor='black', facecolor='none', linewidth=2)

        return ellipse


    def plot_hist(self, hist_data, xlabel, figtitle, figname, credible_intervals=False, kde=False, nbins=30):

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
            plt.legend(fontsize=18)

        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel('Density', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(figtitle, fontsize=20)
        plt.savefig(f"{self.figdir}/{figname}.png")
        plt.close("all")


    def analyse_prior(self, n_samples=1_000):
        """draw samples from the prior distribution and analysize prior assumptions """

        labels, title_id, fig_id = self.get_fig_strings()

        if self.likelihood_dist == 'normal' or self.likelihood_dist == 't':
            prior = self.priors.mu
        elif self.likelihood_dist == 'weibull':
            prior = self.priors.sig0


        prior_samples = prior.rvs(n_samples)

        # plot samples from the prior distribution
        sim_prior = pd.DataFrame(prior_samples, columns=self.pars[:self.ndatasets])

        # ellipse for 98% CI of prior samples
        ellipse = False
        if self.ndatasets == 2:
            ellipse = True
            prior_ellipse = self.get_CI_ellipse([sim_prior[self.pars[0]], sim_prior[self.pars[1]]])

        fig, ax = plt.subplots(1, 2, figsize=(12,8))
        ax[0].scatter(sim_prior[self.pars[0]], sim_prior[self.pars[1]], color='skyblue', alpha=0.4)
        sns.kdeplot(x=sim_prior[self.pars[0]], y=sim_prior[self.pars[1]], levels=20, color='gray', linewidths=1, ax=ax[0])
        ax[0].axline((prior.mean[0], prior.mean[0]), slope=1, color='black', linestyle='--')
        if ellipse:
            ax[0].add_patch(prior_ellipse)
        ax[0].set_xlabel(f'{labels[0]} (MPa)', fontsize=20)
        ax[0].set_ylabel(f'{labels[1]} (MPa)', fontsize=20)
        ax[0].tick_params(axis='both', labelsize=20)
        ax[0].set_title("{} prior samples".format(title_id[0]), fontsize=20)

        sns.kdeplot(x=sim_prior[self.pars[0]], y=sim_prior[self.pars[1]], fill=True, cmap='viridis', thresh=0.001, levels=20, ax=ax[1])
        ax[1].axline((prior.mean[0], prior.mean[0]), slope=1, color='black', linestyle='--')
        ax[1].set_title('{} prior density plot'.format(title_id[0]), fontsize=20)
        ax[1].set_xlabel(f'{labels[0]} (MPa)', fontsize=20)
        ax[1].set_ylabel(f'{labels[1]} (MPa)', fontsize=20)
        ax[1].tick_params(axis='both', labelsize=20)
        plt.tight_layout()
        plt.savefig("{}/{}_prior_samples.png".format(self.figdir, fig_id[0]))
        plt.close("all")

        # histogram of prior mean difference
        par_diff = sim_prior[self.pars[0]] - sim_prior[self.pars[1]]
        xlabel = "{} - {} (MPa)".format(labels[0], labels[1])
        figtitle = "Prior over the difference"
        figname = "prior_{}_{}_diff".format(self.pars[0], self.pars[1])
        self.plot_hist(par_diff, xlabel, figtitle, figname, credible_intervals=True)

        # histogram of prior mean ratio
        par_ratio = sim_prior[self.pars[0]] / sim_prior[self.pars[1]]
        xlabel = "{} / {}".format(labels[0], labels[1])
        figtitle = "Prior over the ratio"
        figname = "prior_{}_{}_ratio".format(self.pars[0], self.pars[1])
        self.plot_hist(par_ratio, xlabel, figtitle, figname, credible_intervals=True)

        credible_intervals = [0.01, 0.10, 0.25, 0.75, 0.90, 0.99]
        prior_diff_CI = self.get_quantiles(par_diff, credible_intervals)
        prior_ratio_CI = self.get_quantiles(par_ratio, credible_intervals)
        prior_prob_diff = sum(par_diff > 0) / n_samples
        prior_prob_ratio = sum(par_ratio > 1) / n_samples
        print("prior probability {} > {}: {}".format(self.pars[0], self.pars[1], prior_prob_diff))
        print("Prior difference quantiles: {}".format(prior_diff_CI))
        print("prior probability {} / {} > 1: {}".format(self.pars[0], self.pars[1], prior_prob_ratio))
        print("prior ratio quantiles: {}".format(prior_ratio_CI))


    def get_quantiles(self, samples, cred_int, axis=None):
        """calculate quantile for given credible level """

        if axis is None:
            return np.quantile(samples, cred_int)
        else:
            return np.quantile(samples, cred_int, axis=axis)


    def analyse_data(self, MAP_post_dist=False):
        """ plot histogram of data """

        print('****************************')
        print('DATA SUMMARY')
        print('****************************')
        plt.figure(figsize=(12,8))
        data_colors = ['seagreen', 'violet']
        for data_idx in range(self.ndatasets):
            sns.histplot(self.data[data_idx], bins=15, kde=False, color=data_colors[data_idx], edgecolor='black', stat='density', alpha=0.5, label=f'{self.data_labels[data_idx]} data')
        plt.xlabel('Strength (MPa)', fontsize=20)
        plt.ylabel('Density', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('Data', fontsize=20)
        if MAP_post_dist:
            xlim = plt.xlim()
            xgrid = np.linspace(xlim[0], xlim[1], 100)
            if self.likelihood_dist == 'normal':
                if self.ndelta == 1:
                    MAP_dists = [norm(self.map_pars[f'mu{i}'], 1/self.map_pars[f'delta2_{i}'] ** 0.5) for i in np.arange(self.ndatasets)+1]
                elif self.ndelta > 1:
                    MAP_dists = [norm(self.map_pars[f'mu{i}'], 1/self.map_pars[f'delta2_{i}'] ** 0.5) for i in np.arange(self.ndatasets)+1]
            elif self.likelihood_dist == 't':
                if self.ndelta == 1:
                    map_var = 1/self.map_pars['delta2']
                    MAP_dists = [t(df=self.map_pars[f'tdf{i}'], loc=self.map_pars[f'mu{i}'], scale=np.sqrt(map_var)) for i in np.arange(self.ndatasets)+1]
                elif self.ndelta > 1:
                    map_var = [1/self.map_pars[f'delta2_{i}'] for i in np.arange(self.ndatasets)+1]
                    MAP_dists = [t(df=self.map_pars[f'tdf{i}'], loc=self.map_pars[f'mu{i}'], scale=np.sqrt(map_var[i])) for i in np.arange(self.ndatasets)+1]
            elif self.likelihood_dist == 'weibull':
                MAP_dists = [weibull_min(self.map_pars[f'rho{i}'], scale=self.map_pars[f'sig0_{i}']) for i in np.arange(self.ndatasets)+1]

            for data_idx in range(self.ndatasets):
                plt.plot(xgrid, MAP_dists[data_idx].pdf(xgrid), color=data_colors[data_idx], linewidth=2, label=f'{self.data_labels[data_idx]} MAP Dist.')
            plt.legend(fontsize=18)
            plt.savefig(f"{self.figdir}/data_hist_with_MAP_dist.png")
        else:
            plt.legend(fontsize=18)
            plt.savefig(f"{self.figdir}/data_hist.png")
        plt.close("all")

        data_mean = [np.mean(self.data[i]) for i in np.arange(self.ndatasets)]
        data_std = [np.std(self.data[i]) for i in np.arange(self.ndatasets)]

        if self.ndatasets == 2:
            print(f"sample1: mean {data_mean[0]}, std {data_std[0]}")
            print(f"sample2: mean {data_mean[1]}, std {data_std[1]}")
            print(f"Difference of sample means: {data_mean[0] - data_mean[1]}")
            print(f"Ratio of sample means: {data_mean[0] / data_mean[1]}")
            n0 = len(self.data[0])
            n1 = len(self.data[1])
            pooled_std = np.sqrt( ((n0 - 1)*data_std[0]**2 + (n1-1)*data_std[1]**2) / (n0 + n1 - 2) )
            print(f"Sample effect size: { (data_mean[0] - data_mean[1]) / pooled_std}")


    def analyse_weibull_data(self):
        """ plot histogram of data """

        print('****************************')
        print('DATA SUMMARY')
        print('****************************')
        fig, ax =plt.subplots(1, 2, figsize=(18,8))
        data_colors = ['seagreen', 'violet']
        for data_idx in range(self.ndatasets):
            sort_indices = np.argsort(self.data[data_idx])
            sorted_data = np.array(self.data[data_idx])[sort_indices]
            nmeas = len(sorted_data)
            prob_fail = np.array([(i + 1/2) / nmeas for i in np.arange(nmeas)])
            ax[0].plot(sorted_data, prob_fail, color=data_colors[data_idx], marker='.', markersize=20, label='Test Data', linestyle='None')
            ax[1].plot(np.log(sorted_data), np.log(-np.log(1-prob_fail)), color=data_colors[data_idx], marker='.', markersize=20, label='Test Data', linestyle='None')
        ax[0].set_xlabel('Strength (MPa)', fontsize=20)
        ax[1].set_xlabel('Strength (MPa)', fontsize=20)
        ax[0].set_title('Weibull Fit', fontsize=24)
        ax[1].set_title('Weibull Fit Linear Space', fontsize=24)
        ax[0].set_ylabel('Probability of Failure', fontsize=20)
        ax[1].set_ylabel('Probability of Failure', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        xlim = ax[0].get_xlim()
        xgrid = np.linspace(xlim[0], xlim[1], 100)
        for idx in range(self.ndatasets):
            rr = self.map_pars[f'rho{idx+1}']
            s0 = self.map_pars[f'sig0_{idx+1}']
            weibull_fit = 1 - np.exp(-(xgrid / s0)**rr)
            linear_weibull_fit = rr * np.log(xgrid) - rr * np.log(s0)
            ax[0].plot(xgrid, weibull_fit, color=data_colors[idx], linewidth=2, label=f'{self.data_labels[idx]} MAP Dist.')
            ax[1].plot(np.log(xgrid), linear_weibull_fit, color=data_colors[idx], linewidth=2, label=f'{self.data_labels[idx]} MAP Dist.')

        plt.legend(fontsize=18)
        plt.savefig(f"{self.figdir}/weibull_data_with_MAP_dist.png")

        for idx in range(self.ndatasets):
            post_draws = (self.sim_post[self.pars][::100])
            s0 = np.array(post_draws[f'sig0_{idx+1}'])
            rr = np.array(post_draws[f'rho{idx+1}'])
            weibull_fits = []
            linear_weibull_fits = []
            for pd_idx in range(len(post_draws)):
                  weibull_fits.append(1 - np.exp(-(xgrid / s0[pd_idx]) ** rr[pd_idx]))
                  linear_weibull_fits.append(rr[pd_idx] * np.log(xgrid) - rr[pd_idx] * np.log(s0[pd_idx]))
            weibull_fits = np.array(weibull_fits)
            linear_weibull_fits = np.array(linear_weibull_fits)
            credible_intervals = [0.01, 0.99]
            weibull_CI = self.get_quantiles(weibull_fits, credible_intervals, axis=0)
            linear_weibull_CI = self.get_quantiles(linear_weibull_fits, credible_intervals, axis=0)
            ax[0].plot(xgrid, weibull_CI.T, color=data_colors[idx], linestyle='--', linewidth=2, label=f'{self.data_labels[idx]} 99% HPI')
            ax[1].plot(np.log(xgrid), linear_weibull_CI.T, color=data_colors[idx], linestyle='--', linewidth=2, label=f'{self.data_labels[idx]} 99% HPI')
            plt.savefig(f"{self.figdir}/weibull_data_with_MAP_dist_HPI.png")

        plt.close("all")

    def perform_MCMC(self, n_post_samples=100_000, burnin=10_000,
        diagnostic_check=20_000, iter_start_diag=10_000, cov_check=2_000,
        verbosity=True, print_frequency=100, prop_var=None, par_blocks=None):
        """ Approximate the posterior distribution with an MCMC simulation """

        print('****************************')
        print('POSTERIOR SIMULATION...')
        print('****************************')
        self.mcmc = MCMC(data=self.data, priors=self.priors,
            likelihood_dist=self.likelihood_dist, par_names=self.pars,
            verbose=verbosity, display=print_frequency, figdir=self.figdir)

        self.mcmc.simulate(nsamples=n_post_samples, burnin=burnin,
            cov_check=cov_check, prop_var=prop_var,
            diagnostic_check=diagnostic_check,
            iter_start_diag=iter_start_diag, par_blocks=par_blocks)

        self.sim_post = pd.DataFrame(data=self.mcmc.trace)[burnin:]
        map_idx = np.argmax(self.sim_post['log_post'])
        self.map_pars = self.sim_post.iloc[map_idx]



    def analyse_posterior(self):

        print('****************************')
        print('POSTERIOR SUMMARY')
        print('****************************')

        labels, title_id, fig_id = self.get_fig_strings()
        par_dims = list(self.mcmc._par_dims.values())
        self.post_corr = self.sim_post[self.pars[:par_dims[0]]].corr()
        self.post_mean = self.sim_post[self.pars[:par_dims[0]]].mean()
        self.post_cov = self.sim_post[self.pars[:par_dims[0]]].cov()
        print(f"posterior correlation: {self.post_corr}")

        thin = 10
        # ellipse for 98% CI
        ellipse = False
        if par_dims[0] == 2:
            ellipse = True
            lambda_, v = np.linalg.eig(self.post_cov)
            lambda_ = np.sqrt(lambda_)
            ell_radius_x = 2*np.sqrt(5.991) * lambda_[0] # 5.991 is the chi-square value for 98% CI
            ell_radius_y = 2 * np.sqrt(5.991) * lambda_[1]
            ell_angle = np.rad2deg(np.arccos(v[0, 0]))
            ellipse = Ellipse((self.post_mean[0], self.post_mean[1]),
                width=ell_radius_x, height=ell_radius_y, angle=ell_angle,
                edgecolor='black', facecolor='none', linewidth=2)

            # density contour of first distribution parameter (mean for normal dist and sig0 for weibull)
            sns.kdeplot(x=self.sim_post[self.pars[0]], y=self.sim_post[self.pars[1]], color='gray', levels=20, linewidths=1)

            # 45 degree line
            fig, ax = plt.subplots(figsize=(12,8))
            ax.scatter(self.sim_post[self.pars[0]][::thin], self.sim_post[self.pars[1]][::thin], color='seagreen', alpha=0.4)
            ax.axline((self.post_mean[0], self.post_mean[0]), slope=1, color='black', linestyle='--')
            ax.set_xlabel(labels[0], fontsize=20)
            ax.set_ylabel(labels[1], fontsize=20)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            ax.set_title(r'Posterior Distribution of {}'.format(title_id[0]), fontsize=20)
            if ellipse:
                ax.add_patch(ellipse)
            plt.savefig(f"{self.figdir}/posterior_distribution_{fig_id[0]}.png")
            plt.close("all")

            fig, ax = plt.subplots(figsize=(12,8))
            sns.kdeplot(x=self.sim_post[self.pars[0]][::thin], y=self.sim_post[self.pars[1]][::thin], fill=True, cmap='viridis', thresh=0.05, levels=20, ax=ax)
            ax.axline((self.post_mean[0], self.post_mean[0]), slope=1, color='black', linestyle='--')
            ax.set_title('Posterior Density Plot', fontsize=20)
            ax.set_xlabel(labels[0], fontsize=20)
            ax.set_ylabel(labels[1], fontsize=20)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            plt.savefig(f"{self.figdir}/posterior_density_{fig_id[0]}.png")
            plt.close("all")

        cmap = 'viridis'
        # contour of all parameters
        fig, ax = plt.subplots(self.ndim, self.ndim, figsize=(16,16))
        thin = 50
        col_ticks = [None] * self.ndim
        for col in range(self.ndim):
            for row in range(self.ndim):
                pars = [self.pars[row], self.pars[col]] #[par_names[row], par_names[col]]
                sub_df = self.sim_post.loc[::thin, pars]
                if row == col:
                    # kde of marginal posterior
                    sns.set_style("ticks")
                    kde = sns.kdeplot(sub_df.iloc[:, 0], linewidth=4,
                        ax=ax[row,col], color='violet')
                    ticks = ax[row, col].get_xticks()
                    col_ticks[col] = [ticks[2], ticks[-3]]
                    ax[row,col].grid(False)
                    ax[row,col].set_facecolor('white')
                    ax[row, col].set_ylabel('')
                    ax[row, col].set_xlabel('')
                    ax[row, col].set_yticks([])
                    ax[row, col].set_xticks([])
                    ax[row, col].set_ylim(bottom=0)
                    ax[row, col].tick_params(axis='both', labelsize=22)

                if row > col:

                # kde of joint-posterior
                    sns.kdeplot(data=sub_df, levels=50,
                        fill=False, cmap=cmap, x=pars[1] , y=pars[0],
                        ax=ax[row,col])
                    ax[row,col].grid(False)
                    ax[row,col].set_facecolor('white')
                    ax[row, col].set_xlabel('')
                    ax[row, col].set_ylabel('')
                    ax[row, col].set_yticks([])
                    ax[row, col].set_xticks([])
                    ax[row, col].tick_params(axis='both', labelsize=22)
                    for key, spine in ax[row, col].spines.items():
                        spine.set_visible(True)

                if row == self.ndim-1:
                    ax[row, col].set_xlabel(self.pars[col], fontsize=24, fontweight='bold')
                    ax[row, col].set_xticks(col_ticks[col])
                if row < col:
                    fig.delaxes(ax[row, col])
                if col == 0:
                    ax[row, col].set_ylabel(self.pars[row], fontsize=24, fontweight='bold')
        for row in range(1, self.ndim):
            ax[row, 0].set_yticks(col_ticks[row])
        plt.savefig(f"{self.figdir}/posterior_contours.png")
        plt.close("all")

        if par_dims[0] == 2:
            self.analyse_post_diff()
            self.analyse_post_ratio()
            self.analyse_post_effect_size()

        if self.likelihood_dist == 'weibull':
            self.analyse_weibull_data()
        else:
            self.analyse_data(MAP_post_dist=True)

    def analyse_post_diff(self):

        labels, title_id, fig_id = self.get_fig_strings()

        n_post_samples = len(self.sim_post)
        self.sim_post[f'{fig_id[0]}_diff'] = self.sim_post[self.pars[0]] - self.sim_post[self.pars[1]]
        self.post_prob_diff = sum(self.sim_post[f'{fig_id[0]}_diff'] > 0) / n_post_samples
        print(f"post probability {fig_id[0]}1 > {fig_id[0]}m2: {self.post_prob_diff}")

        # histogram of posterior mean difference
        xlabel = "{} - {} (MPa)".format(labels[0], labels[1])
        figtitle = "Posterior over the difference"
        figname = "posterior_{}_{}_difference".format(self.pars[0], self.pars[1])
        self.plot_hist(self.sim_post[f'{fig_id[0]}_diff'], xlabel, figtitle, figname, credible_intervals=True, kde=False)

        credible_intervals = [0.01, 0.10, 0.25, 0.75, 0.90, 0.99]
        post_diff_CI = self.get_quantiles(self.sim_post[f'{fig_id[0]}_diff'], credible_intervals)
        print("Posterior difference quantiles: {}".format(post_diff_CI))
        print(f"Mean posterior difference: {self.sim_post[f'{fig_id[0]}_diff'].mean()}")

    def analyse_post_ratio(self):

        labels, title_id, fig_id = self.get_fig_strings()
        n_post_samples = len(self.sim_post)
        self.sim_post[f'{fig_id[0]}_ratio'] = self.sim_post[self.pars[0]] / self.sim_post[self.pars[1]]
        self.post_prob_ratio = sum(self.sim_post[f'{fig_id[0]}_ratio'] > 1) / n_post_samples
        print(f"post probability {fig_id[0]}1 / {fig_id[0]}2 > 1: {self.post_prob_ratio}")

        # histogram of posterior mean ratio
        xlabel = "{} / {}".format(labels[0], labels[1])
        figtitle = "Posterior over the ratio"
        figname = "posterior_{}_{}_ratio".format(self.pars[0], self.pars[1])
        self.plot_hist(self.sim_post[f'{fig_id[0]}_ratio'], xlabel, figtitle, figname, credible_intervals=True, kde=False)

        credible_intervals = [0.01, 0.10, 0.25, 0.75, 0.90, 0.99]
        post_ratio_CI = self.get_quantiles(self.sim_post[f'{fig_id[0]}_ratio'], credible_intervals)
        print("Posterior ratio quantiles: {}".format(post_ratio_CI))
        print(f"Mean posterior ratio: {self.sim_post[f'{fig_id[0]}_ratio'].mean()}")


    def analyse_post_effect_size(self):

        labels, title_id, fig_id = self.get_fig_strings()

        if self.likelihood_dist == 'normal' or self.likelihood_dist == 't':
            if self.ndelta == 1:
                self.sim_post['effect_size'] = self.sim_post[f'{fig_id[0]}_diff'] / np.sqrt(1 / self.sim_post['delta2'])
            elif self.ndelta == 2:
                self.sim_post['pooled_var'] = ((1/self.sim_post['delta2_1']) + (1/self.sim_post['delta2_2'])) / 2
                self.sim_post['effect_size'] = self.sim_post['mu_diff'] / np.sqrt(self.sim_post['pooled_var'])
        elif self.likelihood_dist == 'weibull':
            par_var = self.sim_post[self.pars[:2]].var(axis=0)
            self.sim_post['pooled_var'] = ((1/par_var[self.pars[0]]) + (1/par_var[self.pars[1]])) / 2 
            self.sim_post['effect_size'] = self.sim_post[f'{fig_id[0]}_diff'] / np.sqrt(1 / self.sim_post['pooled_var'])

        # histogram of posterior mean difference effect size
        xlabel = "({} - {}) / {}".format(labels[0], labels[1], r"$\sigma$")
        figtitle = "Posterior over the effect size"
        figname = "posterior_over_effect_size".format(self.pars[0], self.pars[1])
        self.plot_hist(self.sim_post['effect_size'], xlabel, figtitle, figname, credible_intervals=True, kde=False)

        credible_intervals = [0.01, 0.10, 0.25, 0.75, 0.90, 0.99]
        post_effect_size_CI = self.get_quantiles(self.sim_post['effect_size'], credible_intervals)
        print("Posterior effect size quantiles: {}".format(post_effect_size_CI))
        print(f"Mean posterior effect size: {self.sim_post['effect_size'].mean()}")
