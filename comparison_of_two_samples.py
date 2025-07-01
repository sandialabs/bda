import numpy as np
from scipy.stats import norm, gamma, multivariate_normal
from bayes_factor import BayesFactor
import bayesian_data_analysis
from bayesian_data_analysis import BDA
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import sys, os, csv
import pickle
from matplotlib import ticker, colormaps
import seaborn as sns

"""
Bayesian Reasoning and methods
Ch. 16: Comparing Two Samples
Kevin Davisross
"""

def create_dual_contours(bda):
    cmaps = [colormaps['viridis'], colormaps['plasma']]
    colors = ['seagreen', 'blueviolet']
    alphas = [0.4, 0.4]
    thin = 50
    # contour of all parameters
    fig, ax = plt.subplots(bda[0].ndim, bda[0].ndim, figsize=(16,16))
    col_ticks = [None] * 4
    for idx, bayes_analysis in enumerate(bda):
        df = bayes_analysis.sim_post[bayes_analysis.pars][::thin]
        if bayes_analysis.ndim == 2:
            cols_orig = df.columns.tolist()
            cols = cols_orig * 2
            cols = cols[0::2] + cols[1::2]
            df = df[cols]
            df.columns = [f"{col}_{i+1}" for col in cols_orig for i in range(2)]
        ndim = df.shape[1]
        all_pars = df.columns.tolist()
        if idx == 0:
            xlabels = all_pars.copy()
        for col in range(ndim):
            for row in range(ndim):
                pars = [all_pars[row], all_pars[col]] #[bayes_analysis.pars[row], bayes_analysis.pars[col]] #[par_names[row], par_names[col]]
                sub_df = df[pars] #bayes_analysis.sim_post.loc[::thin, pars]
                if row == col:
                    # kde of marginal posterior
                    sns.set_style("ticks")
                    kde = sns.kdeplot(sub_df.iloc[:, 0], linewidth=4,
                        ax=ax[row,col], color=colors[idx])
                    if idx == 0:
                        ticks = ax[row, col].get_xticks()
                        col_ticks[col] = [ticks[2], ticks[-3]]
                    ax[row,col].grid(False)
                    ax[row,col].set_facecolor('white')
                    ax[row, col].set_ylabel('')
                    ax[row, col].set_xlabel('')
                    ax[row, col].set_xticks([])
                    ax[row, col].set_yticks([])
                    ax[row, col].set_ylim(bottom=0)
                    ax[row, col].tick_params(axis='both', labelsize=22)

                if row > col:

                # kde of joint-posterior
                    sns.kdeplot(data=sub_df, levels=50,
                        fill=False, cmap=cmaps[idx], alpha=alphas[idx], x=pars[1] , y=pars[0],
                        ax=ax[row,col])
                    ax[row,col].grid(False)
                    ax[row,col].set_facecolor('white')
                    ax[row, col].set_xlabel('')
                    ax[row, col].set_ylabel('')
                    ax[row, col].set_xticks([])
                    ax[row, col].set_yticks([])
                    ax[row, col].tick_params(axis='both', labelsize=22)
                    for key, spine in ax[row, col].spines.items():
                        spine.set_visible(True)

                if row == ndim-1:
                    ax[row, col].set_xlabel(xlabels[col], fontsize=24, fontweight='bold')
                    ax[row, col].set_xticks(col_ticks[col])
                if row < col and idx == 0:
                    fig.delaxes(ax[row, col])
                if col == 0:
                    ax[row, col].set_ylabel(xlabels[row], fontsize=24, fontweight='bold')

        for row in range(1, ndim):
            ax[row, 0].set_yticks(col_ticks[row])

    plt.savefig(f"figures/{runID}_dual_posterior_contours.png")
    plt.close("all")

def plot_dist(dist, figdir):
    """ plot a distribution over values defined by x"""
    x = np.linspace(0, 1, 100)
    plt.figure()
    plt.plot(x, dist.pdf(x))
    plt.savefig(f"{figdir}/dist.png")
    plt.close()

    plt.figure()
    plt.plot(x, dist.logpdf(x))
    plt.savefig(f"{figdir}/log_dist.png")
    plt.close()

def define_gamma_prior(shape=None, rate=None, mean=None, var=None):

    if shape is not None:
        assert rate is not None
        return gamma(shape, scale=1/rate)
    elif mean is not None:
        assert var is not None
        shape = mean ** 2 / var
        rate = mean / var
        return gamma(shape, scale=1/rate)


def build_prior(prior_mean, prior_std, prior_corr_level):
    """ build prior from mean and standard deviation """

    ndim = len(prior_std)
    prior_corr = np.ones((ndim, ndim)) * prior_corr_level
    np.fill_diagonal(prior_corr, 1)
    prior_cov_mat = create_covariance_matrix(prior_corr, prior_std)
    prior = multivariate_normal(mean=prior_mean, cov=prior_cov_mat)

    return prior


def create_covariance_matrix(corr, std):
    """ create covariance matrix from correleation and standard
        deviation
    """
    ndim = len(std)
    cov_mat = np.zeros((ndim, ndim))

    for i in np.arange(ndim):
        for j in np.arange(ndim):
            cov_mat[i, j] = corr[i, j] * std[i] * std[j]

    return cov_mat


def perform_data_analysis(data, priors, likelihood_dist, data_labels,
    n_post_samples=100_000, burnin=10_000, diagnostic_check=20_000, iter_start_diag=10_000,
    cov_check=2_000, pars=None, figdir=None, verbosity=True, print_frequency=1000,
    par_blocks=None):
    """ compare the means of two samples, mu1 and mu2, and assume a common standard deviation delta"""


    bda = BDA(data, priors, likelihood_dist, data_labels, pars, figdir)
    if len(pars) == 4:
        bda.analyse_prior()
    bda.analyse_data()
    bda.perform_MCMC(n_post_samples=n_post_samples, burnin=burnin,
        diagnostic_check=diagnostic_check, iter_start_diag=iter_start_diag,
        cov_check=cov_check, verbosity=verbosity, print_frequency=print_frequency,
        par_blocks=par_blocks)
    bda.analyse_posterior()
    return bda


if __name__ == '__main__':

    """ two input args required: file names to be imported
        e.g. 'horizontal', 'vertical', 'AlN_dev', 'AlN_sorted' """

    np.random.seed(1)
    plt.close("all")

    n_post_samples = 120_000
    burnin = 20_000
    diagnostic_check = 20_000
    iter_start_diag = 20_000
    cov_check = 2_000
    verbosity = True
    print_frequency = 5_000
    par_blocks = [[0, 1, 2, 3]]
    likelihood_dist = 'weibull' # options: 'weibull', 't', 'normal'
    data_dir = "data"
    data_labels = sys.argv[1:]
    unique_variance = True

    if 'horizontal' in sys.argv:
        runID = "horiz_vert"
        figdir = f"figures/{runID}"
    elif 'AlN_sorted' in sys.argv:
        runID = "AlN"
        figdir = f"figures/{runID}"
    else:
        raise ValueError('Filename not recognized. Make sure two files specified for data comparison.')

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    nfiles = len(sys.argv) - 1
    data = [None] * nfiles
    for idx, arg in enumerate(sys.argv[1:]):
        filename = f"{data_dir}/{arg}.dat"
        with open(filename) as f:
            reader = csv.reader(f, delimiter=' ')
            sample_data = list(reader) 
            data[idx] = [float(item[0]) for item in sample_data]

    if unique_variance:
        ndelta = nfiles
    else:
        ndelta = 1

    # assume prior dependence between the means and
    # independence of the data variance

    # prior on the precision of the distribution
    if likelihood_dist == 't' or likelihood_dist == 'normal':
        # prior on distribution variance
        delta2_mean1 = .001
        delta2_var1 = 1000
        delta2_prior1 = define_gamma_prior(mean=delta2_mean1, var=delta2_var1)

        delta2_mean2 = .001
        delta2_var2 = 1000
        delta2_prior2 = define_gamma_prior(mean=delta2_mean2, var=delta2_var2)
        plot_dist(delta2_prior2, figdir)

        delta2_prior = [delta2_prior1, delta2_prior2]
        #delta2_prior = [delta2_prior1]

        # prior on distribution mean
        mu_corr_level = 0.9
        mu_prior_mean = np.array([290, 290])
        mu_prior_sd = np.array([10, 10])
        mu_prior = build_prior(mu_prior_mean, mu_prior_sd, mu_corr_level)

        priors = [mu_prior, delta2_prior]
        pars = ['mu1', 'mu2', 'delta2_1', 'delta2_2']

        if likelihood_dist == 't':
            # prior on distribution degrees of freedom
            tdf_mean = 15
            tdf_std = 25
            tdf_prior = [norm(20, 10), norm(10, 10)]
            tdf_prior = [norm(15, 15), norm(20, 20)]
            #df = [30, 2] # used for AlN
            #df = [20, 20] # used for horiz/vert
            priors = [mu_prior, delta2_prior, tdf_prior]
            pars = ['mu1', 'mu2', 'delta2_1', 'delta2_2', 'tdf1', 'tdf2']

    elif likelihood_dist == 'weibull':

        if False:
            bda = [None] * 2
            ######################################
            ### ASSUME: two data distributions ###
            ######################################

            figdir = f"figures/{runID}_2dist"
            if not os.path.exists(figdir):
                os.makedirs(figdir)
            # build prior on distribution characteristic strenght
            sig0_corr_level = 0.9
            sig0_prior_mean = np.array([500, 500])
            sig0_prior_sd = np.array([500, 500])
            sig0_prior = build_prior(sig0_prior_mean, sig0_prior_sd, sig0_corr_level)

            # build prior on distribution Weibull modulus
            rho_corr_level = 0.9
            rho_prior_mean = np.array([15, 15])
            rho_prior_sd = np.array([15, 15])
            rho_prior = build_prior(rho_prior_mean, rho_prior_sd, rho_corr_level)
            priors = [sig0_prior, rho_prior]
            pars = ['sig0_1', 'sig0_2', 'rho1', 'rho2']
            par_blocks = [[0, 1, 2, 3]]

            bda[0] = perform_data_analysis(data, priors, likelihood_dist, data_labels,
                n_post_samples=n_post_samples, burnin=burnin, diagnostic_check=diagnostic_check,
                iter_start_diag=iter_start_diag, cov_check=cov_check,
                pars=pars, figdir=figdir, verbosity=verbosity, print_frequency=print_frequency,
                par_blocks=par_blocks)

            ###################################
            ### ASSUME: 1 data distribution ###
            ###################################

            figdir = f"figures/{runID}_1dist"
            if not os.path.exists(figdir):
                os.makedirs(figdir)
            # build prior on distribution characteristic strenght
            sig0_prior_mean = 500
            sig0_prior_sd = 500
            sig0_prior = norm(sig0_prior_mean, sig0_prior_sd)

            # build prior on distribution Weibull modulus
            rho_prior_mean = 15
            rho_prior_sd = 15
            rho_prior = norm(rho_prior_mean, rho_prior_sd)
            priors = [sig0_prior, rho_prior]

            pars = ['sig0_1', 'rho1']
            par_blocks = [[0, 1]]

            data = [data[0] + data[1]]

            bda[1] = perform_data_analysis(data, priors, likelihood_dist, data_labels,
                n_post_samples=n_post_samples, burnin=burnin, diagnostic_check=diagnostic_check,
                iter_start_diag=iter_start_diag, cov_check=cov_check,
                pars=pars, figdir=figdir, verbosity=verbosity, print_frequency=print_frequency,
                par_blocks=par_blocks)

            with open(f'figures/{runID}_bda.pkl', 'wb') as f:
                pickle.dump(bda, f)

        bounds = [[0, np.inf], [0, np.inf]]
        with open(f'figures/{runID}_bda.pkl', 'rb') as f:
            bda = pickle.load(f)
        # estimator options: 'harmonic', 'monte_carlo_integration',
        # 'truncated_harmonic', 'retargeted_harmonic', 'kde_harmonic',  -- Laplace soon
        if True:
            methods = ['monte_carlo_integration', 'harmonic', 'truncated_harmonic', 'retargeted_harmonic', 'kde_harmonic']
            bf = {}
            bayes_fac = {}
            for method in methods:
                print(f"Calculating marginal likelihood via {method} method.")
                bf = BayesFactor(bda, estimator=method, bounds=bounds, figdir=figdir, thin=100)
                if method is None:
                    continue
                bf.calculate_marginal_likelihood()
                bayes_fac[method] = bf.z_estimate[0]/bf.z_estimate[1]
                print(f"Bayes factor from {method} method: {bayes_fac[method]}")
            #bf.perform_likelihood_analysis()
        #with open(f"{figdir}/{runID}_bayes_factor.pkl", 'wb') as f:
        #    pickle.dump([bf, bayes_fac], f)
        #create_dual_contours(bda)



