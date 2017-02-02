'''
Bayesian models for TWAS.

Author: Kunal Bhutani <kunalbhutani@gmail.com>
'''

from scipy.stats import norm
import pymc3 as pm
import numpy as np
from theano import shared
from scipy.stats.distributions import pareto
from scipy import optimize
import theano.tensor as t


def tinvlogit(x):
    return t.exp(x) / (1 + t.exp(x))


def calculate_waic(trace, model=None, r_logp=True):
    """
    Taken directly from PyMC3.
    Reproduced to only take into account the phenotype and not mediator
    variable when calculating logp.

    Calculate the widely available information criterion and the effective
    number of parameters of the samples in trace from model.

    Read more theory here - in a paper by some of the
    leading authorities on Model Selection - http://bit.ly/1W2YJ7c
    """
    log_py = log_post_trace(trace, model)
    lppd = np.sum(np.log(np.mean(np.exp(log_py), axis=0)))
    p_waic = np.sum(np.var(log_py, axis=0))
    if r_logp:
        return -2 * lppd + 2 * p_waic, log_py, lppd
    else:
        return -2 * lppd + 2 * p_waic


def calculate_loo(trace=None, model=None, log_py=None):
    """
    Taken directly from PyMC3.
    Reproduced to only take into account the phenotype and not mediator
    variable when calculating logp.

    Calculates leave-one-out (LOO) cross-validation for out of sample
    predictive model fit, following Vehtari et al. (2015).
    Cross-validation is computed using Pareto-smoothed importance sampling.

    Returns log pointwise predictive density calculated via
    approximated LOO cross-validation.
    """
    if log_py is None:
        log_py = log_post_trace(trace, model)
    # Importance ratios
    r = 1. / np.exp(log_py)
    r_sorted = np.sort(r, axis=0)

    # Extract largest 20% of importance ratios and
    # fit generalized Pareto to each
    # (returns tuple with shape, location, scale)
    q80 = int(len(log_py) * 0.8)
    pareto_fit = np.apply_along_axis(lambda x: pareto.fit(x, floc=0),
                                     0, r_sorted[q80:])
    # Calculate expected values of the order statistics of the fitted Pareto
    S = len(r_sorted)
    M = S - q80
    z = (np.arange(M) + 0.5) / M
    expvals = map(lambda x: pareto.ppf(z, x[0], scale=x[2]), pareto_fit.T)

    # Replace importance ratios with order statistics of fitted Pareto
    r_sorted[q80:] = np.vstack(expvals).T
    # Unsort ratios (within columns) before using them as weights
    r_new = np.array([x[np.argsort(i)]
                     for x, i in zip(r_sorted,
                                     np.argsort(r, axis=0))])

    # Truncate weights to guarantee finite variance
    w = np.minimum(r_new, r_new.mean(axis=0) * S**0.75)
    loo_lppd = np.sum(np.log(np.sum(w * np.exp(log_py), axis=0) / np.sum(w, axis=0)))

    return loo_lppd


def log_post_trace(trace, model):
    '''
    Taken directly from PyMC3.
    Reproduced to only take into account the phenotype and not mediator
    variable when calculating logp.

    Calculate the elementwise log-posterior for the sampled trace.
    '''
    logp = np.hstack([obs.logp_elemwise(pt) for pt in trace]
                     for obs in model.observed_RVs if obs.__repr__() == 'phen')
    if len(logp.shape) > 2:
        logp = logp.squeeze(axis=1)
    return logp


class BayesianModel(object):
    '''
    General Bayesian Model Class for quantifying
    relationship between gene and phenotype

    Adapted from Thomas Wiecki
    https://github.com/pymc-devs/pymc3/issues/511#issuecomment-125935523

    '''

    def __init__(self, variational=True, mb=False,
                 n_chain=50000, n_trace=5000,
                 logistic=False, steps=None):
        """
        Args:
            variational (bool, optional): Use Variational Inference
            mb (bool, optional): Use minibatches
        """
        self.variational = variational
        self.cached_model = None
        self.mb = mb
        self.n_chain = n_chain
        self.n_trace = n_trace
        self.logistic = logistic
        self.steps = steps


    def cache_model(self, **inputs):
        """
        Create a cached model for the Bayesian model using
        shared theano variables for each Bayesian
        input parameter.

        Args:
            **inputs (dict): inputs for Bayesian model

        """
        self.shared_vars = self._create_shared_vars(**inputs)
        self.cached_model = self.create_model(**self.shared_vars)

    def create_model(self, **inputs):
        """
        Each instance of this class needs to define
        their PYMC3 model in here.
        """
        raise NotImplementedError('This method has to be overwritten.')

    def _create_shared_vars(self, **inputs):
        """
        For each input variable, create theano shared variable
        and set their initial values.

        Args:
            **inputs (dict): inputs for Bayesian model

        Returns:
            dict: key, value - var_name, theano.shared variable
        """
        shared_vars = {}
        for name, data in inputs.items():
            shared_vars[name] = shared(data, name=name)
        return shared_vars

    def _clean_inputs(self, inputs):
        """
        Clean the inputs, i.e. remove some
        genotype columns. Useful for some class of Bayesian models
        such as Two-Stage, where first stage involves filtering
        on certain SNPs.

        Args:
            inputs (dict): inputs for Bayesian model
        Returns:
            dict: cleaned inputs for Bayesian model
        """
        return inputs

    def run(self, **inputs):
        """
        Run cached Bayesian model using the inputs

        Args:
            **inputs (dict): inputs for Bayesian model

        Returns:
            trace: Trace of the PyMC3 inference
        """
        if self.cached_model is None:
            self.cache_model(**inputs)
        for name, data in inputs.items():
            self.shared_vars[name].set_value(data)
        if self.mb and self.variational:
            self.minibatches = zip(self._mb_generator(inputs['gwas_gen']),
                                   self._mb_generator(inputs['gwas_phen']))
        self.trace = self._inference()
        return self.trace

    def _inference(self, n_trace=None):
        """
        Perform the inference. Uses ADVI if self.variational
        is True. Also, uses minibatches is self.mb=True based
        on generators defined in self.run.

        Otherwise, uses Metropolis.

        Args:
            n_trace (int, optional): Number of steps used for trace
        Returns:
            trace: Trace of the PyMC3 inference
        """
        if n_trace is None:
            n_trace = self.n_trace

        print(n_trace)
        with self.cached_model:
            if self.variational:
                if self.mb:
                    v_params = pm.variational.advi_minibatch(n=self.n_chain,
                               minibatch_tensors=self.minibatch_tensors,
                               minibatch_RVs=self.minibatch_RVs,
                               minibatches=self.minibatches,)
                else:
                    v_params = pm.variational.advi(n=self.n_chain)
                trace = pm.variational.sample_vp(v_params, draws=n_trace)
                self.v_params = v_params
            else:
                if self.steps is None:
                    self.steps = pm.Metropolis()
                start = pm.find_MAP(fmin=optimize.fmin_powell)
                trace = pm.sample(self.n_chain,
                                  step=self.steps,
                                  start=start,
                                  progressbar=True,
                                  )
                trace = trace[-n_trace:]
        self.trace = trace
        return trace

    def cross_validation(self, k_folds, **inputs):
        """
        Run cross-validation on the inputs and calculate
        statistics for each fold test set.

        Args:
            k_folds (sklearn.cross_validation): Folds of test and train
                                                samples
            **inputs (dict): inputs for Bayesian model

        Returns:
            dict: statistics for each fold
        """
        self.cv_stats, self.cv_traces = [], []
        self.k_folds = k_folds
        inputs = self._clean_inputs(inputs)
        for i, fold in enumerate(k_folds):
            train, test = fold
            input_train, input_test = {}, {}
            for name, data in inputs.items():
                if name in self.cv_vars:
                    input_train[name] = data[train]
                    input_test[name] = data[test]
                else:
                    input_train[name] = data
                    input_test[name] = data
            trace = self.run(**input_train)
            stats = self.calculate_statistics(trace, **input_test)
            self.cv_traces.append(trace)
            self.cv_stats.append(stats)
        return self.cv_traces, self.cv_stats

    def calculate_ppc(self, trace):
        """
        Calculate several post-predictive checks
        based on the trace.
        """
        dic = pm.stats.dic(trace, self.cached_model)
        waic, log_py, logp = calculate_waic(trace, self.cached_model)
        #loo = calculate_loo(log_py=log_py)
        mu, sd, zscore = self._alpha_stats(trace)
        return {'dic': dic,
                'waic': waic,
                'logp': logp,
                #'loo': loo,
                'mu': mu,
                'sd': sd,
                'zscore': zscore}

    def calculate_statistics(self, trace, **input_test):
        """
        Calculate mse and logp statistics on a test set.

        Args:
            **input_test (dict): test set of inputs
            trace (PyMC3.trace): Trace of the inference chain
        Returns:
            dict: logp and mse
        """
        inputs = self._clean_inputs(input_test)
        mc_logp = self._logp(trace, **inputs)
        mean_mse = self._mse(trace, **inputs)
        mse2 = self._mse2(trace, **inputs)
        mu, sd, zscore = self._alpha_stats(trace)
        return {'logp': mc_logp,
                'mse': mean_mse,
                'mse2': mse2,
                'mu': mu,
                'sd': sd,
                'zscore': zscore}

    def calculate_bf(self, trace, var_name='mediator_model'):
        '''
        Calculate Bayes Factor using a Bernoulli variable in the 
        trace.
        '''
        p_alt = trace[var_name].mean()
        bayes_factor = (p_alt/(1-p_alt))
        return bayes_factor


    def _logp(self, trace, **inputs):
        """
        Calculate log likelihood using Monte Carlo integration.

        Args:
            **inputs (dict): inputs used in likelhood calculation
            trace (PyMC3.trace): Trace of the inference chain
        Returns:
            float: Log likelihood as estimated by Monte Carlo integration
        """
        def calc_log(step):
            exp_pred = np.dot(inputs['gwas_gen'],
                              step['beta_med'].T).ravel()
            phen_pred = step['alpha'] * exp_pred
            phen_prob = norm.logpdf(x=inputs['gwas_phen'],
                                    loc=phen_pred,
                                    scale=step['phenotype_sigma'])
            return phen_prob

        phen_probs = [calc_log(trace[idx])
                      for idx in np.random.randint(0, len(self.trace), 500)]
        phen_probs = np.asmatrix(phen_probs)
        mc_logp = phen_probs.sum(axis=1).mean()
        return mc_logp

    def _mse(self, trace, **inputs):
        """
        Calculate mean squared error of the model fit.
        Args:
            **inputs (dict): inputs used in likelhood calculation
            trace (PyMC3.trace): Trace of the inference chain

        Returns:
            float: Mean squared error across all samples
        """
        phen_mse = []
        for idx in np.random.randint(0, len(trace), 500):
            step = self.trace[idx]
            exp_pred = np.dot(inputs['gwas_gen'],
                              step['beta_med'].T).ravel()
            phen_pred = step['alpha'] * exp_pred
            phen_mse = np.mean((inputs['gwas_phen'] - phen_pred) ** 2)
        mean_mse = np.mean(phen_mse)
        return mean_mse

    def _mse2(self, trace, **inputs):
        """
        Calculate mean squared error of the model fit 
        using posterior means of beta_med instead of
        sampling from it.

        Args:
            **inputs (dict): inputs used in likelhood calculation
            trace (PyMC3.trace): Trace of the inference chain

        Returns:
            float: Mean squared error across all samples
        """
        exp = np.dot(inputs['gwas_gen'],
                     trace['beta_med'].mean(axis=0).T)
        phen_pred = exp * trace['alpha'].mean()
        mse = np.mean((inputs['gwas_phen'] - phen_pred) ** 2)
        return mse

    def _alpha_stats(self, trace):
        """
        Calculate statistics of the alpha value in
        the trace.
        """
        mean = np.mean(trace['alpha'])
        sd = np.std(trace['alpha'], ddof=1)
        zscore = mean / sd
        return mean, sd, zscore

    def _mb_generator(self, data, size=500):
        """
        Generator for minibatches
        """
        rng = np.random.RandomState(0)
        while True:
            ixs = rng.randint(len(data), size=size)
            yield data[ixs]


class TwoStage(BayesianModel):
    """
    Two Stage Inference.

    First stage: Bootstrapped ElasticNet
    Second stage: Use loci that were learned in the first stage
                  and their mean and std as priors for a simple
                  Bayesian Linear Regression

    Attributes:

    """
    def __init__(self, coef_mean, coef_sd, p_sigma_beta=10,
                 *args, **kwargs):
        """
        Args:

        """
        self.name = 'TwoStage'
        self.cv_vars = ['gwas_phen', 'gwas_gen']
        self.vars = {'coef_mean': coef_mean,
                     'coef_sd': coef_sd,
                     'p_sigma_beta': p_sigma_beta}
        super(TwoStage, self).__init__(*args, **kwargs)

    def create_model(self, gwas_gen, gwas_phen):
        """
        Simple Bayesian Linear Regression

        Args:
            gwas_gen (pandas.DataFrame): GWAS genotypes
            gwas_phen (pandas.DataFrame): GWAS phenotypes

        Returns:
            pymc3.Model(): The Bayesian model
        """
        n_ind, n_snps = gwas_gen.eval().shape
        with pm.Model() as phenotype_model:
            beta_med = pm.Normal('beta_med',
                                 mu=self.vars['coef_mean'],
                                 sd=self.vars['coef_sd'],
                                 shape=(1, n_snps))
            phenotype_expression_mu = pm.dot(beta_med, gwas_gen.T)
            intercept = pm.Normal('intercept', mu=0, sd=1)
            alpha = pm.Normal('alpha', mu=0, sd=1)
            phenotype_mu = intercept + alpha * phenotype_expression_mu
            if self.logistic:
                p = tinvlogit(phenotype_mu)
                phen = pm.Bernoulli('phen', p=p, observed=gwas_phen)
            else:
                phenotype_sigma = pm.HalfCauchy('phenotype_sigma',
                                beta=self.vars['p_sigma_beta'])
                phen = pm.Normal('phen',
                                 mu=phenotype_mu,
                                 sd=phenotype_sigma,
                                 observed=gwas_phen)
        if self.variational and self.mb:
            self.minibatch_RVs = [phen]
            self.minibatch_tensors = [gwas_gen, gwas_phen]
        return phenotype_model

class TwoStageBF(BayesianModel):
    """
    Two Stage Inference.

    First stage: Bootstrapped ElasticNet
    Second stage: Use loci that were learned in the first stage
                  and their mean and std as priors for a simple
                  Bayesian Linear Regression

    Attributes:

    """
    def __init__(self, coef_mean, coef_sd, p_sigma_beta=10,
                 *args, **kwargs):
        """
        Args:

        """
        self.name = 'TwoStageBF'
        self.cv_vars = ['gwas_phen', 'gwas_gen']
        self.vars = {'coef_mean': coef_mean,
                     'coef_sd': coef_sd,
                     'p_sigma_beta': p_sigma_beta}
        super(TwoStageBF, self).__init__(*args, **kwargs)

    def create_model(self, gwas_gen, gwas_phen):
        """
        Simple Bayesian Linear Regression

        Args:
            gwas_gen (pandas.DataFrame): GWAS genotypes
            gwas_phen (pandas.DataFrame): GWAS phenotypes

        Returns:
            pymc3.Model(): The Bayesian model
        """
        n_ind, n_snps = gwas_gen.eval().shape
        with pm.Model() as phenotype_model:
            beta_med = pm.Normal('beta_med',
                                 mu=self.vars['coef_mean'],
                                 sd=self.vars['coef_sd'],
                                 shape=(1, n_snps))
            
            mediator = pm.dot(beta_med, gwas_gen.T)
            intercept = pm.Normal('intercept', mu=0, sd=1)
            alpha = pm.Normal('alpha', mu=0, sd=1)
            phenotype_sigma = pm.HalfCauchy('phenotype_sigma',
                                beta=self.vars['p_sigma_beta'])
            

            # Model Selection
            p = np.array([0.5, 0.5])
            mediator_model = pm.Bernoulli('mediator_model', p[1])

            # Model 1
            phenotype_mu_null = intercept

            # Model 2
            phenotype_mu_mediator = intercept + alpha * mediator

            phen = pm.DensityDist('phen',
                                lambda value: pm.switch(mediator_model, 
                                    pm.Normal.dist(mu=phenotype_mu_mediator, sd=phenotype_sigma).logp(value), 
                                    pm.Normal.dist(mu=phenotype_mu_null, sd=phenotype_sigma).logp(value)
                                ),
                                observed=gwas_phen)
            self.steps = [pm.BinaryGibbsMetropolis(vars=[mediator_model]),
                          pm.Metropolis()]

            
        if self.variational and self.mb:
            self.minibatch_RVs = [phen]
            self.minibatch_tensors = [gwas_gen, gwas_phen]
        return phenotype_model



class Joint(BayesianModel):
    """
    Jointly model the transcriptional regulation and
    its effect on the phenotype.

    """
    def __init__(self, model_type='laplace', coef_sd=None, coef_mean=None,
                 tau_beta=1, lambda_beta=1, m_sigma_beta=10,
                 p_sigma_beta=10, *args, **kwargs):
        """
            Expression ~ N(X\beta, \sigma_exp)
            P(\beta) ~ Horseshoe (tau_beta, lambda_beta)
            P(\sigma_exp) ~ HalfCauchy(m_sigma_beta)
            Phenotype ~ N(X\beta\alpha, \sigma_phen)
            P(\alpha) ~ Uniform(-10, 10)
            P(\sigma_phen) ~ HalfCauchy(p_sigma_beta)
        Args:
            tau_beta (int): P(\beta) ~ Horseshoe (tau_beta, lambda_beta)
            lambda_beta (int): P(\beta) ~ Horseshoe (tau_beta, lambda_beta)
            m_sigma_beta (int): P(\sigma_exp) ~ HalfCauchy(m_sigma_beta)
            p_sigma_beta (int): P(\sigma_phen) ~ HalfCauchy(p_sigma_beta)

        """
        self.name = 'Joint'
        self.model_type = model_type
        self.cv_vars = ['gwas_phen', 'gwas_gen']
        self.vars = {'coef_mean': coef_mean,
                     'coef_sd': coef_sd,
                     'tau_beta': tau_beta,
                     'lambda_beta': lambda_beta,
                     'm_sigma_beta': m_sigma_beta,
                     'p_sigma_beta': p_sigma_beta
                     }
        if model_type == 'laplace':
            self.create_model = self._create_model_laplace
        elif model_type == 'horseshoe':
            self.create_model = self._create_model_horseshoe
        elif model_type == 'prior':
            # assert((coef_sd is not None) and (coef_mean is not None),
            #        'Must provided coef_mean and coef_sd if using prior')
            self.create_model = self._create_model_prior
        else:
            raise NotImplementedError('Unsupported model type')
        super(Joint, self).__init__(*args, **kwargs)

    def _create_model_prior(self, med_gen, med_phen,
                            gwas_gen, gwas_phen):
        """
        Args:
            med_gen (pandas.DataFrame): Mediator genotypes
            med_phen (pandas.DataFrame): Mediator phenotypes
            gwas_gen (pandas.DataFrame): GWAS genotypes
            gwas_phen (pandas.DataFrame): GWAS phenotypes

        """
        n_snps = gwas_gen.eval().shape[1]
        with pm.Model() as phenotype_model:
            # Expression
            beta_med = pm.Normal('beta_med',
                                 mu=self.vars['coef_mean'],
                                 sd=self.vars['coef_sd'],
                                 shape=(1, n_snps))
            mediator_intercept = pm.Normal('mediator_intercept',
                                           mu=0,
                                           sd=1)
            mediator_mu = mediator_intercept + pm.dot(beta_med, med_gen.T)
            mediator_sigma = pm.HalfCauchy('mediator_sigma',
                                           beta=self.vars['m_sigma_beta'])
            mediator = pm.Normal('mediator',
                                 mu=mediator_mu,
                                 sd=mediator_sigma,
                                 observed=med_phen)
            # Phenotype
            intercept = pm.Normal('intercept', mu=0, sd=1)
            alpha = pm.Normal('alpha', 0, 1)
            # alpha = pm.Uniform('alpha', -10, 10)
            phenotype_expression_mu = pm.dot(beta_med, gwas_gen.T)
            phenotype_sigma = pm.HalfCauchy('phenotype_sigma',
                                            beta=self.vars['p_sigma_beta'])
            phenotype_mu = intercept + alpha * phenotype_expression_mu
            phen = pm.Normal('phen',
                             mu=phenotype_mu,
                             sd=phenotype_sigma,
                             observed=gwas_phen)

        if self.variational and self.mb:
            self.minibatch_RVs = [phen]
            self.minibatch_tensors = [gwas_gen, gwas_phen]

        return phenotype_model

    def _create_model_horseshoe(self, med_gen, med_phen,
                                gwas_gen, gwas_phen):
        """
        Args:
            med_gen (pandas.DataFrame): Mediator genotypes
            med_phen (pandas.DataFrame): Mediator phenotypes
            gwas_gen (pandas.DataFrame): GWAS genotypes
            gwas_phen (pandas.DataFrame): GWAS phenotypes

        """
        n_snps = gwas_gen.eval().shape[1]
        with pm.Model() as phenotype_model:
            # Expression
            tau_beta = pm.HalfCauchy('tau_beta',
                                     beta=self.vars['tau_beta'])
            lambda_beta = pm.HalfCauchy('lambda_beta',
                                        beta=self.vars['lambda_beta'],
                                        shape=(1, n_snps))
            # lambda_beta = pm.StudentT('lambda_beta', nu=3, mu=0,
            #                           lam=1, shape=(1, n_snps))
            total_variance = pm.dot(lambda_beta * lambda_beta,
                                    tau_beta * tau_beta)
            beta_med = pm.Normal('beta_med',
                                 mu=0,
                                 tau=1 / total_variance,
                                 shape=(1, n_snps))
            mediator_intercept = pm.Normal('mediator_intercept',
                                           mu=0,
                                           sd=1)
            mediator_mu = mediator_intercept + pm.dot(beta_med, med_gen.T)
            mediator_sigma = pm.HalfCauchy('mediator_sigma',
                                           beta=self.vars['m_sigma_beta'])
            mediator = pm.Normal('mediator',
                                 mu=mediator_mu,
                                 sd=mediator_sigma,
                                 observed=med_phen)
            # Phenotype
            alpha = pm.Normal('alpha', 0, 1)
            intercept = pm.Normal('intercept', mu=0, sd=1)
            phenotype_expression_mu = pm.dot(beta_med, gwas_gen.T)
            phenotype_sigma = pm.HalfCauchy('phenotype_sigma',
                                            beta=self.vars['p_sigma_beta'])
            phenotype_mu = intercept + alpha * phenotype_expression_mu
            phen = pm.Normal('phen',
                             mu=phenotype_mu,
                             sd=phenotype_sigma,
                             observed=gwas_phen)

        if self.variational and self.mb:
            self.minibatch_RVs = [phen]
            self.minibatch_tensors = [gwas_gen, gwas_phen]

        return phenotype_model

    def _create_model_laplace(self, med_gen, med_phen,
                              gwas_gen, gwas_phen):
        """
        Args:
            med_gen (pandas.DataFrame): Mediator genotypes
            med_phen (pandas.DataFrame): Mediator phenotypes
            gwas_gen (pandas.DataFrame): GWAS genotypes
            gwas_phen (pandas.DataFrame): GWAS phenotypes

        """
        n_snps = gwas_gen.eval().shape[1]
        with pm.Model() as phenotype_model:
            # Expression
            beta_med = pm.Laplace('beta_med', mu=0, b=1, shape=(1, n_snps),)
            mediator_intercept = pm.Normal('mediator_intercept',
                                           mu=0,
                                           sd=1)
            mediator_mu = mediator_intercept + pm.dot(beta_med, med_gen.T)
            mediator_sigma = pm.HalfCauchy('mediator_sigma',
                                           beta=self.vars['m_sigma_beta'])
            mediator = pm.Normal('mediator',
                                 mu=mediator_mu,
                                 sd=mediator_sigma,
                                 observed=med_phen)
            # Phenotype
            intercept = pm.Normal('intercept', mu=0, sd=1)
            alpha = pm.Normal('alpha', 0, 1)
            # alpha = pm.Uniform('alpha', -10, 10)
            phenotype_expression_mu = pm.dot(beta_med, gwas_gen.T)
            phenotype_sigma = pm.HalfCauchy('phenotype_sigma',
                                            beta=self.vars['p_sigma_beta'])
            phenotype_mu = intercept + alpha * phenotype_expression_mu
            phen = pm.Normal('phen',
                             mu=phenotype_mu,
                             sd=phenotype_sigma,
                             observed=gwas_phen)

        if self.variational and self.mb:
            self.minibatch_RVs = [phen]
            self.minibatch_tensors = [gwas_gen, gwas_phen]

        return phenotype_model

class MultiStudyMultiTissue(BayesianModel):
    """
    Jointly model the transcriptional regulation and
    its effect on the phenotype in multiple studies 
    and multiple tissues. Assume that tissues from the same
    individual are independent given the genotypes i.e.

    P(TisA, TisB | G) = P(TisA | G) P(TisB | G)

    """
    def __init__(self,
                 m_laplace_beta=1,
                 m_sigma_beta=10,
                 p_sigma_beta=10, *args, **kwargs):
        """
            Expression ~ N(X\beta, \sigma_exp)
            P(\beta) ~ Horseshoe (tau_beta, lambda_beta)
            P(\sigma_exp) ~ HalfCauchy(m_sigma_beta)
            Phenotype ~ N(X\beta\alpha, \sigma_phen)
            P(\alpha) ~ Uniform(-10, 10)
            P(\sigma_phen) ~ HalfCauchy(p_sigma_beta)
        Args:
            tau_beta (int): P(\beta) ~ Horseshoe (tau_beta, lambda_beta)
            lambda_beta (int): P(\beta) ~ Horseshoe (tau_beta, lambda_beta)
            m_sigma_beta (int): P(\sigma_exp) ~ HalfCauchy(m_sigma_beta)
            p_sigma_beta (int): P(\sigma_phen) ~ HalfCauchy(p_sigma_beta)

        """
        self.name = 'MultiStudyMultiTissue'
        self.cv_vars = ['gwas_phen', 'gwas_gen']
        self.vars = {'m_laplace_beta': m_laplace_beta,
                     'm_sigma_beta': m_sigma_beta,
                     'p_sigma_beta': p_sigma_beta
                     }
        super(MultiStudyMultiTissue, self).__init__(*args, **kwargs)

    def set_idx(self, med_idx, gwas_idx):
        self.med_idx = med_idx
        self.gwas_idx = gwas_idx
        return

    def create_model(self, 
                     med_gen, med_phen, 
                     gwas_gen, gwas_phen):
        n_snps = gwas_gen.eval().shape[1]
        n_tissues = len(np.unique(self.med_idx)) #
        n_studies = len(np.unique(self.gwas_idx))

        with pm.Model() as phenotype_model:
            # Expression
            
            beta_med = pm.Laplace('beta_med',
                                  mu=0,
                                  b=self.vars['m_laplace_beta'],
                                  shape=(1, n_snps),)
            mediator_intercept = pm.Normal('mediator_intercept',
                                           mu=0,
                                           sd=1,
                                           shape=n_tissues)
            mediator_gamma = pm.Uniform('mediator_gamma',
                                        lower=0,
                                        upper=1,
                                        shape=n_tissues)
            mediator_mu = mediator_intercept[self.med_idx] + mediator_gamma[self.med_idx] * pm.dot(beta_med, med_gen.T)            
            mediator_sigma = pm.HalfCauchy('mediator_sigma',
                                           beta=self.vars['m_sigma_beta'],
                                           shape=n_tissues)
            mediator = pm.Normal('mediator',
                                 mu=mediator_mu,
                                 sd=mediator_sigma[self.med_idx],
                                 observed=med_phen)
            # Phenotype
            intercept = pm.Normal('intercept', mu=0, sd=1, shape=n_studies)
            alpha_mu = pm.Normal('alpha_mu', mu=0, sd=1)
            alpha_sd = pm.HalfCauchy('alpha_sd', beta=1)
            alpha = pm.Normal('alpha', mu=alpha_mu, sd=alpha_sd, shape=n_studies)
            # alpha = pm.Uniform('alpha', -10, 10)
            phenotype_expression_mu = pm.dot(beta_med, gwas_gen.T)
            phenotype_sigma = pm.HalfCauchy('phenotype_sigma',
                                            beta=1,
                                            shape=n_studies)
            phen_mu = intercept[self.gwas_idx] + alpha[self.gwas_idx] * phenotype_expression_mu
            phen_sigma = phenotype_sigma[self.gwas_idx]
            phen = pm.Normal('phen',
                             mu=phen_mu,
                             sd=phen_sigma,
                             observed=gwas_phen)

        if self.variational and self.mb:
            self.minibatch_RVs = [phen]
            self.minibatch_tensors = [gwas_gen, gwas_phen]

        return phenotype_model


class NonMediated(BayesianModel):
    """
    Model the relationship between the genotype and
    phenotype without any added information about the 
    mediator. Use it as a basis for getting
    the null distribution under a mediation analysis.
    """
    def __init__(self,
                 g_laplace_beta=1,
                 p_sigma_beta=10, *args, **kwargs):
        self.name = 'NonMediated'
        self.cv_vars = ['gwas_phen', 'gwas_gen']
        self.vars = {'g_laplace_beta': g_laplace_beta,
                     'p_sigma_beta': p_sigma_beta,
                     }
        super(NonMediated, self).__init__(*args, **kwargs)

    def create_model(self, 
                     gwas_gen, gwas_phen):
        n_snps = gwas_gen.eval().shape[1]
        with pm.Model() as phenotype_model:
            beta = pm.Laplace('beta',
                              mu=0,
                              b=self.vars['g_laplace_beta'],
                              shape=(1, n_snps),)
            # Phenotype
            intercept = pm.Normal('intercept', mu=0, sd=1)
            phenotype_sigma = pm.HalfCauchy('phenotype_sigma',
                                            beta=self.vars['p_sigma_beta'])
            phenotype_mu = intercept + pm.dot(beta, gwas_gen.T)
            phen = pm.Normal('phen',
                             mu=phenotype_mu,
                             sd=phenotype_sigma,
                             observed=gwas_phen)

        if self.variational and self.mb:
            self.minibatch_RVs = [phen]
            self.minibatch_tensors = [gwas_gen, gwas_phen]

        return phenotype_model



class MeasurementError(BayesianModel):
    """
    Use the canonical definition of measurement error as described
    in http://andrewgelman.com/2016/09/04/29847/

    """
    def __init__(self,
                 mediator_mu,
                 mediator_sd,
                 m_laplace_beta=1,
                 p_sigma_beta=10, *args, **kwargs):
        self.name = 'MeasurementError'
        self.cv_vars = ['gwas_phen', 'gwas_gen']
        self.vars = {'mediator_mu': mediator_mu,
                     'mediator_sd': mediator_sd,
                     'p_sigma_beta': p_sigma_beta,
                     }
        super(MeasurementError, self).__init__(*args, **kwargs)

    def create_model(self, gwas_mediator, gwas_phen, gwas_error):
        n_samples = gwas_mediator.eval().shape[0]
        with pm.Model() as phenotype_model:
            # Phenotype
            mediator = pm.Normal('mediator',
                                 mu=self.vars['mediator_mu'],
                                 sd=self.vars['mediator_sd'],
                                 shape=n_samples)
            mediator_meas = pm.Normal('mediator_meas',
                                      mu=mediator,
                                      sd=gwas_error,
                                      shape=n_samples,
                                      observed=gwas_mediator)
            intercept = pm.Normal('intercept', mu=0, sd=1)
            alpha = pm.Uniform('alpha', lower=-10, upper=10)
            #alpha = pm.Normal('alpha', mu=0, sd=1)
            phenotype_sigma = pm.HalfCauchy('phenotype_sigma',
                                            beta=self.vars['p_sigma_beta'])
            phenotype_mu = intercept + alpha * mediator
            phen = pm.Normal('phen',
                             mu=phenotype_mu,
                             sd=phenotype_sigma,
                             observed=gwas_phen) 

        if self.variational and self.mb:
            self.minibatch_RVs = [phen]
            self.minibatch_tensors = [gwas_gen, gwas_phen]

        return phenotype_model

class MeasurementErrorBF(BayesianModel):
    """
    Use the canonical definition of measurement error as described
    in http://andrewgelman.com/2016/09/04/29847/

    """
    def __init__(self,
                 mediator_mu,
                 mediator_sd,
                 precomp_med=True,
                 heritability=0.1,
                 p_sigma_beta=10, *args, **kwargs):
        self.name = 'MeasurementErrorBF'
        self.cv_vars = ['gwas_phen', 'gwas_gen']
        self.vars = {'mediator_mu': mediator_mu,
                     'mediator_sd': mediator_sd,
                     'heritability': heritability,
                     'p_sigma_beta': p_sigma_beta,
                     'precomp_med': precomp_med,
                     }
        super(MeasurementErrorBF, self).__init__(*args, **kwargs)

    def create_model(self, gwas_mediator, gwas_phen, gwas_error):
        n_samples = gwas_mediator.eval().shape[0]
        with pm.Model() as phenotype_model:

            # Mediator
            mediator = pm.Normal('mediator',
                                 mu=self.vars['mediator_mu'],
                                 sd=self.vars['mediator_sd'],
                                 shape=n_samples)
            mediator_meas = pm.Normal('mediator_meas',
                                      mu=mediator,
                                      sd=gwas_error,
                                      shape=n_samples,
                                      observed=gwas_mediator)
            intercept = pm.Normal('intercept', mu=0, sd=1)

            phenotype_sigma = pm.HalfCauchy('phenotype_sigma',
                                            beta=self.vars['p_sigma_beta'])

            if self.vars['precomp_med']:
                p_var = t.sqr(phenotype_sigma)
                h = self.vars['heritability']
                var_explained = (p_var*h)/(1-h)
                md_var = np.square(np.mean(self.vars['mediator_sd']))
                md_mean_sq = np.square(np.mean(self.vars['mediator_mu'])) 
                var_alpha = var_explained/(md_var + md_mean_sq)
                alpha = pm.Normal('alpha', mu=0, sd=t.sqrt(var_alpha))
            else:
                p_var = t.sqr(phenotype_sigma)
                h = self.vars['heritability']
                var_explained = (p_var*h)/(1-h)
                md_var = t.var(mediator)
                md_mean_sq = t.sqr(t.mean(mediator))
                var_alpha = var_explained/(md_var + md_mean_sq)
                alpha = pm.Normal('alpha', mu=0, sd=t.sqrt(var_alpha))
 
            # Model Selection
            p = np.array([0.5, 0.5])
            mediator_model = pm.Bernoulli('mediator_model', p[1])

            # Model 1
            phenotype_mu_null = intercept

            # Model 2
            phenotype_mu_mediator = intercept + alpha * mediator

            phen = pm.DensityDist('phen',
                                lambda value: pm.switch(mediator_model, 
                                    pm.Normal.dist(mu=phenotype_mu_mediator, sd=phenotype_sigma).logp(value), 
                                    pm.Normal.dist(mu=phenotype_mu_null, sd=phenotype_sigma).logp(value)
                                ),
                                observed=gwas_phen)
            self.steps = [pm.BinaryGibbsMetropolis(vars=[mediator_model]),
                          pm.Metropolis()]

        if self.variational and self.mb:
            self.minibatch_RVs = [phen]
            self.minibatch_tensors = [gwas_gen, gwas_phen]

        return phenotype_model