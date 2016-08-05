'''
Bayesian models for TWAS.

Author: Kunal Bhutani <kunalbhutani@gmail.com>
'''
from collections import defaultdict
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
import pymc3 as pm
import numpy as np
from theano import shared


class BayesianModel(object):
    '''
    General Bayesian Model Class for quantifying
    relationship between gene and phenotype
    
    Adapted from Thomas Wiecki
    https://github.com/pymc-devs/pymc3/issues/511#issuecomment-125935523
    
    Attributes:
        cached_model (TYPE): Description
        k_folds (TYPE): Description
        mb (TYPE): Description
        minibatches (TYPE): Description
        shared_vars (TYPE): Description
        trace (TYPE): Description
        variational (TYPE): Description
    '''

    def __init__(self, variational=True, mb=False):
        """
        Args:
            variational (bool, optional): Use Variational Inference
            mb (bool, optional): Use minibatches
        """
        self.variational = variational
        self.cached_model = None
        self.mb = mb

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

    def _inference(self, samples=10000, n_trace=5000):
        """
        Perform the inference. Uses ADVI if self.variational
        is True. Also, uses minibatches is self.mb=True based
        on generators defined in self.run.

        Otherwise, uses Metropolis.

        Args:
            samples (int, optional): Number of steps
            n_trace (int, optional): Number of steps used for trace
        Returns:
            trace: Trace of the PyMC3 inference
        """
        with self.cached_model:
            if self.variational:
                if self.mb:
                    v_params = pm.variational.advi_minibatch(n=samples,
                               minibatch_tensors=self.minibatch_tensors,
                               minibatch_RVs=self.minibatch_RVs,
                               minibatches=self.minibatches,)
                else:
                    v_params = pm.variational.advi(n=samples)
                trace = pm.variational.sample_vp(v_params, draws=n_trace)
            else:
                start = pm.find_MAP()
                trace = pm.sample(samples, step=pm.Metropolis(),
                                  start=start, progressbar=True)
                trace = trace[-n_trace:]
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
        return self.cv_stats

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
        return {'logp': mc_logp,
                'mse': mean_mse}

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

        TODO: confirm that it's okay to take mean across the steps
              of the trace.
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

    def _alpha_zscore(self, model):
        """Summary
        
        Args:
            model (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        mean = np.mean(model.trace['alpha'])
        sd = np.std(model.trace['alpha'], ddof=1)
        zscore = mean / sd
        return mean, sd, zscore

    def _mb_generator(self, data, size=50):
        """Summary
        
        Args:
            data (TYPE): Description
            size (int, optional): Description
        
        Returns:
            TYPE: Description
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
        n_snps = gwas_gen.eval().shape[1]
        with pm.Model() as phenotype_model:
            beta_med = pm.Normal('beta_med',
                                 mu=self.vars['coef_mean'],
                                 sd=self.vars['coef_sd'],
                                 shape=(1, n_snps))
            alpha = pm.Uniform('alpha', -10, 10)
            mu = pm.dot(beta_med, gwas_gen.T)
            phenotype_sigma = pm.HalfCauchy('phenotype_sigma',
                                            beta=self.vars['p_sigma_beta'])
            phen = pm.Normal('phen',
                             mu=alpha * mu,
                             sd=phenotype_sigma,
                             observed=gwas_phen)
        if self.variational and self.mb:
            self.minibatch_RVs = [phen]
            self.minibatch_tensors = [gwas_gen, gwas_phen]
        return phenotype_model


class Joint(BayesianModel):
    """
    Jointly model the transcriptional regulation and
    its effect on the phenotype.

    """
    def __init__(self, tau_beta=10, lambda_beta=1, m_sigma_beta=1,
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
        self.cv_vars = ['gwas_phen', 'gwas_gen']
        self.vars = {'tau_beta': tau_beta,
                     'lambda_beta': lambda_beta,
                     'm_sigma_beta': m_sigma_beta,
                     'p_sigma_beta': p_sigma_beta
                     }
        super(Joint, self).__init__(*args, **kwargs)

    def create_model(self, med_gen, med_phen,
                     gwas_gen, gwas_phen):
        """
        Args:
            med_gen (pandas.DataFrame): Mediator genotypes
            med_phen (pandas.DataFrame): Mediator phenotypes
            gwas_gen (pandas.DataFrame): GWAS genotypes
            gwas_phen (pandas.DataFrame): GWAS phenotypes

        Returns:
            pymc3.Model(): The Bayesian model
        """
        n_snps = gwas_gen.eval().shape[1]
        with pm.Model() as phenotype_model:
            # Expression
            tau_beta = pm.HalfCauchy('tau_beta',
                                     beta=self.vars['tau_beta'])
            lambda_beta = pm.HalfCauchy('lambda_beta',
                                        beta=self.vars['lambda_beta'],
                                        shape=(1, ))
            total_variance = pm.dot(lambda_beta * lambda_beta,
                                    tau_beta * tau_beta)
            beta_med = pm.Normal('beta_med',
                                 mu=0,
                                 tau=1 / total_variance,
                                 shape=(1, n_snps))
            mediator_mu = pm.dot(beta_med, med_gen.T)
            mediator_sigma = pm.HalfCauchy('mediator_sigma',
                                           beta=self.vars['m_sigma_beta'])
            mediator = pm.Normal('mediator',
                                 mu=mediator_mu,
                                 sd=mediator_sigma,
                                 observed=med_phen)
            # Phenotype
            alpha = pm.Uniform('alpha', -10, 10)
            phenotype_expression_mu = pm.dot(beta_med, gwas_gen.T)
            phenotype_sigma = pm.HalfCauchy('phenotype_sigma',
                                            beta=self.vars['p_sigma_beta'])
            phenotype_mu = alpha * phenotype_expression_mu
            phen = pm.Normal('phen',
                             mu=phenotype_mu,
                             sd=phenotype_sigma,
                             observed=gwas_phen)

        if self.variational and self.mb:
            self.minibatch_RVs = [phen]
            self.minibatch_tensors = [gwas_gen, gwas_phen]

        return phenotype_model
