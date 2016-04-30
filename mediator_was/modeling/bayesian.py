'''

Two-stage Bayesian analysis for expression

TODO: Simulate different genetic architectures
        -- Assumptions behind BSLMM, Others
      See if including the uncertainty is applicable in certain cases
      MSE or KS test for out-of-sample for simulation
      MSE or KS test for in-sample for testing
      RUN 

'''

import pymc3 as pm
import numpy as np
import pandas as pd
from collections import namedtuple
import theano.tensor as t
import scipy.stats as stats
import scipy.misc as misc


Model = namedtuple('Model', ['model', 'trace', 'beta_exp_trace', 'type'])

def tinvlogit(x):
    return t.exp(x) / (1 + t.exp(x))


def expression_model(genotypes, expression,
                     b=1, cauchy_beta=10):
    '''
    Fit just the expression model.
    '''
    n_snps = genotypes.shape[1]
    with pm.Model() as expression_model:
        beta_exp = pm.Laplace('beta_exp', mu=0, b=b,
                              shape=(1, n_snps))
        mu = pm.dot(beta_exp, genotypes.T)
        sigma = pm.HalfCauchy('sigma', beta=cauchy_beta)
        exp = pm.Normal('exp', mu=mu, sd=sigma, observed=expression)
        start = pm.find_MAP()
        expression_trace = pm.sample(100000, step=pm.Metropolis(), start=start, progressbar=True)

    return Model(expression_model, expression_trace[-10000:],
                 expression_trace['beta_exp'][-10000:], 'expression')


def phenotype_model_with_pm(genotypes, phenotypes, beta_exp_trace,
                            cauchy_beta=10, logistic=False):
    '''
    Fit phenotype model with posterior mean of beta_exp from expression model 
    '''
    n_snps = genotypes.shape[1]
    beta_exp_trace_reshaped = np.array([beta_exp_trace[:,0][:, idx]
                                          for idx in range(n_snps)])
    beta_exp_mean = beta_exp_trace_reshaped.mean(axis=1)
    expression = np.dot(beta_exp_mean, genotypes.T)
    with pm.Model() as phenotype_model:
        beta_phen = pm.Normal('beta_phen', 0, 1)
        
        sigma = pm.HalfCauchy('sigma', beta=cauchy_beta)
        if not logistic:
            phen = pm.Normal('phen', mu=beta_phen*expression, sd=sigma, observed=phenotypes)
        else:
            p = tinvlogit(beta_phen*expression)
            phen = pm.Bernoulli('phen', p=p, observed=phenotypes)
        start = pm.find_MAP()
        phenotype_trace = pm.sample(15000, start=start, step=pm.NUTS(),
                                    progressbar=True)
    return Model(phenotype_model, phenotype_trace[-10000:], beta_exp_trace, 'pm')


def phenotype_model_with_prior(genotypes, phenotypes, beta_exp_trace,
                               cauchy_beta=10, logistic=False):
    '''
    Fit phenotype model with beta_exp priors from expression model as (posterior_mean, posterior_sd)
    '''
    n_snps = genotypes.shape[1]
    beta_exp_trace_reshaped = np.array([beta_exp_trace[:,0][:, idx]
                                          for idx in range(n_snps)])
    beta_exp_mean = beta_exp_trace_reshaped.mean(axis=1)
    beta_exp_sd = beta_exp_trace_reshaped.std(axis=1, ddof=1)
    with pm.Model() as phenotype_model:
        beta_exp = pm.Normal('beta_exp', 
                             mu=beta_exp_mean, 
                             sd=beta_exp_sd, 
                             shape=(1, n_snps))
        beta_phen = pm.Normal('beta_phen', 0, 1)
        exp = pm.dot(beta_exp, genotypes.T)
        sigma = pm.HalfCauchy('sigma', beta=cauchy_beta)
        if not logistic:
            phen = pm.Normal('phen', 
                             mu=beta_phen*exp, 
                             sd=sigma, 
                             observed=phenotypes)
        else:
            p = tinvlogit(beta_phen*exp)
            phen = pm.Bernoulli('phen', p=p, observed=phenotypes)
        start = pm.find_MAP()
        # step1 = pm.NUTS([beta_exp])
        # step2 = pm.Metropolis([beta_phen, sigma])
        # phenotype_trace = pm.sample(20000, step=[step1, step2], start=start,
        #                             progressbar=True)
        phenotype_trace = pm.sample(30000, step=pm.NUTS(), start=start,
                                            progressbar=True)
    return Model(phenotype_model, phenotype_trace[-10000:], phenotype_trace['beta_exp'][-15000:], 'prior')


def full_model(exp_genotypes, expression,
               phen_genotypes, phenotypes,
               b=1, cauchy_beta=10, logistic=False):
    '''
    Fit phenotype and expression model at the same time.
    '''
    n_snps = exp_genotypes.shape[1]
    with pm.Model() as phenotype_model:

        # Expression
        beta_exp = pm.Laplace('beta_exp', mu=0, b=b, shape=(1, n_snps))
        expression_mu = pm.dot(beta_exp, exp_genotypes.T)
        expression_sigma = pm.HalfCauchy('expression_sigma', beta=10)
        exp = pm.Normal('expression', 
                        mu=expression_mu, 
                        sd=expression_sigma, 
                        observed=expression)
        # Phenotype
        beta_phen = pm.Normal('beta_phen', 0, 1)
        phenotype_expression_mu = pm.dot(beta_exp, phen_genotypes.T)
        phenotype_sigma = pm.HalfCauchy('phenotype_sigma', beta=10)
        phenotype_mu = beta_phen * phenotype_expression_mu
        if not logistic:
            phen = pm.Normal('phen', 
                             mu=phenotype_mu, 
                             sd=phenotype_sigma, 
                             observed=phenotypes)

        else:
            p = tinvlogit(phenotype_mu)
            phen = pm.Bernoulli('phen', p=p, observed=phenotypes)
        
        # Fit
        start = pm.find_MAP()
        step1 = pm.Metropolis([beta_exp, expression_sigma])
        step2 = pm.NUTS([beta_phen, phenotype_sigma])
        phenotype_trace = pm.sample(125000, step=pm.Metropolis(), start=start, progressbar=True)
        # phenotype_trace = pm.sample(50000, step=[step1, step2], start=start, progressbar=True)
    return Model(phenotype_model, phenotype_trace[-15000:], phenotype_trace['beta_exp'][-15000:], 'full')


def compute_ppc(model, samples=500, size=1):
    '''
    models: dict((model_name, [model, trace]))
    '''
    values = []
    var = model.model.phen
    for idx in randint(0, len(model.trace), samples):
        param = phenotype_trace[idx]
        values.append(var.distribution.random(point=param, size=size).mean(0))
    values = np.asarray(values).mean(0)
    return values


def compute_mse(phenotypes, model):
    '''
    Compute insample mse
    '''
    squared_error = (phenotypes - compute_ppc(model.model, model.trace))**2
    return numpy.mean(squared_error)


def compute_ppc_oos(genotypes, model, num_steps=10000,):
    '''
    Compute out of sample post predictive checks

    TODO: instead of using value of trace, resample traces at the position
          using pm.distributions as in compute_ppc.
    '''

    if model.type == 'pm':
        exp = np.dot(genotypes,
                     model.beta_exp_trace[-num_steps:].mean(axis=0).T)

    elif model.type == 'prior':
        exp = np.dot(genotypes,
                     model.trace[-num_steps:]['beta_exp'].mean(axis=0).T)

    elif model.type == 'full':
        exp = np.dot(genotypes, 
                     model.trace[-num_steps:]['beta_exp'].mean(axis=0).T)

    else:
        raise ValueError('Unrecognized model type for ppc calculation')

    phen = (exp*model.trace[-num_steps:]['beta_phen'].mean()).ravel()
    return exp, phen


def compute_mse_oos(genotypes, phenotypes, model):
    '''
    Compute out of sample mse
    '''
    exp_hat, phen_hat = compute_ppc_oos(genotypes, model)
    squared_error = (phenotypes - phen_hat)**2
    return np.mean(squared_error)


def ppc_df(phenotype, models, oos=True):
    ppc = {}
    ppc['observed'] = phenotype
    for name, model in model.items():
        if not oos:
            ppc[name] = compute_ppc(model)
        else:
            ppc[name] = compute_ppc_oos(model)
    return pd.DataFrame(ppc)


def phen(model, n_steps=5000, pvalue=False):
    summary = pm.df_summary(model.trace[-n_steps:])
    if not pvalue:
        return summary.ix['beta_phen']['mean'], summary.ix['beta_phen']['sd']
    if pvalue:
        return summary.ix['beta_phen']['mean'], summary.ix['beta_phen']['sd'], 0


def bayes_factor(model, genotypes, phenotype, 
                 train_genotypes=None, train_expression=None):

    harmonic_mean = lambda a: -(np.log(1/len(a)) + misc.logsumexp(-1*a))
    if model.type == 'pm':
        calc_alternate, calc_null = _bf_pm()
    elif model.type == 'prior':
        calc_alternate, calc_null = _bf_prior()
    elif model.type == 'full':
        calc_alternate, calc_null = _bf_full()

    n_snps = genotypes.shape[1]
    beta_exp_trace = model.beta_exp_trace
    beta_exp_trace_reshaped = np.array([beta_exp_trace[:,0][:, idx]
                                          for idx in range(n_snps)])

    beta_exp_mean = beta_exp_trace_reshaped.mean(axis=1)
    beta_exp_sd = beta_exp_trace_reshaped.std(axis=1, ddof=1)
    expression = np.dot(beta_exp_mean, genotypes.T)

    alternate = lambda step: calc_alternate(step, genotypes, phenotype, expression,
                                            beta_exp_mean, beta_exp_sd,
                                            train_genotypes, train_expression)
    null = lambda step: calc_null(step, genotypes, phenotype, expression,
                                  beta_exp_mean, beta_exp_sd,
                                  train_genotypes, train_expression)

    alternate_log_prob = np.array(list((map(alternate, model.trace[-5000:]))))
    null_log_prob = np.array(list((map(null, model.trace[-5000:]))))
    return (harmonic_mean(alternate_log_prob) - harmonic_mean(null_log_prob))


def _bf_pm():
    def compute_logprob_alternate(step, genotypes, phenotype,
                                  expression, 
                                  beta_exp_mean=None, beta_exp_sd=None,
                                  train_genotypes=None, train_expression=None):
        log_p_sigma = np.log(stats.halfcauchy.pdf(step['sigma'], 0, 10))
        log_p_beta = np.log(stats.norm.pdf(step['beta_phen'], 0, 1))
        y_hat = step['beta_phen']*expression
        log_p_y = np.log(stats.norm.pdf(phenotype, y_hat, step['sigma'])).sum()
        return log_p_sigma + log_p_beta + log_p_y

    def compute_logprob_null(step, genotypes, phenotype,
                                    expression=None, 
                                    beta_exp_mean=None, beta_exp_sd=None,
                                    train_genotypes=None, train_expression=None):
            log_p_sigma = np.log(stats.halfcauchy.pdf(step['sigma'], 0, 10))
            log_p_y = np.log(stats.norm.pdf(phenotype, 0, step['sigma'])).sum()
            return log_p_sigma + log_p_y
    
    return compute_logprob_alternate, compute_logprob_null


def _bf_prior():
    def compute_logprob_alternate(step, genotypes, phenotype,
                                  expression=None, 
                                  beta_exp_mean=None, beta_exp_sd=None,
                                  train_genotypes=None, train_expression=None):
        log_p_beta_exp = np.log(stats.norm.pdf(step['beta_exp'], beta_exp_mean, beta_exp_sd)).sum()
        log_p_sigma = np.log(stats.halfcauchy.pdf(step['sigma'], 0, 10))
        log_p_beta = np.log(stats.norm.pdf(step['beta_phen'], 0, 1))
        expression = np.dot(step['beta_exp'], genotypes.T)
        y_hat = step['beta_phen']*expression
        log_p_y = np.log(stats.norm.pdf(phenotype, y_hat, step['sigma'])).sum()
        return log_p_sigma + log_p_beta_exp + log_p_y + log_p_beta 

    def compute_logprob_null(step, genotypes, phenotype,
                                    expression=None, 
                                    beta_exp_mean=None, beta_exp_sd=None,
                                    train_genotypes=None, train_expression=None):
        log_p_beta_exp = np.log(stats.norm.pdf(step['beta_exp'], beta_exp_mean, beta_exp_sd)).sum()
        log_p_sigma = np.log(stats.halfcauchy.pdf(step['sigma'], 0, 10))
        log_p_beta = np.log(stats.norm.pdf(step['beta_phen'], 0, 1))
        log_p_y = np.log(stats.norm.pdf(phenotype, 0, step['sigma'])).sum()
        return log_p_sigma + log_p_y + log_p_beta_exp

    return compute_logprob_alternate, compute_logprob_null


def _bf_full():
    def compute_logprob_alternate(step, genotypes, phenotype,
                                    expression=None, 
                                    beta_exp_mean=None, beta_exp_sd=None,
                                    train_genotypes=None, train_expression=None):
        log_p_beta_exp = np.log(stats.laplace.pdf(step['beta_exp'], 0, 1)).sum()
        log_p_sigma_exp = np.log(stats.halfcauchy.pdf(step['expression_sigma'], 0, 10))
        log_p_sigma_phen = np.log(stats.halfcauchy.pdf(step['phenotype_sigma'], 0, 10))
        log_p_beta = np.log(stats.norm.pdf(step['beta_phen'], 0, 1))
        
        expression_hat = np.dot(step['beta_exp'], train_genotypes.T)
        log_p_expression = np.log(stats.norm.pdf(train_expression, expression_hat, step['expression_sigma'])).sum()
        
        phen_expression = np.dot(step['beta_exp'], genotypes.T)
        phen_hat = step['beta_phen']*phen_expression
        log_p_phen = np.log(stats.norm.pdf(phenotype, phen_hat, step['phenotype_sigma'])).sum()
        return log_p_beta_exp + log_p_sigma_exp + log_p_sigma_phen + log_p_expression + log_p_phen + log_p_beta

    def compute_logprob_null(step, genotypes, phenotype,
                                    expression=None, 
                                    beta_exp_mean=None, beta_exp_sd=None,
                                    train_genotypes=None, train_expression=None):
        log_p_beta_exp = np.log(stats.laplace.pdf(step['beta_exp'], 0, 1)).sum()
        log_p_sigma_exp = np.log(stats.halfcauchy.pdf(step['expression_sigma'], 0, 10))
        log_p_sigma_phen = np.log(stats.halfcauchy.pdf(step['phenotype_sigma'], 0, 10))
        expression_hat = np.dot(step['beta_exp'], train_genotypes.T)
        log_p_expression = np.log(stats.norm.pdf(train_expression, expression_hat, step['expression_sigma'])).sum()
        log_p_phen = np.log(stats.norm.pdf(phenotype, 0, step['phenotype_sigma'])).sum()
        return log_p_beta_exp + log_p_sigma_exp + log_p_sigma_phen + log_p_expression + log_p_phen

    return compute_logprob_alternate, compute_logprob_null