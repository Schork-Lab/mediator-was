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
                expression_trace['beta_exp'], 'expression')


def phenotype_model_with_pm(genotypes, phenotypes, beta_exp_trace,
                            cauchy_beta=10, logistic=False):
    '''
    Fit phenotype model with posterior mean of beta_exp from expression model 
    '''
    n_snps = genotypes.shape[1]
    beta_exp_trace_reshaped = np.array([beta_exp_trace[:,0][:, idx]
                                          for idx in range(n_snps)])
    beta_exp_mean = beta_exp_trace_reshaped.mean(axis=1)
    with pm.Model() as phenotype_model:
        beta_phen = pm.Uniform('beta_phen', -1e3, 1e3)
        expression = np.dot(beta_exp_mean, genotypes.T)
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
        beta_phen = pm.Uniform('beta_phen', -1e3, 1e3)
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
        phenotype_trace = pm.sample(50000, step=pm.NUTS(), start=start,
                                    progressbar=True)
    return Model(phenotype_model, phenotype_trace[-10000:], phenotype_trace['beta_exp'], 'prior')


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
        beta_phen = pm.Uniform('beta_phen', -1e3, 1e3)
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
        phenotype_trace = pm.sample(150000, step=pm.Metropolis(), start=start, progressbar=True)
        # phenotype_trace = pm.sample(50000, step=[step1, step2], start=start, progressbar=True)
    return Model(phenotype_model, phenotype_trace[-10000:], phenotype_trace['beta_exp'], 'full')


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


def compute_mse(phenotypes, models):
    '''
    models: dict(model_name, [trace, model, type={'pm','prior','full'}])
    '''
    mse = dict()
    for name, model in models.items():
        squared_error = (phenotypes - compute_ppc(model.model, model.trace))**2
        mse[name] = np.mean(squared_error)
    return mse


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


def compute_mse_oos(genotypes, phenotypes, models):
    '''
    models: dict(model_name, [trace, model, type={'pm','prior','full'}])
    '''
    mse = dict()
    for name, model in models.items():
        squared_error = (phenotypes - compute_ppc_oss(model))**2
        mse[name] = np.mean(squared_error)
    return mse


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

