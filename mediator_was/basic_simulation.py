'''
Survey of the properties of the measurement error estimators
under different uncertainty and pve of phenotype.

Author: Kunal Bhutani <kunalbhutani@gmail.com>
'''


import pandas as pd
import numpy
import numpy.random as R
from mediator_was.association import *

def generate_association(pve=None, x=None, n_samples=5000, beta=1):
    '''
    Generate X and Y based on user-provided association (default beta: 1).

    X ~ N(0,1) or sampled from a given list of X

    If pve is None, assume independence between the two, and generate 
    Y ~ N(0,1) otherwise Y ~ (betaX, variance determined by pve)
    '''
    if x:
        x = R.choice(x, size=n_samples)
    else:
        x = R.normal(size=n_samples)
    if pve:
        y = beta*x
        sigma_y = numpy.sqrt(numpy.var(y) * (1 / pve - 1))
    else:
        y = numpy.zeros(n_samples)
        sigma_y = 1
    y += R.normal(size=n_samples, scale=sigma_y) 
    return x, y

def add_uncertainty(x, sigma_u, sigma_u_std=None):
    '''
    Add uncertainty to measurements of X* 
    X ~ N(X*, sigma_ui)

    If sigma_u_std not set, sample each x the same.
    sigma_ui = sigma_u
   
    Otherwise:
    sigma_ui = N(sigma_u, sigma_u_std)
    '''
    n_samples = x.shape[0]
    if not sigma_u_std:
        w = x + R.normal(size=n_samples, scale=numpy.sqrt(sigma_u)+.00000001)
        sigma_u = numpy.tile(sigma_u, reps=n_samples)
    else: # Note that this is heteroscedastic, but still non linear.
        sigma_u = abs(R.normal(sigma_u, scale=sigma_u_std+0.0000001, size=n_samples))
        w = [xi+R.normal(scale=numpy.sqrt(sigma_ui))
            for xi, sigma_ui in zip(x, sigma_u) ]
    return w, sigma_u

def model_generator(pve, x=None, n_samples=5000):
    '''
    Generator for creating new measurements of X based on true X* and Y*
    generated once using the provided pve.
    '''
    x, y = generate_association(pve, x=None, n_samples=n_samples)
    while True:
        sigma_u, sigma_u_std = yield
        w, sigma_ui = add_uncertainty(x, sigma_u, sigma_u_std)
        yield w, y, sigma_ui

def trial(model, sigma_u, error_type='homoscedastic'):
    '''
    Generate new values of X using sigma_u and associated error
    type based on model generator. Fit estimators and return
    associated statistics.
    '''
    if error_type == 'homoscedastic':
        sigma_u_std = None
    elif error_type == 'heteroscedastic':
        sigma_u_std = sigma_u
    else: #TODO: Linear/Different heteroscedastic
        raise NotImplementedError
    next(model)
    w, y, sigma_ui = model.send([sigma_u, sigma_u_std])
    results = {'OLS': t(w, y, method="OLS"),
               'WLS': t(w, y, sigma_ui, method="WLS"),
               'Moment': t(w, y, sigma_ui, method="moment"),
               'Moment2': t(w, y, sigma_ui, method="moment2"),
               'Moment-Buo': t(w, y, sigma_ui, method="momentbuo"),
               'RC': t(w, y, sigma_ui, method="rc"),
               'RC-hetero': t(w, y, sigma_ui, method="rc-hetero"),
               'Weighted': t(w, y, sigma_ui, method="weighted")}
    return results

def simulate_model(model, sigma_u_list=[0.5], n_trials=500):
    '''
    For known association model Y* ~ X*, 
    simulate W ~ (X*, u) for trials and fit Y ~ W. Simulate both
    heteroscedastic and homoscedastic u.

    Return a multindex pandas DataFrame:
        indices = ['Sigma U', 'Error Type', 'Trial', 'Estimator']
        columns = ['coeff', 'se', 'wald', 'pvalue']

    '''
    statistics = {}
    for sigma_u in sigma_u_list:
        for error_type in ['homoscedastic', 'heteroscedastic']:
            label = tuple([sigma_u, error_type])
            for i in range(n_trials):
                results = trial(model, sigma_u, error_type)
                for method, result in results.items():
                    result_label = label+tuple([i, method])
                    statistics[result_label] = result
    statistics_df = pd.DataFrame.from_dict(statistics).T 
    statistics_df.index.names = ['SigmaU', 'Error', 'Trial', 'Estimator']
    statistics_df.columns = ['coeff', 'se', 'pvalue']
    statistics_df['wald'] = statistics_df['coeff']**2 / statistics_df['se']**2
    return statistics_df

def run_simulation(pve_list=[0.001], sigma_u_list=[0.5], x=None, n_samples=5000, n_models=500, n_trials=50):
    '''
    Run both homoscedastic and heteroscedatic simulation for the provided model and sigma_u_list.
    Return a multindex pandas DataFrame:
        indices = ['PVE', 'Sigma U', 'Error Type', 'Trial', 'Estimator']
        columns = ['coeff', 'se', 'wald', 'pvalue']
    '''
    def run_model(pve, i):
        model = model_generator(pve, x, n_samples)
        model_df = simulate_model(model, sigma_u_list, n_trials)
        idx_names = model_df.index.names
        index = tuple(model_df.index)
        index = [tuple([i, pve])+idx for idx in index]
        model_df.index = pd.MultiIndex.from_tuples(index)
        model_df.index.names = ['Model', 'PVE']+idx_names
        return model_df

    simulation_df = pd.concat([run_model(pve, i)
                               for pve in pve_list
                               for i in range(n_models)])

    return simulation_df
