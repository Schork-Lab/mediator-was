"""Estimate power of TWAS.

Author: Abhishek Sarkar <aksarkar@mit.edu>
        Kunal Bhutani   <kunalbhutani@gmail.com>
"""
import sys

import numpy
import pandas as pd
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.tools

from mediator_was.simulation import *
from mediator_was.association import *



def trial(simulation, method, n_test=50000, alpha=0.05, ngwas_tests=1e6,
          n_gene_tests=1e3):
    """Count up significant hits for a new test set"""
    # count = numpy.zeros(7)
    # alpha /= numpy.array([ngwas_tests]+[n_gene_tests]*6)

    genotypes, expression, phenotype = getattr(simulation, method)(n_test)
    associations = []
    for g, e, ms, eqtl in zip(genotypes, expression, simulation.models, simulation.eqtls):
        # predicted_expression = numpy.array([m.predict(g) for m in ms])
        # predicted_expression = numpy.apply_along_axis(standardize, 1, predicted_expression)
        # w = numpy.mean(predicted_expression, axis=0)
        # sigma_ui = numpy.var(predicted_expression, axis=0)
        design = statsmodels.tools.add_constant(g)
        w = ms[0].predict(design)
        sigma_ui = wls_prediction_std(ms[0], design)[0]
        sigma_ui = numpy.power(sigma_ui, 2) - ms[0].mse_resid
        sigma_u = numpy.mean(sigma_ui)
        association = {'OLS': t(w, phenotype, method="OLS"),
                   'WLS': t(w, phenotype, sigma_ui, method="WLS"),
                   'Moment': t(w, phenotype, sigma_ui, method="moment"),
                   'Moment2': t(w, phenotype, sigma_ui, method="moment2"),
                   'Moment-Buo': t(w, phenotype, sigma_ui, method="momentbuo"),
                   'RC': t(w, phenotype, sigma_ui, method="rc"),
                   'RC-hetero': t(w, phenotype, sigma_ui, method="rc-hetero"),
                   'Weighted': t(w, phenotype, sigma_ui, method="weighted")}
        associations.append(association)
    return associations

def run_simulation(method, n_causal_eqtls, n_test, n_causal_genes=20, n_trials=25, hapgen=None):
    """Run the simulation with the specified parameters"""
    numpy.random.seed(0)
    n_train = numpy.repeat(1000, 1)
    simulations = {}
    with simulation(n_causal_eqtls, n_train, n_causal_genes, hapgen=hapgen) as sim:
        for i in range(n_trials):
            associations = trial(sim, method, n_test=n_test)
            for gene, association in enumerate(associations):
                for estimator, statistics in association.items():
                    label = tuple(['Trial {}'.format(i), 'Gene {}'.format(gene), estimator])
                    simulations[label] = statistics

    simulations_df = pd.DataFrame.from_dict(simulations).T 
    simulations_df.index.names = ['Trial', 'Gene', 'Estimator']
    simulations_df.columns = ['coeff', 'se', 'pvalue']
    simulations_df['wald'] = simulations_df['coeff']**2 / simulations_df['se']**2
    return simulations_df

def power(n_causal_eqtls, n_test, n_causal_genes, hapgen=None):
    """Estimate the power of the naive and corrected analyses."""
    return run_simulation("simulate_mediated_phenotype", n_causal_eqtls, n_test, n_causal_genes, hapgen=hapgen)

def type_1_error_null(n_causal_eqtls, n_test, n_causal_genes, hapgen=None):
    """Estimate the type 1 error rate of the naive and corrected analyses assuming
a null phenotype."""
    return run_simulation("simulate_null_phenotype", n_causal_eqtls, n_test, n_causal_genes, hapgen=hapgen)

def type_1_error_independent(n_causal_eqtls, n_test, n_causal_genes, hapgen=None):
    """Estimate the type 1 error rate of the naive and corrected analyses assuming
an independent phenotype."""
    return run_simulation("simulate_independent_phenotype", n_causal_eqtls, n_test, n_causal_genes, hapgen=hapgen)

if __name__ == '__main__':
    method = sys.argv[1]
    n_causal_eqtls, n_test = [int(x) for x in sys.argv[2:]]
    getattr(sys.modules[__name__], method)(n_causal_eqtls, n_test)
