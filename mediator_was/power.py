"""Estimate power of TWAS.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import sys

import numpy

from mediator_was.simulation import *
from mediator_was.association import t as _t

def _gwas(genotypes, phenotype):
    """Compute the best GWAS p-value in the locus"""
    return min(_t(g, phenotype)[1] for g in genotypes.T)

def _egwas(genotypes, expression, phenotype):
    """Find the best eQTL in the locus and test it against phenotype"""
    eqtl_p = ((_t(g, expression)[1], g) for g in genotypes.T)
    eqtl = min(eqtl_p, key=operator.itemgetter(0))[1]
    return _t(eqtl, phenotype)[1]

def _trial(params, models, n_total_tests=15000, alpha=0.05, n_gwas_tests=1e6, n_gene_tests=1.5e4):
    """Count up significant hits for a new test set"""
    count = numpy.zeros(4)
    alpha = alpha / numpy.array([n_gwas_tests, n_gene_tests, n_gene_tests, n_gene_tests])
    genotypes, true_expression, phenotype = simulate_test(params)
    for g, e, ms in zip(genotypes, true_expression, zip(*models)):
        predicted_expression = numpy.array([m.predict(g) for m in ms])
        w = numpy.mean(predicted_expression, axis=0)
        sigma_ui = numpy.var(predicted_expression, axis=0)
        sigma_u = numpy.mean(sigma_ui)
        alpha = alpha / n_total_tests
        count += numpy.array([_gwas(g, phenotype), _egwas(g, e, phenotype),
                              _t(w, phenotype)[1], _t(w, phenotype, sigma_u)[1]]) < alpha
    return count

def power(n_causal_eqtls, n_test, n_models=4, n_trials=10):
    """Estimate the power of the naive and corrected analyses."""
    with simulation(n_causal_eqtls, n_models) as sim:
        return sum(_trial(*sim) for _ in range(n_trials)) / (100 * n_trials)

if __name__ == '__main__':
    n_causal_eqtls, n_test = [int(x) for x in sys.argv[1:]]
    numpy.random.seed(0)
    print(n_causal_eqtls, n_test, ' '.join('{:.3f}'.format(p) for p in power(n_causal_eqtls, n_test)))
