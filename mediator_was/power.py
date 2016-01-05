"""Estimate power of TWAS.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import sys

import numpy

from mediator_was.simulation import *
from mediator_was.association import *

def trial(simulation, method, n_test=50000, alpha=0.05, ngwas_tests=1e6,
          n_gene_tests=1.5e4):
    """Count up significant hits for a new test set"""
    count = numpy.zeros(4)
    alpha /= numpy.array([ngwas_tests, n_gene_tests, n_gene_tests, n_gene_tests])
    genotypes, _, phenotype = getattr(simulation, method)(n_test)
    for g, ms, eqtl in zip(genotypes, simulation.models, simulation.eqtls):
        predicted_expression = numpy.array([m.predict(g) for m in ms])
        w = numpy.mean(predicted_expression, axis=0)
        sigma_ui = numpy.var(predicted_expression, axis=0)
        sigma_u = numpy.mean(sigma_ui)
        count += numpy.array([gwas(g, phenotype),
                              gwas(g[:,eqtl], phenotype),
                              t(w, phenotype)[1],
                              t(w, phenotype, sigma_u)[1]]) < alpha
    return count

def run_simulation(method, n_causal_eqtls, n_test, n_trials=10):
    """Run the simulation with the specified parameters"""
    numpy.random.seed(0)
    n_train = numpy.repeat(1000, 4)
    with simulation(n_causal_eqtls, n_train) as sim:
        result = sum(trial(sim, method, n_test=n_test) for _ in range(n_trials)) / (100 * n_trials)
        print(n_causal_eqtls, n_test, ' '.join('{:.3f}'.format(p) for p in result))

def power(n_causal_eqtls, n_test):
    """Estimate the power of the naive and corrected analyses."""
    run_simulation("simulate_mediated_phenotype", n_causal_eqtls, n_test)

def type_1_error_null(n_causal_eqtls, n_test):
    """Estimate the type 1 error rate of the naive and corrected analyses assuming
a null phenotype."""
    run_simulation("simulate_null_phenotype", n_causal_eqtls, n_test)

def type_1_error_independent(n_causal_eqtls, n_test):
    """Estimate the type 1 error rate of the naive and corrected analyses assuming
an independent phenotype."""
    run_simulation("simulate_independent_phenotype", n_causal_eqtls, n_test)

if __name__ == '__main__':
    method = sys.argv[1]
    n_causal_eqtls, n_test = [int(x) for x in sys.argv[2:]]
    getattr(sys.modules[__name__], method)(n_causal_eqtls, n_test)
