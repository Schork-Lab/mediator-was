"""Simulate a TWAS study.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import collections
import contextlib
import pickle
import sys

import numpy
import numpy.random
import sklearn.linear_model
import sklearn.metrics

import mediator_was.association

Gene = collections.namedtuple('Gene', ['maf', 'beta', 'pve'])
Phenotype = collections.namedtuple('Phenotype', ['beta', 'genes', 'pve'])

def _add_noise(genetic_value, pve):
    """Add Gaussian noise to genetic values to achieve desired PVE.

    Assume true effects are N(0, 1) and sample errors from appropriately scaled
    Gaussian.

    """
    sigma = numpy.sqrt(numpy.var(genetic_value) * (1 / pve - 1))
    return genetic_value + numpy.random.normal(size=genetic_value.shape, scale=sigma)

def generate_gene_params(scale_by_maf=False):
    """Return a vector of minor allele frequencies and a vector of effect sizes.

    The average PVE by cis-genotypes on gene expression is 0.17. Assume the
    average number of causal cis-eQTLs is 10.

    """
    p = numpy.random.geometric(1 / 10)
    pve = numpy.random.beta(1, 1 / .17)
    maf = numpy.random.uniform(size=p, low=0.05, high=0.5)
    beta = numpy.random.normal(size=p)
    return Gene(maf, beta, pve)

def generate_phenotype_params(n_causal_genes=100, pve=0.2):
    """Return causal gene effects and cis-eQTL parameters for causal genes.

    Each gene has an effect size sampled from N(0, 1). For each gene, generate
    MAF and (sparse) effect sizes for cis-SNPs.

    The PVE by high-effect loci associated to height is 0.2.

    """
    beta = numpy.random.normal(size=n_causal_genes)
    genes = [generate_gene_params() for _ in beta]
    return Phenotype(beta, genes, pve)

def simulate_gene(params, n=1000):
    """Return genotypes at cis-eQTLs and cis-heritable gene expression.

    params - (maf, effect size, cis_pve) tuple
    n - number of individuals

    """
    genotypes = numpy.random.binomial(2, params.maf, size=(n, params.maf.shape[0]))
    expression = _add_noise(numpy.dot(genotypes, params.beta), params.pve)
    return genotypes, expression

def simulate_train(params, n=1000, model=sklearn.linear_model.LinearRegression):
    """Train models on each cis-window.

    params - list of (maf, effect size, pve) tuples
    n - number of individuals
    model - sklearn estimator

    """
    models = [model() for _ in params.genes]
    for p, m in zip(params.genes, models):
        genotypes, expression = simulate_gene(params=p, n=n)
        m.fit(genotypes, expression)
    return models

def simulate_test(params, n=25000):
    """Return genotypes, true expression, and continuous phenotype.

    params - simulation parameters
    n - number of individuals

    """
    genetic_value = numpy.zeros(n)
    genotypes = []
    true_expression = []
    for p, b in zip(params.genes, params.beta):
        cis_genotypes, total_expression = simulate_gene(params=p, n=n)
        genotypes.append(cis_genotypes)
        true_expression.append(total_expression)
        genetic_value += b * total_expression
    phenotype = _add_noise(genetic_value, params.pve)
    return genotypes, true_expression, phenotype

@contextlib.contextmanager
def simulation(n_models=4):
    """Retrieve a cached simulation, or run one if no cached simulation exists.

    Sample n_models training cohorts to learn linear models of gene expression.

    Sample one test cohort, predict expression using each linear model in turn,
    and compute naive and corrected association statistics for phenotype
    against predicted expression.

    """
    hit = False
    try:
        with open('simulation.pkl', 'rb') as f:
            sim = pickle.load(f)
            hit = True
    except:
        params = generate_phenotype_params()
        models = [simulate_train(params) for _ in range(n_models)]
        genotypes, true_expression, phenotype = simulate_test(params)
        sim = (params, models, genotypes, true_expression, phenotype)
    try:
        yield sim
    except Exception as e:
        raise e
    finally:
        if not hit:
            with open('simulation.pkl', 'wb') as f:
                pickle.dump(sim, f)

def _trial(test, params, models, n_total_tests=15000, alpha=0.05):
    """Count up significant hits for a new test set"""
    count = numpy.zeros(2)
    genotypes, true_expression, phenotype = simulate_test(params)
    for g, e, ms in zip(genotypes, true_expression, zip(*models)):
        predicted_expression = numpy.array([m.predict(g) for m in ms])
        w = numpy.mean(predicted_expression, axis=0)
        sigma_ui = numpy.var(predicted_expression, axis=0)
        sigma_u = numpy.mean(sigma_ui)
        alpha = alpha / n_total_tests
        if test(w, phenotype)[1] < alpha:
            count[0] += 1
        if test(w, phenotype, sigma_u)[1] < alpha:
            count[1] += 1
    return count

def power(test, params, models, *args, n_trials=10):
    """Estimate the power of the naive and corrected analyses."""
    count = numpy.zeros(2)
    for _ in range(n_trials):
        count += _trial(test, params, models)
    return count / (100 * n_trials)

def trace(params, models, genotypes, true_expression, phenotype):
    """Trace through the naive and corrected analyses."""
    T = mediator_was.association.t
    print('naive_p', 'corrected_p', 'naive_se', 'corrected_se', 'sigma_x', 'sigma_u', 'pve')
    for g, e, ms, p in zip(genotypes, true_expression, zip(*models), params.genes):
        predicted_expression = numpy.array([m.predict(g) for m in ms])
        if len(ms) == 1:
            # TODO: errors from linear model
            raise NotImplementedError
        else:
            w = numpy.mean(predicted_expression, axis=0)
            sigma_ui = numpy.var(predicted_expression, axis=0)
            sigma_u = numpy.mean(sigma_ui)
        naive_se, naive_p = T(w, phenotype)
        corrected_se, corrected_p = T(w, phenotype, sigma_u)
        outputs = [naive_p, corrected_p,
                   naive_se, corrected_se,
                   numpy.var(predicted_expression) - numpy.mean(sigma_u),
                   numpy.mean(sigma_u),
                   p.pve,
        ]
        print('\t'.join('{:.3g}'.format(o) for o in outputs))

if __name__ == '__main__':
    numpy.random.seed(0)
    with simulation() as sim:
        trace(*sim)
