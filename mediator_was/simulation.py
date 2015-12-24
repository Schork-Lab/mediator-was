"""Simulate a TWAS study.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import collections
import contextlib
import operator
import pickle
import sys

import numpy
import numpy.random as R
import sklearn.linear_model
import sklearn.metrics

from mediator_was.association import *

Gene = collections.namedtuple('Gene', ['maf', 'beta', 'pve'])
Phenotype = collections.namedtuple('Phenotype', ['beta', 'genes', 'pve'])

def _add_noise(genetic_value, pve):
    """Add Gaussian noise to genetic values to achieve desired PVE.

    Assume true effects are N(0, 1) and sample errors from appropriately scaled
    Gaussian.

    """
    sigma = numpy.sqrt(numpy.var(genetic_value) * (1 / pve - 1))
    return genetic_value + R.normal(size=genetic_value.shape, scale=sigma)

def _generate_gene_params(n_causal_eqtls, pve=0.17):
    """Return a vector of minor allele frequencies and a vector of effect sizes.

    The average PVE by cis-genotypes on gene expression is 0.17. Assume the
    average number of causal cis-eQTLs is 10.

    """
    maf = R.uniform(size=n_causal_eqtls, low=0.05, high=0.5)
    beta = R.normal(size=n_causal_eqtls)
    return Gene(maf, beta, pve)

def _generate_phenotype_params(n_causal_genes=100, n_causal_eqtls=10, pve=0.2):
    """Return causal gene effects and cis-eQTL parameters for causal genes.

    Each gene has an effect size sampled from N(0, 1). For each gene, generate
    MAF and (sparse) effect sizes for cis-SNPs.

    The PVE by high-effect loci associated to height is 0.2.

    """
    beta = R.normal(size=n_causal_genes)
    genes = [_generate_gene_params(n_causal_eqtls) for _ in beta]
    return Phenotype(beta, genes, pve)

def _simulate_gene(params, n=1000, center=True):
    """Return genotypes at cis-eQTLs and cis-heritable gene expression.

    n - number of individuals

    """
    genotypes = R.binomial(2, params.maf, size=(n, params.maf.shape[0])).astype(float)
    if center:
        genotypes -= numpy.mean(genotypes, axis=0)[numpy.newaxis,:]
    expression = _add_noise(numpy.dot(genotypes, params.beta), params.pve)
    return genotypes, expression

class Simulation(object):
    def __init__(self, n_causal_genes=100, n_causal_eqtls=10, n_train=None):
        """Initialize a new simulation"""
        R.seed(0)
        if n_train is None:
            n_train = numpy.repeat(1000, 4)
        elif numpy.isscalar(n_train):
            n_train = numpy.array([n_train])
        self.params = _generate_phenotype_params(n_causal_genes, n_causal_eqtls)
        self.models = list(zip(*[self._train(n) for n in n_train]))
        self.eqtls = self._eqtls(n_train[0])

    def _train(self, n=1000, model=sklearn.linear_model.LinearRegression):
        """Train models on each cis-window.

        n - number of individuals
        model - sklearn estimator

        """
        models = [model() for _ in self.params.genes]
        for p, m in zip(self.params.genes, models):
            genotypes, expression = _simulate_gene(params=p, n=n)
            m.fit(genotypes, expression)
        return models

    def _eqtls(self, n=1000):
        """Identify eQTLs in each cis-window."""
        eqtls = []
        for p, m in zip(self.params.genes, self.models):
            genotypes, expression = _simulate_gene(params=p, n=n)
            eqtls.append(min((t(g, expression)[1], i) for i, g in enumerate(genotypes.T))[1])
        return eqtls

    def _test(self, method, n=50000):
        """Return genotypes, true expression, and continuous phenotype.

        n - number of individuals
        method - phenotype specification

        """
        genotypes, expression = zip(*[_simulate_gene(p, n) for p in self.params.genes])
        phenotype = method(n, genotypes, expression)
        return genotypes, expression, phenotype

    def simulate_null_phenotype(self, n=50000):
        """Simulate unmediated phenotype"""
        return self._test(lambda n, *args: R.normal(size=n), n)

    def simulate_independent_phenotype(self, n=50000):
        """Simulate phenotype with independent effects at causal eQTLs"""
        def phenotype(n, genotypes, expression):
            genetic_value = sum(numpy.dot(g, R.normal(size=p.freq.shape))
                                for g, p in zip(genotypes, self.params.genes))
            return _add_noise(genetic_value, self.params.pve)
        return self._test(phenotype, n)

    def simulate_mediated_phenotype(self, n=50000):
        """Simulate an expression mediated phenotype"""
        def phenotype(n, genotypes, expression):
            genetic_value = sum(b * e for e, b in zip(self.params.beta, expression))
            return _add_noise(genetic_value, self.params.pve)
        return self._test(phenotype, n)

    def trace(self):
        """Trace through the naive and corrected analyses."""
        genotypes, true_expression, phenotype = self.simulate_null_phenotype(n=5000)
        print('naive_p', 'corrected_p', 'true_p', 'sigma_x', 'sigma_u', 'pve')
        for g, e, ms, p in zip(genotypes, true_expression, self.models, self.params.genes):
            predicted_expression = numpy.array([m.predict(g) for m in ms])
            if len(ms) == 1:
                # TODO: errors from linear model
                raise NotImplementedError
            else:
                w = numpy.mean(predicted_expression, axis=0)
                sigma_ui = numpy.var(predicted_expression, axis=0)
                sigma_u = numpy.mean(sigma_ui)
            naive_se, naive_p = t(w, phenotype)
            corrected_se, corrected_p = t(w, phenotype, sigma_u)
            true_se, true_p = t(e, phenotype)
            outputs = [naive_p, corrected_p, true_p,
                       numpy.var(predicted_expression) - numpy.mean(sigma_u),
                       numpy.mean(sigma_u),
                       p.pve,
            ]
            print('\t'.join('{:.3g}'.format(o) for o in outputs))

@contextlib.contextmanager
def simulation(n_causal_eqtls, n_train):
    """Retrieve a cached simulation, or run one if no cached simulation exists.

    Sample n_models training cohorts to learn linear models of gene expression.

    Sample one test cohort, predict expression using each linear model in turn,
    and compute naive and corrected association statistics for phenotype
    against predicted expression.

    """
    key = 'simulation-{}-{}.pkl'.format(n_causal_eqtls, n_train)
    hit = False
    try:
        with open(key, 'rb') as f:
            sim = pickle.load(f)
            hit = True
    except:
        sim = Simulation(n_causal_eqtls=n_causal_eqtls, n_train=n_train)
    try:
        yield sim
    except Exception as e:
        raise e
    finally:
        if not hit:
            with open(key, 'wb') as f:
                pickle.dump(sim, f)

if __name__ == '__main__':
    with simulation(100, numpy.repeat(1000, 4)) as sim:
        sim.trace()
