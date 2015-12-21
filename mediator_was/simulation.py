"""Simulate a TWAS study.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import numpy
import numpy.random
import sklearn.linear_model

import mediator_was.association

def _add_noise(genetic_value, pve):
    """Add Gaussian noise to genetic values to achieve desired PVE.

    Assume true effects are N(0, 1) and sample errors from appropriately scaled
    Gaussian.

    """
    sigma = numpy.var(genetic_value) * (1 / pve - 1)
    return numpy.random.normal(size=genetic_value.shape, scale=sigma)

def generate_gene_params(n_causal_snps, cis_pve=0.17, scale_by_maf=False):
    """Return a vector of minor allele frequencies and a vector of effect sizes.

    The average PVE by cis-genotypes on gene expression is 0.17.

    MAF sampled from U(0.5, 0.5). SNP effect sizes sampled from N(0, 1).

    """
    p = numpy.random.geometric(1 / n_causal_snps)
    maf = numpy.random.uniform(size=p, low=0.05, high=0.5)
    beta = numpy.random.normal(size=p)
    if scale_by_maf:
        # Scale effect sizes so each SNP explains equal variance
        beta /= numpy.sqrt(maf * (1 - maf))
    return maf, beta, cis_pve

def generate_sim_params(n_causal_genes=100, n_causal_snps=10, expression_pve=0.2):
    """Return causal gene effects and cis-eQTL parameters for causal genes.

    Each gene has an effect size sampled from N(0, 1). For each gene, generate
    MAF and (sparse) effect sizes for cis-SNPs.

    The PVE by high-effect loci associated to height is 0.2.

    """
    beta = numpy.random.normal(size=n_causal_genes)
    gene_params = [generate_gene_params(n_causal_snps) for _ in beta]
    return beta, gene_params, expression_pve

def simulate_gene(params, n=1000, pve=0.17):
    """Return genotypes at cis-eQTLs and cis-heritable gene expression.

    params - (maf, effect size, cis_pve) tuple
    n - number of individuals
    pve - proportion of variance explained by cis-eQTLs

    """
    maf, beta, pve = params
    genotypes = numpy.random.binomial(2, maf, size=(n, maf.shape[0]))
    expression = _add_noise(numpy.dot(genotypes, beta), pve)
    return genotypes, expression

def train(params, n=300, model=sklearn.linear_model.LinearRegression):
    """Train models on each cis-window.

    params - list of (maf, effect size, pve) tuples
    n - number of individuals
    model - sklearn estimator

    """
    _, gene_params, _ = params
    models = [model() for p in gene_params]
    for p, m in zip(gene_params, models):
        m.fit(*simulate_gene(params=p, n=n))
    return models

def test(params, n=5000):
    """Return genotypes, true expression, and continuous phenotype.

    params - simulation parameters
    n - number of individuals

    """
    beta, gene_params, pve = params
    phenotype = numpy.zeros(n)
    genotypes = []
    true_expression = []
    for p, b in zip(gene_params, beta):
        cis_genotypes, expression = simulate_gene(params=p, n=n)
        genotypes.append(cis_genotypes)
        true_expression.append(expression)
        phenotype += b * expression
    phenotype = _add_noise(phenotype, pve)
    return genotypes, true_expression, phenotype

def simulate(n_models=1):
    """Run a simulation.

    Sample n_models training cohorts to learn linear models of gene expression.

    Sample one test cohort, predict expression using each linear model in turn,
    and compute naive and corrected association statistics for phenotype
    against predicted expression.

    """
    params = generate_sim_params()
    models = [train(params) for _ in range(n_models)]
    genotypes, true_expression, phenotype = test(params)
    T = mediator_was.association.t
    for g, e, ms in zip(genotypes, true_expression, zip(*models)):
        predicted_expression = [m.predict(g) for m in ms]
        if len(ms) == 1:
            # TODO: what if you don't have real expression?
            raise NotImplementedError
        else:
            sigma_ui = numpy.var(numpy.array(predicted_expression), axis=0)
            sigma_u = numpy.var(sigma_ui)
        pvalues = [[T(p, phenotype) for p in predicted_expression],
                   [T(p, phenotype, sigma_u) for p in predicted_expression],
                   [T(p, phenotype, sigma_ui) for p in predicted_expression],]
        for p in zip(*pvalues):
            print('\t'.join('{:.3e}'.format(_) for _ in p))

if __name__ == '__main__':
    simulate(n_models=4)
