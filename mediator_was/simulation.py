"""Simulate a TWAS study.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import numpy
import numpy.random
import sklearn.linear_model

import mediator_was.association

def _errors(n, pve):
    """Sample errors to achieve desired PVE.

    Assume true effects are N(0, 1) and sample errors from appropriately scaled
    Gaussian.

    """
    return numpy.random.normal(size=n, scale=(1 / pve - 1))

def generate_gene_params(n_cis_snps, n_causal_snps, cis_pve=0.17, scale_by_maf=False):
    """Return a vector of minor allele frequencies and a vector of effect sizes.

    The average PVE by cis-genotypes on gene expression is 0.17.

    MAF sampled from U(0.5, 0.5). SNP effect sizes sampled from N(0, 1).

    """
    maf = numpy.random.uniform(size=n_cis_snps, low=0.05, high=0.5)
    beta = numpy.zeros(n_cis_snps)
    beta[:n_causal_snps] = numpy.random.normal(size=n_causal_snps)
    if scale_by_maf:
        # Scale effect sizes so each SNP explains equal variance
        beta /= numpy.sqrt(maf * (1 - maf))
    return maf, beta, cis_pve

def generate_sim_params(n_causal_genes=100, n_cis_snps=200, n_causal_snps=10,
                        expression_pve=0.2):
    """Return causal gene effects and cis-eQTL parameters for causal genes.

    Each gene has an effect size sampled from N(0, 1). For each gene, generate
    MAF and (sparse) effect sizes for cis-SNPs.

    The PVE by high-effect loci associated to height is 0.2.

    """
    beta = numpy.random.normal(size=n_causal_genes)
    genes = [generate_gene_params(n_cis_snps, n_causal_snps) for _ in beta]
    return beta, genes, expression_pve

def simulate_gene(params, n=1000, pve=0.17):
    """Return a (n,) array of cis-heritable gene expression and (n,p) array of
    cis-genotypes.

    n - number of individuals
    p - number of SNPs
    params - (maf, effect size, cis_pve) tuple
    pve - proportion of variance explained by cis-eQTLs

    """
    maf, beta, pve = params
    genotypes = numpy.random.binomial(2, maf, size=(n, maf.shape[0]))
    expression = numpy.dot(genotypes, beta) + _errors(n, pve)
    return genotypes, expression

def train(params, n=300, model=sklearn.linear_model.ElasticNet):
    """Train models on each cis-window.

    n - number of individuals
    model - sklearn estimator
    params - list of (maf, effect size, pve) tuples

    """
    models = [model() for p in params]
    for p, m in zip(params, models):
        m.fit(*simulate_gene(params=p, n=n))
    return models

def test(params, models, n=5000):
    """Return a vector of continuous phenotype values.

    n - number of individuals
    pve - proportion of variance explained by cis-heritable expression of causal genes
    sim_params - gene effect sizes, cis-eQTL parameters

    """
    beta, gene_params, pve = sim_params
    phenotype = numpy.zeros(n)
    true_expression = []
    predicted_expression = []
    for p, m in zip(gene_params, models):
        cis_genotypes, expression = simulate_gene(params=p, n=n)
        true_expression.append(expression)
        predicted_expression.append(m.predict(cis_genotypes))
        phenotype += numpy.dot(expression, beta)
    phenotype += _errors(n, pve)

    L = mediator_was.association.lrt
    for true, predicted in zip(true_expression, predicted_expression):
        p = [mediator_was.association.lrt_naive(true, phenotype),
             L(predicted, phenotype, numpy.std(predicted - true)),
             L(predicted, phenotype, numpy.std(predicted - numpy.mean(predicted))),
        ]
        print(*p)
