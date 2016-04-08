"""Simulate a TWAS study.

Author: Abhishek Sarkar <aksarkar@mit.edu>
        Kunal Bhutani   <kunalbhutani@gmail.com>
"""
from __future__ import print_function

import collections
import contextlib
import operator
import pickle
import sys

import numpy
import numpy.random as R
import sklearn.linear_model
import sklearn.metrics
import sklearn.utils


from mediator_was.association import *
from mediator_was.processing.helpers import load_hapgen, load_plink


Gene = collections.namedtuple('Gene', ['maf', 'haps', 'beta', 'pve', 'pve_se'])
Phenotype = collections.namedtuple('Phenotype', ['beta', 'genes', 'pve'])



def _sample_causal_loci(haps, n_causal_snps, min_maf=0.05, max_maf=0.5):
    """
    Sample gene specific causal loci based on a haplotype dataframe
    restricted to a minor allele frequency range and total number
    of causal snps
    """
    # maf = haps.apply(lambda x: numpy.sum(x)/len(x), axis=1)
    # causal_loci = numpy.where((maf > min_maf) & (maf < max_maf))[0]
    # if len(causal_loci) < n_causal_snps:
    #     print('Warning: not enough causal loci in provided haplotypes', file=sys.stderr)
    # causal_loci = R.choice(causal_loci, size=n_causal_snps)
    causal_loci = R.choice(haps.index, size=n_causal_snps)
    return causal_loci

def _add_noise(genetic_value, pve):
    """Add Gaussian noise to genetic values to achieve desired PVE.

    Assume true effects are N(0, 1) and sample errors from appropriately scaled
    Gaussian.

    """
    sigma = numpy.sqrt(numpy.var(genetic_value) * (1 / pve - 1))
    return genetic_value + R.normal(size=genetic_value.shape, scale=sigma)

def _generate_gene_params(p_causal_eqtls, hap_df=None, pve=0.17, pve_se=0.05, n_snps=80):
    """Return a matrix of haplotypes or a vector minor allele frequencies and a vector of effect sizes.
    The average PVE by cis-genotypes on gene expression is 0.17. Assume the
    average number of causal cis-eQTLs is 10.

    """
    print(p_causal_eqtls)
    if hap_df is not None:
        haps = hap_df
        maf = None
        n_snps = len(haps)
        haps.index = range(n_snps)
        n_causal_eqtls = numpy.round(n_snps*p_causal_eqtls)
        causal_loci = _sample_causal_loci(haps, n_causal_eqtls)
    else:
        haps = None
        maf = R.uniform(size=n_snps, low=0.05, high=0.5)
        n_causal_eqtls = numpy.round(n_snps*p_causal_eqtls)
        causal_loci = R.random_integers(0, n_snps-1, size=n_causal_eqtls)

    beta = numpy.zeros(n_snps)
    beta[causal_loci] = R.normal(size=n_causal_eqtls)
    return Gene(maf, haps, beta, pve, pve_se)

def _generate_phenotype_params(n_causal_genes=100, p_causal_eqtls=.1, pve=0.2, hapgen=None, plink=None):
    """Return causal gene effects and cis-eQTL parameters for causal genes.

    Each gene has an effect size sampled from N(0, 1). For each gene, generate
    MAF and (sparse) effect sizes for cis-SNPs.

    The PVE by high-effect loci associated to height is 0.2.

    """
    if hapgen:
        n_causal_genes = len(hapgen)
        genes = [_generate_gene_params(p_causal_eqtls, load_hapgen(gene)) 
                 for gene in hapgen]
    elif plink:
        n_causal_genes = len(plink)
        genes = [_generate_gene_params(p_causal_eqtls, load_plink(gene)) 
                 for gene in plink]
    else:
        genes = [_generate_gene_params(p_causal_eqtls) 
                 for _ in range(n_causal_genes)]  

    beta = R.normal(size=n_causal_genes)
    return Phenotype(beta, genes, pve)

def _simulate_gene(params, n=1000, center=True):
    """Return genotypes at cis-eQTLs and cis-heritable gene expression.

    n - number of individuals
    same_cohort - boolean (default:False), choose the first n individuals
            so that the training cohort remains the same.

    """
    if params.maf is not None:
        genotypes = R.binomial(2, params.maf, size=(n, params.maf.shape[0])).astype(float)
    else:
        genotypes = params.haps[R.random_integers(low=0, high=params.haps.shape[1]-2, size=n)]
        genotypes = genotypes.T.values.astype(float)
    if center:
        genotypes -= numpy.mean(genotypes, axis=0)[numpy.newaxis,:]
    pve = R.normal(params.pve, params.pve_se)
    if pve <= 0: # Negative contribution of genotype to expression
        pve = 0.01
    expression = _add_noise(numpy.dot(genotypes, params.beta), pve)
    return genotypes, expression

class Simulation(object):
    def __init__(self, n_causal_genes=100, p_causal_eqtls=.1, 
                 bootstrap=None, n_train=None,
                 hapgen=None, plink=None, pve=0.2/50):
        """Initialize a new simulation.

        
        There are two ways to train the G -> E models.

        interstudy:
        n_train - a numpy array specifying # of individuals in different studies
        
        intrastudy:
        bootstrap - tuple (# samples in study, # samples per bootstrap, # bootstraps) 
                     specifying models using bootstrapped samples from the same study. 

        There are three ways to generate the genotypes:

        1. Use independent genotypes generated using a maf range of 0.05 to 0.5
        2. Use a separate hapgen2 genotype file for each gene  
        3. Use a separate plink genotype file for each gene 

        Note if hapgen or plink are set, the number of causal genes is
        determined by the number of items in hapgen or plink lists.

        hapgen - list of path to the hapgen2 generated haps file. Default: None
        plink - list of paths to plink-prefix for bed files. Default: None


        """
        #R.seed(0)
        if n_train is None:
            n_train = numpy.repeat(1000, 4)
        elif numpy.isscalar(n_train):
            n_train = numpy.array([n_train])
        self.params = _generate_phenotype_params(n_causal_genes, p_causal_eqtls, 
                                                 hapgen=hapgen, plink=plink, pve=pve)
        self.models = list(zip(*[self._train(n) for n in n_train]))
        if bootstrap:
            self.b_models = self._b_train(bootstrap)

        #self.eqtls = self._eqtls(n_train[0])

    def _b_train(self, bootstrap):
        b_models = []
        for p in self.params.genes:
            genotypes, expression = _simulate_gene(params=p, n=bootstrap[0])
            gene_models = []
            for i in range(bootstrap[2]):
                b_genotypes, b_expression = sklearn.utils.resample(genotypes, expression, 
                                                                   n_samples=bootstrap[1])
                model = sklearn.linear_model.ElasticNetCV()
                model.fit(b_genotypes, b_expression)
                gene_models.append(model)
            b_models.append(gene_models)
        return b_models

    def _train(self, n=1000, model=statsmodels.api.OLS):
        """Train models on each cis-window.

        n - number of individuals

        """
        # models = [model() for _ in self.params.genes]
        # for p, m in zip(self.params.genes, models):
        #     genotypes, expression = _simulate_gene(params=p, n=n)
        #     m.fit(genotypes, expression)
        models = []
        for p in self.params.genes:
            genotypes, expression = _simulate_gene(params=p, n=n)
            
            design = statsmodels.tools.add_constant(genotypes)
            
            models.append(model(expression, design).fit())

        return models

    def _eqtls(self, n=1000):
        """Identify eQTLs in each cis-window."""
        eqtls = []
        for p, m in zip(self.params.genes, self.models):
            genotypes, expression = _simulate_gene(params=p, n=n)
            eqtls.append(min((t(g, expression, method="OLS")[2], i) 
                            for i, g in enumerate(genotypes.T))[1])
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
            genetic_value = sum(numpy.dot(g, R.normal(size=p.beta.shape))
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
            if sigma_u > numpy.var(w):
                # TODO: Not sure if necessary to handle this.
                print("Warning: sigma_u less than sigma_w")
                continue
            naive_coeff, naive_se, naive_p = t(w, phenotype)
            corrected_coeff, corrected_se, corrected_p = t(w, phenotype, sigma_u)
            true_coeff, true_se, true_p = t(e, phenotype)
            outputs = [naive_p, corrected_p, true_p,
                       numpy.var(predicted_expression) - numpy.mean(sigma_u),
                       numpy.mean(sigma_u),
                       p.pve,
                       ]
            print('\t'.join('{:.3g}'.format(o) for o in outputs))

@contextlib.contextmanager
def simulation(p_causal_eqtls, n_train, n_causal_genes, bootstrap=None, hapgen=None, plink=None):
    """Retrieve a cached simulation, or run one if no cached simulation exists.

    Sample n_models training cohorts to learn linear models of gene expression.

    Sample one test cohort, predict expression using each linear model in turn,
    and compute naive and corrected association statistics for phenotype
    against predicted expression.

    """
    key = 'simulation-{}-{}-{}.pkl'.format(p_causal_eqtls, n_train, n_causal_genes)
    hit = False
    # try:
    #     with open(key, 'rb') as f:
    #         sim = pickle.load(f)
    #         hit = True
    # except:
    hit = False
    sim = Simulation(n_causal_genes=n_causal_genes, p_causal_eqtls=p_causal_eqtls, 
                     n_train=n_train, bootstrap=bootstrap, hapgen=hapgen, plink=plink)
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
