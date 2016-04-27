"""
Simulate a TWAS study design modularily.

Author: Kunal Bhutani   <kunalbhutani@gmail.com>
        Abhishek Sarkar <aksarkar@mit.edu>
"""
from __future__ import print_function

import collections
import contextlib
import operator
import pickle
import sys
import time

import numpy
import numpy.random as R
import sklearn.linear_model
import sklearn.metrics
import sklearn.utils


from mediator_was.modeling.association import *
from mediator_was.processing.helpers import load_hapgen, load_plink
import mediator_was.modeling.bayesian as bay


current_milli_time = lambda: int(round(time.time() * 1000))

def _add_noise(genetic_value, pve):
    """Add Gaussian noise to genetic values to achieve desired PVE.

    Assume true effects are N(0, 1) and sample errors from appropriately scaled
    Gaussian.

    """
    sigma = numpy.sqrt(numpy.var(genetic_value) * (1 / pve - 1))
    return genetic_value + R.normal(size=genetic_value.shape, scale=sigma)

def _fit_bootstrapped_EN(bootstrap_params, genotypes, expression):
    '''
    Fit bootstrapped ElasticNetCV

    Args:
        bootstrap_params: tuple(number of individuals per boostrap, number of bootstraps)
        genotypes
        expression

    Returns:
        array of ElasticNetCV models
    '''

    b_models = []
    model=sklearn.linear_model.ElasticNetCV
    for i in range(bootstrap_params[1]):
        b_genotypes, b_expression = sklearn.utils.resample(genotypes, expression, 
                                                           n_samples=bootstrap_params[0])
        b_model = model(max_iter=10000).fit(b_genotypes, b_expression)
        b_models.append(b_model)
    return b_models

def _fit_OLS(genotypes, expression):
    design = statsmodels.tools.add_constant(genotypes)
    return statsmodels.api.OLS(expression, design).fit()

class Gene():
    def __init__(self, name, plink_file, n_train=500, p_causal_eqtls=0.1, pve=0.17, pve_se=0.05,
                 n_causal=0, bootstrap_params=(350, 25)):
        '''
        A gene to simulate properties and train models for gene expression


        Args:
            If n_causal > 0, p_causal_eqtls is not used.
            bootstrap_params: tuple(number of individuals per boostrap, number of bootstraps)
 

        Attributes:
            haps: n x p pandas.DataFrame of n individual and p genotypes     
            causal_loci: numpy.array of size causal loci
                         with values as indices out of the p genotypes
            beta: numpy.array of size p of beta_exp
            train_ind: numpy.array of individual indices used for training
            train_genotypes, train_expression: training genotypes and expression array
            bootstrap_models: a list of bootstrapped ElasticNetCV models fit to training data
            bayesian_model: a bayesian Model of type expression

       
        TODO: allow for more complex noise models
       
        '''

        # Set parameters
        self.name = name
        self.id = '_'.join(map(str, [name, p_causal_eqtls, pve, current_milli_time()]))
        self.plink_file = plink_file
        self.n_train = n_train
        self.p_causal_eqtls = p_causal_eqtls
        self.pve = pve
        self.pve_se = pve_se
        self.n_causal = n_causal
        self.bootstrap_params = bootstrap_params

        # Generate data
        self._generate_params()
        self._train()

        return

    def _generate_params(self,):
        '''
        Load gene haplotypes and generate causal snps
        '''
        self.haps = load_plink(self.plink_file)
        n_snps = len(self.haps)
        self.haps.index = range(n_snps)
        if self.n_causal == 0:
            n_causal_eqtls = numpy.round(n_snps*self.p_causal_eqtls)
        else:
            n_causal_eqtls = self.n_causal
        self.causal_loci = self._sample_causal_loci(n_causal_eqtls)
        self.beta = numpy.zeros(n_snps)
        self.beta[self.causal_loci] = R.normal(size=n_causal_eqtls)
        self.train_ind = R.random_integers(low=0, high=self.haps.shape[1]-2, size=self.n_train)
        self.not_train = numpy.setdiff1d(numpy.arange(self.haps.shape[1]-2), self.train_ind)
        return

    def _sample_causal_loci(self, n_causal_snps, min_maf=0.05, max_maf=0.5):
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
        causal_loci = R.choice(self.haps.index, size=n_causal_snps)
        return causal_loci

    def _train(self):
        self.train_genotypes, self.train_expression = self.simulate(train=True)
        self.bootstrap_models = _fit_bootstrapped_EN(self.bootstrap_params,
                                                     self.train_genotypes,
                                                     self.train_expression)
        self.bayesian_model = bay.expression_model(self.train_genotypes,
                                                   self.train_expression)
        self.ols_model = _fit_OLS(self.train_genotypes, self.train_expression)
        return


    def simulate(self, n=1000, center=True, train=False):
        """Return genotypes at cis-eQTLs and cis-heritable gene expression.

        n - number of individuals
        ind_indices - array of individual indices [optional]
        """
        if train:
            ind_indices = self.train_ind
        else:
            ind_indices = R.choice(self.not_train, n)

        genotypes = self.haps[ind_indices]
        genotypes = genotypes.T.values.astype(float)
        if center:
            genotypes -= numpy.mean(genotypes, axis=0)[numpy.newaxis,:]
        pve = self.pve
        #pve = R.normal(params.pve, params.pve_se)
        #if pve <= 0: # Negative contribution of genotype to expression
        #    pve = 0.01
        expression = _add_noise(numpy.dot(genotypes, self.beta), pve)
        return genotypes, expression


class Study(object):
    def __init__(self, name, causal_genes, pve=0.2/100,
                 n_samples=5000):
        self.name = name
        self.causal_genes = causal_genes
        self.beta = R.normal(size=len(causal_genes))
        self.pve = pve
        self.n_samples = n_samples
        self._generate()
        return

    def _generate(self):
        '''
        Creates a gene map for future usage and simulates a mediated phenotype.
        '''
        self.gene_map = dict((gene.id, idx)
                             for idx, gene in enumerate(self.causal_genes))
        self.genotypes, self.expression, self.phenotype = self.simulate(self.n_samples)


    def simulate(self, n=50000, method="mediated"):
        """
        Return genotypes, true expression, and continuous phenotype.
            n - number of individuals
            method - phenotype specification
        """

        if method == "mediated":
            def phenotype(n, genotypes, expression):
                genetic_value = sum(b * e
                                    for e, b in zip(self.beta, expression))
                return _add_noise(genetic_value, self.pve)

        elif method == "independent":
            def phenotype(n, genotypes, expression):
                genetic_value = sum(numpy.dot(g, R.normal(size=gene.beta.shape))
                                    for g, gene in zip(genotypes, self.causal_genes))
                return _add_noise(genetic_value, self.pve)

        genotypes, expression = zip(*[gene.simulate(n, train=False)
                                    for gene in self.causal_genes])
        phenotype = phenotype(n, genotypes, expression)

        return genotypes, expression, phenotype


class Association(object):
    def __init__(self, name, gene, study):
        self.name = name
        self.gene = gene
        self.study = study
        self.phenotype = self.study.phenotype

        if gene.id in self.study.gene_map:
            self.genotype = self.study.genotypes[self.study.gene_map[gene.id]]
        else:
            self.genotype = self.gene.simulate(n=self.phenotypes.shape,
                                                train=False)
        self._fit_frequentist()
        self._fit_bayesian()

        return

    def _fit_frequentist(self):
        '''
        Fit frequentist methods for association
        '''
        genotype = self.genotype
        phenotype = self.phenotype

        # Bootstrapped
        bms = self.gene.bootstrap_models
        pred_expr = numpy.array([m.predict(genotype) for m in bms])
        w_bootstrap = numpy.mean(pred_expr, axis=0)
        sigma_ui_bootstrap = numpy.var(pred_expr, ddof=1, axis=0)

        # Single Model Fit
        ms = self.gene.ols_model
        design = statsmodels.tools.add_constant(genotype)
        w = ms.predict(design)
        sigma_ui = (design * numpy.dot(ms.cov_params(), design.T).T).sum(1)

        # Bayesian Multiple Imputation
        exp_model = self.gene.bayesian_model
        beta_exp_trace = exp_model.trace[-5000::100]['beta_exp']
        beta_exp_trace = numpy.array([beta_exp_trace[:, 0][:, idx]
                                     for idx in range(genotype.shape[1])])
        bay_pred_expr = numpy.apply_along_axis(lambda x: genotype.dot(x.T),
                                               0,
                                               beta_exp_trace).T


        association = {'OLS': t(w, phenotype, method="OLS"),
                       'OLS-ElasticNet': t(pred_expr[0], phenotype,
                                           method="OLS"),
                       'WLS': t(w, phenotype, sigma_ui, method="WLS"),
                       'Moment': t(w, phenotype, sigma_ui, method="moment"),
                       'Moment2': t(w, phenotype, sigma_ui, method="moment2"),
                       'Moment-Buo': t(w, phenotype, sigma_ui,
                                       method="momentbuo"),
                       'RC': t(w, phenotype, sigma_ui, method="rc"),
                       'RC-hetero-bootstrapped': t(w_bootstrap, phenotype,
                                                   sigma_ui_bootstrap,
                                                   method='rc-hetero'),
                       'RC-hetero-se': t(w, phenotype, sigma_ui,
                                         method="rc-hetero"),
                       # TODO: 'RC-Log' to have multiplicative error.
                       'Weighted': t(w, phenotype, sigma_ui,
                                     method="weighted"),
                       'MI-Bootstrapped': multiple_imputation(pred_expr, phenotype),
                       'MI-Bayesian': multiple_imputation(bay_pred_expr, phenotype),
                       }
        self.f_association = association
        return

    def _fit_bayesian(self, n_steps=10000):
        exp_model = self.gene.bayesian_model
        exp_trace = exp_model.beta_exp_trace[-n_steps:]
        pm_model = bay.phenotype_model_with_pm(self.genotype,
                                               self.phenotype,
                                               exp_trace)
        prior_model = bay.phenotype_model_with_prior(self.genotype,
                                                     self.phenotype,
                                                     exp_trace)
        full_model = bay.full_model(self.gene.train_genotypes,
                                    self.gene.train_expression,
                                    self.genotype,
                                    self.phenotype)
        self.bayesian_models = [pm_model, prior_model, full_model]
        return

class Power(object):
    def __init__(self, associations):
        return




