"""
Simulate a TWAS study design modularily.

Author: Kunal Bhutani   <kunalbhutani@gmail.com>
        Abhishek Sarkar <aksarkar@mit.edu>
"""
from __future__ import print_function

import pickle
import time
import glob

import numpy
import numpy.random as R
import sklearn.linear_model
import sklearn.metrics
import sklearn.utils
import pandas as pd
import pymc3 as pm


from mediator_was.modeling.association import *
from mediator_was.processing.helpers import load_plink
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
        bootstrap_params: tuple(number of individuals per bootstrap, number of bootstraps)
        genotypes
        expression

    Returns:
        array of ElasticNetCV models
    '''

    b_models = []
    model = sklearn.linear_model.ElasticNetCV
    
    # Fit entire model to get estimates of /alpha and /lambda
    l1_ratio_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    full_model = model(l1_ratio=l1_ratio_range, max_iter=10000)
    full_model.fit(genotypes, expression)

    model = sklearn.linear_model.ElasticNet
    for i in range(bootstrap_params[1]):
        b_genotypes, b_expression = sklearn.utils.resample(genotypes, expression, 
                                                           n_samples=bootstrap_params[0])
        b_model = model(max_iter=10000, alpha=full_model.alpha_, l1_ratio=full_model.l1_ratio_).fit(b_genotypes, b_expression)
        b_models.append(b_model)
    return b_models

def _fit_OLS(genotypes, expression):
    design = statsmodels.tools.add_constant(genotypes)
    return statsmodels.api.OLS(expression, design).fit()

class Gene():
    '''
     A gene to simulate properties and train models for gene expression


    Args:
        If p_causal_eqtls = 0, number of causal snps = 1
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

    def __init__(self, name, plink_file, n_train=500, p_causal_eqtls=0.1, pve=0.17, pve_se=0.05,
                 bootstrap_params=(350, 25), seed=0, haps=None):

        # Set parameters
        R.seed(seed)
        self.name = name
        self.id = '_'.join(map(str, [name, p_causal_eqtls,
         pve, seed, current_milli_time()]))
        self.plink_file = plink_file
        self.n_train = n_train
        self.p_causal_eqtls = p_causal_eqtls
        self.pve = pve
        self.pve_se = pve_se
        self.bootstrap_params = bootstrap_params

        # Generate data
        self._generate_params(haps)
        self._train()

        return

    def _generate_params(self, haps=None):
        '''
        Load gene haplotypes and generate causal snps
        '''
        if haps is None:
            self.haps = load_plink(self.plink_file) 
        else:
            self.haps = haps
        n_snps = len(self.haps)
        self.haps.index = range(n_snps)
        if self.p_causal_eqtls == 0:
            n_causal_eqtls = 1
        else:
            n_causal_eqtls = int(numpy.round(n_snps*self.p_causal_eqtls))
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
        #self.bayesian_model = bay.expression_model(self.train_genotypes,
        #                                           self.train_expression)
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
    '''
    Study class for saving causal genes, genotypes, and phenotypes

    Args:
        causal_genes - list of gene objects, length g 
                       NOTE: not saving this anymore to save space.
        pve - percentage of variance explained by genotypes
        n_samples - number of samples

    Attributes:
        beta - beta_phen for each gene
        genotypes - array of genotypes, length g ; 
                    each entry is of length g.snp x n
        expression - g x n array of expression 
        phenotype - 1 x n array of phenotype
        gene_map - dict((gene_id, gene_idx))
    '''

    def __init__(self, name, causal_genes, pve=0.2,
                 n_samples=5000, seed=0):
        R.seed(seed)
        self.name = name
        self.id = '_'.join(map(str, [name, pve, len(causal_genes), seed, current_milli_time(),]))
        self.causal_genes = [gene.id for gene in causal_genes]
        #self.causal_genes = causal_genes
        self.beta = numpy.ones(len(causal_genes))
        #self.beta = R.normal(size=len(causal_genes))
        self.pve = pve
        self.n_samples = n_samples
        self._generate(causal_genes)
        return

    def _generate(self, causal_genes):
        '''
        Creates a gene map for future usage and simulates a mediated phenotype.
        '''
        self.gene_map = dict((gene.id, idx)
                             for idx, gene in enumerate(causal_genes))
        self.genotypes, self.expression, self.phenotype = self.simulate(causal_genes, self.n_samples)
        self.oos_genotypes, self.oos_expression, self.oos_phenotype = self.simulate(causal_genes, self.n_samples)

    def simulate(self, causal_genes, n=50000, method="mediated"):
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
                                    for g, gene in zip(genotypes, causal_genes))
                return _add_noise(genetic_value, self.pve)

        genotypes, expression = zip(*[gene.simulate(n, train=False)
                                    for gene in causal_genes])
        phenotype = phenotype(n, genotypes, expression)

        return genotypes, expression, phenotype


class Association(object):
    '''
    Each gene - study relationship has its own association object.   
    Fit frequentist associations and bayesian models.


    Args:
        gene - gene.id
        study - study.id
        
    Attributes:
        phenotype - study.phenotype
        genotype -   
            If study.causal_genes does not include gene,
                 generate random set of genotypes from gene
            Otherwise, use the genotypes used to generate study.phenotype
        beta - simulated beta from study for gene
        expected_pve - expected pve based on other gene's beta and study pve
        f_association - dictionary of association statistics from frequentist tests
        bayesian_models - list(pm_model, prior_model, full_model)


    '''
    def __init__(self, name, gene, study, seed=0):
        R.seed(seed)
        self.name = name
        self.gene = gene.id
        self.study = study.id
        self.phenotype = study.phenotype
        self.oos_phenotype = study.oos_phenotype

        if gene.id in study.gene_map:
            self.genotype = study.genotypes[study.gene_map[gene.id]]
            self.oos_genotype = study.oos_genotypes[study.gene_map[gene.id]]
            self.beta = study.beta[study.gene_map[gene.id]]
            self.expected_pve = (self.beta/sum(study.beta))*study.pve
        else:
            self.genotype, _ = gene.simulate(n=self.phenotype.shape,
                                             train=False)
            self.oos_genotype, _ = gene.simulate(n=self.phenotype.shape,
                                             train=False)
            self.beta = 0
            self.expected_pve = 0 


        self._test_frequentist(gene)
        self._fit_bayesian(gene, pm=False, prior=False, full=False, ts=False)
        self._test_bayesian(gene)
        
        return

    def _test_frequentist(self, gene):
        '''
        Fit frequentist methods for association
        '''
        genotype = self.genotype
        phenotype = self.phenotype

        # Single Model Fit
        ms = gene.ols_model
        design = statsmodels.tools.add_constant(genotype)
        w = ms.predict(design)
        sigma_ui = (design * numpy.dot(ms.cov_params(), design.T).T).sum(1)

        # Bootstrapped
        bms = gene.bootstrap_models
        pred_expr = numpy.array([m.predict(genotype) for m in bms])
        w_bootstrap = numpy.mean(pred_expr, axis=0)
        sigma_ui_bootstrap = numpy.var(pred_expr, ddof=1, axis=0)

        association = {'OLS': t(w, phenotype, method="OLS"),
                       'OLS-ElasticNet': t(pred_expr[0], phenotype,
                                           method="OLS"),
                       'RC-hetero-bootstrapped': t(w_bootstrap, phenotype,
                                                   sigma_ui_bootstrap),
                       'MI-Bootstrapped': multiple_imputation(pred_expr, phenotype),
                       }
        self.f_association = association
        return

    def _fit_bayesian(self, gene, n_steps=10000, 
                      pm=True, prior=True, full=True,
                      ts=True, ts_variational=True):
        '''
        Fit Bayesian Models

            pm: Posterior mean
            prior: Moment matching
            full: Joint using MCMC
            ts: two stage (first stage: ElasticNet)
            ts_variational: two 
        '''

        models = []
        if pm or prior:
            exp_model = gene.bayesian_model
            exp_trace = exp_model.beta_exp_trace[-n_steps:]
        if pm:
            pm_model = bay.phenotype_model_with_pm(self.genotype,
                                                   self.phenotype,
                                                   exp_trace)
            models.append(pm_model)
        if prior:
            prior_model = bay.phenotype_model_with_prior(self.genotype,
                                                         self.phenotype,
                                                         exp_trace)
            models.append(prior_model)
        if full:
            full_model = bay.full_model(gene.train_genotypes,
                                        gene.train_expression,
                                        self.genotype,
                                        self.phenotype)
            models.append(full_model)
        if ts:
            coefs = pd.DataFrame([model.coef_ for model in gene.bootstrap_models])
            ts_model = bay.two_stage_model(coefs,
                                           self.genotype,
                                           self.phenotype)
            models.append(ts_model)
        if ts_variational:
            coefs = pd.DataFrame([model.coef_ for model in gene.bootstrap_models])
            ts_variational_model = bay.two_stage_variational_model(coefs,
                                                                   self.genotype,
                                                                   self.phenotype)
            full_variational = bay.full_variational_model(gene.train_genotypes,
                                                            gene.train_expression,
                                                            self.genotype,
                                                            self.phenotype)
            full_variational_hs = bay.full_variational_hs_model(gene.train_genotypes,
                                                gene.train_expression,
                                                self.genotype,
                                                self.phenotype)
            full_variational_mb = bay.full_variational_mb_model(gene.train_genotypes,
                                                            gene.train_expression,
                                                            self.genotype,
                                                            self.phenotype)
            full_variational_hs_mb = bay.full_variational_hs_mb_model(gene.train_genotypes,
                                                            gene.train_expression,
                                                            self.genotype,
                                                            self.phenotype)

            models.append(ts_variational_model)
            models.append(full_variational)
            models.append(full_variational_hs)
            models.append(full_variational_mb)
            models.append(full_variational_hs_mb)
        self.bayesian_models = models
        return

    def _test_bayesian(self, gene, mse=True, bf=False):
        if mse:
            mse = dict((model.type,
                        bay.compute_mse_oos(self.oos_genotype,
                                            self.oos_phenotype,
                                            model))
                       for model in self.bayesian_models)
            self.b_mse = mse


        if bf:
            bf = dict(('BF-'+model.type,
                       bay.bayes_factor(model, self.genotype, self.phenotype,
                                        gene.train_genotypes, gene.train_expression))
                        for model in self.bayesian_models)
            self.b_bf = bf

        return

    def create_frequentist_df(self):
        f_df = pd.DataFrame.from_dict(self.f_association, orient='index')
        f_df.columns = ['coeff', 'se', 'pvalue']
        f_df.index = pd.MultiIndex.from_tuples([(index, self.gene) for index in
                                                f_df.index])
        return f_df

    def create_mse_df(self):
        b_df = pd.DataFrame.from_dict(self.b_mse, orient='index')
        b_df.columns = ['mse']
        b_df.index = pd.MultiIndex.from_tuples([(index, self.gene) for index in
                                                b_df.index])
        return b_df

    def create_bf_df(self):
        bf_df = pd.DataFrame.from_dict(self.b_bf, orient='index')
        bf_df.columns = ['psuedo_bf']
        bf_df.index = pd.MultiIndex.from_tuples([(index, self.gene) for index in
                                                bf_df.index])
        return bf_df

class Power():
    def __init__(self, study=None, associations=None, association_dir=None):
        if association_dir:
            self.association_dir = association_dir
            study_file = glob.glob(association_dir+'*study.pkl')[0]
            with pm.Model():
                #study_file = os.path.join(association_dir, "study.pkl")
                self.study = pickle.load(open(study_file, 'rb'))
        else:
            self.study = study
        self._create_association_dfs(associations, association_dir)   
        self.pr_df = self.precision_recall_df()  
        self.roc_df = self.roc_df()
        return

    def posterior_alpha_inclusion(self, association):
        '''
        For Bayesian models, compute 95% credible interval for alpha
        '''
        alpha = {}
        for model in association.bayesian_models:
            sd = numpy.std(model.trace['alpha'])
            mean = numpy.mean(model.trace['alpha'])
            alpha[(model.type, association.gene)] = (mean, sd)
        df = pd.DataFrame.from_dict(alpha, orient='index')
        df.index = pd.MultiIndex.from_tuples(df.index)
        return df

    def precision_recall_df(self):   
        precision_recall_df = pd.concat([self.f_estimator_df[['estimator', 'precision', 'recall']],
                                         self.b_estimator_df[['estimator', 'precision', 'recall']],
                                         # self.bf_estimator_df[['estimator', 'precision', 'recall']]
                                         ])
        return precision_recall_df

    def roc_df(self):   
        roc_df = pd.concat([self.f_estimator_df[['estimator', 'fpr', 'recall']],
                            self.b_estimator_df[['estimator', 'fpr', 'recall']],
                                         # self.bf_estimator_df[['estimator', 'fpr', 'recall']]
                            ])
        return roc_df



    def _calc_bayes_df(self, association):
        bf = dict(('BF-'+model.type,
                    bay.bayes_factor(model, 
                                     association.genotype,
                                     association.phenotype))
                    for model in association.bayesian_models[:2])
        bf_df = pd.DataFrame.from_dict(bf, orient='index')
        bf_df.columns = ['psuedo_bf']
        bf_df.index = pd.MultiIndex.from_tuples([(index, association.gene) for index in
                                                bf_df.index])
        return bf_df

    def _create_association_dfs(self, associations=None, association_dir=None):
        print(association_dir)
        f_association_dfs = []
        b_association_dfs = []
        b_alpha_dfs = []
        bf_association_dfs = []
        if association_dir:
            with pm.Model():
                for fn in glob.glob(association_dir + '/assoc*.pkl'):
                    association = pickle.load(open(fn, 'rb'))
                    f_association_dfs.append(association.create_frequentist_df())
                    b_association_dfs.append(association.create_mse_df())
                    b_alpha_dfs.append(self.posterior_alpha_inclusion(association))
                    # bf_association_dfs.append(association.create_bf_df())
                    del association
            self.f_association_df = pd.concat(f_association_dfs)
            self.b_association_df = pd.concat(b_association_dfs)
            self.b_alpha_dfs = pd.concat(b_alpha_dfs)
            # self.bf_association_df = pd.concat(bf_association_dfs)
        else:
            self.f_association_df = pd.concat([association.create_frequentist_df()
                                              for association in associations])
            self.b_association_df = pd.concat([association.create_mse_df()
                                              for association in associations])
            self.b_alpha_dfs = pd.concat([self.posterior_alpha_inclusion(association)
                                  for association in associations])
            # self.bf_association_df = pd.concat([association.create_bf_df()
            #                                    for association in associations])



        f_estimator =  lambda x: self._calculate_estimator_df(self.f_association_df, x)
        self.f_estimator_df = pd.concat(map(f_estimator,
                                            self.f_association_df.index.levels[0]))

        b_sort = lambda x: x.sort_values('mse')
        b_estimator = lambda x: self._calculate_estimator_df(self.b_association_df, 
                                                              x,
                                                              b_sort)
        self.b_estimator_df = pd.concat(map(b_estimator,
                                            self.b_association_df.index.levels[0]))


        # bf_sort = lambda x: x.sort_values('psuedo_bf', ascending=False)
        # bf_estimator = lambda x: self._calculate_estimator_df(self.bf_association_df, 
        #                                                       x,
        #                                                       bf_sort)
        # self.bf_estimator_df = pd.concat(map(bf_estimator,
        #                                     self.bf_association_df.index.levels[0]))

        return


    def _calculate_estimator_df(self, association_df, estimator, sort_func=lambda x: x.sort_values('pvalue')):
        estimator_df = pd.DataFrame.copy(association_df.ix[estimator])
        estimator_df['in_study'] = estimator_df.index.map(lambda x: True if x in self.study.gene_map else False)
        total_alternate = estimator_df['in_study'].sum()
        total_null = estimator_df.shape[0]-total_alternate
        fprs, fdrs, recalls, precisions = [], [], [], []
        estimator_df = sort_func(estimator_df)
        for i in range(1, estimator_df.shape[0]+1):
            num_correct = float(sum(estimator_df['in_study'][:i]))
            num_incorrect = i - num_correct
            fprs.append(num_incorrect / total_null)
            fdrs.append(num_incorrect / i)
            precisions.append(num_correct / i)
            recalls.append(num_correct / total_alternate)
        estimator_df['fdr'] = fdrs
        estimator_df['fpr'] = fprs
        estimator_df['precision'] = precisions
        estimator_df['recall'] = recalls
        estimator_df['estimator'] = estimator
        return estimator_df
