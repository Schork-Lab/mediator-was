"""
Simulate a TWAS study design modularily.

TODO:
    1. Implement permutation testing
    2. Implement an Ensemble gene ranking to see how well it performs


Author: Kunal Bhutani   <kunalbhutani@gmail.com>
        Abhishek Sarkar <aksarkar@mit.edu>
"""
from __future__ import print_function

import pickle
import time
import glob
import os

import numpy as np
import numpy.random as R
import sklearn.linear_model
import sklearn.metrics
import sklearn.utils
import pandas as pd
import pymc3 as pm
from sklearn.cross_validation import KFold, ShuffleSplit
from scipy import stats


from mediator_was.association.frequentist import *
from mediator_was.processing.helpers import load_plink

import mediator_was.association.bayesian as bay


current_milli_time = lambda: int(round(time.time() * 1000))


def _add_noise(genetic_value, pve):
    """Add Gaussian noise to genetic values to achieve desired PVE.

    Assume true effects are N(0, 1) and sample errors from appropriately scaled
    Gaussian.

    """
    sigma = np.sqrt(np.var(genetic_value) * (1 / pve - 1))
    return genetic_value + R.normal(size=genetic_value.shape, scale=sigma)


def _fit_bootstrapped_EN(bootstrap_params, genotypes, expression):
    '''
    Fit bootstrapped ElasticNetCV

    Args:
        bootstrap_params: 
            tuple(number of individuals per bootstrap, number of bootstraps)
        genotypes
        expression

    Returns:
        array of ElasticNetCV models
    '''

    b_models = []
    model = sklearn.linear_model.ElasticNetCV
    n = bootstrap_params[0]

    # Fit entire model to get estimates of /alpha and /lambda
    l1_ratio_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    full_model = model(l1_ratio=l1_ratio_range, max_iter=10000)
    full_model.fit(genotypes, expression)

    model = sklearn.linear_model.ElasticNet
    for i in range(bootstrap_params[1]):
        b_genotypes, b_expression = sklearn.utils.resample(genotypes,
                                                           expression,
                                                           n_samples=n)
        b_model = model(max_iter=10000,
                        alpha=full_model.alpha_,
                        l1_ratio=full_model.l1_ratio_)
        b_model = b_model.fit(b_genotypes, b_expression)
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
        causal_loci: np.array of size causal loci
                     with values as indices out of the p genotypes
        beta: np.array of size p of beta_exp
        train_ind: np.array of individual indices used for training
        train_genotypes, train_expression: arrays of genotypes and expression
        bootstrap_models: a list of fitted bootstrapped ElasticNet models
        bayesian_model: a bayesian Model of type expression

        TODO: allow for more complex noise models
    '''

    def __init__(self, name, plink_file,
                 n_train=500, p_causal_eqtls=0.1, pve=0.17, pve_se=0.05,
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
            n_causal_eqtls = int(np.round(n_snps * self.p_causal_eqtls))
        self.causal_loci = self._sample_causal_loci(n_causal_eqtls)
        self.beta = np.zeros(n_snps)
        #self.beta[self.causal_loci] = R.normal(size=n_causal_eqtls)
        self.beta[self.causal_loci] = np.ones(n_causal_eqtls)
        self.train_ind = R.random_integers(low=0,
                                           high=self.haps.shape[1] - 2,
                                           size=self.n_train)
        self.not_train = np.setdiff1d(np.arange(self.haps.shape[1] - 2),
                                      self.train_ind)
        return

    def _sample_causal_loci(self, n_causal_snps, min_maf=0.05, max_maf=0.5):
        """
        Sample gene specific causal loci based on a haplotype dataframe
        restricted to a minor allele frequency range and total number
        of causal snps
        """
        # maf = haps.apply(lambda x: np.sum(x)/len(x), axis=1)
        # causal_loci = np.where((maf > min_maf) & (maf < max_maf))[0]
        # if len(causal_loci) < n_causal_snps:
        #     print('Warning: not enough causal loci in provided haplotypes',
        #            file=sys.stderr)
        # causal_loci = R.choice(causal_loci, size=n_causal_snps)
        causal_loci = R.choice(self.haps.index, size=n_causal_snps)
        return causal_loci

    def _train(self):
        self.train_genotypes, self.train_expression = self.simulate(train=True)
        self.bootstrap_models = _fit_bootstrapped_EN(self.bootstrap_params,
                                                     self.train_genotypes,
                                                     self.train_expression)
        # self.bayesian_model = bay.expression_model(self.train_genotypes,
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
            genotypes -= np.mean(genotypes, axis=0)[np.newaxis, :]
        pve = self.pve
        # pve = R.normal(params.pve, params.pve_se)
        # if pve <= 0: # Negative contribution of genotype to expression
        #     pve = 0.01
        expression = _add_noise(np.dot(genotypes, self.beta), pve)
        return genotypes, expression


class Study(object):
    '''
    Study class for saving causal genes, genotypes, and phenotypes

    Args:
        causal_genes - list of gene objects, length g
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
        self.id = '_'.join(map(str,
                               [name, pve,
                                len(causal_genes),
                                seed,
                                current_milli_time()]))
        self.causal_genes = [gene.id for gene in causal_genes]
        # self.causal_genes = causal_genes
        self.beta = np.ones(len(causal_genes))
        # self.beta = R.normal(size=len(causal_genes))
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
                genetic_value = sum(np.dot(g, R.normal(size=gene.beta.shape))
                                    for g, gene in zip(genotypes, causal_genes)
                                    )
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
        gene - Gene object
        study - Study object
        seed - seed passed to different random functions
        associate - compute associations
        me - use measurement error Bayesian model

    Attributes:
        phenotype - study.phenotype
        genotype -   
            If study.causal_genes does not include gene,
                 generate random set of genotypes from gene
            Otherwise, use the genotypes used to generate study.phenotype
        beta - simulated beta from study for gene
        expected_pve - expected pve based on other gene's beta and study pve
        f_association - dictionary of association statistics from frequentist tests
        bayesian_models - list of bayesian models
        b_mse - mse using out of sample samples
        b_zscore - zscore equivalent statistic for alpha
    '''
    def __init__(self, name, gene, study, seed=0, associate=True, me=True):
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
            self.expected_pve = (self.beta / sum(study.beta)) * study.pve
        else:
            self.genotype, _ = gene.simulate(n=self.phenotype.shape,
                                             train=False)
            self.oos_genotype, _ = gene.simulate(n=self.phenotype.shape,
                                                 train=False)
            self.beta = 0
            self.expected_pve = 0

        if associate:
            self._frequentist(gene)
            self.b_stats = {}
            if me:
                self._bayesian_me(gene)
            else:
                self._bayesian(gene)
        return

    def _generate_kfolds(self, k=5, seed=0):
        '''
        Generate train and testing folds foruse in cross-validation
        for Bayesian inference. Uses a seed to insure the same folds are used
        for each association task.

        TODO: StratifiedKFold for case/control studies

        Args:
            k (int, optional): number of folds (default: 10)
            seed (int, optional): consistent random state (default: 0)
        '''
        # self.kfolds = list(KFold(self.genotype.shape[0],
        #                         n_folds=k, random_state=seed))
        self.kfolds = list(ShuffleSplit(self.genotype.shape[0],
                                        n_iter=1, test_size=0.2,
                                        random_state=seed))
        return

    def _frequentist(self, gene):
        '''
        Fit frequentist methods for association
        '''
        genotype = self.genotype
        phenotype = self.phenotype

        # Bootstrapped
        bms = gene.bootstrap_models
        pred_expr = np.array([m.predict(genotype) for m in bms])
        w_bootstrap = np.mean(pred_expr, axis=0)
        sigma_ui_bootstrap = np.var(pred_expr, ddof=1, axis=0)

        # Two-stage bootstrap
        # elasticnet = pd.DataFrame([model.coef_
        #                            for model in gene.bootstrap_models])
        # columns = np.where(((elasticnet != 0).sum(axis=0) / elasticnet.shape[0]) > 0.5)[0]
        # model = sklearn.linear_model.ElasticNetCV
        # # Fit entire model to get estimates of /alpha and /lambda
        # l1_ratio_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        # full_model = model(l1_ratio=l1_ratio_range, max_iter=10000)
        # full_model.fit(gene.train_genotypes[:,columns], gene.train_expression)
        # ts_expr = full_model.predict(genotype[:, columns])

        association = {'OLS-Mean': t(w_bootstrap, phenotype, method="OLS"),
                       'OLS-ElasticNet': t(pred_expr[0], phenotype,
                                           method="OLS"),
                       'RC-hetero-bootstrapped': t(w_bootstrap, phenotype,
                                                   sigma_ui_bootstrap,
                                                   method='rc-hetero'),
                       'MI-Bootstrapped': multiple_imputation(pred_expr,
                                                              phenotype),
       #                'OLS-TwoStage': t(ts_expr, phenotype, method="OLS")
                       }
        self.f_association = association
        return


    def _bayesian_me(self, gene):
        '''
        Fit measurement error Bayesian model and compute a Bayes Factor
        and other statistics
        '''
        genotype = self.genotype
        phenotype = self.phenotype

        # Bootstrapped
        bms = gene.bootstrap_models
        pred_expr = np.array([m.predict(genotype) for m in bms])
        w_bootstrap = np.mean(pred_expr, axis=0)
        sd_ui_bootstrap = np.std(pred_expr, ddof=1, axis=0)

        model = bay.MeasurementErrorBF(mediator_mu=w_bootstrap.mean(),
                                       mediator_sd=w_bootstrap.std(),
                                       variational=False,
                                       n_chain=75000)
        trace = model.run(gwas_phen=phenotype,
                          gwas_mediator=w_bootstrap,
                          gwas_error=np.sqrt(sd_ui_bootstrap))

        stats = model.calculate_ppc(trace)
        p_alt = model.trace['mediator_model'].mean()
        bayes_factor = (p_alt/(1-p_alt))
        stats['bayes_factor'] = bayes_factor
        self.b_stats[model.name] = stats
        return self.b_stats

    def _bayesian(self, gene, ):
        '''
        Fit Two Stage and Joint Bayesian models and calculate statistics based on both
        out of sample MSE and cross-validation.
        '''

        elasticnet = pd.DataFrame([model.coef_
                                   for model in gene.bootstrap_models])
        columns = np.where(((elasticnet != 0).sum(axis=0) / elasticnet.shape[0]) > 0.5)[0]
        self.included_snps = columns
        coef_mean = elasticnet[columns].mean(axis=0).values
        coef_sd = elasticnet[columns].std(axis=0, ddof=1).values
        ts_model = bay.TwoStage(coef_mean, coef_sd,
                                variational=True, n_chain=50000)
        ts_trace = ts_model.run(gwas_gen=self.genotype[:, columns],
                                gwas_phen=self.phenotype)
        ts_stats = ts_model.calculate_ppc(ts_trace)
        ts_stats['bayes_factor'] = 0
        self.b_stats[ts_model.name] = ts_stats

        j_model = bay.Joint(variational=True, mb=True, n_chain=50000)
        j_trace = j_model.run(med_gen=gene.train_genotypes[:, columns],
                              med_phen=gene.train_expression,
                              gwas_gen=self.genotype[:, columns],
                              gwas_phen=self.phenotype)
        j_stats = j_model.calculate_ppc(j_trace)
        j_stats['bayes_factor'] = 0
        self.b_stats[j_model.name] = j_stats
        return self.b_stats

    def _generate_null_phen(self, permuted_noise=True):
        model = bay.NonMediated()
        model.run(gwas_gen=self.genotype, gwas_phen=self.phenotype)
        self.null_phenotypes = []
        for idx in range(0, 10):
            # if permuted_noise:
            #     phenotype_hat = self.genotype.dot(model.trace[idx]['beta'].ravel())
            #     residuals = self.phenotype - phenotype_hat
            #     phenotype_new = phenotype_hat + np.random.permutation(residuals)
            # else:
            #     phen = model.cached_model.observed_RVs[0]
            #     phenotype_new = phen.distribution.random(point=model.trace[idx]).ravel()
            # self.null_phenotypes.append(phenotype_new)
            permuted_phenotype = np.random.permutation(self.phenotype)
            self.null_phenotypes.append(permuted_phenotype)
        return self.null_phenotypes

    def _generate_null_statistics(self, gene):
        self.null_stats = []
        elasticnet = pd.DataFrame([model.coef_
                                   for model in gene.bootstrap_models])
        columns = np.where(((elasticnet != 0).sum(axis=0) / elasticnet.shape[0]) > 0.5)[0]
        self.included_snps = columns
        coef_mean = elasticnet[columns].mean(axis=0).values
        coef_sd = elasticnet[columns].std(axis=0, ddof=1).values
        ts_model = bay.TwoStage(coef_mean, coef_sd,
                                variational=True,
                                n_chain=50000)

        for phenotype in self.null_phenotypes:
            ts_trace = ts_model.run(gwas_gen=self.genotype[:, columns],
                                    gwas_phen=phenotype)
            ts_stats = ts_model.calculate_ppc(ts_trace)
            self.null_stats.append(ts_stats)
        return self.null_stats


    def create_frequentist_df(self):
        '''
        Create dataframe from f_association dict
        '''
        f_df = pd.DataFrame.from_dict(self.f_association, orient='index')
        f_df.columns = ['coeff', 'se', 'pvalue']
        f_df.index = pd.MultiIndex.from_tuples([(index, self.gene) for index in
                                                f_df.index])
        return f_df

    def create_bayesian_df(self):
        def create_stat_df(stat='waic'):
            b_stat = dict((model, stats[stat])
                          for model, stats in self.b_stats.items()
                          )
            b_df = pd.DataFrame.from_dict(b_stat, orient='index')
            b_df.columns = ['value']
            b_df.index = pd.MultiIndex.from_tuples([(index, self.gene, stat)
                                                   for index in b_df.index])
            return b_df

        b_df = pd.concat([create_stat_df('dic'),
                          create_stat_df('waic'),
                          create_stat_df('logp'),
                          create_stat_df('mu'),
                          create_stat_df('sd'),
                          create_stat_df('zscore'),
                          create_stat_df('bayes_factor')
                          ])
        return b_df

    def save(self, fn):
        f_df = self.create_frequentist_df()
        f_df.to_csv(fn+'.fassoc.tsv', sep='\t')
        b_df = self.create_bayesian_df()
        b_df.to_csv(fn+'.bassoc.tsv', sep='\t')
        return



class Power():
    '''
    Calculate precision/recall and other power related statistics
    for a given study and calculated associations.

    Args:
        study - Study object
        associatinos - list of associations objects
        association_dir - path to associations, if provided, also looks
                          for a *study.pkl pickled Study object in
                          directory
    Attributes:
        study - Study object
        pr_df - precision recall DataFrame
        roc_df - sensitivity specificity DataFrame
        {f, mse, zscore}_estimator_df - sorted DataFrames with statistics

    '''

    def __init__(self, study=None, associations=None, association_dir=None,
                 pickled=True):
        if association_dir:
            self.association_dir = association_dir
            study_file = glob.glob(association_dir + '*study.pkl')[0]
            with pm.Model():
                study = pickle.load(open(study_file, 'rb'))
                self.study = study.id
                self.study_genes = study.gene_map
        else:
            self.study = study.id
            self.study_genes = study.gene_map
        self._create_association_dfs(associations, association_dir, pickled=pickled)
        self.pr_df = self.precision_recall_df()
        self.roc_df = self.roc_df()
        return

    def precision_recall_df(self):
        columns = ['estimator', 'precision', 'recall']
        precision_recall_df = pd.concat([self.f_estimator_df[columns],
                                         self.waic_estimator_df[columns]
                                         ])
        return precision_recall_df

    def roc_df(self):
        columns = ['estimator', 'fpr', 'recall']
        roc_df = pd.concat([self.f_estimator_df[columns],
                            self.waic_estimator_df[columns]])
        return roc_df

    def power(self, alpha=0.05):
        def frequentist(f_df):
            def calculate(estimator):
                e_df = f_df.ix[estimator]
                total = len(e_df)
                e_df = e_df[e_df['in_study']]
                power = sum(e_df['pvalue'] < alpha/total) / len(e_df)
                return power
            return map(calculate, f_df.index.levels[0])
        
        def bayesian(zscore_df):
            def calculate(estimator):
                e_df = zscore_df.ix[estimator]
                e_df = e_df[e_df['in_study']]
                power = sum(e_df['zscore'] > stats.norm.ppf(1 - alpha)) / len(e_df)
                return power
            return map(calculate, zscore_df.index.levels[0])

        f_power = frequentist(self.f_association_df)
        b_power = bayesian(self.zscore_df)
        return f_power, b_power

    def type_i_error(self, alpha=0.05):
        ## Should do this one at a gene at a time and then aggregate.
        def frequentist(f_df):
            def calculate(estimator):
                e_df = f_df.ix[estimator]
                e_df = e_df[e_df['pvalue'] < alpha/len(e_df)]
                error = sum(~e_df['in_study']) / len(e_df)
                return error
            return map(calculate, f_df.index.levels[0])
        
        def bayesian(zscore_df):
            def calculate(estimator):
                e_df = zscore_df.ix[estimator]
                e_df = e_df[e_df['zscore'] > stats.norm.ppf(1 - alpha)]
                error = sum(~e_df['in_study']) / len(e_df)
                return error
            return map(calculate, zscore_df.index.levels[0])

        f_error = frequentist(self.f_association_df)
        b_error = bayesian(self.zscore_df)
        return f_error, b_error


    def _create_association_dfs(self, associations=None, association_dir=None,
                                pickled=False):
        '''
        Create dataframes that combine all association statistics
        '''
        def create_estimator_df(association_df,
                                sort_func=lambda x: x.sort_values('pvalue')):
            estimator = lambda x: self._calculate_estimator_df(association_df,
                                                               x,
                                                               sort_func)
            estimator_df = pd.concat(map(estimator,
                                         association_df.index.levels[0]))
            return estimator_df

        def get_associations(association_dir):
            with pm.Model():
                for fn in glob.glob(os.path.join(association_dir,
                                                 'assoc*.pkl')):
                    yield pickle.load(open(fn, 'rb'))

        def create_stat_df(association, stat='waic'):
            b_stat = dict((model, stats[stat])
                          for model, stats in association.b_stats.items())
            b_df = pd.DataFrame.from_dict(b_stat, orient='index')
            b_df.columns = [stat]
            b_df.index = pd.MultiIndex.from_tuples([(index, association.gene)
                                                   for index in b_df.index])
            return b_df

        if pickled:
            # Load pickled association
            if association_dir:
                associations = get_associations(association_dir)

            freq, dic, waic, logp, zscore, bf = [], [], [], [], [], []
            for association in associations:
                dic.append(create_stat_df(association, 'dic'))
                waic.append(create_stat_df(association, 'waic'))
                logp.append(create_stat_df(association, 'logp'))
                zscore.append(create_stat_df(association, 'zscore'))
                bf.append(create_stat_df(association, 'bayes_factor'))
                freq.append(association.create_frequentist_df())
                
            self.f_association_df = pd.concat(freq)
            self.b_dic_df = pd.concat(dic)
            self.b_waic_df = pd.concat(waic)
            self.b_logp_df = pd.concat(logp)
            self.b_zscore_df = pd.concat(zscore)
            self.b_bf_df = pd.concat(bf)

            self.f_estimator_df = create_estimator_df(self.f_association_df)
            logp_sort = lambda x: x.sort_values('logp', ascending=False)
            self.logp_estimator_df = create_estimator_df(self.b_logp_df, logp_sort)
            loo_sort = lambda x: x.sort_values('loo', ascending=False)
            self.loo_estimator_df = create_estimator_df(self.b_loo_df, loo_sort)
            waic_sort = lambda x: x.sort_values('waic')
            self.waic_estimator_df = create_estimator_df(self.b_waic_df, waic_sort)
            zscore_sort = lambda x: x.sort_values('zscore', ascending=False)
            self.zscore_estimator_df = create_estimator_df(self.b_zscore_df, zscore_sort)
            bf_sort = lambda x: x.sort_values('bayes_factor', ascending=False)
            self.bf_estimator_df = create_estimator_df(self.b_bf_df, bf_sort)


        else:
            # Load intermediate files generated
            f_df = pd.concat([pd.read_table(fn, index_col=[0, 1])
                              for fn in glob.glob(os.path.join(association_dir,
                                                               '*.fassoc.tsv'))
                              ])
            self.f_estimator_df = create_estimator_df(f_df)

            b_df = pd.concat([pd.read_table(fn, index_col=[0, 1, 2])
                             for fn in glob.glob(os.path.join(association_dir, 
                                                              '*.bassoc.tsv'))
                             ])
            b_df['value'] = b_df['value'].astype(float)
            des_sort = lambda x: x.sort_values('value', ascending=False)
            asc_sort = lambda x: x.sort_values('value')
            self.dic_estimator_df = create_estimator_df(b_df.xs('dic', level=2), asc_sort)
            self.logp_estimator_df = create_estimator_df(b_df.xs('logp', level=2), des_sort)
            self.waic_estimator_df = create_estimator_df(b_df.xs('waic', level=2), asc_sort)
            self.bf_estimator_df = create_estimator_df(b_df.xs('bayes_factor', level=2), des_sort)
            if 'zscore' in b_df.index.levels[2]:
                self.zscore_estimator_df = create_estimator_df(b_df.xs('zscore', level=2), des_sort)
            else:
                zscore_df = (b_df.xs('mu', level=2) / b_df.xs('sd', level=2)).applymap(abs)
                self.zscore_estimator_df = create_estimator_df(zscore_df, des_sort)

        return

    def _calculate_estimator_df(self, association_df,
                                estimator,
                                sort_func=lambda x: x.sort_values('pvalue')):
        '''
        Calculate fdr, fpr, precision, and recall for an estimator
        and its associated statistics
        '''
        estimator_df = pd.DataFrame.copy(association_df.ix[estimator])
        estimator_df['in_study'] = estimator_df.index.map(lambda x: x in self.study_genes)

        total_alternate = estimator_df['in_study'].sum()
        total_null = estimator_df.shape[0] - total_alternate
        fprs, fdrs, recalls, precisions = [], [], [], []
        estimator_df = sort_func(estimator_df)
        for i in range(1, estimator_df.shape[0] + 1):
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


