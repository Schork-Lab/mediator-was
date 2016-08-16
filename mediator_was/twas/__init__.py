'''
Module for computations and visualizations related
to Transcriptome-wide Association Studies

TODO: Fix handling of associations and saving of attributes. 
      Currently a mismatch due to time constraints.

Author: Kunal Bhutani <kunalbhutani@gmail.com>
'''
import os
import glob
import pysam
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.cross_validation import KFold
from sklearn.utils import resample

from mediator_was.association.frequentist import *
import mediator_was.association.bayesian as bay
from mediator_was.twas.elasticnet import fit_models


class Gene():
    """
    TWAS mediator, which contains expression of a gene and
    the genotypes +- 500KB of TSS/TSE of gene for a GTEx or equivalent
    study.

    Attributes:
        alleles (pandas.DataFrame): genotypes of training individuals
        expression (pandas.DataFrame): expression of training individuals
        covariates (pandas.DataFrame): covariates of training individuals
        elasticnet (pandas.DataFrame): output from fitted bootstrapped models
                                       of transcriptional regulation
        chromosome (str): chromosome of gene
        loci (pandas.DataFrame): information of loci +- 500KB of gene
        main_dir (str): path to gene directory
        name (str): name of gene
        samples (list): list of training sample ids

    """
    def __init__(self, main_dir, gtex=True):
        """
        Currently only loads in processed genes that have been
        processed using scripts/train_gtex_EN.py
        
        TODO: start from vcf and expression file, and do
              all processing.
        
        Args:
            main_dir (str): path to directory with processed information
            gtex (bool, optional): use gtex normalized or rlog normalized
        """
        self.main_dir = main_dir
        self.name = os.path.basename(main_dir)
        self.gtex = gtex
        self.chromosome = main_dir.split('/')[-2]  # Assumes no trailing /
        self._load_data()
        self._retrain()
        self._save()
        return

    def _load_data(self):
        """
        Load processed data from scripts/train_gtex_EN.py. Optionally
        load gtex expression fitted data or rlog transformed fitted data.

        Args:
            gtex (bool, optional): Load gtex processed expression data or
                                   rlog transformed

        """
        self.alleles = self._load_gtex_alleles()
        self.loci = self._load_gtex_loci()
        self.alleles.columns = self.loci.index
        if self.gtex:
            data = self._load_gtex_expression()
        else:
            data = self._load_rlog_expression()
        self.expression, self.covariates, self.elasticnet = data
        self.samples = self.expression.index
        self.alleles = self.alleles.ix[self.samples]
        return

    def _retrain(self, twostage=True):
        if self.elasticnet is None:
            self.elasticnet, self.alpha, self.l1_ratio = fit_models(self.alleles,
                                                                    self.expression,
                                                                    self.covariates)
            self.elasticnet['gene'] = self.name
        if twostage:
            if 'twostage' not in self.elasticnet.bootstrap.unique():
                self.elasticnet = self._twostage_train()

        return

    def _twostage_train(self, min_p_inclusion=0.5):
        '''
        Retrain ElasticNet based on convergence across bootstrapped ElasticNet
        '''
        p_inclusion = self.elasticnet['id'].value_counts() / len(self.elasticnet.bootstrap.unique())
        ts_positions = p_inclusion[p_inclusion > min_p_inclusion].index
        ts_model = fit_models(self.alleles[ts_positions],
                              self.expression,
                              self.covariates, n_bootstraps=0)[0]
        ts_model['bootstrap'] = 'twostage'
        ts_model['gene'] = self.name
        elasticnet = pd.concat([self.elasticnet, ts_model])
        return elasticnet

    def _save(self):
        if self.gtex:
            label = "gtex_normalized"
        else:
            label = "rlog"
        fn = os.path.join(self.main_dir,
                          "{}.{}.elasticnet.tsv".format(self.name, label))
        self.elasticnet.to_csv(fn, sep='\t', index=False)

        if hasattr(self, 'alpha'):
            param_fn = os.path.join(self.main_dir,
                                    "{}.{}.params.txt".format(self.name, label))
            with open(param_fn, "w") as OUT:
                OUT.write("L1 ratio: {} \n".format(self.alpha))
                OUT.write("Alpha: {} \n".format(self.l1_ratio))
        return

    def _load_gtex_alleles(self):
        alleles_file = os.path.join(self.main_dir,
                                    self.name + '.alleles.tsv')
        allele_df = pd.read_table(alleles_file, sep='\t',
                                  index_col=0, header=None)
        return allele_df

    def _load_gtex_loci(self):
        loci_file = os.path.join(self.main_dir,
                                 self.name + '.locinames')
        loci_df = pd.read_table(loci_file, sep='\t', index_col=0)
        return loci_df

    def _load_gtex_expression(self):
        gtex_file = os.path.join(self.main_dir,
                                 self.name + ".gtex_normalized.phen.tsv")
        covariates_file = gtex_file.replace(".phen.tsv",
                                            ".covariates.tsv")
        en_file = gtex_file.replace(".phen.tsv",
                                    ".elasticnet.tsv")
        phen_df = pd.read_table(gtex_file, sep='\t', index_col=0)
        covariates_df = pd.read_table(covariates_file, sep='\t', index_col=0)
        try:
            en_df = pd.read_table(en_file, sep='\t')
        except OSError:
            en_df = None

        return phen_df, covariates_df, en_df

    def _load_rlog_expression(self):
        rlog_file = os.path.join(self.main_dir,
                                 self.name + ".rlog.phen.tsv")
        covariates_file = rlog_file.replace(".phen.tsv",
                                            ".covariates.tsv")
        en_file = rlog_file.replace(".phen.tsv",
                                    ".elasticnet.tsv")
        phen_df = pd.read_table(rlog_file, sep='\t', index_col=0)
        covariates_df = pd.read_table(covariates_file, sep='\t', index_col=0)
        try:
            en_df = pd.read_table(en_file, sep='\t')
        except OSError:
            en_df = None
        return phen_df, covariates_df, en_df


class Study():
    """
    A GWAS type study with genotypes and a continuous phenotype.

    Attributes:
        name (str): name of study
        phen (dict): a pandas.DataFrame of phenotype for each
                     chromosome
        samples (list): list of sample ids of individuals
        vcf (pysam.VariantFile): a reader for the study vcf
    """
    def __init__(self, study_path_prefix):
        """
        Initiates pysam object and loads phenotypes

        Args:
            study_path_prefix (str): prefix of path to study
        """
        self.name = os.path.basename(study_path_prefix)
        vcf_path = study_path_prefix + '.vcf.gz'
        self.vcf = pysam.VariantFile(vcf_path)
        # self.samples = list(map(lambda x: x.split('_')[0],
        #                     self.vcf.header.samples))
        self.samples = list(self.vcf.header.samples)
        self._load_phenotypes(study_path_prefix)
        return

    def _load_phenotypes(self, phen_prefix_path):
        '''
        Load LEAP generated phenotypes, one for each chromosome as
        created by scripts/fit_leap.py.

        The files have the same prefix and are tab-separated files
        with two columns: (sample, phenotype) just as used by LEAP.

        sample1\t0
        sample2\t1
        sample3\t2

        Args:
            phen_prefix_path (str): prefix path for all phen files
        '''
        self.phen = {}
        for fn in glob.glob(phen_prefix_path + '*.liab'):
            chrom = fn.split('.')[-2]
            df = pd.read_table(fn, index_col=0, names=['phen'])
            self.phen[chrom] = df.ix[self.samples]
        # self.case_control = pd.read_table(phen_prefix_path + '.fam',
        #                                   names=['fam', 'id', '1', '2', '3', 'phen'],
        #                                   sep='\t')
        # self.case_control.index = self.case_control['fam']+'_'+self.case_control['id']
        # self.case_control = self.case_control.ix[self.samples]['phen']
        return

    def get_alleles(self, chrom, position, ref=None, alt=None, missing_filter=0.05):
        '''
        Get alleles for a particular locus using the vcf reader.
        If the locus is not found, return np.NaN. Optionally, checks
        if the vcf matches the user-provided ref and alt. If it does not match,
        returns np.NaN

        NOTE: Missing values are giving the value of the alt allele frequency
              of the original alt allele. This is most likely to be the minor
              allele and as such each missing person is set to have 2 * MAF 
              as their dosage.
        Args:
            chrom (str): chromosome
            position (int): position of loci
            ref (None, optional): reference allele
            alt (None, optional): alternate allele

        Returns:
            LIST: number of alt alleles per sample, nan if not found
        '''
        def parse_alleles(samples, switch=False):
            allele_counts = Counter()
            for sample in samples:
                allele_counts.update([gt for gt in sample.values()[0]])
            n_chroms = (allele_counts[0] + allele_counts[1])
            t_chroms = n_chroms + allele_counts[None]
            # aaf = 0
            aaf = float(allele_counts[1]) / n_chroms
            missing_af = float(allele_counts[None]) / t_chroms
            if missing_af > missing_filter:
                print('Missing freq {} is higher than {} for {}: {}, skipping.'.format(missing_af, missing_filter, chrom, position))
                raise
            def replace_missing(allele):
                if allele is None:
                    return aaf
                if switch:
                    return 1 - allele
                return allele

            alleles = [np.sum([replace_missing(gt)
                       for gt in sample.values()[0]])
                       for sample in samples]
            return alleles
        try:
            record = next(self.vcf.fetch(str(chrom),
                                         int(position) - 1,
                                         int(position)))
            if ref and alt:
                # Matches ref and alt
                if (record.alleles[0] == ref) and (record.alleles[1] == alt):
                    alleles = parse_alleles(record.samples.values())
                # Switched reference and alt
                elif (record.alleles[0] == alt) and (record.alleles[1] == ref):
                    print('Switching ref, alt at {}:{}'.format(chrom, position))
                    alleles = parse_alleles(record.samples.values(), switch=True)
                # Not matching ref and alt
                else:
                    print('Not matching ref, alt at {}:{}'.format(chrom, position))

                    raise
        except:
             alleles = np.nan
        return alleles


class Association():
    """
    Compute association statistics between a study's phenotype and
    gene.

    Attributes:
        f_assoc (TYPE): Description
        gene (TYPE): Description
        gtex_gen (TYPE): Description
        gtex_phen (TYPE): Description
        gwas_gen (TYPE): Description
        gwas_phen (TYPE): Description
        study (TYPE): Description
    """
    def __init__(self, gene, study,
                 associate=True,
                 permute=None,
                 min_p_inclusion=0.5,
                 missing_filter=0.05):
        """Summary

        Args:
            gene (TYPE): Description
            study (TYPE): Description
        """
        self.gene = gene.name
        self.study = study.name
        self.elasticnet = gene.elasticnet
        self.min_p_inclusion = min_p_inclusion
        self.missing_filter = missing_filter
        self._load_genotypes(gene, study)
        self._load_phenotypes(gene, study)
        #self._generate_kfolds()
        self._predict_expression(gene)
        self.f_stats = None
        self.b_stats = None
        self.permute_stats = None

        if associate:
            self.associate()
        if permute is not None:
            n_permutations, random_state = permute
            self.permute(n_permutations, random_state)

        return

    def _load_genotypes(self, gene, study):
        """
        Load genotypes based on overlap between study and gene
        loci.

        Args:
            gene (Gene):
            study (Study):
        """
        def _get_alleles(loci):
            alleles = study.get_alleles(loci['chromosome'],
                                        loci['position'],
                                        loci['ref'],
                                        loci['alt'],
                                        self.missing_filter)
            return alleles
        alleles = gene.loci[gene.loci.index.isin(gene.elasticnet['id'])].apply(_get_alleles, axis=1)
        alleles = alleles[~alleles.isnull()]
        self.gwas_gen = pd.DataFrame([np.array(x) for x in alleles],
                                     index=alleles.index).T
        loci = set(self.gwas_gen.columns).intersection(gene.alleles.columns)
        self.loci = list(loci)
        self.gwas_gen = self.gwas_gen[self.loci]
        self.gwas_gen.index = study.samples
        self.gtex_gen = gene.alleles[self.loci]
        # # NOTE :: included_snps should be part of gene object too.
        ## CLEAN THIS UP.
        self.included_snps = self.elasticnet[self.elasticnet.bootstrap == 'twostage']['id']
        self.included_snps = list(set(self.loci).intersection(self.included_snps))

        return

    def _load_phenotypes(self, gene, study):
        """
        Load phenotypes associated with gene and study
        """
        self.gwas_phen = study.phen[gene.chromosome]['phen']
        self.gtex_phen = gene.expression
        #self.gwas_case_control = study.case_control
        return

    def _predict_expression(self, gene):
        """
        Predict expression using ElasticNet bootstraps in
        gene.

        Args:
            gene (Gene): Description

        """
        pred_expr = []
        for bootstrap, df in gene.elasticnet.groupby('bootstrap'):
            df = df[df['id'].isin(self.gwas_gen.columns)]
            loci = df.id
            beta = df.beta.values.astype(float)
            contributions = self.gwas_gen[loci] * beta
            expression = pd.DataFrame(contributions.sum(axis=1),
                                      columns=[bootstrap])
            pred_expr.append(expression)
        self.pred_expr = pd.concat(pred_expr, axis=1)

    def _generate_kfolds(self, k=10, seed=0):
        '''
        Generate train and testing folds foruse in cross-validation
        for Bayesian inference. Uses a seed to insure the same folds are used
        for each association task.

        TODO: StratifiedKFold for case/control studies

        Args:
            study (Study): Study object
            k (int, optional): number of folds (default: 10)
            seed (int, optional): consistent random state (default: 0)
        '''
        self.kfolds = KFold(self.gwas_gen.shape[0], n_folds=k,
                            random_state=seed)
        return

    def associate(self, **bayesian_args):
        """Summary

        Returns:
            TYPE: Description
        """
        inter_var = self.pred_expr.mean(axis=1).var(ddof=1)
        intra_var = self.pred_expr.var(axis=1, ddof=1).mean()
        print('Intra-variance: {}'.format(intra_var))
        print('Inter-variance: {}'.format(inter_var))
        if inter_var > intra_var:
            print('Heritable, running associations.')
            self._frequentist(self.pred_expr)
            self._bayesian(**bayesian_args)
        return

    def _bayesian(self, joint=True, ts=False, joint_prior=False):
        """
        Fit three different Bayesian Linear Regressions using
        only SNPS filtered using ElasticNet.

        1. Simple Bayesian Regression (Using EN Priors)
        2. Joint Bayesian Linear Regression (LaPlace)
        3. Joint Bayesian Linear Regression (Using EN Priors)

        Returns:
            dict: Bayesian Statistics
        """

        # # First stage filter for TwoStage Model
        # self.included_snps = self.elasticnet[self.elasticnet.bootstrap == 'twostage']['id']
        # self.included_snps = list(set(self.loci).intersection(self.included_snps))
        self.b_stats = {}
        self.b_traces = {}

        coefs = self.elasticnet[self.elasticnet['id'].isin(self.included_snps)]
        coefs = coefs[~coefs['bootstrap'].isin(['full', 'twostage'])]
        coef_mean = coefs.groupby('id')['beta'].mean().values
        coef_sd = coefs.groupby('id')['beta'].std(ddof=1).values

        if ts:
            ts_model = bay.TwoStage(coef_mean, coef_sd,
                                    variational=True, mb=False, n_chain=50000)
            ts_trace = ts_model.run(gwas_gen=self.gwas_gen[self.included_snps].values,
                                    gwas_phen=self.gwas_phen.values)
            ts_stats = ts_model.calculate_ppc(ts_trace)
            self.b_traces['ts'] = ts_trace
            self.b_stats['ts'] = ts_stats

        if joint:
            j_model = bay.Joint(model_type='laplace',
                                variational=True, mb=True, n_chain=50000)
            j_trace = j_model.run(med_gen=self.gtex_gen[self.included_snps].values,
                                  med_phen=self.gtex_phen.values.ravel(),
                                  gwas_gen=self.gwas_gen[self.included_snps].values,
                                  gwas_phen=self.gwas_phen.values)
            j_stats = j_model.calculate_ppc(j_trace)
            self.b_traces['Joint_LaPlace'] = j_trace
            self.b_stats['Joint_LaPlace'] = j_stats

        if joint_prior:
            j_model2 = bay.Joint(model_type='prior',
                                 coef_mean=coef_mean,
                                 coef_sd=coef_sd,
                                 variational=True, mb=True, n_chain=50000)
            j_trace2 = j_model2.run(med_gen=self.gtex_gen[self.included_snps].values,
                                   med_phen=self.gtex_phen.values.ravel(),
                                   gwas_gen=self.gwas_gen[self.included_snps].values,
                                   gwas_phen=self.gwas_phen.values)
            j_stats2 = j_model2.calculate_ppc(j_trace2)
            self.b_traces['Joint_Prior'] = j_trace2
            self.b_stats['Joint_Prior'] = j_stats2

        print(self.b_stats)
        return

    def permute(self, n_permutations=10, random_state=0):
        self.permute_stats = []
        j_model = bay.Joint(model_type='laplace',
                            variational=True,
                            mb=True,
                            n_chain=50000)

        for i in range(n_permutations):
            random_state = random_state * i + i
            gwas_phen = resample(self.gwas_phen.values,
                                 replace=False,
                                 random_state=random_state)

            j_trace = j_model.run(med_gen=self.gtex_gen[self.included_snps].values,
                                  med_phen=self.gtex_phen.values.ravel(),
                                  gwas_gen=self.gwas_gen[self.included_snps].values,
                                  gwas_phen=gwas_phen)
            j_stats = j_model.calculate_ppc(j_trace)
            self.permute_stats.append({'Permuted-Joint-LaPlace': j_stats})

        return

    def _frequentist(self, pred_expr):
        """
        Compute frequentist statistics using OLS, Regression Calibration,
        and Multiple Imputation.

        Returns:
            dict: Frequentist statistics
        """

        bootstraps = [column for column in pred_expr.columns
                      if column not in ('full', 'twostage')]
        phen = self.gwas_phen.values
        try:
            full_expr = pred_expr['full']
        except KeyError:
            full_expr = np.zeros(shape=phen.shape)
        try:
            ts_expr = pred_expr['twostage']
        except KeyError:
            ts_expr = np.zeros(shape=phen.shape)
        bootstrap_expr = pred_expr[bootstraps]
        mean_expr = bootstrap_expr.mean(axis=1)
        sigma_ui = bootstrap_expr.var(ddof=1, axis=1)

        association = {'OLS-Full': t(full_expr, phen, method="OLS"),
                       'OLS-TwoStage': t(ts_expr, phen, method="OLS"),
                       'OLS-Mean': t(mean_expr, phen,
                                     method="OLS"),
                       'RC-hetero-bootstrapped': t(mean_expr, phen,
                                                   sigma_ui,
                                                   method='rc-hetero'),
                       'MI-Bootstrapped': multiple_imputation(bootstrap_expr.T,
                                                              phen),
                       }
        self.f_stats = association
        print(self.f_stats)
        return

    def save(self, file_prefix):
        if self.f_stats is not None:
            f_df = pd.DataFrame.from_dict(self.f_stats, orient='index')
            f_df.columns = ['coeff', 'se', 'pvalue']
            f_df.index = pd.MultiIndex.from_tuples([(index, self.gene)
                                                    for index in f_df.index])
            f_df.to_csv(file_prefix + '.fstats.tsv', sep='\t')
        if self.b_stats is not None:
            b_df = pd.DataFrame(self.b_stats).T
            b_df.index = pd.MultiIndex.from_tuples([(index, self.gene)
                                                    for index in b_df.index])
            b_df.to_csv(file_prefix + '.bstats.tsv', sep='\t')

        if self.permute_stats is not None:
            permute_df = pd.concat([pd.DataFrame(stats).T
                                    for stats in self.permute_stats])
            index = [('Permutation {}'.format(i), test, self.gene)
                     for i, test in enumerate(permute_df.index)]
            permute_df.index = pd.MultiIndex.from_tuples(index)
            permute_df.to_csv(file_prefix + '.permutations.tsv', sep='\t')
