'''
Computations and visualizations related
to Transcriptome-wide Association Studies

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
from mediator_was.twas.elasticnet import fit_models, calculate_cv_r2


class Gene():
    """
    TWAS mediator, which contains expression of a gene and
    the genotypes +- 500KB of TSS/TSE of gene for a GTEx or equivalent
    study.

    Attributes:
        name (str): name of gene
        chromosome (str): chromosome of gene
        elasticnet (pandas.DataFrame): output from fitted bootstrapped models
                                       of transcriptional regulation
        main_dir (str): path to gene directory
        gtex (bool): use gtex normalized evalues
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
        self._load()
        return

    def _load(self):
        """
        Load processed data.
        """
        def load_params(fn):
            with open(fn, 'r') as IN:
                get_value = lambda x: float(x.rstrip().split(' ')[-1])
                data = tuple(map(get_value, IN))
                try:
                    l1_ratio, alpha, r2 = data
                except:
                    l1_ratio, alpha = data
                    r2 = None
            return l1_ratio, alpha, r2
        if self.gtex:
            label = "gtex_normalized"
        else:
            label = "rlog"
        en_file = os.path.join(self.main_dir,
                               "{}.{}.elasticnet.tsv".format(self.name, label))
        self.elasticnet = pd.read_table(en_file, sep='\t')
        param_file = en_file.replace('elasticnet.tsv', 'params.txt')
        self.l1_ratio, self.alpha, self.r2 = load_params(param_file)
        return

class Study():
    """
    A GWAS type study with genotypes and a continuous phenotype. 
    Requires that a VCF and the phenotypes share the prefix.

    Look for examples of the phenotype in the example directory. 

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
        self.samples = list(self.vcf.header.samples)
        self._load_phenotypes(study_path_prefix)
        return

    def _load_phenotypes(self, phen_prefix_path):
        '''
        Load LEAP generated phenotypes, one for each chromosome as
        created in the format: phen_prefix.chromosome.liab

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
            self.phen[chrom] = self.phen[chrom].fillna(self.phen[chrom].mean())

        return

    def get_alleles(self, chrom, position, ref=None, alt=None, missing_filter=0.1):
        '''
        Get alleles for a particular locus using the vcf reader.
        If the locus is not found, return np.NaN. Optionally, checks
        if the vcf matches the user-provided ref and alt. If the alleles are
        switched, it changes the dosages to reflect that. If it does not match,
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
    """
    def __init__(self, gene, study,
                 associate=True,
                 permute=None,
                 min_p_inclusion=0.5,
                 missing_filter=0.1):

        self.gene = gene.name
        self.study = study.name
        self.elasticnet = gene.elasticnet
        self.min_p_inclusion = min_p_inclusion
        self.missing_filter = missing_filter
        self._load_genotypes(gene, study)
        
        self._predict_expression(gene)
        self._load_phenotypes(gene, study)

        self.f_stats = None
        self.b_stats = None
        if associate:
            self.associate()

        return

    def _load_genotypes(self, gene, study):
        """
        Load genotypes based on overlap between study and gene loci.
        """
        def _get_alleles(loci):
            alleles = study.get_alleles(loci['chromosome'],
                                        loci['position'],
                                        loci['ref'],
                                        loci['alt'],
                                        self.missing_filter)
            return alleles

        loci = gene.elasticnet[['id', 'chromosome', 'position', 'ref', 'alt']].drop_duplicates().set_index('id')
        alleles = loci.apply(_get_alleles, axis=1)
        alleles = alleles[~alleles.isnull()]
        if len(alleles) == 0:
            raise Exception("No matching alleles between study and gene elasticnet models.")
        self.loci = list(alleles.index)
        self.gwas_gen = pd.DataFrame([np.array(x) for x in alleles],
                                     index=self.loci).T
        self.gwas_gen.index = study.samples
        print(self.gwas_gen.columns)
        return

    def _load_phenotypes(self, gene, study):
        """
        Load phenotypes associated with gene and study
        """
        self.gwas_phen = study.phen[gene.chromosome]['phen']
        return

    def _predict_expression(self, gene):
        """
        Predict expression using ElasticNet bootstraps in gene.
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

    def associate(self, inter_gt=1):
        """
        Run Bayesian and Frequentist measurement error associations.
        Filter heritable genes based on
        inter_gt * inter-individual variance > intra-ind variance.

        inter_gt - constant (default: 1)

        """
        bootstraps = [column for column in self.pred_expr.columns
                      if column not in ('full', 'twostage')]
        inter_var = self.pred_expr[bootstraps].mean(axis=1).var(ddof=1)
        intra_var = self.pred_expr[bootstraps].var(axis=1, ddof=1).mean()
        print('Intra-variance: {}'.format(intra_var))
        print('Inter-variance: {}'.format(inter_var))
        if inter_var > intra_var:
            self._frequentist(self.pred_expr)
            self._bayesian(self.pred_expr)
        return

    def _bayesian(self, pred_expr,  min_inclusion=0.5):
        '''
        Fit Two Stage model and compute a Bayes Factor
        and other statistics.

        Returns:
            dict: Bayesian Statistics
        '''
        self.b_stats, self.b_traces = {}, {}
        phen = self.gwas_phen.values

        # Filter based on min_inclusion
        coefs = self.elasticnet[~self.elasticnet['bootstrap'].isin(['full', 'twostage'])] # Only bootstraps
        coefs = coefs[coefs['id'].isin(self.loci)] # Only overlapping with GWAS      
        bootstraps_per_snp = coefs['id'].value_counts()
        n_bootstraps = len(self.elasticnet['bootstrap'].unique()) - 2
        min_bootstraps = min_inclusion * n_bootstraps
        self.bay_snps =  bootstraps_per_snp[bootstraps_per_snp > min_bootstraps].index
        coefs = coefs[coefs['id'].isin(self.bay_snps)] # Min inclusion
        
        # Empirical priors
        coef_mean = coefs.groupby('id')['beta'].mean().values
        coef_sd = coefs.groupby('id')['beta'].std(ddof=1).values
        
        # Fit model
        ts_model = bay.TwoStageBF(coef_mean, coef_sd,
                                variational=False,
                                n_chain=100000,
                                n_trace=10000)
        ts_trace = ts_model.run(gwas_gen=self.gwas_gen[self.bay_snps].values,
                                gwas_phen=phen)
        ts_stats = ts_model.calculate_ppc(ts_trace)
        ts_stats['bayes_factor'] = ts_model.calculate_bf(ts_trace)
        self.b_stats[ts_model.name] = ts_stats
        self.b_traces[ts_model.name] = ts_trace
        
        return self.b_stats


    def _frequentist(self, pred_expr):
        """
        Compute frequentist statistics using OLS, Regression Calibration,
        and Multiple Imputation.

        Returns:
            dict: Frequentist statistics
        """

        bootstraps = [column for column in pred_expr.columns
                      if column not in ['full', 'twostage']]
        phen = self.gwas_phen.values
        try:
            full_expr = pred_expr['full']
        except KeyError:
            full_expr = np.zeros(shape=phen.shape)

        bootstrap_expr = pred_expr[bootstraps]
        mean_expr = bootstrap_expr.mean(axis=1)
        sigma_ui = bootstrap_expr.var(ddof=1, axis=1)

        self.f_stats = {'OLS-E': t(full_expr, phen,
                                  method="OLS"),
                       'OLS-M': t(mean_expr, phen,
                                  method="OLS"),
                       'RC': t(mean_expr, phen, sigma_ui,
                               method='rc-hetero'),
                       'MI': multiple_imputation(bootstrap_expr.T,
                                                 phen)}
        print(self.f_stats)
        return self.f_stats



    def save(self, file_prefix):
        bootstraps = [column for column in self.pred_expr.columns
                      if column not in ('full', 'twostage')]
        inter_var = self.pred_expr[bootstraps].mean(axis=1).var(ddof=1)
        intra_var = self.pred_expr[bootstraps].var(axis=1, ddof=1).mean()
        with open(file_prefix + '.variance', 'w') as OUT:
            OUT.write('Intra-variance\t{}\t{}\n'.format(self.gene, intra_var))
            OUT.write('Inter-variance\t{}\t{}\n'.format(self.gene, inter_var))
        
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
