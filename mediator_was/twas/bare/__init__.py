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
from mediator_was.association.frequentist import *
import mediator_was.association.bayesian as bay
import logging
import argparse

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)

class Gene():
    """
    TWAS mediator, which contains expression of a gene and
    the genotypes +- 500KB of TSS/TSE of gene for a GTEx or equivalent
    study.

    Attributes:
        chromosome (str): chromosome of gene
        elasticnet (pandas.DataFrame): output from fitted bootstrapped models
                                       of transcriptional regulation
        gtex (bool): use gtex normalized evalues
        main_dir (str): path to gene directory
        name (str): name of gene
        elasticnet (pandas.DataFrame): effect sizes of bootstrapped elasticnet
        l1_ratio (float): l1/l2 ratio determined using 3-fold cross validation
        alpha (float): learning rate determined using 3-fold cross validation
        r2 (float): "out-of-sample" r2 using 5-fold cross validation
    """

    def __init__(self, main_dir, gtex=True):
        """
        Currently only loads in processed genes that have been
        processed using scripts/train_gtex_EN.py

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
        Load processed data for the gene. Each gene was fit using ElasticNetCV
        to calculate hyperparameters L1 Ratio and Alpha and the "full" model
        parameters, and then 50 times using 300 bootstrapped samples and the
        same hyper parameters
        """
        def load_params(fn):
            """
            Load the transcriptional regulation ElasticNet fit parameters
            Args:
                fn (str): Path to the file
            Returns:
                tuple: L1 Ratio, Alpha, R2
            """
            with open(fn, 'r') as IN:
                data = tuple(map(lambda x: float(x.rstrip().split(' ')[-1]),
                                 IN))
            return data

        if self.gtex:
            label = "gtex_normalized"
        else:
            label = "rlog"

        logger.info('Loaded Gene %s', self.name)
        # Load the bootstrapped elasticnet weights
        en_file = os.path.join(self.main_dir,
                               "{}.{}.elasticnet.tsv".format(self.name, label))
        self.elasticnet = pd.read_table(en_file, sep='\t')

        # Load the model fitting parameters
        param_file = en_file.replace('elasticnet.tsv', 'params.txt')
        self.l1_ratio, self.alpha, self.r2 = load_params(param_file)

        return


class Study():
    """
    A GWAS type study with genotypes and a continuous phenotype.
    Requires that a VCF and the phenotypes share the same prefix.

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

    def get_alleles(self, chrom, position,
                    ref=None, alt=None, missing_filter=0.1):
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
            missing_filter (float, optional): Description

        Returns:
            LIST: number of alt alleles per sample, nan if not found
        '''
        def parse_alleles(samples, switch=False):
            """
            Parse pysam samples into dosage of alt-allele counts

            Args:
                samples (LIST): pysam sample records
                switch (bool, optional): switch alt and ref alleles

            Returns:
                TYPE: Description
            """
            def replace_missing(allele, aaf):
                """
                Replaces missing alleles with the provided aaf.
                Switches to 1-aaf is the ref and alt are switched
                between VCF and user-provided ref and alt

                Args:
                    allele

                """
                if allele is None:
                    return aaf
                if switch:
                    return 1 - allele
                return allele

            allele_counts = Counter()
            for sample in samples:
                allele_counts.update([gt for gt in sample.values()[0]])
            n_chroms = (allele_counts[0] + allele_counts[1])
            t_chroms = n_chroms + allele_counts[None]
            aaf = float(allele_counts[1]) / n_chroms
            missing_af = float(allele_counts[None]) / t_chroms
            if missing_af > missing_filter:
                logger.warn("Skpping {}: {}".format(chrom, position))
                logger.warn('Missing freq {} is higher than {}'
                             .format(missing_af, missing_filter))
                raise

            alleles = [np.sum([replace_missing(gt, aaf)
                               for gt in sample.values()[0]])
                       for sample in samples]
            return alleles

        # Try to fetch record and calculate alleles, otherwise
        # raise exception
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
                    logger.warn('Switching ref, alt at {}:{}'.
                                 format(chrom, position))

                    alleles = parse_alleles(record.samples.values(),
                                            switch=True)
                # Not matching ref and alt
                else:
                    logger.warn('Not matching ref, alt at {}:{}'
                                 .format(chrom, position))
                    raise
        except:
            # All alleles are nan if not found
            alleles = np.nan
        return alleles


class Association():
    """
    Compute association statistics using imputed expression
    values from a study vcf and gene elasticnet models.

    Attributes:
        b_stats (dict): Bayesian statistics
        bay_snps (list): List of SNPs included in BAY-TS
        elasticnet (pandas.DataFrame): eQTLs from bootstrapped models
        f_stats (dict): Frequentist statistics
        gene (str): Name of gene
        gwas_gen (pandas.DataFrame): n x p matrix of p genotypes, n individuals
        gwas_phen (pandas.DataFrame): n x 1 array of individual phenotypes
        loci (list): Union of loci in all elasticnet models
        min_p_inclusion (float): Minimum probability of inclusion for BAY-TS
        missing_filter (float): Maximum missingness allowed for GWAS loci
        pred_expr (pandas.DataFrame): n x m matrix of predicted expression,
                                      n individuals
                                      m-1 bootstrapped models
                                      1 complete, full model

        study (str): Name of study
    """

    def __init__(self, gene, study,
                 associate=True,
                 min_p_inclusion=0.5,
                 missing_filter=0.1):
        """
        Load in study genotypes and phenotypes based on
        gene, predict expression, and compute association
        statistics.

        Args:
            gene (Gene): A Gene object
            study (Study): A Study object
            associate (bool, optional): Compute association statistics
            min_p_inclusion (float, optional): Minimum probability of inclusion
                                               for BAY-TS
            missing_filter (float, optional): Maximum missingness for GWAS loci
        """
        self.gene = gene.name
        self.study = study.name
        self.elasticnet = gene.elasticnet
        self.min_p_inclusion = min_p_inclusion
        self.missing_filter = missing_filter

        logger.info("Starting association for {}: {}"
                     .format(self.gene, self.study))

        # Load genotypes and predict expression
        logger.info("Reading GWAS genotypes")
        self._load_genotypes(gene, study)
        logger.info("Predicting expression")
        self._predict_expression(gene)
        logger.info("Load phenotypes")
        self._load_phenotypes(gene, study)

        # Compute association statistics
        self.f_stats = None
        self.b_stats = None
        if associate:
            logger.info("Calculating association statistics")
            self.associate()
        return

    def _load_genotypes(self, gene, study):
        """
        Load genotypes for GWAS individuals
        based on overlap between study VCF and gene elasticnet
        bootstrapped loci.

        Args:
            gene (Gene): Gene object
            study (Study): Study object

        Raises:
            ValueError: There is no overlap between study VCF
                        and gene elasticnet bootstrapped loci
        """
        def _get_alleles(loci):
            alleles = study.get_alleles(loci['chromosome'],
                                        loci['position'],
                                        loci['ref'],
                                        loci['alt'],
                                        self.missing_filter)
            return alleles

        # Get unique loci
        loci = gene.elasticnet[['id', 'chromosome', 'position', 'ref', 'alt']]
        loci = loci.drop_duplicates().set_index('id')

        # Get alleles from study VCF using unique loci
        alleles = loci.apply(_get_alleles, axis=1)
        alleles = alleles[~alleles.isnull()]

        # Raise exception if no matching loci between study VCF and gene
        # elasticnet bootstrapped models
        if len(alleles) == 0:
            msg = "No matching alleles between study and gene elasticnet models."
            raise ValueError(msg)

        # Create gwas_gene pandas.DataFrame attribute
        self.loci = list(alleles.index)
        self.gwas_gen = pd.DataFrame([np.array(x) for x in alleles],
                                     index=self.loci).T
        self.gwas_gen.index = study.samples
        return

    def _load_phenotypes(self, gene, study):
        """
        Load phenotypes associated with gene and study
        based on gene's chromosome

        Args:
            gene (Gene): Gene object
            study (Study): Study object
        """
        self.gwas_phen = study.phen[gene.chromosome]['phen']
        return

    def _predict_expression(self, gene):
        """
        Predict expression using ElasticNet bootstraps in gene.
        and previously calculated gwas_gen attribute.

        Set pred_expr attribute

        Args:
            gene (Gene): Gene object
        """
        logger.info("Predicting expression for {}".format(self.gene))
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

    def associate(self):
        """
        Run Bayesian and Frequentist measurement error associations.
        """
        # Calculate inter-individual and intra-individual variance
        bootstraps = [column for column in self.pred_expr.columns
                      if column not in ('full', 'twostage')]
        inter_var = self.pred_expr[bootstraps].mean(axis=1).var(ddof=1)
        intra_var = self.pred_expr[bootstraps].var(axis=1, ddof=1).mean()
        logger.info('Intra-variance: {}'.format(intra_var))
        logger.info('Inter-variance: {}'.format(inter_var))

        # Calculate statistics
        self._frequentist(self.pred_expr)
        self._bayesian(self.min_p_inclusion)
        return

    def _bayesian(self, min_inclusion=0.5):
        '''
        Fit Two Stage model and compute a Bayes Factor
        and other statistics.

        Returns:
            dict: Bayesian Statistics

        Args:
            min_inclusion (float, optional): minimum fraction of bootstrapped
                                             models a SNP must be included in
                                             for it be used in BAY-TS
        '''
        self.b_stats, self.b_traces = {}, {}
        phen = self.gwas_phen.values


        logger.info('Calculating Bayesian Statistics')
        # Filter based on overlap with GWAS and minimum inclusion
        not_bs = ['full', 'twostage']
        coefs = self.elasticnet[~self.elasticnet['bootstrap'].isin(not_bs)]
        coefs = coefs[coefs['id'].isin(self.loci)]
        bs_per_snp = coefs['id'].value_counts()
        n_bootstraps = len(self.elasticnet['bootstrap'].unique()) - 2
        min_bootstraps = min_inclusion * n_bootstraps
        self.bay_snps = bs_per_snp[bs_per_snp > min_bootstraps].index
        coefs = coefs[coefs['id'].isin(self.bay_snps)]
        logger.info('Filtered down to {} SNPs for BAY-TS'.format(len(coefs)))


        # Empirical priors
        coef_mean = coefs.groupby('id')['beta'].mean().values
        coef_sd = coefs.groupby('id')['beta'].std(ddof=1).values

        # Fit model
        logger.info('Running BAY-TS')
        ts_model = bay.TwoStageBF(coef_mean, coef_sd,
                                  variational=False,
                                  n_chain=50000,
                                  n_trace=10000)
        ts_trace = ts_model.run(gwas_gen=self.gwas_gen[self.bay_snps].values,
                                gwas_phen=phen)
        ts_stats = ts_model.calculate_ppc(ts_trace)
        ts_stats['bayes_factor'] = ts_model.calculate_bf(ts_trace)
        self.b_stats[ts_model.name] = ts_stats
        self.b_traces[ts_model.name] = ts_trace
        logger.info('Finished BAY-TS')
        return self.b_stats


    def _frequentist(self, pred_expr):
        """
        Compute frequentist statistics using OLS, Regression Calibration,
        and Multiple Imputation.
        
        Returns:
            dict: Frequentist statistics
        
        Args:
            pred_expr (pandas.DataFrame): rows: individuals,
                                          columns: elasticnet models
        """

        logger.info('Calculating Frequentist Statistics')
        bootstraps = [column for column in pred_expr.columns
                      if column not in ['full', 'twostage']]
        phen = self.gwas_phen.values
        try:
            full_expr = pred_expr['full']
        except KeyError:
            logger.warn('Full ElasticNet model not found')
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
        return self.f_stats



    def save(self, file_prefix):
        """
        Save frequentist and bayesian statistics as 
        well as variance ratio with file_prefix and suffixes:
        '.fstats.tsv', '.bstats.tsv', '.variance', respectively.
        
        Args:
            file_prefix (str): Description
        
        """
        logger.info('Saving statistics with prefix: {}'.format(file_prefix))
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
                                                    for index in f_df.index],
                                                    names=('estimator', 'gene'))
            f_df.to_csv(file_prefix + '.fstats.tsv', sep='\t')


        if self.b_stats is not None:
            b_df = pd.DataFrame(self.b_stats).T
            b_df.index = pd.MultiIndex.from_tuples([(index, self.gene)
                                                    for index in b_df.index],
                                                    names=('estimator', 'gene'))
            b_df.to_csv(file_prefix + '.bstats.tsv', sep='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene', help='Path to gene directory', required=True)
    parser.add_argument('--study', help='Path to study vcf prefix', required=True)
    parser.add_argument('--out', help='Path to out files', required=True)
    parser.add_argument('--rlog', help='Use RLog transformed values, default', 
                        dest='gtex', action='store_false')
    parser.add_argument('--gtex', help='Use GTEx-Norm values', 
                        dest='gtex', action='store_true')
    parser.add_argument('--min_inclusion', help='Minimum inclusion probability for BAY-TS, default: 0.5',
                        type=float, default=0.5)
    parser.add_argument('--max_missing', help='Maximum missing genotypes for exclusion for study genotypes, default: 0.1',
                        type=float, default=0.1)
    parser.set_defaults(gtex=False)
    args = parser.parse_args()

    logger.info('Loading gene {}'.format(args.gene))
    gene = Gene(args.gene, args.gtex)

    logger.info('Loading Study {}'.format(args.study))
    study = Study(args.study)

    logger.info('Associating')
    association = Association(gene, study, min_p_inclusion=args.min_inclusion, missing_filter=args.max_missing)
    association.save(args.out)