'''
Train bootstrapped ElasticNet using GTEx normalized and rlog transformed GTEx Whole Blood Samples
Author: Kunal Bhutani <kunalbhutani@gmail.com>
Date: 06/20/2016
'''

import sys
import os
import pandas as pd
import numpy
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.utils import resample
from plinkio import plinkfile
import yaml

# Relative paths
full_path = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(open(os.path.join(full_path, '../../config.yaml')))

# WTCCC Imputed Loci
wtccc_dir = config['data']['wtccc']['dir']
imputed_loci = pd.read_table(os.path.join(wtccc_dir, "RA.bim"), header=None)
imputed_loci = set(imputed_loci[0].astype(str)+'_'+imputed_loci[3].astype(str))

# GTEx Normalized
gtex_dir = config['data']['gtex']['dir']
gtex_expression = config['data']['gtex']['files']['normalized']
gtex_covariates = config['data']['gtex']['files']['covariates']
gtex_data_df = pd.read_table(os.path.join(gtex_dir, gtex_expression))
gtex_covariates_df = pd.read_table(os.path.join(gtex_dir, gtex_covariates),
                                   index_col=0).T

# Rlog Transformed   
rlog_dir = os.path.join(gtex_dir, 'tis53')
rlog_data_dfs = dict((chrom, pd.read_table(os.path.join(rlog_dir, 'chr{}.data.gz'.format(chrom)),
                        header=None, ))
                     for chrom in range(1, 23)) # Autosomal
rlog_gene_dfs = dict((chrom, pd.read_table(os.path.join(rlog_dir, 'chr{}.info.gz'.format(chrom)), 
                       names=['chromosome', 'start', 'end', 'ensemblid', 'name']))
                     for chrom in range(1, 23))
rlog_pcs_dfs = dict((chrom, pd.read_table(os.path.join(rlog_dir, 'chr{}.pcs.tsv'.format(chrom)),
                      index_col=0))
                     for chrom in range(1, 23))

rlog_sample_df = pd.read_table(os.path.join(rlog_dir, 'tis53.samples.txt.gz'), header=None)
for i in range(1, 23):
    rlog_data_dfs[i].columns = rlog_gene_dfs[i].name
    rlog_data_dfs[i].index = rlog_sample_df[0].map(lambda x: "-".join(x.split('-')[:2]))


# Processed Dir
processed_dir = config['analysis']['processed']['gtex']


def load_alleles_df(plink_file):
    '''
    Load alleles from a plink file. Note that they are counts of reference,
    so reverse into counts of alternate.

    Return an allele dataframe

    '''
    def reverse_alleles(x):
        if x == 0:
            return 2
        elif x == 2:
            return 0
        else:
            return x
    reverse_alleles = numpy.vectorize(reverse_alleles)

    plink_file = plinkfile.open(plink_file)
    sample_ids = map(lambda x: x.iid, plink_file.get_samples())
    allele_ids = map(lambda x: x.name, plink_file.get_loci())
    alleles = numpy.array(list(plink_file)).T
    alleles = reverse_alleles(alleles)
    allele_df = pd.DataFrame(alleles, columns=allele_ids, index=sample_ids)
    return allele_df


def fit_gene(gene, chromosome, subset_loci=None):
    ''' 
    Load in gene information and fit bootstrapped ElasticNet models.

    '''
    def create_coeff_df(model, allele_df):
        coef_df = pd.DataFrame(model.coef_[:allele_df.shape[1]],
                               index=allele_df.columns,
                               columns=['beta'])
        coef_df['chromosome'] = coef_df.index.map(lambda x: x.split('_')[0])
        coef_df['position'] = coef_df.index.map(lambda x: x.split('_')[1])
        coef_df['ref'] = coef_df.index.map(lambda x: x.split('_')[2])
        coef_df['alt'] = coef_df.index.map(lambda x: x.split('_')[3])
        coef_df['gene'] = gene
        coef_df = coef_df[coef_df.beta != 0]
        return coef_df

    def load_gtex_normalized(gene_dir):
        gene = os.path.basename(gene_dir)
        phen_df = pd.read_table(os.path.join(gene_dir, gene+".gtex_normalized.phen.tsv"), sep="\t")
        covariates_df = pd.read_table(os.path.join(gene_dir, gene+".gtex_normalized.phen.tsv"), sep="\t")
        return phen_df, covariates_df

    def load_rlog(gene_dir):
        gene = os.path.basename(gene_dir)
        phen_df = pd.read_table(os.path.join(gene_dir, gene+".rlog.phen.tsv"), sep="\t")
        covariates_df = pd.read_table(os.path.join(gene_dir, gene+".rlog.covariates.tsv"), sep="\t")
        return phen_df, covariates_df

    def fit_models(allele_df, phen_df, covariates_df,
                   n_samples=300, n_bootstraps=50,):
        coef_dfs = []
        samples = phen_df.index

        design = pd.concat([allele_df.ix[samples], covariates_df.ix[samples]],
                           axis=1)
        l1_ratio_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                          0.7, 0.8, 0.9, 0.95, 0.99]
        full_model = ElasticNetCV(l1_ratio=l1_ratio_range, max_iter=10000)
        full_model.fit(design, phen_df.values.ravel())
        coef_df = create_coeff_df(full_model, allele_df)
        coef_df['bootstrap'] = 'full'
        coef_dfs.append(coef_df)

        for i in range(n_bootstraps):
            b_samples = resample(samples, replace=False, n_samples=n_samples)
            model = ElasticNet(alpha=full_model.alpha_,
                               l1_ratio=full_model.l1_ratio_,
                               max_iter=10000)
            model.fit(design.ix[b_samples],
                      phen_df.ix[b_samples].values.ravel())
            coef_df = create_coeff_df(model, allele_df)
            coef_df['bootstrap'] = str(i)
            coef_dfs.append(coef_df)

        return pd.concat(coef_dfs), full_model.alpha_, full_model.l1_ratio_

    gene_dir = os.path.join(processed_dir, "{}/{}".format(chromosome, gene))

    # Load Genotypes
    allele_df = pd.read_table(os.path.join(gene_dir, "{}.alleles.tsv".format(gene)))
    # !! Treat missing as 0
    allele_df = allele_df.applymap(lambda x: 0 if x == 3 else x) 
    loci = pd.read_table(os.path.join(gene_dir, '{}.locinames'.format(gene)))
    loci['tag'] = loci['chromosome'].astype(str) + '_' + loci['position'].astype(str)
    if not subset_loci:
        subset_loci = loci['tag']
    loci['subset'] = loci['tag'].map(lambda x: True if x in subset_loci else False)
    allele_df.columns = loci['id']
    allele_df = allele_df[loci[loci['overlapping']]['id']]


    gtex_phen_df, gtex_covariates_df = load_gtex_normalized(gene_dir)   
    gtex_coeff_dfs, gtex_alpha, gtex_l1_ratio = fit_models(allele_df, gtex_phen_df, gtex_covariates_df)
    with open(os.path.join(gene_dir, "bootstrap_params.gtex_normalized.txt"), "w") as OUT:
        OUT.write("L1 ratio: {} \n".format(gtex_alpha))
        OUT.write("Alpha: {} \n".format(gtex_l1_ratio))

    rlog_phen_df, rlog_covariates_df = load_rlog(gene_dir)
    rlog_coeff_dfs, rlog_alpha, rlog_l1_ratio = fit_models(allele_df, rlog_phen_df, rlog_covariates_df)
    with open(os.path.join(gene_dir, "bootstrap_params.rlog.txt"), "w") as OUT:
        OUT.write("L1 ratio: {} \n".format(rlog_alpha))
        OUT.write("Alpha: {} \n".format(rlog_l1_ratio))

    return gtex_coeff_dfs, rlog_coeff_dfs


def process_gene(gene, chromosome):
    '''
    Process gene to create intermediates that are used when fitting ElasticNet.
    '''
    def generate_gtex_intermediates(gene, chromosome, samples=None):
        phen_df = gtex_data_df[gtex_data_df.gene == gene]
        if not samples:
            samples = [column for column in phen_df.columns
                       if column.startswith('GTEX-')]
        phen_df = phen_df[samples].T
        phen_df.columns = [gene]
        covariates_df = gtex_covariates_df.ix[samples]
        return phen_df, covariates_df

    def generate_rlog_intermediates(gene, chromosome, samples=None):
        '''
        '''
        rlog_data_df = rlog_data_dfs[chromosome]
        rlog_pcs_df = rlog_pcs_dfs[chromosome]
        covariates = ['C1', 'C2', 'C3', 'gender', 'Platform']
        covariates_df = pd.concat([gtex_covariates_df[covariates],
                                   rlog_pcs_df], axis=1).dropna()
        phen_df = rlog_data_df[[gene]].dropna()
        if not samples:
            samples = list(set(covariates_df.index).intersection(phen_df.index))
        phen_df = phen_df.ix[samples]
        covariates_df = covariates_df.ix[samples]
        return phen_df, covariates_df

   # Genotypes
    plink_file = os.path.join(processed_dir,
                              "{}/{}/{}".format(chromosome, gene, gene))
    allele_df = load_alleles_df(plink_file)
    alleles_file = '{}.alleles.tsv'.format(plink_file)
    allele_df.to_csv(alleles_file, sep="\t", index=False, header=False)

    loci_file = '{}.locinames'.format(plink_file)
    loci_df = pd.DataFrame(allele_df.columns, columns=['id'])
    loci_df['chromosome'] = loci_df['id'].map(lambda x: x.split('_')[0])
    loci_df['position'] = loci_df['id'].map(lambda x: x.split('_')[1])
    loci_df['ref'] = loci_df['id'].map(lambda x: x.split('_')[2])
    loci_df['alt'] = loci_df['id'].map(lambda x: x.split('_')[3])
    loci_df['gene'] = gene
    loci_df.to_csv(loci_file, sep="\t", index=False)

    # GTEx Normalized Intermediates
    gtex_phen_df, gtex_covariates_df = generate_gtex_intermediates(gene, chromosome)
    phen_file = '{}.gtex_normalized.phen.tsv'.format(plink_file)
    gtex_phen_df.to_csv(phen_file, sep='\t')
    covariates_file = '{}.gtex_normalized.covariates.tsv'.format(plink_file)
    gtex_covariates_df.to_csv(covariates_file, sep='\t')

    # RLog Transformed Intermediates
    rlog_phen_df, rlog_covariates_df = generate_rlog_intermediates(gene, chromosome)
    phen_file = '{}.rlog.phen.tsv'.format(plink_file)
    rlog_phen_df.to_csv(phen_file, sep='\t')
    covariates_file = '{}.rlog.covariates.tsv'.format(plink_file)
    rlog_covariates_df.to_csv(covariates_file, sep='\t')
    return

def main(gene_list_file, out_file_prefix, process=True):
    '''
    Train models for all genes in gene_list_file (gene, chromosome)
    using both GTEx normalized and rlog transformed values.
    '''
    gene_list = pd.read_table(gene_list_file, names=['gene', 'chromosome'])
    if process:
        for gene, chromosome in gene_list:
            process_gene(gene, chromosome)
    gtex_norm_coeffs_list, rlog_coeffs_list = [], []
    for gene, chromosome in gene_list:
        gtex_coeffs, rlog_coeffs = fit_gene(gene, chromosome)
        gtex_norm_coeffs_list.append(gtex_coeffs)
        rlog_coeffs_list.append(rlog_coeffs)
    pd.concat(gtex_norm_coeffs_list).to_csv('{}.gtex_normalized.tsv'.format(out_file_prefix),
                                            sep="\t")
    pd.concat(rlog_coeffs_list).to_csv('{}.rlog.tsv'.format(out_file_prefix),
                                       sep="\t")
    return


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage: python train_gtex_EN.py gene_list_file out_prefix')
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
