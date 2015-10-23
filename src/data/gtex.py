import pandas as pd
import statsmodels.api as sm
import os

# Main directories
main_dir = "/projects/gtex"
data_dir = os.path.join(main_dir, "data")
analysis_dir = os.path.join(main_dir, "analysis")

# Genotype files
sample_vcf = "phg000520.v1.GTEx_ExomeSeq_SNP.genotype-calls-vcf.c1/"
sample_vcf += "GTEx_Data_20140613_ExomeSeq_180Indiv_GATK_UG_SNPs_annot.vcf.gz"
sample_vcf = os.path.join(data_dir, "GenotypeFiles", sample_vcf)
rsid_file = sample_vcf+".rsids"

# Expression files
expression_folder = 'phe000006.v2.GTEx_RNAseq.expression-data-matrixfmt.c1'
expression_folder = os.path.join(data_dir, 'ExpressionFiles', expression_folder)
expression_file = 'GTEx_Data_20150112_RNAseq_RNASeQCv1.1.8_gene_rpkm.gct.gz'
expression_file = os.path.join(expression_folder, expression_file)

# Covariate Information files
covariate_folder = os.path.join(data_dir, "ExpressionFiles",
                               "gtex.portal.covariates")
covariate_files = [os.path.join(covariate_folder, fn)
                  for fn in os.listdir(covariate_folder)]

# Residual Expression files
residuals_folder = os.path.join(analysis_dir, "residuals")
residual_files = [os.path.join(residuals_folder, fn)
                  for fn in os.listdir(residuals_folder)]

# Multi-Tissue EQTLs files
multi_tissue_folder = os.path.join(data_dir, "MultiTissueEQTLs")
multi_tissue_eqtl_file = "res_final_uc_com_genes_com_snps.txt.gz"
multi_tissue_eqtl_file = os.path.join(multi_tissue_folder,
                                      multi_tissue_eqtl_file)

# Single Tissue EQTLs files
single_tissue_folder= os.path.join(data_dir, "SingleTissueEQTLs")
single_tissue_eqtls_files = [os.path.join(single_tissue_folder, fn)
                             for fn in os.listdir(single_tissue_folder)
                             if fn.endswith('.eqtl')]
# Sample Information files
sample_info_file = "phs000424.v5.pht002743.v5.p1.c1.GTEx_Sample_Attributes.GRU"
sample_info_file += ".txt.gz"
sample_info_file = os.path.join(data_dir, "PhenotypeFiles",
                                sample_info_file)






def load_sample_genotypes(sample_vcf=sample_vcf, add_rsid=False):
    # Load in files
    sample_ids = pd.read_table(sample_vcf+'.012.indv', names=['Sample'])
    loci = pd.read_table(sample_vcf+'.012.pos',
                         names=['Chromosome', 'Position'])
    sample_genotypes = pd.read_table(sample_vcf+'.012',
                                     sep="\t", header=None, index_col=0).T

    # Fix columns and indices of genotypes dataframe
    sample_genotypes.columns = sample_ids['Sample']
    sample_genotypes['Chromosome'] = loci['Chromosome'].astype(str).values
    sample_genotypes['Position'] = loci['Position'].values

    if add_rsid:
        rsid_file = sample_vcf+'.rsids'
        rsids = pd.read_table(rsid_file, 
                              names=['Chromosome', 'Position', 'Rsid'],
                              sep=" ")
        rsids.set_index(['Chromosome', 'Position'],
                         inplace=True, drop=True)
        sample_genotypes['RSID'] = rsids.ix[sample_genotypes.index]['Rsid']
        sample_genotypes.set_index(['Chromosome', 'Position', 'RSID'],
                                     inplace=True, drop=True)
    else:
        sample_genotypes.set_index(['Chromosome', 'Position'],
                                     inplace=True, drop=True)


    return sample_genotypes


def load_expression(expression_file=expression_file, only_samples=True):
    expression_df = pd.read_table(expression_file,
                                  compression='gzip',
                                  skiprows=2, sep="\t")
    expression_df.set_index('Name', drop=False, inplace=True)
    if only_samples:
        expression_df = expression_df[expression_df.columns[2:]]
    return expression_df


def load_eqtls(eqtl_file, filetype="multi"):
    if filetype == "multi":
        eqtls = pd.read_table(eqtl_file,
                              compression="gzip", sep="\t")
    else:
        eqtls = pd.read_table(eqtl_file, sep="\t")
        # To get it in same format as multisample eqtls
        eqtls['snp'] = eqtls['SNP']
        eqtls['gene'] = eqtls['Gen_ID']

        eqtls['sample'] = os.path.basename(eqtl_file).split('.')[0]
    return eqtls


def load_sample_info(sample_info_file=sample_info_file,
                     only_pilot_tissues=False,
                     subset_columns=['dbGaP_Sample_ID', 'SAMPID', 'SMTS', 'SMTSD']):
    sample_info = pd.read_table(sample_info_file,
                                compression="gzip", skiprows=10, sep="\t")
    sample_map = {'Blood Vessel': 'Artery',
                  'Adipose Tissue': 'Adipose'}
    sample_mapper = lambda x: sample_map[x] if x in sample_map else x         
    sample_info['SMTS'] = sample_info['SMTS'].map(sample_mapper)
    if only_pilot_tissues:
        interesting_tissues = ['Adipose', 'Artery', 'Blood', 'Heart',
                               'Lung', 'Muscle', 'Nerve', 'Skin', 'Thyroid']
        sample_info = sample_info[sample_info.SMTS.isin(interesting_tissues)]

    if subset_columns:
        sample_info = sample_info[subset_columns]

    return sample_info


def load_covariates(covariate_file):
    covariate_df = pd.read_table(covariate_file)
    covariate_df.set_index('ID', inplace=True, drop=True)
    covariate_df = covariate_df.T
    return covariate_df


def add_sample_info(covariate_df, sample_info, covariate):
    covariate_name = lambda x: x.replace(' - ','_').replace(' ','_')
    sample_info['covariate_name'] = sample_info.SMTSD.map(covariate_name)
    covariate_samples = sample_info.covariate_name == covariate
    covariate_samples = sample_info[covariate_samples]['SAMPID']
    sample_map = dict(("-".join(sample.split('-')[:2]), sample)
                      for sample in covariate_samples)
    get_sample_id = lambda x: sample_map[x] if x in sample_map else None
    covariate_df['Sample'] = covariate_df.index.map(get_sample_id)
    covariate_df.dropna()
    covariate_df.set_index('Sample', inplace=True, drop=True)
    return covariate_df

def calculate_residuals(covariate_df, expression_df, gene, overlapping_samples):
    covariates = covariate_df.ix[overlapping_samples]
    phenotype = expression_df.ix[gene]
    phenotype = phenotype[overlapping_samples].astype(float)
    covariates = sm.add_constant(covariates)
    model = sm.OLS(phenotype, covariates)
    model_fit = model.fit()
    residuals = pd.DataFrame(model_fit.resid, columns=[gene])
    return residuals

def load_residuals(residual_files):
    residuals_df = pd.concat([pd.read_table(fn, index_col=0) 
                              for fn in residual_files]).T
    return residuals_df

def get_samples_by_genotype(sample_genotypes, loci, genotype=None):
    '''
    loci is tuple ('chromosome', position)
    '''
    def get_samples(loci_genotypes, genotype):
        return loci_genotypes[loci_genotypes == genotype].index
    loci_genotypes = sample_genotypes.ix[loci]
    if not genotype:
        return dict((genotype, get_samples(loci_genotypes, genotype))
                     for genotype in [0, 1, 2])
    else:
        return get_samples(loci_genotypes, genotype)


def get_gene_df(expression_df, gene, interesting_tissues):
    gene_df = pd.DataFrame(expression_df.ix[[gene,'Tissue']]).T
    gene_df['Sample'] = gene_df.index.map(lambda x: x.split('-')[1])
    gene_df[gene] = gene_df[gene].astype(float)
    gene_df = gene_df[gene_df.Tissue.isin(interesting_tissues)]
    gene_df = gene_df.pivot_table(values=gene, index='Sample', columns=['Tissue'])
    gene_df = gene_df.dropna()
    return gene_df

