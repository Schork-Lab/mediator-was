import os

import yaml
import pandas as pd
from pyensembl import EnsemblRelease

full_path = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(open(os.path.join(full_path,
                                     '../../config.yaml')))
main_dir = config['data']['gtex']['dir']
genotypes_dir = config['data']['gtex']['files']['genotypes']['by_chromosome']
genotypes_path = os.path.join(main_dir, genotypes_dir)
expression_fn = config['data']['gtex']['files']['expression']
expression_path = os.path.join(main_dir, expression_fn)




def load_expression(fn=expression_path):
    expression_df = pd.read_table(fn)
    ensembl = EnsemblRelease(76)

    def get_gene(x):
        try:
            return ensembl.gene_by_id(x.split('.')[0]).name
        except:
            return 'None'

    expression_df['name'] = expression_df['Id'].map(get_gene)
    del expression_df['Id']
    # For now, let's just remove all cases, where there is more
    # than one instance of a gene. Something funky with Ensembl.
    instances = expression_df['name'].value_counts()
    more_than_one = instances[instances > 1].index
    expression_df = expression_df[~expression_df.isin(more_than_one)]
    expression_df = expression_df.dropna(subset=['name'])
    expression_df.set_index('name', inplace=True, drop=True)
    return expression_df

def _load_012(fn, rsids=None):
    '''
    Loads in vcf-generated .012 files. Optionally reads and
    appends rsids file that has been generated using the first 3 columns
    of a vcf file (chromosome, position, rs-id)
    '''
    # Load in files
    sample_ids = pd.read_table(fn+'.indv', names=['Sample'])
    loci = pd.read_table(fn+'.pos',
                         names=['Chromosome', 'Position'])
    genotypes_df = pd.read_table(fn,
                                sep="\t", header=None, index_col=0).T

    # Fix columns and indices of genotypes dataframe
    genotypes_df.columns = sample_ids['Sample']
    genotypes_df['Chromosome'] = loci['Chromosome'].astype(str).values
    genotypes_df['Position'] = loci['Position'].values
    genotypes_df.set_index(['Chromosome', 'Position'],
                            inplace=True, drop=False)

    if rsids:
        rsids = pd.read_table(rsids, 
                              names=['Chromosome', 'Position', 'Rsid'],
                              sep="\t")
        rsids['Chromosome'] = rsids['Chromosome'].astype(str).values
        # Remove indels
        rsids.drop_duplicates(['Chromosome', 'Position'], inplace=True)
        rsids.set_index(['Chromosome', 'Position'],
                         inplace=True, drop=True)
        genotypes_df['RSID'] = rsids.ix[genotypes_df.index]['Rsid']
        genotypes_df.set_index(['Chromosome', 'Position', 'RSID'],
                                     inplace=True, drop=True)
    else:
        genotypes_df.set_index(['Chromosome', 'Position'],
                                     inplace=True, drop=True)

    return genotypes_df

def load_genotypes(chromosome="1", rsids=False):
    """
    Load in chromosome-specific gtex files created. Optionally,
    add rsid to the sample genotypes index
    """
    if rsids:
        rsids = os.path.join(genotypes_path, "all.rsids")
    chromosome_fn = os.path.join(genotypes_path, "chr"+chromosome+".012")
    return _load_012(chromosome_fn, rsids)

