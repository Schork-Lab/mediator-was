# Python libraries
import os
# Packages
import yaml
import pandas as pd
import sqlite3


# Relative paths
full_path = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(open(os.path.join(full_path,
                                     '../../config.yaml')))
dbsnp = config['data']['references']['dbsnp']
gencode = config['data']['references']['gencode']
main_dir = config['data']['predixcan']['dir']
en_db = os.path.join(main_dir,
                     config['data']['predixcan']['files']['database'])


def add_positions_to_database(en_df,
                              dbsnp=dbsnp,):
    '''
    The database only contains rsids. To predict gene expression, need
    to know locations to match with VCF files. Add positions to the database
    file.
    '''

    dbsnp_df = pd.read_table(dbsnp, compression='gzip', header=None)
    columns = ['id', 'chr', 'start', 'end', 'rsid']
    columns += list(dbsnp_df.columns[5:])
    dbsnp_df.columns = columns
    dbsnp_df = dbsnp_df[['id', 'chr', 'start', 'end', 'rsid']]
    dbsnp_df = dbsnp_df.drop_duplicates('rsid')
    en_df = en_df.merge(dbsnp_df,
                        left_on='rsid', right_on='rsid',
                        how='inner')
    return en_df

def load_gencode(gencode=gencode):
    gencode_df = pd.read_table(gencode, header=None, skiprows=5, sep="\t")
    gencode_df.rename(columns={0: 'chromosome', 2: 'type', 3: 'start', 4: 'end', 8: 'info'}, inplace=True)
    gencode_df = gencode_df[gencode_df['type'] == 'gene']
    gencode_df['gene'] = gencode_df['info'].map(lambda x: x.split('gene_id')[1].split()[0].split('"')[1])
    gencode_df['gene_name'] = gencode_df['info'].map(lambda x: x.split('gene_name')[1].split()[0].split('"')[1])

    gencode_df = gencode_df[['chromosome', 'start', 'end', 'gene', 'gene_name']].drop_duplicates().set_index(['gene'])
    return gencode_df



def load_database(fn=en_db,
                  alpha=1.0):
    '''
    Load database
    '''

    en_df = pd.read_table(fn, sep="\t")
    en_df = en_df[en_df.alpha == alpha]
    en_df = add_positions_to_database(en_df)
    en_df.index = range(len(en_df))
    en_df.rename(columns={'refAllele': 'ref', 'chr': 'chromosome', 'end': 'position'}, inplace=True)
    return en_df



def load_sql_database(fn, gencode_df=None):
    '''
    Load SQL Predixcan Databases into Pandas Dataframes
    '''
    def get_gene_name(gene):
        if gene.startswith('ENSG'):
            try:
                gene = gencode_df.ix[gene.split('.')[0]]['gene_name']
            except:
                pass
        return gene

    conn = sqlite3.connect(fn)
    en_df = pd.read_sql('SELECT * from weights', conn)
    en_df = add_positions_to_database(en_df)
    if gencode_df is None:
        gencode_df = load_gencode()
    en_df['gene'] = en_df['gene'].map(get_gene_name)
    en_df.index = range(len(en_df))
    en_df.rename(columns={'ref_allele': 'ref', 'eff_allele': 'alt', 'weight': 'beta',
                          'chr': 'chromosome', 'end': 'position'}, inplace=True)
    return en_df