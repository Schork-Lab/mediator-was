# Python libraries
import os
# Packages
import yaml
import pandas as pd


# Relative paths
full_path = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(open(os.path.join(full_path,
                                     '../../config.yaml')))
dbsnp = config['data']['references']['dbsnp']
main_dir = config['data']['predixcan']['dir']
en_db = os.path.join(main_dir,
                     config['data']['predixcan']['files']['database'])


def add_positions_to_database(en_df,
                              dbsnp=dbsnp,
                              fn=os.path.join(main_dir, "EN.withpos.tsv")):
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
