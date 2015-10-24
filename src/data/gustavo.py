# Python libraries
import os
# Packages
import yaml
import pandas as pd
# Project
import gtex

# Relative paths
config = yaml.load(open('../config.yaml'))
dbsnp = config['data']['references']['dbsnp']
main_dir = config['data']['gustavo']['dir']
en_db = os.path.join(main_dir,
                     config['data']['gustavo']['files']['database'])


def add_positions_to_database(en_db=en_db,
                              dbsnp=dbsnp,
                              fn=os.path.join(main_dir, "EN.withpos.tsv")):
    '''
    The database only contains rsids. To predict gene expression, need
    to know locations to match with VCF files. Add positions to the database
    file.
    '''

    en_df = pd.read_table(en_db)  # Elastic Net DF
    dbsnp_df = pd.read_table(dbsnp, compression='gzip')
    columns = ['id', 'chr', 'start', 'end', 'rsid']
    columns += list(dbsnp_df.columns[5:])
    dbsnp_df.columns = columns
    en_df = en_df.merge(dbsnp_df[['id', 'chr', 'start', 'end', 'rsid']],
                        left_on='rsid', right_on='rsid',
                        how='inner')
    en_df.to_csv(fn, sep="\t", index=False)
    return fn


def load_database(fn=os.path.join(main_dir, "EN.withpos.tsv")):
    '''
    Load database
    '''

    en_df = pd.read_table('../data/DGN-WB_0.5.withpos.txt', sep="\t")
    en_df['chr'] = en_df['chr'].map(lambda x: x[3:])
    return en_df
