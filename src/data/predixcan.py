# Python libraries
import os
import gzip
# Packages
import yaml
import pandas as pd
# Project
import gtex
import helpers as h

# Relative paths
full_path = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(open(os.path.join(full_path,
                                     '../../config.yaml')))
dbsnp = config['data']['references']['dbsnp']
main_dir = config['data']['predixcan']['dir']
en_db = os.path.join(main_dir,
                     config['data']['predixcan']['files']['database'])


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


def load_database(fn=os.path.join(main_dir, "EN.withpos.tsv"),
                 alpha=1.0):
    '''
    Load database
    '''

    en_df = pd.read_table(fn, sep="\t")
    en_df = en_df[en_df.alpha == alpha]
    en_df['chromosome'] = en_df['chr']
    en_df['position'] = en_df['end']
    en_df.index = range(len(en_df))

    return en_df


def predict_expression(en_df, vcf_file):
    predictions_df, not_found, incorrect, switched = h.stream_predict(en_df, vcf_file)
    return predictions_df, not_found, incorrect, switched

def load_predicted(fn):
    if fn.endswith('.gz'):
        compression = 'gzip'
    else:
        compression = None
    predicted_df = pd.read_table(fn, 
                                 index_col=0,
                                 compression=compression,
                                 sep='\t')
    return predicted_df