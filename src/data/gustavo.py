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

    en_df = pd.read_table(fn, sep="\t")
    en_df['chromosome'] = en_df['chr'].map(lambda x: str(x)[-1])
    en_df['position'] = en_df['end']
    en_df = en_df.sort(['chromosome', 'position'])
    en_df.index = range(len(en_df))
    return en_df


def predict_expression(en_df, vcf_file):
    predictions_df, not_found = h.stream_predict(en_df, vcf_file)
    return predictions_df, not_found
