import os

import yaml
import pandas as pd
from pyensembl import EnsemblRelease

full_path = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(open(os.path.join(full_path,
                                     '../../config.yaml')))
main_dir = config['data']['gtex']['dir']
genotype_fn = os.path.join(main_dir,
                           config['data']['gtex']['files']['genotypes'])
expression_fn = os.path.join(main_dir,
                             config['data']['gtex']['files']['expression'])


def load_expression(fn=expression_fn):
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
