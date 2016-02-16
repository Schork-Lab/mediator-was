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
main_dir = config['data']['bslmm']['dir']
studies = config['data']['bslmm']['studies'].items()


def load_study(study):
    '''
    Loads a BSLMM study by going through all
    the gene specific folders and aggregating
    them into a single dataframe.
    '''
    def load_gene(gene_dir):
        gene = os.path.basename(gene_dir)
        try:
            snps = pd.read_table(os.path.join(gene_dir, gene+".wgt.map"),
                                 sep=" ")
            betas = pd.read_table(os.path.join(gene_dir, gene+".wgt.cor"),
                                  names=['beta'])
        except:
            return None
        gene_df = pd.concat([snps, betas], axis=1)
        gene_df['Gene'] = gene
        return gene_df

    study_dir = os.path.join(main_dir, study[1]['dir'])
    gene_dfs = [load_gene(os.path.join(study_dir, gene))
                for gene in os.listdir(study_dir)]
    study_df = pd.concat(gene_dfs)
    study_df.sort(['SNP_Pos'])
    return study_df


def parse_studies():
    '''
    Parses all studies in the BSLMM data.
    '''
    study_data = {}
    dbsnp_df = pd.read_table(dbsnp, compression='gzip')
    columns = ['id', 'chromosome', 'start', 'end', 'rsid']
    columns += list(dbsnp_df.columns[5:])
    dbsnp_df.columns = columns
    for study in studies:
        study_df = load_study(study)
        study_df = study_df.merge(dbsnp_df[['id', 'chromosome', 'start', 'end', 'rsid']],
                                    left_on='SNP_ID', right_on='rsid',
                                    how='inner')
        study_df.rename(columns={'SNP_Pos': 'position', 'Gene': 'gene'}, inplace=True)
        study_data[study[0]] = study_df
    return study_data
m