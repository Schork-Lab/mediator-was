import seaborn as sns
import pandas as pd


def multi_gene_plot(merged_df, genes):
    '''
    Creates a predicted vs measured plot for a list
    of genes found in merged_df

    merged_df created from modeling.helpers.merged_df has
    the form:
                            sample_1   sample_2  sample_3
         gene_id-1:
            Measured
            Predicted
         gene_id-2:
            Measured
            Predicted
    '''
    combined_gene_df = []
    for gene in genes:
        gene_df = merged_df.ix[gene].T
        gene_df['Gene'] = gene
        combined_gene_df.append(gene_df)
    combined_gene_df = pd.concat(combined_gene_df)
    ax = sns.lmplot('Predicted', 'Measured',
                    col='Gene', data=combined_gene_df)
    return ax
