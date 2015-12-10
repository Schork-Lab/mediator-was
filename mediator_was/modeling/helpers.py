import pandas as pd
import numpy as np


def merge_dfs(measured, predicted):
    '''
    Merge measured and predicted dataframes.
    The dataframe should be in the form: gene_names X sample_names.

    Removes all genes which do not appear in both.

    :returns: a merged dataframe with a multi-index:
                gene_id-1:
                    Measured
                    Predicted
                gene_id-2:
                    Measured
                    Predicted
    '''
    merged = pd.concat([measured, predicted],
                       keys=['Measured', 'Predicted'])
    merged = merged.swaplevel(0, 1)
    found_in_both = set(measured.index).intersection(predicted.index)
    merged = merged.ix[found_in_both, :]
    merged = merged.dropna(axis=1,)
    return merged


def calculate_correlations(merged_df, genes):
    '''
    Calculate the correlation between measured and predicted
    for a set of genes.
    '''
    correlations = merged_df.ix[genes].T.corr()
    correlations = correlations.stack()
    idx = pd.IndexSlice
    idx = (idx[:, 'Measured', 'Predicted'],
           idx[:])
    correlations = pd.DataFrame(np.diag(correlations.loc[idx]),
                                index=genes,
                                columns=['Correlation'])

    return correlations.T
