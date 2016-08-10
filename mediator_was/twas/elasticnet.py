'''
Methods related to fitting ElasticNet models for gene expression.

Author: Kunal Bhutani <kunalbhutani@gmail.com>
'''
import pandas as pd
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.utils import resample



def create_coeff_df(model, allele_df):
    coef_df = pd.DataFrame(model.coef_[:allele_df.shape[1]],
                           columns=['beta'])
    coef_df['id'] = allele_df.columns
    coef_df['chromosome'] = coef_df['id'].map(lambda x: x.split('_')[0])
    coef_df['position'] = coef_df['id'].map(lambda x: x.split('_')[1])
    coef_df['ref'] = coef_df['id'].map(lambda x: x.split('_')[2])
    coef_df['alt'] = coef_df['id'].map(lambda x: x.split('_')[3])
    coef_df = coef_df[coef_df.beta != 0]
    return coef_df


def fit_models(allele_df, phen_df, covariates_df,
               n_samples=300, n_bootstraps=5,):
    coef_dfs = []
    covariates_df = covariates_df.dropna()
    samples = list(set(phen_df.index).intersection(covariates_df.index))
    design = pd.concat([allele_df.ix[samples], covariates_df.ix[samples]],
                       axis=1)
    design['constant'] = 1
    l1_ratio_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                      0.7, 0.8, 0.9, 0.95, 0.99]
    full_model = ElasticNetCV(l1_ratio=l1_ratio_range, max_iter=10000)
    full_model.fit(design, phen_df.ix[samples].values.ravel())
    coef_df = create_coeff_df(full_model, allele_df)
    coef_df['bootstrap'] = 'full'
    coef_dfs.append(coef_df)

    for i in range(n_bootstraps):
        b_samples = resample(samples, replace=False, n_samples=n_samples)
        model = ElasticNet(alpha=full_model.alpha_,
                           l1_ratio=full_model.l1_ratio_,
                           max_iter=10000)
        model.fit(design.ix[b_samples],
                  phen_df.ix[b_samples].values.ravel())
        coef_df = create_coeff_df(model, allele_df)
        coef_df['bootstrap'] = str(i)
        coef_dfs.append(coef_df)

    return pd.concat(coef_dfs), full_model.alpha_, full_model.l1_ratio_
