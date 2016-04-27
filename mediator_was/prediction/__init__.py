"""
Analysis of our predictions before incorporating them into association methods

Author: Kunal Bhutani <kunalbhutani@gmail.com>
"""
from itertools import combinations
import pandas as pd
import mediator_was.prediction.plotting
import numpy
from mediator_was.prediction.predict import stream

class Predictions():
    '''
    Prediction class for calculating and analyzing different predicted expression for the same samples.
    '''
    def __init__(self, 
                 genotype_file=None, genotype_filetype='vcf', weights={}, 
                 predicted_files=None, sample_map=None):
        '''
        Calculate predicted expression using provided genotype file and weights or
        load in already predicted expression values.
        '''       
        # Calculate or load predicted values
        self.predictions = {}
        if predicted_files:
            for method, fn in predicted_files.items():
                self.predictions[method] = self.load(fn, sample_map)
        else:
            error = "Calculating Predictions requires genotype_file and weights"
            assert genotype_file is not None and weights is not None, error
            self.genotype_file = genotype_file
            self.genotype_filetype = genotype_filetype
            self.weights = weights


    def predict(self):
        '''
        Stream predict expression
        '''
        for method, weights in self.weights.items():
            print('Predicting for {}'.format(method))
            predictions_df = stream(weights, self.genotype_file, self.genotype_filetype)
            self.predictions[method] = predictions_df

    def load(self, fn, sample_map=None):
        '''
        Load in previously calculated predicted expression.
        '''
        if fn.endswith('.gz'):
            compression = 'gzip'
        else:
            compression = None
        predicted_df = pd.read_table(fn,
                                     index_col=0,
                                     compression=compression,
                                     sep='\t')
        if sample_map:
            predicted_df.columns = predicted_df.columns.map(sample_map)
        return predicted_df

    def create_attenuation_df(self, studies=None):
        '''
        Calculate attenuation based on overlapping predicted gene expression across studies. 
        Attenuation is defined as (sigma_x + sigma_u) / (sigma_x) . sigma_u is estimated using the variance
        of prediction across different studies.

        TODO: This simplication does not apply for our purposes, because our error is not additive.
        '''
        if not studies:
            studies = self.predictions.keys()
        shared_genes = [set(self.predictions[study].index) for study in studies]
        shared_genes = shared_genes[0].intersection(*shared_genes[1:])
        attenuations = []
        for gene in shared_genes:
            gene_df = pd.concat([self.predictions[study].ix[gene]
                                 for study in studies], axis=1)
            # standardize
            #gene_df = gene_df.apply(lambda x: x+100, axis=0) # Center at 100+mu so everything is positive
            #gene_df = gene_df.apply(lambda x: boxcox(x)[0], axis=0)
            gene_df = gene_df.apply(standardize, axis=0)
            expression = gene_df.mean(axis=1)
            sigma_W = numpy.var(expression, ddof=1)

            error = gene_df.var(axis=1, ddof=1)
            error = error / len(self.predictions)
            sigma_u = numpy.mean(error)
            sigma_u_var = numpy.sqrt(numpy.var(error, ddof=1))
            sigma_u_corr = numpy.corrcoef(expression, error)[0,1]

            sigma_X = sigma_W - sigma_u
            gene_attenuation = sigma_X/(sigma_u + sigma_X)
            attenuations.append((gene, 1./gene_attenuation, sigma_W, sigma_X, sigma_u, sigma_u_var, sigma_u_corr))
        attenuations = pd.DataFrame(attenuations, columns=['Gene', 'Attenuation', 'Sigma W', 'Sigma X', 
                                                       'Sigma U', 'Sigma U Std', 'Sigma U Correlation'])
        return attenuations

    def create_corr_df(self, studies=None, shared_genes=None, samples=None):
        '''
        Calculate correlation across studies for each gene. If shared_genes list is provided,
        only look at those genes. Otherwise, look at genes that have been predicted across all studies.

        Optionally, only look at certain samples.

        '''

        if not studies:
            studies = list(self.predictions.keys())
        if not shared_genes:
            shared_genes = [set(self.predictions[study].index) for study in studies]
            shared_genes = shared_genes[0].intersection(*shared_genes[1:])
        if not samples:
            samples = self.predictions[studies[0]].columns

        print('Calculating correlation for studies: {}'.format(studies))
            
        corrs = []
        genes = []
        for gene in shared_genes:
            try:
                gene_df = pd.concat([self.predictions[study].ix[gene][samples]
                                    for study in studies], axis=1)
            except:
                continue
            gene_df.columns = studies
            #gene_df = gene_df.apply(standardize, axis=0)
            corr = gene_df.corr()
            corr = corr.values[numpy.triu_indices(corr.shape[0], k=1)]
            genes.append(gene)
            corrs.append(corr)
        corr_df = pd.DataFrame(corrs, index=genes, columns=pd.MultiIndex.from_tuples(list(combinations(studies, 2))))
        #corr_df.index = shared_genes
        return corr_df

    def get_avg_corr(self, studies=None, shared_genes=None, samples=None):
        '''
        Calculate average (across genes) pairwise correlation across studies. Looks at genes 
        that are shared across all studies.
        '''
        if not studies:
            studies = self.predictions.keys()

        corr_df = self.create_corr_df(studies, shared_genes, samples)
        corr_df = corr_df.describe().ix['mean']
        sample_combos = combinations(gtex_pred.predictions.keys(), 2)
        corr_df = corr_df.unstack()
        for study in studies:
            corr_df.loc[study, study] = 1
        for study1, study2 in combinations(studies, 2):
            corr_df.loc[study2, study1] = corr_df.loc[study1, study2]
        return corr_df

    def get_pairwise_avg_corr(self, studies=None, shared_genes=None, samples=None):
        '''
        Calculate average (across genes) pairwise correlation across studies. Looks at genes
        that are shared pairwise. 
        '''
        if not studies:
            studies = self.predictions.keys()
        study_combos = list(combinations(studies, 2))
        correlations = list(map(lambda x: str(self.create_corr_df(x, shared_genes, samples).describe().ix['mean']),
                                 study_combos))
        corr_df = pd.DataFrame(list(map(lambda x: float(x.split()[2]), correlations)), 
                               index=pd.MultiIndex.from_tuples(study_combos))
        corr_df = corr_df.unstack()
        corr_df.columns = corr_df.columns.droplevel(0)
        for study in studies:
            corr_df.loc[study, study] = 1
        for study1, study2 in combinations(studies, 2):
            corr_df.loc[study2, study1] = corr_df.loc[study1, study2]
        return corr_df
