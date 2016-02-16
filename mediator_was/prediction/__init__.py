"""
Analysis of our predictions before incorporating them into  association methods
"""

import pandas as pd
import mediator_was.prediction.plotting
from mediator_was.prediction.predict import stream

class Predictions():
    def __init__(self, vcf, weights={},
                 predicted_files=None):

        # Calculate or load predicted values
        self.predictions = {}
        if predicted_files:
            for method, fn in predicted_files.iteritems():
                self.predictions[method] = self.load(fn)
        else:
            error = "Calculating Predictions requires vcf and weights"
            assert vcf is not None and weights is not None, error
            self.vcf = vcf
            self.weights = weights
            self.predict()

    def predict(self):
        for method, weights in self.weights.items():
            predictions_df = stream(weights, self.vcf, 'vcf')
            self.predictions[method] = predictions_df

    def load(fn):
        if fn.endswith('.gz'):
            compression = 'gzip'
        else:
            compression = None
        predicted_df = pd.read_table(fn,
                                     index_col=0,
                                     compression=compression,
                                     sep='\t')
        return predicted_df
