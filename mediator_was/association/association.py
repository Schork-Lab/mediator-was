"""Compute TWAS p-values, corrected for prediction error

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import sys

import numpy
import scipy.stats
import statsmodels.api
import statsmodels.tools

# Freeze this for efficiency
_sf = scipy.stats.chi2(1).sf

def lrt(expression, phenotype, sigma_u):
    """Return p-value of likelihood ratio test corrected by the reliability ratio.

    expression - n x 1 array of predicted expression
    phenotype - n x 1 array of phenotypes
    sigma_u - scalar estimate of prediction error

    The naive slope estimate is off by a multiplicative factor (attenuation
    bias), so rescale it in the alternative model.

    The bias is sigma_x / (sigma_x + sigma_u), and the correction is its
    inverse.

    After correction, perform a nested model comparison dropping one variable
    and compute tail probabilities from the chi-square distribution with one
    degree of freedom.

    """
    design = statsmodels.tools.add_constant(expression)
    model = statsmodels.api.Logit(phenotype, design)
    fit = model.fit()
    llnull = fit.llnull
    sigma_x = numpy.var(expression)
    fit.params[1] *= (sigma_x + sigma_u) / sigma_x
    llalt = model.loglike(fit.params)
    return _sf(-2 * (llnull - llalt))
