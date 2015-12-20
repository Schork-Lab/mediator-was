"""Compute TWAS p-values, corrected for prediction error

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import sys

import numpy
import scipy.stats
import statsmodels.api
import statsmodels.tools

# Freeze this for efficiency
_chi2 = scipy.stats.chi2(1).sf
_t = scipy.stats.t(2).sf

def _wrap(model):
    """Wrapper function around statsmodels API fit()"""
    def fit(phenotype, expression):
        assert phenotype.shape == expression.shape
        design = statsmodels.tools.add_constant(expression)
        return model(phenotype, design).fit()
    return fit

_ols = _wrap(statsmodels.api.OLS)
_logit = _wrap(statsmodels.api.Logit)

def reliability_ratio(expression, sigma_u):
    """Return the reliability ratio given observed values and estimated error variance"""
    sigma_x = numpy.var(expression) - sigma_u
    return sigma_x / (sigma_x + sigma_u)

def t(expression, phenotype, sigma_u=None):
    """Test for association of continuous phenotype to expression.

    If sigma_u is None, perform a naive analysis.

    If sigma_u is a scalar, rescale the naive estimate of the regression
    coefficient by the reliability ratio.

    If sigma_u is a vector, regress phenotype against imputed (true) expression
    values.

    """
    fit = _ols(phenotype, expression)
    assert(fit.params.shape[0] > 1)
    if sigma_u is None:
        return fit.pvalues[0]
    elif numpy.isscalar(sigma_u):
        return _t(fit.params[1] / reliability_ratio(expression, sigma_u))
    else:
        raise NotImplementedError

def lr(expression, phenotype, sigma_u=None):
    """Test for association between binary phenotype and expression.

    If sigma_u is None, perform a naive analysis.

    If sigma_u is a scalar, rescale the naive estimate of the regression
    coefficient by the reliability ratio.

    If sigma_u is a vector, regress phenotype against imputed (true) expression
    values.

    """
    fit = _logit(phenotype, expression)
    if sigma_u is None:
        return fit.llr_pvalue
    elif numpy.isscalar(sigma_u):
        llnull = fit.llnull
        fit.params[1] /= reliability_ratio(expression, sigma_u)
        llalt = model.loglike(fit.params)
        return _chi2(-2 * (llnull - llalt))
    else:
        raise NotImplementedError
