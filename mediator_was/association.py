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

def reliability(expression, sigma_u):
    """Return the reliability matrix given observed values and estimated errors."""
    sigma_x = numpy.var(expression) - numpy.mean(sigma_u)
    return sigma_x / (sigma_x + sigma_u)

def _regression_calibration(model, expression, phenotype, sigma_u=None):
    """Compute regression calibration estimates for model given expression,
    phenotype, and estimated errors.

    If sigma_u is None, perform a naive analysis.

    If sigma_u is a scalar, assume homoskedastic errors and rescale the naive
    estimate of the regression coefficient by the reliability ratio.

    If sigma_u is a vector, assume heteroskedastic errors regress phenotype
    against imputed (true) expression values.

    """
    if not isinstance(sigma_u, numpy.ndarray):
        fit = model(phenotype, expression)
        assert(fit.params.shape[0] > 1)
        if sigma_u is None:
            # Naive regression
            pass
        elif numpy.isscalar(sigma_u):
            # Homoskedastic errors
            fit.params[1] /= reliability(expression, sigma_u)
        else:
            raise ValueError("Expecting none or scalar")
    else:
        # Heteroskedastic errors
        mu_x = numpy.mean(expression)
        lambda_1 = reliability(expression, sigma_u)
        lambda_0 = mu_x - lambda_1 * mu_x
        imputed_expression = lambda_0 + lambda_1 * expression
        fit = model(phenotype, imputed_expression)
    return fit

def t(expression, phenotype, sigma_u=None):
    """Test for association of continuous phenotype to expression."""
    fit = _regression_calibration(_ols, expression, phenotype, sigma_u)
    if numpy.isscalar(sigma_u):
        return _t(fit.params[1])
    else:
        return fit.pvalues[0]

def lr(expression, phenotype, sigma_u=None):
    """Test for association between binary phenotype and expression."""
    fit = _regression_calibration(_logit, expression, phenotype, sigma_u)
    if numpy.isscalar(sigma_u):
        llnull = fit.llnull
        llalt = model.loglike(fit.params)
        return _chi2(-2 * (llnull - llalt))
    else:
        return fit.llr_pvalue
