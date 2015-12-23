"""Compute TWAS p-values, corrected for prediction error

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import math
import sys

import numpy
import numpy.linalg
import scipy.stats
import statsmodels.api
import statsmodels.tools

# Freeze this for efficiency
_chi2 = scipy.stats.chi2(1).sf

def _regression_calibration(model, expression, phenotype, sigma_u=None):
    """Compute regression calibration estimates for model given expression,
    phenotype, and estimated errors.

    If sigma_u is None, perform a naive analysis.

    If sigma_u is a scalar, assume homoskedastic errors and rescale the naive
    estimate of the regression coefficient by the reliability ratio.

    If sigma_u is a vector, assume heteroskedastic errors and regress phenotype
    against imputed (true) expression values.

    """
    design = statsmodels.tools.add_constant(expression)
    if sigma_u is None:
        f = model(phenotype, design).fit()
        assert f.pvalues.shape == (2,)
        return f.bse[1], f.pvalues[1]
    else:
        n = phenotype.shape[0]
        error_cov = numpy.array([[0, 0], [0, numpy.mean(sigma_u)]])
        # TODO: this could be "negative". See Buonaccorsi p. 121
        expression_cov = design.T.dot(design) / n - error_cov
        expr_phen_cov = design.T.dot(phenotype) / n
        # TODO: handle heteroskedastic error
        reliability = numpy.linalg.inv(expression_cov + error_cov).dot(expression_cov)
        mu_x = numpy.tile(numpy.mean(design, axis=0), reps=(n, 1))
        imputed_expression = mu_x - (design - mu_x).dot(reliability)
        fit = model(phenotype, imputed_expression).fit()
        pseudo_residuals = phenotype - numpy.dot(design, fit.params)
        delta = pseudo_residuals[:,numpy.newaxis] * design + error_cov.dot(fit.params)
        H = delta.T.dot(delta) / (expression.shape[0] * (expression.shape[0] - 2))
        M = numpy.linalg.inv(expression_cov)
        coeff_cov = M.dot(H).dot(M)
        return numpy.sqrt(coeff_cov[1, 1]), _chi2(math.pow(fit.params[1] / coeff_cov[1, 1], 2))

def t(expression, phenotype, sigma_u=None):
    """Test for association of continuous phenotype to expression."""
    return _regression_calibration(statsmodels.api.OLS, expression, phenotype, sigma_u)

def lr(expression, phenotype, sigma_u=None):
    """Test for association between binary phenotype and expression."""
    return _regression_calibration(statsmodels.api.Logit, expression, phenotype, sigma_u)
