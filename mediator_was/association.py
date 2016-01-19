"""Compute TWAS p-values, corrected for prediction error

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
from __future__ import print_function
import math
import pdb
import sys

import numpy
import numpy.linalg
import scipy.stats
import statsmodels.api
import statsmodels.tools

# Freeze this for efficiency
_chi2 = scipy.stats.chi2(1).sf

def gwas(genotypes, phenotype):
    """Compute the best GWAS p-value in the locus"""
    if len(genotypes.shape) == 1:
        return t(genotypes, phenotype)[1]
    else:
        return min(t(g, phenotype)[1] for g in genotypes.T)

def _moment_estimator(expression, phenotype, sigma_u=None):
    """Compute method of moments estimator for model.

    The algorithm is due to Fuller 1987, pp. 13-15

    """
    if sigma_u is None:
        expression_error_var = 0
    else:
        expression_error_var = numpy.mean(sigma_u)
    n = phenotype.shape[0]
    phenotype_mean = numpy.mean(phenotype)
    expression_mean = numpy.mean(expression)
    phenotype_expression_cov = sum((p - phenotype_mean) * (e - expression_mean) for p, e in zip(phenotype, expression)) / n
    expression_var = numpy.var(expression) - expression_error_var
    coeff = phenotype_expression_cov / expression_var
    phenotype_var = numpy.var(phenotype)
    equation_error_var =  phenotype_var - coeff * coeff * expression_var
    if equation_error_var < 0:
        print("Warning: correcting negative estimate", file=sys.stderr)
        expression_var = phenotype_expression_cov / phenotype_var
        coeff = phenotype_expression_cov / expression_var
    psuedo_residual_var = sum((p - phenotype_mean - (e - expression_mean) * coeff) ** 2 for p, e in zip(phenotype, expression)) / (n - 2)
    coeff_var = (expression_var * psuedo_residual_var + coeff ** 2 * expression_error_var) / (n - 1)
    se = math.sqrt(coeff_var)
    return coeff, se

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
        imputed_expression = mu_x + (design - mu_x).dot(reliability)
        fit = model(phenotype, imputed_expression).fit()
        pseudo_residuals = phenotype - numpy.dot(design, fit.params)
        delta = pseudo_residuals[:,numpy.newaxis] * design + error_cov.dot(fit.params)
        H = delta.T.dot(delta) / (expression.shape[0] * (expression.shape[0] - 2))
        M = numpy.linalg.inv(expression_cov)
        coeff_cov = M.dot(H).dot(M)
        coeff, se = fit.params[1], coeff_cov[1, 1]
        return coeff, se
        
def t(expression, phenotype, sigma_u=None):
    """Test for association of continuous phenotype to expression."""
    coeff, se =  _moment_estimator(expression, phenotype, sigma_u)
    return se, _chi2(coeff * coeff / (se * se))

def lr(expression, phenotype, sigma_u=None):
    """Test for association between binary phenotype and expression."""
    coeff, se = _regression_calibration(statsmodels.api.Logit, expression, phenotype, sigma_u)
    return se, _chi2(coeff * coeff / (se * se))