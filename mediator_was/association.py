"""Compute TWAS p-values, corrected for prediction error

Author: Abhishek Sarkar <aksarkar@mit.edu>
        Kunal Bhutani   <kunalbhutani@gmail.com>

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

def standardize(values, center=True, unit_variance=True):
    if center:
        values = values - numpy.mean(values)
    if unit_variance:
        values = values / numpy.std(values)
    return values

def gwas(genotypes, phenotype):
    """Compute the best GWAS p-value in the locus"""
    if len(genotypes.shape) == 1:
        return t(genotypes, phenotype, method="OLS")[2]
    else:
        return min(t(g, phenotype, method="OLS")[2] for g in genotypes.T)

def _moment_estimator_buonaccorsi(expression, phenotype, sigma_u, num_replicates=4):
    '''
    Moment estimator based on Chapter 5 Buonaccorsi.
    '''
    n = phenotype.shape[0]

    S_WW = numpy.var(expression, ddof=1)
    sigma_u = numpy.mean(sigma_u)
    Sigma_hat_XX = S_WW - sigma_u
    Sigma_hat_XY = numpy.cov(expression, phenotype, ddof=1)[0, 1]
    
    # Eq 5.9
    coeff = Sigma_hat_XY / Sigma_hat_XX 
    intercept = numpy.mean(phenotype) - coeff*numpy.mean(expression) 
    coeff_column = numpy.array([intercept, coeff])
    coeff_column.shape = (2, 1)

    # Eq 5.11
    residual_variance = numpy.sum([numpy.power(D_i - (intercept + coeff*W_i), 2)
                                   for D_i, W_i in zip(phenotype, expression)])
    residual_variance /= (n - 1)
    residual_variance -= coeff * coeff * sigma_u

  
    design = statsmodels.tools.add_constant(expression)
    Sigma_hat_u = numpy.array([[0, 0], [0, sigma_u]])
    M_hat_XX = numpy.dot(design.T, design) / n - Sigma_hat_u
    M_hat_XY = numpy.dot(design.T, phenotype) / n

    # Robust Moment Estimator
    # Eq 5.12
    pseudo_residuals = phenotype - numpy.dot(design, coeff_column).T
    # Hacky, because numpy shape matching is weird.
    # Unclear best way to do matrix subtraction using numpy a
    constant_error = numpy.array(Sigma_hat_u.dot(coeff_column))
    constant_error.shape = (2, 1)
    constant_error = numpy.tile(constant_error, reps=n).T

    delta = pseudo_residuals.T*design - constant_error
    H = delta.T.dot(delta) / (n*n)
    M = numpy.linalg.inv(M_hat_XX)
    coeff_cov = M.dot(H).dot(M)
    se = numpy.sqrt(coeff_cov[1, 1])


    # Normal based with constant, estimated measurement error parameters.
    # Eq. 5.14
    # individual_residual_variance = numpy.tile(residual_variance + coeff*coeff*sigma_u, reps=n)
    # if individual_residual_variance.any() < 0:
    #     individual_residual_variance = 0

    # coeff_column = numpy.array([intercept, coeff])
    # coeff_column.shape = (2, 1)
    # Z = numpy.tile(-1*numpy.dot(Sigma_hat_u, coeff_column), reps=n)
    # Z = Z.T
    # H_hat_N = numpy.sum([numpy.dot(design[i,:], design[i, :].T)*individual_residual_variance + numpy.dot(Z[i, :], Z[i,:].T)
    #                      for i in range(n)])
    # H_hat_N /= n*n
    # MSE_c = numpy.sum([numpy.power(D_i - (intercept + coeff*W_i), 2) 
    #                   for D_i, W_i in zip(phenotype, expression)]) / (n - 1)
    # H_hat_NC = H_hat_N
    # H_hat_NC += (Sigma_hat_u*(residual_variance - MSE_c) + Z.T.dot(Z))/(n * n * (num_replicates-1))
    # coeff_cov = numpy.linalg.inv(M_hat_XX).dot(H_hat_NC).dot(M_hat_XX)
    # se = numpy.sqrt(coeff_cov[1, 1])

    return coeff, se

def _moment_estimator_2(expression, phenotype, sigma_u=None):
    '''Compute robust moment estimator based on Fuller pp. 166-170. 
    It is less impacted by high error expression variance.
    '''
    if sigma_u is None:
        expression_error_var = 0
    else:
        expression_error_var = numpy.mean(sigma_u)
   
    n = phenotype.shape[0]
    expression = expression - numpy.mean(expression)
    phenotype = phenotype - numpy.mean(phenotype)
    expression_var = numpy.var(expression, ddof=1)
    phenotype_var = numpy.var(phenotype, ddof=1)
    phenotype_expression_covar = sum((p * e) for p, e in zip(phenotype, expression)) / (n-1)
    lambda_hat = (expression_var - phenotype_expression_covar**2/phenotype_var)
    lambda_hat /= expression_error_var
    alpha = 4-2*((expression_var - expression_error_var)/expression_var)
    #alpha = 2
    if lambda_hat >= (1 + 1.0/(n+1)):
        H_hat_xx = expression_var - expression_error_var
    else:
        H_hat_xx = expression_var - (lambda_hat - 1/(n+1))*expression_error_var
    coeff = 1/(H_hat_xx + alpha*expression_error_var/(n+1))
    coeff *= phenotype_expression_covar
    sigma_vv = phenotype_var - 2*coeff*phenotype_expression_covar + coeff*coeff*expression_var
    sigma_vv *= (n+1)/(n+2)
    coeff_var = sigma_vv/H_hat_xx
    coeff_var += (H_hat_xx**-2)*(expression_error_var*sigma_vv + coeff**2*expression_error_var**2)    
    coeff_var /= n+1
    se = numpy.sqrt(coeff_var)
    return coeff, se

def _moment_estimator(expression, phenotype, sigma_u=None,
                      model_check=False):
    """Compute method of moments estimator for model. Optionally
    return true expression and residuals if interested in
    model checking.

    The algorithm is due to Fuller 1987, pp. 13-15


    """
    if sigma_u is None:
        expression_error_var = 0
    else:
        expression_error_var = numpy.mean(sigma_u)
    n = phenotype.shape[0]
    phenotype_mean = numpy.mean(phenotype)
    expression_mean = numpy.mean(expression)
    phenotype_expression_cov = sum([(p - phenotype_mean) * (e - expression_mean) for p, e in zip(phenotype, expression)]) / n
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
    if not model_check:
        return coeff, se
    else:
        intercept = numpy.mean(phenotype) - coeff*numpy.mean(expression)
        true_expression_ratio = (1./(coeff*coeff*expression_error_var + equation_error_var))
        true_expression = [true_expression_ratio*(expression_error_var*(p - intercept)*coeff + equation_error_var*e)
                            for p, e in zip(phenotype, expression)]
        residuals = [p - intercept - e*coeff
                    for p, e in zip(phenotype, expression)]
        return coeff, se, true_expression, residuals


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
        coeff, se = fit.params[1], numpy.sqrt(coeff_cov[1, 1])
        return coeff, se

def t(expression, phenotype, sigma_u=None, method="moment"):
    """Test for association of continuous phenotype to expression."""
    if method == "moment":
        coeff, se = _moment_estimator(expression, phenotype, sigma_u)
    elif method == "moment2":
        coeff, se = _moment_estimator_2(expression, phenotype, sigma_u)
    elif method == "rc":
        coeff, se = _regression_calibration(statsmodels.api.OLS, expression, phenotype, sigma_u)
    elif method == "OLS":
        design = statsmodels.tools.add_constant(expression)
        fit = statsmodels.api.OLS(phenotype, design).fit()
        coeff, se = fit.params[1], fit.bse[1]
    else:
        raise NotImplementedError
    return coeff, se, _chi2(coeff * coeff / (se * se))

def lr(expression, phenotype, sigma_u=None):
    """Test for association between binary phenotype and expression."""
    coeff, se = _regression_calibration(statsmodels.api.Logit, expression, phenotype, sigma_u)
    return se, _chi2(coeff * coeff / (se * se))
