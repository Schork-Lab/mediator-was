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

def _weighted_moment_estimator(expression, phenotype, sigma_u):
    '''
    Weighted moment estimator based on p. 123 Buonaccorsi
    '''

    # Fit regular moment estimator and get residual variance
    n = phenotype.shape[0]
    design = statsmodels.tools.add_constant(expression)
    S_WW = numpy.var(expression, ddof=1)
    constant_sigma_u = numpy.mean(sigma_u)
    Sigma_hat_XX = S_WW - constant_sigma_u
    Sigma_hat_XY = numpy.cov(expression, phenotype, ddof=1)[0, 1]
    
    # Eq 5.9
    coeff = Sigma_hat_XY / Sigma_hat_XX 
    intercept = numpy.mean(phenotype) - coeff*numpy.mean(expression) 
    coeff_column = numpy.array([intercept, coeff])

    # Eq 5.11
    residual_variance = numpy.sum([numpy.power(D_i - (intercept + coeff*W_i), 2)
                                   for D_i, W_i in zip(phenotype, expression)])
    residual_variance /= (n - 2)
    residual_variance -= coeff * coeff * numpy.mean(sigma_u)

    # 5.14 Calculate \sigma_e and weights \pi for each sample.
    sigma_e = numpy.array([residual_variance + coeff*coeff*ui for ui in sigma_u])
    pi = 1./sigma_e
    
    # Calculate coefficients p. 123
    M_hat_XXpi = numpy.sum(pi[i]*design[i,:,numpy.newaxis]*design[i,:] - numpy.array([[0, 0],[0, ui]])
                       for i, ui in enumerate(sigma_u))/n
    M_hat_XYpi = numpy.sum(pi[i]*design[i,:,numpy.newaxis]*phenotype[i]
                       for i in range(n))/n
    M_hat_XXpi_inv = numpy.linalg.inv(M_hat_XXpi)
    coeffs = M_hat_XXpi_inv.dot(M_hat_XYpi)
    coeffs.shape = (2)

    # Calculate coefficients covariance
    error_matrix = numpy.array([numpy.array([[0, 0], [0, ui]]).dot(coeffs) for ui in sigma_u])
    pseudo_residuals = phenotype - numpy.dot(design, coeffs)
    delta = pseudo_residuals[:,numpy.newaxis]*design - error_matrix
    delta = pi[:,numpy.newaxis] * delta
    H_hat_Rpi = delta.T.dot(delta) / (n*(n-design.shape[1]))
    coeff_cov = M_hat_XXpi_inv.dot(H_hat_Rpi).dot(M_hat_XXpi_inv)
    coeff = coeffs[1]
    se = numpy.sqrt(coeff_cov[1,1])
    return coeff, se


def _moment_estimator_buonaccorsi(expression, phenotype, sigma_u, homoscedastic=True, num_replicates=4):
    '''
    Moment estimator based on Chapter 5 Buonaccorsi.
    '''
    n = phenotype.shape[0]

    S_WW = numpy.var(expression, ddof=1)
    constant_sigma_u = numpy.mean(sigma_u)
    Sigma_hat_XX = S_WW - constant_sigma_u
    Sigma_hat_XY = numpy.cov(expression, phenotype, ddof=1)[0, 1]
    
    # Eq 5.9
    coeff = Sigma_hat_XY / Sigma_hat_XX 
    intercept = numpy.mean(phenotype) - coeff*numpy.mean(expression) 
    coeff_column = numpy.array([intercept, coeff])

    # Eq 5.11
    residual_variance = numpy.sum([numpy.power(D_i - (intercept + coeff*W_i), 2)
                                   for D_i, W_i in zip(phenotype, expression)])
    residual_variance /= (n - 2)
    residual_variance -= coeff * coeff * sigma_u

  
    design = statsmodels.tools.add_constant(expression)
    Sigma_hat_u = numpy.array([[0, 0], [0, constant_sigma_u]])
    M_hat_XX = numpy.dot(design.T, design) / n - Sigma_hat_u
    if M_hat_XX[1, 1] < 0:
        return 0, 1
    M_hat_XY = numpy.dot(design.T, phenotype) / n

    # Robust Moment Estimator
    # Eq 5.12
    pseudo_residuals = phenotype - numpy.dot(design, coeff_column)

    if homoscedastic:
        error_matrix = Sigma_hat_u.dot(coeff_column)
    else:
        error_matrix = numpy.array([numpy.array([[0, 0], [0, u]]).dot(coeff_column) for u in sigma_u])
    delta = pseudo_residuals[:,numpy.newaxis]*design - error_matrix
    H = delta.T.dot(delta) / (n*(n-design.shape[1]))
    try:
        M = numpy.linalg.inv(M_hat_XX)
    except:
        return 0, 1 
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
        error_var = 0
    else:
        error_var = numpy.mean(sigma_u)
    n = phenotype.shape[0]

    sample_expression_var = numpy.var(expression, ddof=1)
    phenotype_expression_cov = numpy.cov(expression, phenotype, ddof=1)[0,1]
    
    # Eq 1.2.3
    expression_var =  sample_expression_var - error_var
    coeff = phenotype_expression_cov / expression_var
    phenotype_var = numpy.var(phenotype, ddof=1)

    # Eq 1.2.4
    equation_error_var =  phenotype_var*expression_var - phenotype_expression_cov**2
    if equation_error_var < 0:
        print("Warning: correcting negative estimate", file=sys.stderr)
        expression_var = phenotype_expression_cov ** 2 / phenotype_var
        coeff = phenotype_var / phenotype_expression_cov
        return 0, 1

    phenotype_mean = numpy.mean(phenotype)
    expression_mean = numpy.mean(expression)
    psuedo_residual_var = sum((p - phenotype_mean - (e - expression_mean) * coeff) ** 2 for p, e in zip(phenotype, expression)) / (n - 2)
    coeff_var = (sample_expression_var * psuedo_residual_var + coeff ** 2 * error_var ** 2) / ((n - 1) * expression_var ** 2)
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


def _regression_calibration(model, expression, phenotype, sigma_u, homoscedastic=True):
    """Compute regression calibration estimates for model given expression,
    phenotype, and estimated errors.

    If sigma_u is None, perform a naive analysis.

    If sigma_u is a scalar, assume homoskedastic errors and rescale the naive
    estimate of the regression coefficient by the reliability ratio.

    If sigma_u is a vector, assume heteroskedastic errors and regress phenotype
    against imputed (true) expression values.

    """
    design = statsmodels.tools.add_constant(expression)
    n = phenotype.shape[0]
    if homoscedastic:
        error_cov = numpy.array([[0, 0], [0, numpy.mean(sigma_u)]])
        # TODO: this could be "negative". See Buonaccorsi p. 121
        expression_cov = numpy.cov(design.T, ddof=1) - error_cov
        if expression_cov[1,1] < 0:
            #print("Warning: correcting negative estimate", file=sys.stderr)
            return 0, 1
        # Hack to make reliability make sense
        expression_cov[0,0] = 1
        try:
            reliability = numpy.linalg.inv(expression_cov + error_cov).dot(expression_cov)
        except:
            print(expression_cov, error_cov)
        mu_x = numpy.tile(numpy.mean(design, axis=0), reps=(n, 1))
        imputed_expression = mu_x + (design - mu_x).dot(reliability)
        fit = model(phenotype, imputed_expression).fit()
        pseudo_residuals = phenotype - numpy.dot(design, fit.params)
        delta = pseudo_residuals[:,numpy.newaxis] * design + error_cov.dot(fit.params)
        H = delta.T.dot(delta) / (n * (n - design.shape[1]))
        M = numpy.linalg.inv(expression_cov)
        coeff_cov = M.dot(H).dot(M)
        coeff, se = fit.params[1], numpy.sqrt(coeff_cov[1, 1])
    else:

        # Speed up so fewer computations
        expression_cov = numpy.var(expression, ddof=1) - numpy.mean(sigma_u)
        if expression_cov < 0:
            #print("Warning: correcting negative estimate", file=sys.stderr)
            return 0, 1
        mu_x = numpy.mean(design, axis=0)
        imputed_expression = numpy.array([[1, mu_x[1] + expression_cov/(expression_cov + ui) * wi]
                                          for ui, wi in zip(sigma_u, expression)])
        fit = model(phenotype, imputed_expression).fit()
        pseudo_residuals = phenotype - numpy.dot(design, fit.params)
        error_matrix = numpy.array([numpy.array([[0, 0], [0, ui]]).dot(fit.params) for ui in sigma_u])
        delta = pseudo_residuals[:,numpy.newaxis] * design + error_matrix
        H = delta.T.dot(delta) / (n * (n - design.shape[1]))
        error_cov = numpy.array([[0, 0], [0, numpy.mean(sigma_u)]])
        # TODO: this could be "negative". See Buonaccorsi p. 121
        expression_cov = numpy.cov(design.T, ddof=1) - error_cov
        expression_cov[0, 0] = 1
        M = numpy.linalg.inv(expression_cov)
        coeff_cov = M.dot(H).dot(M)
        coeff, se = fit.params[1], numpy.sqrt(coeff_cov[1, 1])

    return coeff, se

def t(expression, phenotype, sigma_u=None, method="OLS"):
    """Test for association of continuous phenotype to expression."""
    if method == "moment":
        coeff, se = _moment_estimator(expression, phenotype, sigma_u)
    elif method == "moment2":
        coeff, se = _moment_estimator_buonaccorsi(expression, phenotype, sigma_u)
    elif method == "momentbuo":
        coeff, se = _moment_estimator_buonaccorsi(expression, phenotype, sigma_u)
    elif method == "rc":
        coeff, se = _regression_calibration(statsmodels.api.OLS, expression, phenotype, sigma_u)
    elif method == "rc-hetero":
        coeff, se = _regression_calibration(statsmodels.api.OLS, expression, phenotype, sigma_u, homoscedastic=False)
    elif method == "weighted":
        coeff, se = _weighted_moment_estimator(expression, phenotype, sigma_u)
    elif method == "OLS":
        design = statsmodels.tools.add_constant(expression)
        fit = statsmodels.api.OLS(phenotype, design).fit()
        coeff, se = fit.params[1], fit.bse[1]
    elif method == "WLS":
        design = statsmodels.tools.add_constant(expression)
        sigma_u = numpy.sqrt(sigma_u)
        weights = 1./sigma_u
        fit = statsmodels.api.WLS(phenotype, design, weights=weights).fit()
        coeff, se = fit.params[1], fit.bse[1]
    else:
        raise NotImplementedError
    return coeff, se, _chi2(coeff * coeff / (se * se))

def lr(expression, phenotype, sigma_u=None):
    """Test for association between binary phenotype and expression."""
    coeff, se = _regression_calibration(statsmodels.api.Logit, expression, phenotype, sigma_u)
    return se, _chi2(coeff * coeff / (se * se))
