"""
Statistical Tests and Information Criteria Calculations

This module provides functions to perform the following statistical tests:
- Kolmogorov-Smirnov (K-S) Test
- Anderson-Darling Test
- Chi-Square Goodness-of-Fit Test
- Calculation of Akaike Information Criterion (AIC)
- Calculation of Bayesian Information Criterion (BIC)
"""

import numpy as np
from scipy import stats

# -----------------------------------
# Kolmogorov-Smirnov Test Function
# -----------------------------------

def ks_test(data, dist_name, params):
    """
    Perform the Kolmogorov-Smirnov test.

    Parameters:
    - data: array-like, the sample data.
    - dist_name: str, name of the distribution in scipy.stats.
    - params: tuple, parameters of the distribution.

    Returns:
    - ks_statistic: float, the K-S test statistic.
    - ks_p_value: float, the p-value of the test.
    """
    ks_statistic, ks_p_value = stats.kstest(data, dist_name, args=params)
    return ks_statistic, ks_p_value

# -----------------------------------
# Anderson-Darling Test Function
# -----------------------------------

def ad_test(data, dist_name):
    """
    Perform the Anderson-Darling test.

    Parameters:
    - data: array-like, the sample data.
    - dist_name: str, name of the distribution in scipy.stats.

    Returns:
    - ad_statistic: float, the A-D test statistic.
    - ad_critical_values: array-like, critical values for significance levels.
    - ad_significance_levels: array-like, significance levels corresponding to the critical values.
    """
    ad_result = stats.anderson(data, dist=dist_name)
    ad_statistic = ad_result.statistic
    ad_critical_values = ad_result.critical_values
    ad_significance_levels = ad_result.significance_level
    return ad_statistic, ad_critical_values, ad_significance_levels

# -----------------------------------
# Chi-Square Goodness-of-Fit Test Function
# -----------------------------------

def chi_square_test(data, dist_name, params, num_bins):
    """
    Perform the Chi-Square Goodness-of-Fit test.

    Parameters:
    - data: array-like, the sample data.
    - dist_name: str, name of the distribution in scipy.stats.
    - params: tuple, parameters of the distribution.
    - num_bins: int, number of bins to use in the histogram.

    Returns:
    - chi_statistic: float, the Chi-Square test statistic.
    - chi_p_value: float, the p-value of the test.
    """
    # Create histogram of observed data
    observed_freq, bin_edges = np.histogram(data, bins=num_bins)
    # Calculate expected frequencies based on the fitted distribution
    cdf_vals = getattr(stats, dist_name).cdf(bin_edges, *params)
    expected_freq = len(data) * np.diff(cdf_vals)
    # Adjust expected frequencies to avoid zeros
    expected_freq[expected_freq == 0] = 1e-6
    # Chi-Square Test
    chi_statistic, chi_p_value = stats.chisquare(f_obs=observed_freq, f_exp=expected_freq)
    return chi_statistic, chi_p_value

# -----------------------------------
# Information Criteria Calculation Functions
# -----------------------------------

def calculate_aic(log_likelihood, num_params):
    """
    Calculate Akaike Information Criterion (AIC).

    Parameters:
    - log_likelihood: float, the log-likelihood of the model.
    - num_params: int, number of parameters in the model.

    Returns:
    - aic: float, the AIC value.
    """
    aic = 2 * num_params - 2 * log_likelihood
    return aic

def calculate_bic(log_likelihood, num_params, num_samples):
    """
    Calculate Bayesian Information Criterion (BIC).

    Parameters:
    - log_likelihood: float, the log-likelihood of the model.
    - num_params: int, number of parameters in the model.
    - num_samples: int, number of observations.

    Returns:
    - bic: float, the BIC value.
    """
    bic = num_params * np.log(num_samples) - 2 * log_likelihood
    return bic
