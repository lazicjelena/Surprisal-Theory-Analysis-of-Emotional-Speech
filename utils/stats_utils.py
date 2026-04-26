# -*- coding: utf-8 -*-
"""utils.stats_utils

Centralizovani statisticki helperi koje koriste vise foldera projekta.

P-012 (Faza 2-C): cross-folder konsolidacija. Funkcije
``calculate_log_Likelihood`` i ``calculate_aic`` su prethodno postojale
kao byte-identicne kopije u 5 fajlova (additional_analysis/my_functions,
duration_prediction/surprisal_results, information_metrics/my_functions,
linear_regression/stats_utils, split_over_effect/surprisal_results).
Tijela funkcija NISU mijenjana - samo premjestena na jedno centralno mjesto
(zero-change).

Pipeline role
-------------
Project-wide shared utility module imported under the package path
``utils.stats_utils``. Holds the two identical statistical utilities --
:func:`calculate_log_Likelihood` (per-residual normal log-pdf) and
:func:`calculate_aic` (Akaike information criterion plus mean / std
log-likelihood summaries) -- that previously lived as duplicate copies
in regression / surprisal-results plotting scripts. Bodies were not
modified during the P-012 cross-folder consolidation; only their
location.
"""

import numpy as np
from scipy.stats import norm


def calculate_log_Likelihood(data):
    """Per-element normal log-pdf using the empirical mean / std of ``data``.

    Estimates ``mu = mean(data)`` and ``sigma = std(data)`` from
    ``data`` itself and returns the log-density
    ``norm.logpdf(data, loc=mu, scale=sigma)`` evaluated on every
    element. Used by :func:`calculate_aic` on residuals to score
    the Gaussian-noise assumption of the linear regression
    pipeline.

    Parameters
    ----------
    data : array_like of float
        Input vector (typically a residual array). Mean and std
        are estimated in-sample.

    Returns
    -------
    numpy.ndarray
        Per-element log-density values; ``shape == data.shape``.
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    return norm.logpdf(data, loc=mean, scale=std_dev)


def calculate_aic(real_values, results, k):
    """AIC plus mean / std log-likelihood for one model column.

    Computes per-row residuals ``real_values - results``, scores
    them under a Gaussian noise model with
    :func:`calculate_log_Likelihood`, and forms the per-row
    ``aic = 2 * k - 2 * log_likelihood``. Returns the per-row AIC
    array together with the mean and std of the log-likelihoods,
    which the calling :func:`akaike_for_column` reduces to a
    ``\Delta\log\mathcal{L}`` summary.

    Parameters
    ----------
    real_values : array_like of float
        Observed durations (or any target values).
    results : array_like of float
        Model predictions, same shape as ``real_values``.
    k : int
        Number of effective parameters of the model (used only
        in the AIC offset; the mean / std log-likelihoods are
        independent of ``k``).

    Returns
    -------
    tuple of (numpy.ndarray, float, float)
        ``(aic_per_row, mean_log_likelihood, std_log_likelihood)``.
    """
    residuals = np.array(real_values) - np.array(results)
    log_likelihood = calculate_log_Likelihood(residuals)
    aic = 2 * k - 2 * log_likelihood
    return aic, np.mean(log_likelihood), np.std(log_likelihood)
