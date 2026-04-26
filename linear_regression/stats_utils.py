# -*- coding: utf-8 -*-
"""stats_utils.py
Pomocne statisticke funkcije izdvojene iz:
  - Linear regression/final_graphs.py
  - Linear regression/results.py

P-008 (Faza 2-B): zajednicke IDENTICNO funkcije se centralizuju unutar
foldera 'Linear regression/'. Tijelo funkcija NIJE mijenjano - samo
premjesteno (zero-change).

Napomena: calculate_log_Likelihood, calculate_aic postoje i u drugim
folderima (Additional files after recension, Different information
measurement parameters, Duration Prediction based on Surprisals,
Split-over effect). Cross-folder konsolidacija nije dio P-008 -
ostaje za P-009.

Pipeline role
-------------
Folder-local helper module imported by ``final_graphs.py`` and
``results.py`` inside ``Linear regression/``. Holds the two
identical statistical utilities -- :func:`calculate_log_Likelihood`
(per-residual normal log-pdf) and :func:`calculate_aic`
(Akaike information criterion plus mean / std log-likelihood
summaries) -- that previously lived as duplicate copies in both
plotting scripts. Bodies were not modified during the P-008
consolidation; only their location.
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
