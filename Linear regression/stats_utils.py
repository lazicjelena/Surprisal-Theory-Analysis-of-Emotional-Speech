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
"""

import numpy as np
from scipy.stats import norm


def calculate_log_Likelihood(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    return norm.logpdf(data, loc=mean, scale=std_dev)


def calculate_aic(real_values, results, k):

    residuals = np.array(real_values) - np.array(results)
    log_likelihood = calculate_log_Likelihood(residuals)
    aic = 2 * k - 2 * log_likelihood
    return aic, np.mean(log_likelihood), np.std(log_likelihood)
