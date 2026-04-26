# -*- coding: utf-8 -*-
"""generate_graphs_utils.py
Zajednicke pomocne funkcije koje su ranije bile duplicirane kroz skripte
u folderu 'Generate graphs/'.

P-007 (Faza 2-A refaktorisanja):
  padding_sequence je izdvojen iz:
    - frequency_over_time.py
    - frequency_over_time_plots.py
    - rms_over_time.py
    - rms_over_time_plots.py
  Tijelo funkcije NIJE mijenjano - samo premjesteno.
"""

import numpy as np


def padding_sequence(f0_all_files):
    """Pad ragged per-recording sequences to a common length with NaNs.

    Computes the maximum length over all sublists in ``f0_all_files``
    and right-pads every shorter sublist with ``np.nan`` so that the
    returned list of arrays is rectangular and can be passed to
    :func:`numpy.nanmean` / :func:`numpy.nanstd` along the time axis.
    The original values are preserved as-is; only trailing ``NaN``
    samples are appended.

    Parameters
    ----------
    f0_all_files : list of array-like
        One sequence per recording. Can mix lists, ``numpy`` arrays
        and nested lists; each inner sequence must support
        :func:`len` and :func:`numpy.concatenate`.

    Returns
    -------
    list of numpy.ndarray
        One padded array per input sublist, all of equal length
        ``max(len(s) for s in f0_all_files)``. Entries beyond the
        original length of each sublist are ``np.nan``.
    """
    # Finding the maximum length of sublists
    max_length = max(len(sublist) for sublist in f0_all_files)
    # Pad each sublist individually
    padded_list = []
    for sublist in f0_all_files:
        padding = [np.nan] * (max_length - len(sublist))
        padded_sublist = np.concatenate((sublist, padding))
        padded_list.append(padded_sublist)

    return padded_list
