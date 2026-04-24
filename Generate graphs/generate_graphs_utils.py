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

    # Finding the maximum length of sublists
    max_length = max(len(sublist) for sublist in f0_all_files)
    # Pad each sublist individually
    padded_list = []
    for sublist in f0_all_files:
        padding = [np.nan] * (max_length - len(sublist))
        padded_sublist = np.concatenate((sublist, padding))
        padded_list.append(padded_sublist)

    return padded_list
