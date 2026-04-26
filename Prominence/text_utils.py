# -*- coding: utf-8 -*-
"""text_utils.py
Lexical/tekst pomocne funkcije izdvojene iz:
  - Prominence/librosa_estimated_parameters.py
  - Prominence/prominence_build_dataset.py

P-008 (Faza 2-B): zajednicke IDENTICNO funkcije unutar foldera
'Prominence/'. Tijelo funkcija NIJE mijenjano.

Napomena: find_subword postoji i u:
  - Additional files after recension/build_prominence_datasets.py
  - Pervious Surprisals/prominence_build_dataset.py
Cross-folder konsolidacija nije dio P-008 - ostaje za P-009.

Pipeline role
-------------
Shared lexical helper module for the ``Prominence/`` chain.
Hosts :func:`find_subword`, the longest-suffix match used by both
``librosa_estimated_parameters.py`` and
``prominence_build_dataset.py`` to re-split conjoint
wavelet-GUI tokens against the canonical word list of
``../podaci/target_sentences.csv``.
"""


def find_subword(word, unique_words):
    """Return the longest suffix of ``word`` that lies in ``unique_words``.

    Greedy longest-suffix match. The loop iterates over every
    possible suffix length; the last matching suffix wins, so the
    returned value is the longest suffix of ``word`` that is also
    a member of ``unique_words``. If no suffix matches, the empty
    string is returned.

    Parameters
    ----------
    word : str
        Conjoint token produced by the wavelet GUI alignment.
    unique_words : set of str
        Canonical (lower-cased) vocabulary derived from
        ``target_sentences.csv``.

    Returns
    -------
    str
        The longest matching suffix, or ``""`` when none matches.
    """
    subword = ''
    for i in range(1,len(word)+1):
        if word[-i:] in unique_words:
            subword = word[-i:]

    return subword
