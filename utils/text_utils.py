# -*- coding: utf-8 -*-
"""utils.text_utils

Centralizovani leksicki/tekstualni helperi koje koriste vise foldera projekta.

P-012 (Faza 2-C): cross-folder konsolidacija. Funkcija ``find_subword`` je
prethodno postojala kao byte-identicna kopija u 3 fajla
(additional_analysis/build_prominence_datasets, previous_surprisals/
prominence_build_dataset, prominence/text_utils). Tijelo funkcije NIJE
mijenjano - samo premjesteno na jedno centralno mjesto (zero-change).

Pipeline role
-------------
Project-wide shared lexical helper module imported under the package path
``utils.text_utils``. Hosts :func:`find_subword`, the longest-suffix match
used by ``librosa_estimated_parameters.py``,
``prominence_build_dataset.py`` (in both the ``prominence/`` and
``previous_surprisals/`` chains) and
``additional_analysis/build_prominence_datasets.py`` to re-split conjoint
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
