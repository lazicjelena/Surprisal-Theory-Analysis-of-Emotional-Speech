# -*- coding: utf-8 -*-
"""text_similarity_utils.py

Folder-local helpers for orthographic similarity scoring.

P-013-NB (Faza 4-A): konsolidacija identicnih helper funkcija unutar
foldera ``information_metrics/parameter_estimations/``. Funkcije su
prethodno postojale kao byte-identicne kopije u 2 fajla
(``adjusted_surprisals.py``, ``information_value.py``). Tijela funkcija
NISU mijenjana - samo premjestena na jedno centralno mjesto unutar
istog foldera (zero-change).

Napomena: ``get_pos_for_word_at_index`` i ``pos_tags_similarity``
NISU premjestene ovdje jer koriste modulni-level globalnu varijablu
``nlp`` (CLASSLA pipeline) koja se inicijalizuje u pozivajucem
skriptu. Konsolidacija bi zahtijevala dodavanje ``nlp`` kao
eksplicitnog argumenta sto bi promijenilo signature funkcija (krsi
zero-change pravilo). Ostaju kao lokalne kopije u oba fajla.

Pipeline role
-------------
Folder-local helper module imported by ``adjusted_surprisals.py`` and
``information_value.py`` inside
``information_metrics/parameter_estimations/``. Hosts three small
orthographic-similarity utilities used to score similarity between
candidate vocabulary words and a target word for the
information-value / adjusted-surprisal estimators:

- :func:`levenshtein_distance` — classic edit distance.
- :func:`orthographic_similarity` — normalized 1 - LD/maxlen score.
- :func:`sequence_matcher` — character-bigram set overlap (Dice).

Bodies were not modified during the P-013-NB cross-file consolidation;
only their location.
"""


def levenshtein_distance(str1, str2):
    """Classic Levenshtein edit distance between two strings.

    Uses the standard dynamic-programming formulation: the
    distance equals the minimum number of single-character
    insertions, deletions or substitutions required to transform
    ``str1`` into ``str2``.

    Parameters
    ----------
    str1 : str
        First input string.
    str2 : str
        Second input string.

    Returns
    -------
    int
        Edit distance ``>= 0``.
    """
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def orthographic_similarity(word1, word2):
    """Length-normalized orthographic similarity in ``[0, 1]``.

    Computed as ``1 - levenshtein(word1, word2) / max(|word1|, |word2|)``;
    identical strings score ``1`` and completely different
    strings of the same length score ``0``.

    Parameters
    ----------
    word1 : str
        First word.
    word2 : str
        Second word.

    Returns
    -------
    float
        Similarity in ``[0, 1]``.
    """
    distance = levenshtein_distance(word1, word2)
    max_len = max(len(word1), len(word2))
    similarity = 1 - distance / max_len
    return similarity


def sequence_matcher(word1, word2):
    """Character-bigram set-overlap similarity in ``[0, 1]``.

    Builds the set of consecutive-character bigrams for each
    word and returns
    ``2 * |bigrams1 ∩ bigrams2| / (|bigrams1| + |bigrams2|)``
    (Dice coefficient over bigrams). Used as a complementary
    orthographic feature alongside :func:`orthographic_similarity`.

    Parameters
    ----------
    word1 : str
        First word.
    word2 : str
        Second word.

    Returns
    -------
    float
        Bigram overlap in ``[0, 1]``.
    """
    bigrams1 = set([word1[i:i+2] for i in range(len(word1) - 1)])
    bigrams2 = set([word2[i:i+2] for i in range(len(word2) - 1)])
    intersection = bigrams1.intersection(bigrams2)
    return 2 * len(intersection) / (len(bigrams1) + len(bigrams2))
