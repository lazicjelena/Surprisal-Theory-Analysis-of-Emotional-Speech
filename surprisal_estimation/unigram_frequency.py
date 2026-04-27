# -*- coding: utf-8 -*-
"""unigram_frequency.py

Unigram frequency / log-probability estimation against a Serbian
wordlist. Converted from ``Unigram frequency.ipynb`` (P-013-NB).

Pipeline role
-------------
Frequency-based baseline for the ``surprisal_estimation/`` chain.
Lemmatizes every token of every sentence in
``../podaci/target_sentences.csv`` with the CLASSLA Serbian pipeline,
looks up the lemma's frequency in the CLASSLA Wikipedia wordlist
``../podaci/wordlist_classlawiki_sr_20240122114347.csv``, applies
Laplace smoothing for unseen lemmas, computes the unigram
log-probability ``log(freq / total_freq)`` and writes the resulting
``(Sentence, Word, Lemma, Word Frequency, Log Probability)`` table to
``../podaci/wordlist_frequencies.csv``. Acts as the unigram baseline
that LLM-based surprisals (BERT, BERTic, GPT-2, GPT-3, YugoGPT) are
compared against in the downstream regression analyses.

Notes
-----
- Requires the ``classla`` package; the Serbian model is downloaded
  on first use.
"""

import math
import os

import classla
import pandas as pd

# Paths (relative to project root).
WORDLIST_CSV_PATH = os.path.join('..', 'podaci', 'wordlist_classlawiki_sr_20240122114347.csv')
TARGET_SENTENCES_PATH = os.path.join('..', 'podaci', 'target_sentences.csv')
OUTPUT_CSV_PATH = os.path.join('..', 'podaci', 'wordlist_frequencies.csv')

# Laplace smoothing constant for unseen lemmas.
ALPHA = 1


def lemmatize_word(word, nlp):
    """Return the canonical lemma for ``word`` via the CLASSLA pipeline.

    Runs ``word`` through the supplied CLASSLA pipeline (which
    must include the ``lemma`` processor) and extracts the lemma
    from the CoNLL-U output. Used by :func:`main` to normalize
    each surface form before looking it up in the unigram
    frequency wordlist.

    Parameters
    ----------
    word : str
        Surface form of a single word.
    nlp : classla.Pipeline
        CLASSLA pipeline configured with at least
        ``processors='tokenize, lemma, pos'``.

    Returns
    -------
    str
        The canonical lemma extracted from the CoNLL-U output.
    """
    doc = nlp(word)     # run the pipeline
    lemma = doc.to_conll().split()[-8]
    return lemma


def lookup_frequency(lemma, freq_df, alpha=ALPHA):
    """Look up ``lemma`` in ``freq_df`` with Laplace smoothing.

    Returns the matching ``freq`` value if ``lemma`` appears in
    ``freq_df['word']``; otherwise returns the smoothing constant
    ``alpha`` and reports the OOV via the second return value, so
    that the caller can keep the running ``num_new_words``
    counter used in the smoothed total.

    Parameters
    ----------
    lemma : str
        Canonical lemma to look up.
    freq_df : pandas.DataFrame
        Wordlist with at least ``word`` and ``freq`` columns.
    alpha : int, optional
        Smoothing constant for OOV lemmas. Defaults to
        :data:`ALPHA`.

    Returns
    -------
    frequency : int
        The stored frequency, or ``alpha`` if not present.
    is_new : bool
        ``True`` iff the lemma was not found.
    """
    filtered_df = freq_df[freq_df['word'] == lemma]
    if not filtered_df.empty:
        frequency = filtered_df['freq'].values[0]
        return frequency, False
    return alpha, True


def main():
    """Build and save the per-word unigram frequency / log-probability table.

    Loads the CLASSLA Wikipedia wordlist
    (:data:`WORDLIST_CSV_PATH`) and the target sentences
    (:data:`TARGET_SENTENCES_PATH`), instantiates a Serbian
    CLASSLA pipeline, lemmatizes each token, looks up the
    frequency, applies Laplace smoothing for OOVs, computes the
    unigram log-probability, and writes the per-word table to
    :data:`OUTPUT_CSV_PATH`.
    """
    freq_df = pd.read_csv(WORDLIST_CSV_PATH)
    target_sentences_df = pd.read_csv(TARGET_SENTENCES_PATH)

    nlp = classla.Pipeline('sr', processors='tokenize, lemma, pos')

    words_list = []
    frequency_list = []
    target_sentence_list = []
    lemma_list = []
    num_new_words = 0

    for i in range(0, len(target_sentences_df)):
        sentence = target_sentences_df['Text'][i].lower()
        words = sentence.split(' ')

        for word in words:
            if word == '':
                continue
            words_list.append(word)
            target_sentence_list.append(i)

            lemma = lemmatize_word(word, nlp)
            lemma_list.append(lemma)

            frequency, is_new = lookup_frequency(lemma, freq_df, ALPHA)
            frequency_list.append(frequency)
            if is_new:
                num_new_words += 1

    total_freq = sum(freq_df['freq']) + (num_new_words + len(freq_df)) * ALPHA
    log_probability_list = [math.log(freq / total_freq) for freq in frequency_list]

    # Create a DataFrame
    df = pd.DataFrame({'Sentence': target_sentence_list,
                       'Word': words_list,
                       'Lemma': lemma_list,
                       'Word Frequency': frequency_list,
                       'Log Probability': log_probability_list})
    df.to_csv(OUTPUT_CSV_PATH, index=False)


if __name__ == "__main__":
    main()
