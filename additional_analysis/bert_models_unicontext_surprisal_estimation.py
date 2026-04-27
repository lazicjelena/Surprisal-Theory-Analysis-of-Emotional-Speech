# -*- coding: utf-8 -*-
"""bert_models_unicontext_surprisal_estimation.py

Per-word surprisal estimation under an *incremental* (uni-context)
prefix instead of the standard whole-sentence masking. Converted from
``BERT_models_UniContext_surprisal_estimation.ipynb`` (P-013-NB).

Pipeline role
-------------
Variant of the BERT-family masked-LM surprisal estimator (sister of
``surprisal_estimation/bert_surprisal_estimation.py``) used in the
post-recension additional analyses. Instead of masking each token in
the full sentence, it builds the sentence one word at a time and
scores the *last* word in each growing prefix; the resulting per-word
surprisal is therefore conditioned on the prefix only, mimicking a
unidirectional / left-to-right context. The output column
``Surprisal ALBERT uni`` is written to
``../podaci/word_surprisals_albert_uni.csv``.

The original notebook contained four model-loading cells (BERT,
BERTic, RoBERTa, ALBERT) executed in sequence; the last cell to run
overwrites the module-level ``model`` and ``tokenizer``. The
notebook's final outputs use the ALBERT configuration -- preserved
verbatim here. To re-run with a different model, swap the model-loading
block at the top of the script.

Notes
-----
- Requires the ``transformers`` and ``torch`` packages.
- Helper functions are shared with the standard BERT pipeline; their
  bodies are byte-identical to ``bert_surprisal_estimation.py``.
"""

import math
import os

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AlbertForMaskedLM, AutoConfig, AutoTokenizer

# Paths (relative to project root).
TARGET_SENTENCES_PATH = os.path.join('..', 'podaci', 'target_sentences.csv')
OUTPUT_CSV_PATH = os.path.join('..', 'podaci', 'word_surprisals_albert_uni.csv')

# Model configuration (ALBERT — last-loaded variant in the source
# notebook). Alternative checkpoints from the original notebook were
# 'bert-base-multilingual-uncased', 'classla/bcms-bertic' and
# 'FacebookAI/roberta-base'; swap the load lines below to pick one.
MODEL_NAME = "albert-base-v2"

# Module-level model / tokenizer.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
# ALBERT does not have a causal LM head, so we use it for Masked LM instead
model = AlbertForMaskedLM.from_pretrained(MODEL_NAME, config=config)


def mask_each_word(sentence):
    """Produce one masked-sentence variant per token in ``sentence``.

    Tokenizes ``sentence`` with the module-level tokenizer and,
    for each tokenized position, returns a copy of the sentence
    with that token replaced by ``[MASK]``. The original masked
    token is also returned alongside, so that the caller can
    later score its probability under the masked LM.

    Parameters
    ----------
    sentence : str
        Raw input sentence (lower-cased upstream).

    Returns
    -------
    masked_sentences : list of str
        Length equals the tokenized sentence length; element ``i``
        is the sentence with token ``i`` replaced by ``[MASK]``.
    masked_words : list of str
        The original sub-tokens that were replaced, in the same
        order as ``masked_sentences``.
    """
    # Tokenize the sentence
    tokenized_sentence = tokenizer.tokenize(sentence)

    # Lists to store masked sentences and masked words
    masked_sentences = []
    masked_words = []

    # Iterate through each word and replace it with [MASK]
    for i in range(len(tokenized_sentence)):
        masked_sentence = list(tokenized_sentence)  # Create a copy of the tokenized sentence
        masked_sentence[i] = tokenizer.mask_token  # Replace the i-th word with [MASK]

        # Add the masked sentence to the list
        masked_sentences.append(tokenizer.convert_tokens_to_string(masked_sentence))

        # Add the masked word to the list
        masked_words.append(tokenized_sentence[i])

    return masked_sentences, masked_words


def estimate_masked_probability(sentence, candidate_word, model=model, tokenizer=tokenizer):
    """Probability of ``candidate_word`` at the ``[MASK]`` position.

    Encodes ``sentence`` (which must contain exactly one ``[MASK]``
    token) into input ids, replaces the mask id with the id of
    ``candidate_word``, runs the masked LM forward pass, applies
    softmax over the vocabulary at the masked position, and
    returns the probability assigned to ``candidate_word``.

    Parameters
    ----------
    sentence : str
        Sentence containing one ``[MASK]`` token.
    candidate_word : str
        The sub-token whose conditional probability is to be
        scored.
    model : transformers.PreTrainedModel, optional
        Masked-LM checkpoint. Defaults to the module-level
        ``model``.
    tokenizer : transformers.PreTrainedTokenizer, optional
        Matching tokenizer. Defaults to the module-level
        ``tokenizer``.

    Returns
    -------
    float
        Softmax probability of ``candidate_word`` at the masked
        position; in ``[0, 1]``.
    """
    # Tokenize the input sentence
    tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=True)

    # Find the index of the [MASK] token
    mask_index = tokenized_sentence.index(tokenizer.mask_token_id)

    # Replace [MASK] with the candidate word
    tokenized_sentence[mask_index] = tokenizer.convert_tokens_to_ids(candidate_word)

    # Convert tokenized sequence to PyTorch tensor
    input_ids = torch.tensor([tokenized_sentence])

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Apply softmax to obtain probabilities
    probabilities = F.softmax(logits[0, mask_index], dim=0)

    # Find the probability of the candidate word
    candidate_index = tokenizer.convert_tokens_to_ids(candidate_word)
    candidate_probability = probabilities[candidate_index].item()

    return candidate_probability


def concatenate_words_and_probabilities(words, probabilities):
    """Merge WordPiece sub-tokens back into orthographic words.

    BERT-family tokenizers split unknown tokens into ``"head"``
    plus one or more ``"##suffix"`` continuations. This routine
    glues each ``##``-prefixed continuation to the preceding head
    and multiplies the matching per-piece probabilities to obtain
    a single probability for the merged orthographic word.

    Parameters
    ----------
    words : list of str
        WordPiece tokens as produced by the tokenizer.
    probabilities : list of float
        Per-piece probabilities aligned with ``words``.

    Returns
    -------
    new_words : list of str
        Merged orthographic words, with ``##`` prefixes stripped.
    new_probabilities : list of float
        Product of the per-piece probabilities for each merged
        word; length matches ``new_words``.
    """
    new_words = []
    new_probabilities = []

    i = 0
    while i < len(words):
        current_word = words[i]
        current_probability = probabilities[i]

        while i + 1 < len(words) and words[i + 1].startswith('##'):
            next_word = words[i + 1][2:]
            current_word += next_word
            current_probability *= probabilities[i + 1]
            i += 1  # Move to the next word in the sequence

        new_words.append(current_word)
        new_probabilities.append(current_probability)

        i += 1

    return new_words, new_probabilities


def calculate_word_probabilities(sentence):
    """Per-word probabilities under the masked LM for one sentence.

    Pipeline: :func:`mask_each_word` produces one mask variant per
    sub-token; :func:`estimate_masked_probability` scores each
    masked token under the model;
    :func:`concatenate_words_and_probabilities` merges sub-token
    fragments back to orthographic words.

    Parameters
    ----------
    sentence : str
        Input sentence (lower-cased upstream).

    Returns
    -------
    words : list of str
        Orthographic words after sub-token merging.
    probabilities : list of float
        Per-word probabilities aligned with ``words``.
    total : float
        Product of all per-piece probabilities (kept for parity
        with the original notebook output).
    """
    masked_sentences, masked_words = mask_each_word(sentence)

    list_probabilities = []
    total = 1
    for candidate, masked_sentence in zip(masked_words, masked_sentences):
        probability = estimate_masked_probability(masked_sentence, candidate)
        list_probabilities.append(probability)
        total = total * probability

    words, probabilities = concatenate_words_and_probabilities(masked_words, list_probabilities)

    return words, probabilities, total


def main():
    """Build and save the per-word ALBERT-uni surprisal table.

    Reads :data:`TARGET_SENTENCES_PATH`, iterates each row's
    lower-cased sentence, builds incremental prefixes
    (``uni_context_words``) one word at a time, and for each
    prefix calls :func:`calculate_word_probabilities` to score
    the *last* word's probability under the masked LM. The
    surprisal ``-log2(p_last)`` is recorded per prefix step,
    yielding a strictly left-to-right surprisal sequence. The
    resulting ``(Sentence, Word, 'Surprisal ALBERT uni')`` table
    is written to :data:`OUTPUT_CSV_PATH`.
    """
    target_sentences_df = pd.read_csv(TARGET_SENTENCES_PATH)

    words_list = []
    probabilities_list = []
    target_sentence_list = []

    for i in range(0, len(target_sentences_df)):
        sentence = target_sentences_df['Text'][i].lower()
        words = sentence.split(' ')

        uni_context_words = []
        for word in words:
            uni_context_words.append(word)
            new_sentence = " ".join(uni_context_words)
            _, probabilities, total = calculate_word_probabilities(new_sentence)

            words_list.append(word)
            probabilities_list.append(-math.log2(probabilities[-1]))
            target_sentence_list.append(i)

    # Create a DataFrame
    df = pd.DataFrame({'Sentence': target_sentence_list, 'Word': words_list, 'Surprisal ALBERT uni': probabilities_list})

    # Save the DataFrame to a CSV file
    df.to_csv(OUTPUT_CSV_PATH, index=False)


if __name__ == "__main__":
    main()
