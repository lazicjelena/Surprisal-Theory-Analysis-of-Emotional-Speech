# -*- coding: utf-8 -*-
"""utils.analysis_utils

Centralizovani helperi za agregaciju/analizu preko emocija.

P-012 (Faza 2-C): cross-folder konsolidacija. Funkcija
``extraxt_parameter_over_emotion`` je premjestena ovdje iz
``prominence/analysis_utils.py`` da bi cijela ``utils/`` familija
bila na jednom mjestu. Tijelo funkcije NIJE mijenjano (zero-change).

Napomena: funkcija ``extraxt_parameter_over_emotion`` ima typo u imenu
(extraxt -> extract). Zadrzano radi zero-change principa. Preimenovanje
ide u posebnom proposalu.

Pipeline role
-------------
Project-wide shared helper module imported under the package path
``utils.analysis_utils``. Hosts :func:`extraxt_parameter_over_emotion`,
the per-emotion neutral-vs-emotional aggregator that the three
``prominence/plot_energy.py``, ``prominence/plot_frequency.py`` and
``prominence/plot_speech_time.py`` plotting scripts use to align each
emotion's prosodic measurement with its neutral counterpart on the
same ``(speaker, target sentence, word)`` key.
"""


def extraxt_parameter_over_emotion(data, parameter):
    """Build a neutral-vs-emotional table for one prosodic ``parameter``.

    For every neutral utterance row (``emotion == 0``), the same
    word-token in every other emotional state
    ``emotion in {1, 2, 3, 4}`` is looked up by matching
    ``(speaker, target sentence, word)``. Repeated occurrences of
    the same word in a sentence are resolved positionally by
    counting how many times the word has already been consumed
    in the sentence so far. The matching emotional value of
    ``parameter`` is added as a new column named after the
    emotion id; rows for which no emotional match is found are
    dropped (the sentinel ``487923472842101`` is filtered out).

    Note: the misspelled name ``extraxt_parameter_over_emotion``
    is preserved verbatim under the project's zero-change rule.

    Parameters
    ----------
    data : pandas.DataFrame
        Per-utterance prominence/duration table with at least
        ``emotion``, ``speaker``, ``target sentence``, ``word``
        and ``parameter`` columns.
    parameter : str
        Name of the prosodic column to aggregate (e.g.
        ``"prominence"`` or ``"duration"``).

    Returns
    -------
    pandas.DataFrame
        The neutral subset of ``data`` augmented with one
        ``int``-keyed column per emotional state in
        ``{1, 2, 3, 4}`` containing the matching emotional value
        of ``parameter``.
    """
    print(f"Parameter: {parameter}")
    neutral_data = data[data['emotion'] == 0]

    for emotion in [1,2,3,4]:

        print(f"Emotional state: {emotion}")
        duration_list = []
        none_values = 0
        ind = 0
        last_sentence = 932947234
        words = []

        for _,row in neutral_data.iterrows():

            speaker = row['speaker']
            sentence = row['target sentence']
            if sentence != last_sentence:
                last_sentence = sentence
                words = []
            word = row['word']
            words.append(word)

            search = data[data['emotion']==emotion]
            search = search[search['target sentence']==sentence]
            search = search[search['speaker']==speaker]
            search = search[search['word']==word]

            if len(search) > 1:
                index = words.count(word)-1
                duration = search[parameter].values[index]
                ind += 1
            else:
                if len(search)==1:
                    duration = search[parameter].values[0]
                else:
                    duration = 487923472842101
                    none_values += 1
            duration_list.append(duration)

        neutral_data[emotion] = duration_list
        neutral_data = neutral_data[neutral_data[emotion] != 487923472842101]
        print(f"None values count: {none_values}")
        print(f"Double words count: {ind}")

    print(f"Final number of words: {len(neutral_data)}")

    return neutral_data
