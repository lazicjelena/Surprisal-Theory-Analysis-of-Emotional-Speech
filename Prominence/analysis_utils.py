# -*- coding: utf-8 -*-
"""analysis_utils.py
Pomocne funkcije za agregaciju/analizu preko emocija izdvojene iz:
  - Prominence/plot energy.py
  - Prominence/plot frequency.py
  - Prominence/plot spe\u0435ch time.py

P-008 (Faza 2-B): zajednicke IDENTICNO funkcije unutar foldera
'Prominence/'. Tijelo funkcije NIJE mijenjano.

Napomena: funkcija extraxt_parameter_over_emotion ima typo u imenu
(extraxt -> extract). Zadrzano radi zero-change principa. Preimenovanje
ide u posebnom proposalu.

Ova IDENTICNO grupa ima samo 3 kopije - sve u folderu 'Prominence/' -
pa je u potpunosti eliminisana ovim P-008 korakom.
"""


def extraxt_parameter_over_emotion(data, parameter):

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
