# 02 — Detaljna analiza duplikata funkcija

**Projekat:** Surprisal-Theory-Analysis-of-Emotional-Speech
**Datum:** 2026-04-24
**Metoda:** AST ekstrakcija tijela funkcija + kanonska normalizacija (strip komentara/praznih linija) + MD5 hash + pairwise `diff -u`
**Pratnja:** `01_project_map.md`, `01_dependency_overview.md`
**Status:** READ-ONLY analiza. Ne mijenja se kod. Ne predlažu se refactori. Samo razumijevanje razlika.

---

## Legenda klasifikacija

- **IDENTIČNO** — sva tijela su ista kad se ignorišu komentari/prazne linije → jedan hash. Spajanje je semantički bezbjedno.
- **VARIJACIJA (kozmetička)** — logika identična, razlikuju se samo imena lokalnih varijabli/poruka/komentara. Spajanje je bezbjedno uz izbor kanonskog imena.
- **VARIJACIJA (funkcionalna)** — različiti parametri koji ipak rade istu stvar na drugim ulazima/tokenizerima/kolonama. Razlika je eksplicitna, svjesna. Spajanje moguće samo kroz novi argument — ili čuvati odvojeno.
- **KONFLIKT** — isto ime, **stvarno različita logika ili različiti povratni tipovi**. Spajanje je OPASNO jer bi tiho promijenilo rezultat na nekim call-site-ovima.
- **METODA KLASE (nije pravi duplikat)** — isti `name`, ali metoda **različitih klasa** (`__init__`, `forward`, `__len__`, `__getitem__`). Python ih razlikuje po klasi — nema konflikta.

---

## Pregled (21 duplikatno ime)

| # | Ime funkcije | Broj kopija | Distinct varijanti | Klasifikacija | Preporuka |
|---|---|---|---|---|---|
| 1 | `calculate_log_Likelihood` | 6 | 1 | IDENTIČNO | Spojiti |
| 2 | `calculate_aic` | 6 | 1 | IDENTIČNO | Spojiti |
| 3 | `find_subword` | 4 | 1 | IDENTIČNO | Spojiti |
| 4 | `extraxt_parameter_over_emotion` | 3 | 1 | IDENTIČNO (typo u imenu!) | Spojiti (i ispraviti typo) |
| 5 | `get_fixed_length_mel_spectrogram` | 3 | 1 | IDENTIČNO | Spojiti |
| 6 | `extract_mel_spectrogram` | 2 | 1 | IDENTIČNO | Spojiti |
| 7 | `add_column` | 2 | 1 | IDENTIČNO | Spojiti |
| 8 | `add_word_type` | 3 | 2 | VARIJACIJA (kozmetička) | Spojiti (odabrati kanonsko ime varijable) |
| 9 | `extract_words_and_probabilities` | 4 | 2 | VARIJACIJA (funkcionalna — tokenizer marker) | Čuvati odvojeno ILI uvesti argument |
| 10 | `akaike_for_column` | 6 | 4 | KONFLIKT (različiti potpisi i semantika) | Preimenovati / čuvati odvojeno |
| 11 | `calculate_delta_ll` | 6 | 5 | KONFLIKT | Preimenovati / čuvati odvojeno |
| 12 | `inf_k_model` | 5 | 3 | KONFLIKT | Preimenovati / čuvati odvojeno |
| 13 | `lookup_features` | 9 | 7 | KONFLIKT | Preimenovati / čuvati odvojeno |
| 14 | `add_column_with_surprisal` | 2 | 2 | KONFLIKT | Preimenovati / čuvati odvojeno |
| 15 | `calculate_word_probabilities` | 3 | 2 | KONFLIKT (GPT vs n-gram) | Čuvati odvojeno |
| 16 | `find_target_sentence` | 2 | 2 | KONFLIKT (vraća index vs. tekst) | Preimenovati |
| 17 | `process_directory` | 2 | 2 | KONFLIKT (potpuno druga svrha) | Preimenovati |
| 18 | `__init__` | 5 | 5 | METODA KLASE | Ostaviti (nije duplikat) |
| 19 | `__len__` | 2 | 2 | METODA KLASE | Ostaviti (nije duplikat) |
| 20 | `__getitem__` | 2 | 2 | METODA KLASE | Ostaviti (nije duplikat) |
| 21 | `forward` | 3 | 3 | METODA KLASE | Ostaviti (nije duplikat) |

Zbirno:
- **7** funkcija je 100% IDENTIČNO → sigurni kandidati za jedan utils modul.
- **2** funkcije su kozmetičke/funkcionalne VARIJACIJE.
- **8** funkcija su KONFLIKTI → opasno za automatsko spajanje.
- **4** funkcije su metode različitih klasa → **nisu duplikati**, samo dijele ime.

---

# 1. IDENTIČNO (safe za spajanje)

## 1.1 `calculate_log_Likelihood(data)` — 6 kopija

**Lokacije:**

| idx | Fajl | Linija |
|---|---|---|
| 00 | `Additional files after recension/my_functions.py` | 59 |
| 01 | `Different information measurement parameters/my_functions.py` | 53 |
| 02 | `Duration Prediction based on Surprisals/surprisal_results.py` | 50 |
| 03 | `Linear regression/final_graphs.py` | 50 |
| 04 | `Linear regression/results.py` | 49 |
| 05 | `Split-over effect/surprisal_results.py` | 15 |

**Potpis (isti u svim):** `calculate_log_Likelihood(data)`

**Kanonsko tijelo (sve 6 identične):**
```python
def calculate_log_Likelihood(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    return norm.logpdf(data, loc=mean, scale=std_dev)
```

**Razlike logike:** nema.
**Klasifikacija:** IDENTIČNO.
**Preporuka:** **spojiti** u jedan `utils` modul (zajedno sa `calculate_aic`). Napomena: naziv nepravilno kapitalizovan (`_Likelihood`) — ostaviti tako radi zero-change ili tretirati kao zaseban re-name u novom proposal-u.

---

## 1.2 `calculate_aic(real_values, results, k)` — 6 kopija

**Lokacije:**

| idx | Fajl | Linija |
|---|---|---|
| 00 | `Additional files after recension/my_functions.py` | 65 |
| 01 | `Different information measurement parameters/my_functions.py` | 59 |
| 02 | `Duration Prediction based on Surprisals/surprisal_results.py` | 56 |
| 03 | `Linear regression/final_graphs.py` | 56 |
| 04 | `Linear regression/results.py` | 55 |
| 05 | `Split-over effect/surprisal_results.py` | 21 |

**Potpis:** `calculate_aic(real_values, results, k)` u svim.

**Kanonsko tijelo:**
```python
def calculate_aic(real_values, results, k):
    residuals = np.array(real_values) - np.array(results)
    log_likelihood = calculate_log_Likelihood(residuals)
    aic = 2 * k - 2 * log_likelihood
    return aic, np.mean(log_likelihood), np.std(log_likelihood)
```

**Razlike logike:** nema.
**Klasifikacija:** IDENTIČNO.
**Preporuka:** **spojiti** zajedno sa `calculate_log_Likelihood`.

---

## 1.3 `find_subword(word, unique_words)` — 4 kopije

**Lokacije:**

| idx | Fajl | Linija |
|---|---|---|
| 00 | `Additional files after recension/build_prominence_datasets.py` | 77 |
| 01 | `Pervious Surprisals/prominence_build_dataset.py` | 84 |
| 02 | `Prominence/librosa_estimated_parameters.py` | 113 |
| 03 | `Prominence/prominence_build_dataset.py` | 86 |

**Potpis:** `find_subword(word, unique_words)` u svim.

**Kanonsko tijelo:**
```python
def find_subword(word, unique_words):
    subword = ''
    for i in range(1, len(word)+1):
        if word[-i:] in unique_words:
            subword = word[-i:]
    return subword
```

**Klasifikacija:** IDENTIČNO.
**Preporuka:** **spojiti**.

---

## 1.4 `extraxt_parameter_over_emotion(data, parameter)` — 3 kopije

**Lokacije:**

| idx | Fajl | Linija |
|---|---|---|
| 00 | `Prominence/plot energy.py` | 16 |
| 01 | `Prominence/plot frequency.py` | 16 |
| 02 | `Prominence/plot speеch time.py` | 19 |

**Potpis:** `extraxt_parameter_over_emotion(data, parameter)` u svim.

**Klasifikacija:** IDENTIČNO.
**Napomena:** ime funkcije sadrži **typo** (`extraxt` umjesto `extract`) — replikovan u sve 3 kopije (indicira copy-paste, a ne razvoj).
**Preporuka:** **spojiti** u `Prominence/_utils.py` (ili lokalni utils fajl u `Prominence/` folderu). Typo ostaviti radi zero-change principa; posebnim proposal-om kasnije preimenovati.

---

## 1.5 `get_fixed_length_mel_spectrogram(y, sr, n_mels, fixed_length)` — 3 kopije

**Lokacije:**

| idx | Fajl | Linija |
|---|---|---|
| 00 | `Emotion recognition/audiodataset.py` | 16 |
| 01 | `Emotion recognition/prosody_parameters_and_mfcc.py` | 24 |
| 02 | `Mel coefficients and surprisals/calculate_mel_spectrum.py` | 34 |

**Klasifikacija:** IDENTIČNO.
**Preporuka:** **spojiti** — najprirodniji dom je `Mel coefficients and surprisals/` ili novi `audio_utils.py`. Napomena: ove dvije `Emotion recognition/` kopije već su u istom folderu → lokalna konsolidacija trivijalna.

---

## 1.6 `extract_mel_spectrogram(audio_file)` — 2 kopije

**Lokacije:**

| idx | Fajl | Linija |
|---|---|---|
| 00 | `Emotion recognition/prosody_parameters_and_mfcc.py` | 46 |
| 01 | `Mel coefficients and surprisals/calculate_mel_spectrum.py` | 56 |

**Tijelo (identično):**
```python
def extract_mel_spectrogram(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mel_spectrogram = get_fixed_length_mel_spectrogram(y, sr, n_mels=mel_dim, fixed_length=fixed_length)
    return mel_spectrogram
```

**Napomena:** zavisi od modulnih konstanti `mel_dim` i `fixed_length` — pri spajanju utils funkcija mora ih primiti kao argumente **ili** konstante moraju biti definisane u utils modulu.

**Klasifikacija:** IDENTIČNO (po bodyju).
**Preporuka:** **spojiti** zajedno sa `get_fixed_length_mel_spectrogram`. Pažnja na global constants.

---

## 1.7 `add_column(df, k=0)` — 2 kopije

**Lokacije:**

| idx | Fajl | Linija |
|---|---|---|
| 00 | `Different information measurement parameters/my_functions.py` | 240 |
| 01 | `Split-over effect/baseline_model.py` | 19 |

**Potpis:** `add_column(df, k=0)` u oba.

**Klasifikacija:** IDENTIČNO.
**Preporuka:** **spojiti** u `utils` (zajedno sa ostalim `baseline`/LR pipeline funkcijama). Napomena: ova funkcija je baseline LR model bez surprisala — ne miješati sa `add_column_with_surprisal`.

---

# 2. VARIJACIJA (funkcionalna, ne kozmetička u korak 8)

## 2.1 `add_word_type(data, freq_df, column_name)` — 3 kopije, 2 varijante

**Lokacije:**

| idx | Fajl | Linija |
|---|---|---|
| 00 | `Additional files after recension/my_functions.py` | 153 |
| 01 | `Duration Prediction based on Surprisals/transform_data_into_dataframe.py` | 15 |
| 02 | `Linear regression/build_dataset.py` | 51 |

**Canonical grupe:**
- **Grupa A** [hash c2e971140cb9]: `00`, `02` — identične
- **Grupa B** [hash 7e4d88c4f190]: `01` — varijanta

**Diff (Grupa A → B):**
```diff
-    log_prob_list = []
+    word_type_list = []
 ...
-        log_probability_value = ''
+        word_type_value = ''
 ...
-                log_probability_value += freq[column_name].values[...]
+                word_type_value += freq[column_name].values[...]
 ...
-        log_prob_list.append(log_probability_value.strip())
-    return log_prob_list
+        word_type_list.append(word_type_value.strip())
+    return word_type_list
```

**Analiza:** **isključivo kozmetička razlika** — `01` je preimenovao lokalnu varijablu `log_prob_list → word_type_list` i `log_probability_value → word_type_value`. Logika, potpis i povratni tip su identični.

**Klasifikacija:** VARIJACIJA (kozmetička).
**Preporuka:** **spojiti**. Odabrati kanonsko ime varijable — `word_type_*` je semantički tačnije (funkcija se zove `add_word_type`, ali radi sa kolonom `column_name` koja može biti bilo šta — obje varijante su generalne). Zero-risk merge.

---

## 2.2 `extract_words_and_probabilities(subwords, subword_probabilities)` — 4 kopije, 2 varijante

**Lokacije:**

| idx | Fajl | Linija |
|---|---|---|
| 00 | `Different information measurement parameters/information_and_distance_functions.py` | 65 |
| 01 | `Different information measurement parameters/parameters estimations/Yugo_GPT_contextual_entropy.py` | 44 |
| 02 | `Surprisal estimation/llama.py` | 40 |
| 03 | `Surprisal estimation/Yugo GPT-3 surprisal estimation.py` | 48 |

**Canonical grupe:**
- **Grupa A** [9610aafec4e3]: `00` — marker `'Ġ'`
- **Grupa B** [3ff3fcee542b]: `01`, `02`, `03` — marker `'▁'`

**Diff (A → B):**
```diff
-        if subword.startswith('Ġ'):
+        if subword.startswith('▁'):
```

**Analiza:** razlika je u **token-boundary markeru**:
- `Ġ` — GPT-2 / RoBERTa stil (BPE tokenizer)
- `▁` — SentencePiece stil (Llama, T5, YugoGPT)

Ovo je **semantički različito** — pogrešan marker znači pogrešno segmentiranje riječi. Nije kozmetika; zavisi od modela koji poziva funkciju.

**Klasifikacija:** VARIJACIJA (funkcionalna, tokenizer-specifična).
**Preporuka:** **ne spajati automatski**. Dvije opcije:
- a) čuvati odvojeno (trenutno stanje — svaka estimacija koristi svoj tokenizer marker);
- b) spojiti sa dodatnim argumentom `marker='▁'` (default) — ali to zahtijeva promjenu poziva → tretirati kroz zaseban proposal.

---

# 3. KONFLIKT (isto ime, druga logika — opasno za merge)

## 3.1 `akaike_for_column` — 6 kopija, 4 distinct varijante

**Lokacije i potpisi:**

| idx | Fajl | Linija | Potpis |
|---|---|---|---|
| 00 | `Additional files after recension/my_functions.py` | 72 | `akaike_for_column(data, prominence, model_name, baseline_model='baseline')` |
| 01 | `Different information measurement parameters/my_functions.py` | 65 | `akaike_for_column(data, model_name, baseline_model='baseline')` |
| 02 | `Duration Prediction based on Surprisals/surprisal_results.py` | 62 | `akaike_for_column(data, model_name, baseline_model='baseline')` |
| 03 | `Linear regression/final_graphs.py` | 62 | `akaike_for_column(data, model_name, baseline_model='baseline')` |
| 04 | `Linear regression/results.py` | 61 | `akaike_for_column(column_name, model_name, baseline_model='baseline')` ⚠ |
| 05 | `Split-over effect/surprisal_results.py` | 27 | `akaike_for_column(data, model_name, baseline_model='baseline')` |

**Canonical grupe:**
- **Grupa A** [f04975359970]: `00` — tri argumenta (`data, prominence, model_name`), koristi `data[prominence]` kao real-value kolonu, vraća `(difference, std_difference)` gdje je `std_difference = std_ll_1 - std_ll_2`.
- **Grupa B** [06f4ad3750ff]: `01`, `05` — dva argumenta (`data, model_name`), hardcodovan `'time'` kao real-value, **dodaje `data.dropna(...)`**, vraća `(difference, std_ll_2)`.
- **Grupa C** [8fea0785d387]: `02`, `03` — isti potpis kao B, ali **bez dropna**. Vraća `(difference, std_ll_2)`.
- **Grupa D** [023f00b1fffd]: `04` — **potpuno druga semantika**: prvi argument je `column_name`, petlja ide kroz unique vrijednosti te kolone (tipično `'emotion'`), vraća **listu** `difference` i jedan `std_ll_2`.

**Ključni diff-ovi:**

A→B:
```diff
-def akaike_for_column(data, prominence, model_name, baseline_model = 'baseline'):
-    _, mean_ll_1, std_ll_1 = calculate_aic(data[prominence], data[baseline_model], 2)
-    _, mean_ll_2, std_ll_2 = calculate_aic(data[prominence], data[model_name], 3)
-    std_difference = std_ll_1 - std_ll_2
-    return difference, std_difference
+def akaike_for_column(data, model_name, baseline_model = 'baseline'):
+    data = data.dropna(subset=[model_name, baseline_model])
+    _, mean_ll_1, std_ll_1 = calculate_aic(data['time'], data[baseline_model], 2)
+    _, mean_ll_2, std_ll_2 = calculate_aic(data['time'], data[model_name], 3)
+    return difference, std_ll_2
```

B→C (razlikuju se samo u `dropna`):
```diff
-    data = data.dropna(subset=[model_name, baseline_model])
```
(C nema dropna → različito ponašanje na NaN-ovima)

B→D:
```diff
-def akaike_for_column(data, model_name, baseline_model='baseline'):
-    _, mean_ll_1, std_ll_1 = calculate_aic(data['time'], data[baseline_model], 2)
-    _, mean_ll_2, std_ll_2 = calculate_aic(data['time'], data[model_name], 3)
-    difference = mean_ll_1 - mean_ll_2
-    return difference, std_ll_2
+def akaike_for_column(column_name, model_name, baseline_model='baseline'):
+    difference = []
+    for gender in df[column_name].unique():   # ⚠ koristi vanjski globalni `df`
+        data = df[df[column_name]==gender]
+        _, mean_ll_1, std_ll_1 = calculate_aic(data['time'], data[baseline_model], 2)
+        _, mean_ll_2, std_ll_2 = calculate_aic(data['time'], data[model_name], 3)
+        difference.append(mean_ll_1-mean_ll_2)
+    return difference, std_ll_2
```

**Analiza:**
- A: stariji, generalni (bilo koji `prominence` stupac, bilo koji `real_value`, vraća razliku std-ova).
- B, C: sužena verzija na `'time'`, sa/bez `dropna`. B i C razlikuju se **samo po dropna** — mogu producirati različite rezultate kad postoje NaN vrijednosti.
- D: **potpuno drugi povratni tip** — vraća listu diferenca po grupama (npr. po emociji), koristi vanjski `df`.

**Povratni tipovi se razlikuju:** A vraća skalar + skalar (razlika std), B/C vraćaju skalar + skalar (drugi std), D vraća listu + skalar.

**Klasifikacija:** KONFLIKT.
**Preporuka:** **čuvati odvojeno** ili **preimenovati**:
- A → `akaike_for_column_generic` (može preko argumenta `prominence`);
- B → `akaike_for_column_with_dropna`;
- C → `akaike_for_column`;
- D → `akaike_for_column_per_group` (koristi vanjski `df` — odmah je sumnjiva — trebala bi imati `df` kao argument; ovo je bug-risk).

Automatsko spajanje je **opasno** jer call-site u `Linear regression/results.py` očekuje listu kao povratnu vrijednost, a ostali skalar.

---

## 3.2 `calculate_delta_ll` — 6 kopija, 5 distinct varijanti

**Lokacije i potpisi:**

| idx | Fajl | Linija | Potpis |
|---|---|---|---|
| 00 | `Additional files after recension/my_functions.py` | 82 | `calculate_delta_ll(data, surprisal, k, emotion_data, std_data, prominence='time', function='power')` |
| 01 | `Different information measurement parameters/my_functions.py` | 74 | `calculate_delta_ll(data, model_name, baseline='baseline -3')` |
| 02 | `Duration Prediction based on Surprisals/surprisal_results.py` | 70 | `calculate_delta_ll(data, surprisal_name, k)` |
| 03 | `Linear regression/final_graphs.py` | 70 | `calculate_delta_ll(data, surprisal_name, k)` (= isti hash kao 02) |
| 04 | `Linear regression/results.py` | 74 | `calculate_delta_ll(surprisal, k, emotion_data, std_data)` |
| 05 | `Split-over effect/surprisal_results.py` | 36 | `calculate_delta_ll(data, surprisal_name, k)` (potpis isti kao 02/03, ali drugo tijelo) |

**Canonical grupe:**
- A [497d67059aa1]: `00`
- B [2e89341ff742]: `01`
- C [9326806b6f75]: `02` = `03`
- D [b27d3df201b7]: `04`
- E [46a9bf1bd908]: `05`

**Ključne razlike:**

- **A (`00`)**: 7 argumenata, gradi `model_name = surprisal + ' ' + str(k) + ' model' + function`, zove `akaike_for_column(data, prominence, model_name, 'baseline')` (verzija A iz §3.1), puni `emotion_data[emotion]` i `std_data[emotion]` po emociji, **vraća `None`** (side-effect).

- **B (`01`)**: 3 argumenta, baseline default je `'baseline -3'`, zove `akaike_for_column(data, model_name, baseline)`, **vraća (delta_ll, std_element)**.

- **C (`02`, `03`)**: 3 argumenta, gradi `model_name = surprisal_name + ' ' + str(k) + ' model'`, baseline je hardcoded `'baseline'`, vraća `(delta_ll, std_element)`.

- **D (`04`)**: 4 argumenta (bez `data`!), zove `akaike_for_column('emotion', ...)` (verzija D iz §3.1 — koristi vanjski `df`), puni `emotion_data`/`std_data`, **vraća `None`**.

- **E (`05`)**: 3 argumenta, gradi `model_name = f"{surprisal_name} -{k} model"` i baseline `f"baseline -{k}"` (split-over-effect format), vraća `(delta_ll, std_element)`.

**Važni diff — A vs C:**
```diff
-def calculate_delta_ll(data, surprisal, k, emotion_data, std_data, prominence='time', function='power'):
-    model_name = surprisal + ' ' + str(k) + ' model'
-    if function != 'power':
-        model_name+= function
-    try:
-      delta_ll, std_list = akaike_for_column(data, prominence, model_name, 'baseline')
-    except:
-      delta_ll = [0,0,0,0,0]
-      std_list = [1,1,1,1,1]
-    for emotion in range(0,5):
-      emotion_data[emotion].append(delta_ll[emotion])
-      std_data[emotion].append(std_list)
-    return                              # ← vraća None
+def calculate_delta_ll(data, surprisal_name, k):
+    try:
+      delta_ll, std_element = akaike_for_column(data, surprisal_name+' '+str(k)+' model', 'baseline')
+      return delta_ll, std_element      # ← vraća tuple
+    except:
+      print(...)
+      return 0, 0
```

**Važni diff — C vs E (split-over effect):**
```diff
-      delta_ll, std_element = akaike_for_column(data, surprisal_name + ' ' + str(k) + ' model', 'baseline')
+      delta_ll, std_element = akaike_for_column(data, f"{surprisal_name} -{k} model", f"baseline -{k}")
```
C koristi kolone `"{name} {k} model"` + `baseline`, E koristi `"{name} -{k} model"` + `baseline -{k}`. **To su fizički druge kolone u dataframu** — spajanjem bi jedno od ovoga prestalo raditi.

**Klasifikacija:** KONFLIKT (različiti potpisi, različiti povratni tipovi, različita imena kolona koja se pretražuju).
**Preporuka:** **čuvati odvojeno**, po mogućnosti **preimenovati** da se vidi o kojoj varijanti je riječ:
- A → `calculate_delta_ll_per_emotion_inplace` (ili `_side_effect`)
- B → `calculate_delta_ll_generic_baseline`
- C (`02`, `03`) → `calculate_delta_ll_standard`
- D → `calculate_delta_ll_per_emotion_results_style`
- E → `calculate_delta_ll_split_over_effect`

Spajanje je **vrlo opasno** zbog različitih povratnih tipova — tihi bug ako bi se zamijenilo.

---

## 3.3 `inf_k_model` — 5 kopija, 3 distinct varijante

**Lokacije i potpisi:**

| idx | Fajl | Linija | Potpis |
|---|---|---|---|
| 00 | `Additional files after recension/my_functions.py` | 14 | `inf_k_model(df, k, surprisal, prosody='time', function='power')` |
| 01 | `Duration Prediction based on Surprisals/surprisal_results.py` | 17 | `inf_k_model(df, k, surprisal)` |
| 02 | `Linear regression/final_graphs.py` | 17 | `inf_k_model(df, k, surprisal)` (= isti hash kao 01) |
| 03 | `Linear regression/residual_distribution.py` | 17 | `inf_k_model(df, k, surprisal)` |
| 04 | `Linear regression/results.py` | 17 | `inf_k_model(df, k, surprisal)` (= isti hash kao 03) |

**Canonical grupe:**
- A [852a805e1b1d]: `00` — "puna" verzija sa `prosody` i `function` parametrima (podržava `power`/`linear`/`logarithmic`/`exponential`).
- B [12a5fdc51111]: `01`, `02` — "sužena" verzija, hardcoded `'time'`, hardcoded `** k`, uključuje `test_data = test_data.drop(columns=[surprisal_name])` prije `concat`.
- C [eedcc52f3694]: `03`, `04` — ista kao B, **ali bez** `test_data.drop(columns=[surprisal_name])`.

**Ključni diff (A → B):**
```diff
-def inf_k_model(df, k, surprisal, prosody='time', function='power'):
-    if function != 'power':
-        model_name += function
-    if function == 'power':
-        df[surprisal_name] = df[surprisal] ** k
-    if function == 'linear':
-        df[surprisal_name] = df[surprisal] * k
-    if function == 'logarithmic':
-        df[surprisal_name] = np.log(df[surprisal])
-    if function == 'exponential':
-        df[surprisal_name] = np.exp(df[surprisal])
+def inf_k_model(df, k, surprisal):
+    df[surprisal_name] = df[surprisal] ** k
 ...
-    y_train = df[df['fold'] != fold][[prosody]]
+    y_train = df[df['fold'] != fold][['time']]
```

**Diff B → C:**
```diff
-        test_data = test_data.drop(columns=[surprisal_name])
         results_df = pd.concat([results_df, test_data], axis=0)
```
C zadržava privremenu kolonu `surprisal_name` u rezultatu.

**Klasifikacija:** KONFLIKT.
- A je superset B (podržava više funkcija i `prosody` kolona).
- B i C se razlikuju samo po jednom `drop(columns=...)` — razlika je u schemi izlaznog DF-a, može uticati na downstream.

**Preporuka:** **čuvati odvojeno**. Teoretski se A može zadržati kao kanonska (ostale pozovu s defaultima), ali:
- gubi se `drop(columns=[surprisal_name])` iz varijante B, što može biti namjerno;
- ne znamo pouzdano je li A zapravo korištena ili je to stariji eksperiment.

Preporučeno preimenovanje:
- A → `inf_k_model_flexible`
- B → `inf_k_model_drop`
- C → `inf_k_model`

---

## 3.4 `lookup_features` — 9 kopija, 7 distinct varijanti

**Najkompleksniji slučaj.**

**Lokacije i potpisi:**

| idx | Fajl | Linija | Potpis |
|---|---|---|---|
| 00 | `Additional files after recension/my_functions.py` | 113 | `lookup_features(data, freq_df, column_name)` |
| 01 | `Different information measurement parameters/my_functions.py` | 14 | `lookup_features(data, freq_df, column_name)` |
| 02 | `Duration Prediction based on Surprisals/build_surprisal_datasets.py` | 12 | `lookup_features(data, freq_df, column_name)` |
| 03 | `Linear regression/build_dataset.py` | 15 | `lookup_features(data, freq_df, column_name)` |
| 04 | `Pervious Surprisals/build_dataset.py` | 16 | `lookup_features(data, surprisal_df, column_name)` ⚠ drugo ime 2. arg |
| 05 | `Pervious Surprisals/conjoint_data.py` | 12 | `lookup_features(data, surprisal_df, column_name)` |
| 06 | `Pervious Surprisals/correlation_coefficient.py` | 18 | `lookup_features(data, surprisal_df, column_name)` (= isti hash kao 05) |
| 07 | `Prominence/librosa_estimated_parameters.py` | 145 | `lookup_features(data, freq_df, column_name)` |
| 08 | `Prominence/prominence_build_dataset.py` | 118 | `lookup_features(data, freq_df, column_name)` (= isti hash kao 07) |

**Canonical grupe:**
- A [46e872a1239d]: `00` — koristi `if/else` granu na `len(freq)==1`, ima bug `break` prije `log_probability_value += 0` (nikad ne izvršava), `if len(list_of_words)==len(freq_s) or word==freq_s['Word'].iloc[-1]` (dva uslova).
- B [92a7aec86d0f]: `01` — `try/except` umjesto `if len(freq)==1`, bez `break`-bug-a, samo jedan uslov za reset `list_of_words`.
- C [a420b31cc120]: `02` — jednostavniji: bez try/except, direktno `.values[0 + list_of_words.count(word)]`.
- D [3bad30fcc2b3]: `03` — identičan C po logici, samo različit whitespace/komentari (bljeski razlike).
- E [d75d64b2a6a9]: `04` — **drugo ime parametra** (`surprisal_df`), različita imena varijabli (`surprisal_list`, `surprisal_value`, `surprisal_s`, `surprisal_w`), filter `surprisal_df['Sentence']` (uppercase).
- F [68883a87095d]: `05`, `06` — kao E, ali filter koristi **druga imena kolona**: `'target sentence'` i `'word'` (lowercase) umjesto `'Sentence'`/`'Word'`, i `print('error')` je zakomentarisan.
- G [72f9eb3984db]: `07`, `08` — kao C, ali koristi `try/except` kao dummy if/else (`try: log_probability_value += freq[...]; except: log_probability_value += 0`), i ima `print(index)` (nije zakomentarisan).

**Ključni diffovi:**

**C vs E** (freq_df → surprisal_df, različita imena kolona u filteru):
```diff
-def lookup_features(data, freq_df, column_name):
-    log_prob_list = []
+def lookup_features(data, surprisal_df, column_name):
+    surprisal_list = []
 ...
-            freq_s = freq_df[freq_df['Sentence'] == sentence]
-            freq = freq_s[freq_s['Word'] == word]
+            surprisal_s = surprisal_df[surprisal_df['Sentence'] == sentence]
+            surprisal_w = surprisal_s[surprisal_s['Word'] == word]
 ...
-    return log_prob_list
+    return surprisal_list
```
Ovo je kozmetika (preimenovanje varijabli), osim ako su kolone stvarno drugačije — što jeste slučaj u **F**.

**E vs F** (lookup kolone `Sentence`/`Word` → `target sentence`/`word`):
```diff
-            surprisal_s = surprisal_df[surprisal_df['Sentence'] == sentence]
-            surprisal_w = surprisal_s[surprisal_s['Word'] == word]
+            surprisal_s = surprisal_df[surprisal_df['target sentence'] == sentence]
+            surprisal_w = surprisal_s[surprisal_s['word'] == word]
```
Ovo je **stvarna funkcionalna razlika** — pretražuju se drugi stupci u drugom tipu DataFrame-a.

**A vs C** (bug `break` i dvostruki uslov):
```diff
-            if not freq.empty:
-                if len(freq) == 1:
-                    log_probability_value += freq[column_name].values[0]
-                else:
-                    log_probability_value += freq[column_name].values[0 + list_of_words.count(word)]
-            else:
-                break                           # ← dead code; early exit
-                log_probability_value += 0
-                print('error')
-                print(word)
+            if not freq.empty:
+                log_probability_value += freq[column_name].values[0 + list_of_words.count(word)]
+            else:
+                log_probability_value += 0
+                print('error')
+                print(word)
 ...
-            if len(list_of_words) == len(freq_s) or word == freq_s['Word'].iloc[-1]:
+            if len(list_of_words) == len(freq_s):
```
A ima **aktivnu razliku**: `break` prije `log_probability_value += 0` prekida petlju kad se naleti na riječ koja nije u freq_df (nekad je to namjerno da preskoči cijelu rečenicu); dodatno `or word==freq_s['Word'].iloc[-1]` resetuje listu riječi i kad se naiđe na zadnju. Ovo je stvarno drugi algoritam.

**Klasifikacija:** KONFLIKT (sedmostruka podjela: različite kolone u filteru, različita obrada edge case-ova, različita imena parametara).

**Mapa varijanti prema folderu (šta radi u kojem eksperimentu):**

| Grupa | Folderi koji je koriste | Suština |
|---|---|---|
| A | Additional files after recension | Stara verzija s bug-ovima (`break` early exit + dvostruki reset) |
| B | Different information measurement parameters | `try/except` oko `.values[count]` |
| C | Duration Prediction based on Surprisals | Jednostavna, `freq_df['Sentence']/['Word']` |
| D | Linear regression | Ista kao C |
| E | Pervious Surprisals/build_dataset.py | Preimenovano na `surprisal_df`, iste kolone (Sentence/Word) |
| F | Pervious Surprisals/conjoint_data.py, correlation_coefficient.py | `surprisal_df['target sentence']/['word']` — drugi schema |
| G | Prominence | `try/except` kao dummy if/else + `print(index)` |

**Preporuka:** **čuvati odvojeno** ili **preimenovati po grupama**:
- A → `lookup_features_legacy` (zadržati bug-fix verziju samo ako je stvarno korištena)
- B → `lookup_features_try_except` 
- C/D → `lookup_features` (kanonska)
- E → `lookup_features_surprisal_schema` (alias)
- F → `lookup_features_target_sentence_schema` (**različite kolone — stvaran konflikt**)
- G → `lookup_features_silent` (try/except as if/else, bez `print('error')`)

Merge bez imenovanja bi u najmanju ruku tiho razbio F (drugačije ime kolone).

---

## 3.5 `add_column_with_surprisal` — 2 kopije, 2 varijante

**Lokacije:**

| idx | Fajl | Linija | Potpis |
|---|---|---|---|
| 00 | `Different information measurement parameters/my_functions.py` | 117 | `add_column_with_surprisal(df, parameter='', surprisal='', k=3)` |
| 01 | `Split-over effect/surprisal_results.py` | 46 | `add_column_with_surprisal(df, surprisal, k=0)` |

**Razlike:**

- `00`: 4 argumenta s defaultima; gradi training kolone dinamički (`parameter`, `surprisal`), generiše `result_column_name` iz imena parametra/surprisala + `' model'`. Default `k=3`. Ima docstring. Koristi `y_test` je **zakomentarisan** (bug?).
- `01`: 3 argumenta; fiksno `training_columns = ['length', 'log probability', surprisal]`, split-over effect dodaje `-{i}` sufikse za `length`, `log probability` **i** surprisal. Kolona rezultata je `f"{surprisal} -{k} model"`. Ima dead code ispod `return` — ostatak funkcije (drugi blok) je nedostižan (baseline model). Default `k=0`.

**Diff (skraćen):**
```diff
-def add_column_with_surprisal(df, parameter='', surprisal='', k=3):
+def add_column_with_surprisal(df, surprisal, k=0):
 ...
-    if parameter != '':
-        training_columns = ['length', 'log probability', parameter]
-    else:
-        training_columns = ['length', 'log probability']
-    if surprisal != '': 
-        training_columns.append(surprisal)
+    training_columns = ['length', 'log probability', surprisal]
+    if k:
+        for i in range(1,k+1):
+            training_columns.append(f"length -{i}")
+            training_columns.append(f"log probability -{i}")
+            training_columns.append(f"{surprisal} -{i}")
 ...
-    basic_columns = training_columns.copy()
-    for i in range(1,k+1):
-        for column in basic_columns:
-            training_columns.append(f"{column} -{i}")
 ...
-        test_data.loc[:, result_column_name] = y_pred
+        test_data.loc[:, f"{surprisal} -{k} model"] = y_pred
+    return results_df.drop_duplicates()
+    # ↓ NEDOSTIŽAN KOD ispod return:
+    ...
+        test_data.loc[:, f"baseline {k}"] = y_pred
```

**Analiza:** različita logika, različit default `k`, različita schema izlazne kolone (`result_column_name` vs `f"{surprisal} -{k} model"`). Kod `01` ima **dead code** nakon `return` (baseline fallback).

**Klasifikacija:** KONFLIKT.
**Preporuka:** **čuvati odvojeno** / preimenovati:
- `00` → `add_column_with_surprisal_flexible` (sa opcionalnim `parameter`)
- `01` → `add_column_with_surprisal_splitover`

Spajanje bi zahtijevalo svjesnu odluku o split-over schema stringu.

---

## 3.6 `calculate_word_probabilities` — 3 kopije, 2 varijante

**Lokacije:**

| idx | Fajl | Linija | Potpis |
|---|---|---|---|
| 00 | `Surprisal estimation/llama.py` | 72 | `calculate_word_probabilities(sentence, tokenizer=tokenizer, model=model)` |
| 01 | `Surprisal estimation/Yugo GPT-3 surprisal estimation.py` | 80 | `calculate_word_probabilities(sentence, tokenizer=tokenizer, model=model)` |
| 02 | `Surprisal estimation/ngram surprisal estimation/surprisal_estimation_n_gram_model.py` | 36 | `calculate_word_probabilities(sentence, n_gram_counts, vocabulary_size, n=3)` |

**Canonical grupe:**
- A [972664e06347]: `00`, `01` — PyTorch/HF verzija: tokenizuje, poziva model, softmax, poziva `extract_words_and_probabilities` za subword agregaciju.
- B [4a5cba3d7403]: `02` — n-gram verzija: generiše n-grame, računa Laplace-smoothed probability.

**Ključni diff:**
```diff
-def calculate_word_probabilities(sentence, tokenizer=tokenizer, model=model):
-    input_ids = tokenizer.encode(sentence, return_tensors='pt')
-    with torch.no_grad():
-        outputs = model(input_ids)
-        logits = outputs.logits
-    word_probabilities = torch.softmax(logits, dim=-1).mean(dim=1)
-    ...
-    words, probabilities = extract_words_and_probabilities(subwords, subwords_probabilities)
-    return words, probabilities, total_probability
+def calculate_word_probabilities(sentence, n_gram_counts, vocabulary_size, n=3):
+    n_grams = list(ngrams(sentence, n))
+    probabilities = []
+    for n_gram in n_grams:
+        n_gram_count = n_gram_counts.get(n_gram, 0)
+        n_gram_count_1 = n_gram_counts_1.get(n_gram[:-1], 0)
+        probability = (n_gram_count + alpha) / (n_gram_count_1 + alpha * vocabulary_size)
+        probabilities.append((n_gram[-1], probability))
+    return probabilities
```

**Analiza:** potpuno različiti algoritmi i povratni tipovi:
- A: HF LM inference, vraća `(words, probabilities, total_probability)`.
- B: n-gram sa Laplace smoothing, vraća `probabilities` (lista `(word, prob)` tuplova).

**Klasifikacija:** KONFLIKT (različiti modeli, različita semantika, različit povratni tip).
**Preporuka:** **čuvati odvojeno**. Ovo su dva različita eksperimenta — n-gram baseline vs LLM. Nikakvo spajanje nije smisleno; samo preimenovati radi jasnoće:
- A → ostati `calculate_word_probabilities` (ili `calculate_word_probabilities_lm`)
- B → `calculate_word_probabilities_ngram`

---

## 3.7 `find_target_sentence` — 2 kopije, 2 varijante

**Lokacije:**

| idx | Fajl | Linija | Potpis |
|---|---|---|---|
| 00 | `Fetures extraction/text_features_extraction.py` | 43 | `find_target_sentence(sentence, df=target_sentences_df)` |
| 01 | `Transcript - correct/transcription_alignment.py` | 211 | `find_target_sentence(sentence, df)` |

**Diff:**
```diff
-def find_target_sentence(sentence, df=target_sentences_df):
+def find_target_sentence(sentence, df):
     max_similarity = 0
-    target_index = -1
+    target_sentence = ""
-    for index, row in df.iterrows():
+    for _, row in df.iterrows():
         current_similarity = fuzz.ratio(sentence, row['Text'])
         if current_similarity > max_similarity:
             max_similarity = current_similarity
-            target_index = index
+            target_sentence = row['Text']
-    return target_index
+    return target_sentence
```

**Analiza:** logika traženja (fuzzy najbolja podudarnost) je ista, **ali povratna vrijednost je različita**:
- `00`: vraća **index reda** u `df`.
- `01`: vraća **tekst rečenice** (`row['Text']`).

Call-site-ovi apsolutno različito interpretiraju rezultat.

**Klasifikacija:** KONFLIKT (različiti povratni tip).
**Preporuka:** **preimenovati**:
- `00` → `find_target_sentence_index`
- `01` → `find_target_sentence_text`

Spajanje bi uveo argument `return_index=True/False`, ali to je promjena poziva — ostaviti odvojeno dok se ne dogovori.

---

## 3.8 `process_directory` — 2 kopije, 2 varijante

**Lokacije:**

| idx | Fajl | Linija | Potpis |
|---|---|---|---|
| 00 | `Fetures extraction/text_features_extraction.py` | 103 | `process_directory(directory_path, output_path)` |
| 01 | `Prominence/convert_txt_to_lib.py` | 39 | `process_directory(directory)` |

**Diff:** (pokazuje potpuno drugu funkciju)

`00`: gradi DataFrame iz .txt transkripata, čuva .csv per-speaker+emotion. Koristi `pd.DataFrame.append`, `warnings.filterwarnings`, `process_txt_file`, `calculate_word_length`, `get_gender`.

`01`: rekurzivno prolazi kroz folder, za svaki `.txt` poziva `read_transcript_file` + `time_convert`, piše `.lab` fajl. Rekurzivno ulazi u subdirektorijume.

**Analiza:** iste ime, **potpuno različita funkcija** — nijedna linija se ne poklapa. Samo dijele generičko ime.

**Klasifikacija:** KONFLIKT (name collision, ne pravi duplikat).
**Preporuka:** **preimenovati**:
- `00` → `process_directory_to_csv` (ili `build_csv_from_directory`)
- `01` → `convert_txt_directory_to_lab` (ili slično)

---

# 4. METODA KLASE (nije pravi duplikat)

Ove "kopije" su metode različitih klasa — Python ih razlikuje po `self.__class__`. Grupisane su zbog istog `name`-a, ali **nisu kandidati za merge**.

## 4.1 `__init__` — 5 "kopija", 5 klasa

| idx | Fajl | Linija | Klasa | Potpis |
|---|---|---|---|---|
| 00 | `Emotion recognition/audiodataset.py` | 37 | `AudioDataset` | `__init__(self, dataframe, max_length=250)` |
| 01 | `Emotion recognition/mymodel.py` | 14 | `LFLB` (nn.Module) | `__init__(self, in_channels, out_channels, kernel_size=3, pool_size=4, pool_stride=4)` |
| 02 | `Emotion recognition/mymodel.py` | 29 | `MyModel` (nn.Module) | `__init__(self, num_classes)` |
| 03 | `Mel coefficients and surprisals/model.py` | 42 | `MelDataset` | `__init__(self, data)` |
| 04 | `Mel coefficients and surprisals/model.py` | 67 | `TacotronWithSurprisal` (nn.Module) | `__init__(self, vocab_size, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, mel_dim, batch_size)` |

**Klasifikacija:** METODA KLASE.
**Preporuka:** **ostaviti odvojeno**. Ovo su konstruktori potpuno različitih klasa.

## 4.2 `__len__` — 2 "kopije", 2 Dataset klase

| idx | Fajl | Linija | Klasa |
|---|---|---|---|
| 00 | `Emotion recognition/audiodataset.py` | 41 | `AudioDataset` — `return len(self.dataframe)` |
| 01 | `Mel coefficients and surprisals/model.py` | 45 | `MelDataset` — `return len(self.data)` |

**Klasifikacija:** METODA KLASE. **Preporuka:** ostaviti.

## 4.3 `__getitem__` — 2 "kopije", 2 Dataset klase

| idx | Fajl | Linija | Klasa |
|---|---|---|---|
| 00 | `Emotion recognition/audiodataset.py` | 44 | `AudioDataset` — učitava .wav, računa mel spectrogram |
| 01 | `Mel coefficients and surprisals/model.py` | 48 | `MelDataset` — vraća tekst/surprisal/mel tenzore iz DF-a |

**Klasifikacija:** METODA KLASE. **Preporuka:** ostaviti.

## 4.4 `forward` — 3 "kopije", 3 nn.Module klase

| idx | Fajl | Linija | Klasa | Suština |
|---|---|---|---|---|
| 00 | `Emotion recognition/mymodel.py` | 21 | `LFLB` | conv → bn → elu → pool |
| 01 | `Emotion recognition/mymodel.py` | 40 | `MyModel` | LFLB-1..3 → LSTM → FC → softmax |
| 02 | `Mel coefficients and surprisals/model.py` | 92 | `TacotronWithSurprisal` | embed text+surprisal → encoder LSTM → attention → decoder LSTMCell → mel |

**Klasifikacija:** METODA KLASE. **Preporuka:** ostaviti.

---

# 5. Agregirani pregled po folderu

| Folder | Duplikati koje drži | Tipično stanje |
|---|---|---|
| `Additional files after recension/` | IDENTIČNO × 4 (`calculate_*`, `find_subword`, `add_word_type`); KONFLIKT × 4 (`akaike_for_column A`, `calculate_delta_ll A`, `inf_k_model A`, `lookup_features A`) | Stara, "puna" verzija sa generalizacijom (prosody, function); referenca A |
| `Different information measurement parameters/` | IDENTIČNO × 4; KONFLIKT × 4 (varijanta B ili C) | Stripped-down "time" verzija bez function/prosody |
| `Duration Prediction based on Surprisals/` | IDENTIČNO × 3; KONFLIKT × 3 (varijanta C, B `inf_k_model`) | Slično kao "Linear regression" — "standardne" varijante |
| `Linear regression/` | IDENTIČNO × 4; KONFLIKT × 5 (uključuje D `akaike/calculate_delta_ll` sa petljom po emocijama + C `inf_k_model`) | Najviše varijanti. `results.py` je *outlier* sa `emotion` petljom |
| `Split-over effect/` | IDENTIČNO × 4; KONFLIKT × 2 (`add_column_with_surprisal`, `calculate_delta_ll E`) | Split-over `-{k}` kolona schema |
| `Pervious Surprisals/` | IDENTIČNO × 1 (`find_subword`); KONFLIKT × 3 (`lookup_features` E/F) | `surprisal_df` schema, neki sa `target sentence`/`word` malim slovima |
| `Prominence/` | IDENTIČNO × 4 (`find_subword`, `extraxt_parameter_over_emotion` ×3); KONFLIKT × 1 (`lookup_features G` try/except) | `extraxt_` typo × 3, silent try/except u lookup |
| `Surprisal estimation/` | — IDENTIČNO; KONFLIKT × 2 (`calculate_word_probabilities` A vs B + `extract_words_and_probabilities` tokenizer) | LM vs n-gram eksperimenti |
| `Emotion recognition/` | IDENTIČNO × 3 (mel/spec); METODA KLASE × 5 | Čisto + torch modeli |
| `Mel coefficients and surprisals/` | IDENTIČNO × 3; METODA KLASE × 5 | Čisto + Tacotron model |
| `Fetures extraction/` | KONFLIKT × 2 (`find_target_sentence` indeks vs tekst, `process_directory` drugi posao) | Legacy/first-pass |
| `Transcript - correct/` | KONFLIKT × 1 (`find_target_sentence` text-return) | Corrected verzija |

---

# 6. Zaključak i prioriteti za dalje

## 6.1 Sigurno za spajanje (nikakav refactor poziva)

Sljedeće grupe su IDENTIČNO i mogu se spojiti u jedan `utils` modul bez promjene ponašanja:

1. `calculate_log_Likelihood` — 6 kopija → 1 definicija
2. `calculate_aic` — 6 → 1
3. `find_subword` — 4 → 1
4. `get_fixed_length_mel_spectrogram` — 3 → 1
5. `extract_mel_spectrogram` — 2 → 1 (pažljivo sa global `mel_dim`/`fixed_length`)
6. `extraxt_parameter_over_emotion` — 3 → 1 (Prominence folder — lokalni utils)
7. `add_column` — 2 → 1

→ **Ukupno eliminisano duplikata: 26 kopija funkcija → 7 kanonskih.**

## 6.2 Kozmetička razlika (trivijalan merge)

8. `add_word_type` — 3 kopije, razlike samo u imenima varijabli → 1 sa dogovorenim imenom.

## 6.3 Funkcionalne varijacije (spojive samo uz argument)

9. `extract_words_and_probabilities` — razlika u markeru `Ġ` vs `▁`. Može se spojiti sa argumentom `marker=...`, ali to zahtijeva promjenu poziva.

## 6.4 KONFLIKTI — NE SPAJATI automatski

10. `akaike_for_column` — 4 varijante, različiti potpisi i povratni tipovi
11. `calculate_delta_ll` — 5 varijanti, različiti povratni tipovi
12. `inf_k_model` — 3 varijante, jedna podržava više "function" tipova
13. `lookup_features` — 7 varijanti, **različiti schema-ovi kolona u DF-u**
14. `add_column_with_surprisal` — 2 varijante (generic vs split-over format)
15. `calculate_word_probabilities` — 2 potpuno različita algoritma (LM vs n-gram)
16. `find_target_sentence` — vraća index vs tekst
17. `process_directory` — dvije potpuno različite funkcije (csv builder vs .lab writer)

→ Preporuka: **preimenovati** po folderu/semantici prije bilo kakvog refaktora.

## 6.5 Nisu duplikati

18–21. `__init__`, `__len__`, `__getitem__`, `forward` — metode različitih klasa. Ostaviti kako jeste.

---

## Napomena

Ovaj dokument je **samo analiza**. Ništa u kodu nije promijenjeno. Sljedeći korak je definisanje konkretnih proposala (`[P-XXX]`) — npr. `[P-008]` za IDENTIČNO grupu (bezbjedan merge), pa odvojeni proposali za svaki KONFLIKT. Prije toga, treba odlučiti **gdje** utils module postaviti (root `utils/` folder? per-folder `_utils.py`?) i **koja imena** koristiti.

Svaki proposal bi trebao imati:
- spisak fajlova koji se mijenjaju,
- spisak import-linija koje se dodaju,
- garanciju zero-change ponašanja (npr. hash tijela isti prije i nakon),
- plan validacije (import test + postojeći output fajlovi ne smiju se promijeniti).
