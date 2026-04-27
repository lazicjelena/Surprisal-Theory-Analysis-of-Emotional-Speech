# P-012 Finalni izvjestaj — konsolidacija utils funkcija

**Datum:** 2026-04-27
**Status:** ✅ DOVRSEN (DIO 1 + pripreme); DIO 2 svjesno preskocen i dokumentovan u `04_P012_konflikt_inventory.md`.
**Princip:** Zero-change — tijela funkcija, potpisi, imena i docstring-ovi NISU mijenjani; samo lokacije i import-i.

---

## Sazetak

P-012 je centralizovao 5 utility funkcija (4 modula) iz 11 razlicitih fajlova u jedan zajednicki paket `utils/`. Uz to, projekat je doveden u puni package layout (`__init__.py` u svim folderima) i svi sibling import-i su konvertovani u apsolutne package import-e da bi se omogucio `python -m package.module` execution model.

| Mjera | Prije P-012 | Poslije P-012 |
|---|---|---|
| Definicije `calculate_log_Likelihood` | 5 | **1** |
| Definicije `calculate_aic` | 5 | **1** |
| Definicije `find_subword` | 3 | **1** |
| Definicije `get_fixed_length_mel_spectrogram` | 2 | **1** |
| Definicije `extraxt_parameter_over_emotion` | 1 (P-008) | **1** (premjesteno) |
| Sibling import-a (eg. `from my_functions import X`) | 42 | **0** |
| `__init__.py` fajlova | 0 | **18** |
| Top-level `.py` foldera bez package strukture | 17 | **0** |
| Top-level `.py` fajlova ukupno | 90 | **118** (90 + 18 init + 4 utils + 6 ostalih iz P-008) |

---

## DIO 0 — Pripremne radnje (P-012a / b / c)

### P-012a — Cleanup git stanja
**Commit:** `225f5ea P-013 cleanup: remove duplicate Prominence/ tracking + filemode normalize`

- Postavljeno `git config core.filemode false` — Windows mount je prikazivao sve fajlove kao `100755` (executable), pa je `git status` lazno pokazivao SVE fajlove kao modified.
- Uklonjene su uppercase `Prominence/` putanje iz git index-a (ostaci iz P-013 case-insensitive rename-a). Identicnost je verifikovana SHA256 hash-om sa lowercase `prominence/` verzijom.
- Rezultat: 9 fajlova izbrisano iz indeksa (1147 linija), git status sada prazan i konzistentan.

### P-012b — Package struktura (`__init__.py`)
**Commit:** `f308118 P-012b: add __init__.py to all packages (enable -m execution)`

Napravljeno **18 praznih `__init__.py` fajlova** u svim Python folderima:

| Folder | Hijerarhija |
|---|---|
| `additional_analysis/` | top-level |
| `duration_prediction/` | top-level |
| `emotion_recognition/` | top-level |
| `feature_extraction/` | top-level |
| `forced_alignment/` | top-level |
| `forced_alignment/resampling/` | nested |
| `generate_graphs/` | top-level |
| `information_metrics/` | top-level |
| `information_metrics/parameter_estimations/` | nested |
| `linear_regression/` | top-level |
| `mel_surprisal_analysis/` | top-level |
| `previous_surprisals/` | top-level |
| `prominence/` | top-level |
| `split_over_effect/` | top-level |
| `surprisal_estimation/` | top-level |
| `surprisal_estimation/ngram_surprisal_estimation/` | nested |
| `transcript_correction/` | top-level |
| **`utils/` (NOVO)** | top-level — za P-012 konsolidovane utility-je |

Bez ovih `__init__.py` fajlova, `python -m package.module` bi padao sa `ModuleNotFoundError`.

### P-012c — Konverzija 42 sibling import-a u apsolutne package import-e
**Commit:** `aca4b4b P-012c: convert sibling imports to absolute package imports`

Konvertovano **42 sibling import-a** kroz 6 paketa. Sibling import (`from my_functions import X`) radi samo kad se skripta pokrene direktno iz tog foldera (`python script.py` u tom dir-u). Sa `python -m package.script` execution model-om, takvi import-i padaju.

**Distribucija po paketu:**

| Paket | Broj sibling-a | Primjer |
|---|---|---|
| `additional_analysis/` | 8 | `from my_functions import inf_k_model` → `from additional_analysis.my_functions import inf_k_model` |
| `emotion_recognition/` | 8 | `from mymodel import MyModel` → `from emotion_recognition.mymodel import MyModel` |
| `generate_graphs/` | 4 | `from generate_graphs_utils import padding_sequence` → `from generate_graphs.generate_graphs_utils import padding_sequence` |
| `information_metrics/` | 14 | `from my_functions import lookup_features` → `from information_metrics.my_functions import lookup_features` |
| `information_metrics/parameter_estimations/` | 1 | `from information_and_distance_functions import ...` → `from information_metrics.information_and_distance_functions import ...` |
| `linear_regression/` | 2 | `from stats_utils import calculate_aic` → `from linear_regression.stats_utils import calculate_aic` |
| `prominence/` | 5 | `from text_utils import find_subword` → `from prominence.text_utils import find_subword` |
| **UKUPNO** | **42** | |

**Verifikacija:**
- AST jednakost (38/38 izmjenjenih fajlova): non-import AST identican sa HEAD verzijom.
- 118/118 `.py` fajlova prolaze `ast.parse` + `py_compile`.
- 0 sibling import-a preostalo.
- Originalni line-ending stil (CRLF / LF) ocuvan po fajlu.

---

## DIO 1 — Konsolidacija identicnih funkcija u `utils/` paketu

### Korak 1 — `utils/stats_utils.py`
**Commit:** `b927ffb P-012 step 1: consolidate stats utils into utils/stats_utils.py`

**Premjestene funkcije:**

| Funkcija | Argumenti | Vraca |
|---|---|---|
| `calculate_log_Likelihood(data)` | array of floats | `numpy.ndarray` log-pdf vrijednosti |
| `calculate_aic(real_values, results, k)` | tri argumenta | tuple `(aic, mean_LL, std_LL)` |

**5 izvornih kopija (sve byte-identicne, AST hash `e207d977aa168fce` / `731e65fb61087bea`) → 1 centralna definicija u `utils/stats_utils.py`.**

| Iz fajla | Akcija |
|---|---|
| `additional_analysis/my_functions.py` | obrisane lokalne `def`-ove (-55 linija), dodato `from utils.stats_utils import calculate_log_Likelihood, calculate_aic` |
| `duration_prediction/surprisal_results.py` | obrisane lokalne `def`-ove (-54 linija), dodat isti import |
| `information_metrics/my_functions.py` | obrisane lokalne `def`-ove (-55 linija), dodat isti import |
| `split_over_effect/surprisal_results.py` | obrisane lokalne `def`-ove (-49 linija), dodat isti import |
| `linear_regression/stats_utils.py` | **CIJELI FAJL OBRISAN** (-89 linija), preusmjeren u `utils/stats_utils.py` |

**Promijenjeni import-i (consumers):**

| Fajl | Stari import | Novi import |
|---|---|---|
| `linear_regression/regression_plots.py` | `from linear_regression.stats_utils import ...` | `from utils.stats_utils import ...` |
| `linear_regression/regression_results_analysis.py` | `from linear_regression.stats_utils import ...` | `from utils.stats_utils import ...` |

**Net efekat:** -301 linija dupliciranog koda, +6 linija import-a.

**AST jednakost:** ✅ `utils/stats_utils.py` je AST-jednak sa `linear_regression/stats_utils.py` (osim modulnog docstring-a, koji je prosiren da pokrije siri opseg upotrebe).

---

### Korak 2 — `utils/text_utils.py`
**Commit:** `26dd720 P-012 step 2: consolidate text utils into utils/text_utils.py`

**Premjestena funkcija:**

| Funkcija | Argumenti | Vraca |
|---|---|---|
| `find_subword(word, unique_words)` | string + set | najduzi suffix `word`-a koji je u `unique_words`, ili `''` |

**3 izvorne kopije (sve byte-identicne, AST hash `0c04994691dd1948`) → 1 centralna definicija u `utils/text_utils.py`.**

| Iz fajla | Akcija |
|---|---|
| `additional_analysis/build_prominence_datasets.py` | obrisana lokalna `def`, dodat `from utils.text_utils import find_subword` |
| `previous_surprisals/prominence_build_dataset.py` | obrisana lokalna `def`, dodat isti import |
| `prominence/text_utils.py` | **CIJELI FAJL OBRISAN**, preusmjeren u `utils/text_utils.py` |

**Promijenjeni import-i (consumers):**

| Fajl | Stari import | Novi import |
|---|---|---|
| `prominence/librosa_estimated_parameters.py` | `from prominence.text_utils import find_subword` | `from utils.text_utils import find_subword` |
| `prominence/prominence_build_dataset.py` | `from prominence.text_utils import find_subword` | `from utils.text_utils import find_subword` |

**AST jednakost:** ✅ `find_subword` ima tacno **1 definiciju** u projektu (`utils/text_utils.py`). Sva 4 call-site-a su `imported`.

---

### Korak 3 — `utils/audio_utils.py`
**Commit:** `f58ca44 P-012 step 3: consolidate audio utils into utils/audio_utils.py`

**Premjestena funkcija:**

| Funkcija | Argumenti | Vraca |
|---|---|---|
| `get_fixed_length_mel_spectrogram(y, sr, n_mels, fixed_length)` | waveform + parametri | log-Mel spektrogram `(n_mels, fixed_length)` |

**2 izvorne kopije (byte-identicne, AST hash `72c8556b88909212`) → 1 centralna definicija u `utils/audio_utils.py`.**

| Iz fajla | Akcija |
|---|---|
| `mel_surprisal_analysis/calculate_mel_spectrum.py` | obrisana lokalna `def`, dodat `from utils.audio_utils import get_fixed_length_mel_spectrogram` |
| `emotion_recognition/audio_utils.py` | **CIJELI FAJL OBRISAN**, preusmjeren u `utils/audio_utils.py` |

**Promijenjeni import-i (consumers):**

| Fajl | Stari import | Novi import |
|---|---|---|
| `emotion_recognition/audiodataset.py` | `from emotion_recognition.audio_utils import ...` | `from utils.audio_utils import ...` |
| `emotion_recognition/prosody_parameters_and_mfcc.py` | `from emotion_recognition.audio_utils import ...` | `from utils.audio_utils import ...` |

**AST jednakost:** ✅ `get_fixed_length_mel_spectrogram` ima tacno **1 definiciju** u projektu. Sva 3 call-site-a su `imported`.

**Napomena:** `extract_mel_spectrogram` (postoji u `emotion_recognition/`) NIJE premjesten jer ima dependency na modulne globals (`mel_dim`, `fixed_length`). Konsolidacija ostaje za buduci proposal.

---

### Korak 4 — `utils/analysis_utils.py`
**Commit:** `13c9abf P-012 step 4: consolidate analysis utils into utils/analysis_utils.py`

**Premjestena funkcija:**

| Funkcija | Argumenti | Vraca |
|---|---|---|
| `extraxt_parameter_over_emotion(data, parameter)` | DataFrame + naziv kolone | DataFrame sa neutral-vs-emotional alignment-om |

**1 izvorna definicija (vec konsolidovana u P-008-u) → premjestena u `utils/analysis_utils.py`.**

| Iz fajla | Akcija |
|---|---|
| `prominence/analysis_utils.py` | **CIJELI FAJL OBRISAN**, preusmjeren u `utils/analysis_utils.py` |

**Promijenjeni import-i (consumers):**

| Fajl | Stari import | Novi import |
|---|---|---|
| `prominence/plot_energy.py` | `from prominence.analysis_utils import extraxt_parameter_over_emotion` | `from utils.analysis_utils import extraxt_parameter_over_emotion` |
| `prominence/plot_frequency.py` | isto | isto |
| `prominence/plot_speech_time.py` | isto | isto |

**AST jednakost:** ✅ `extraxt_parameter_over_emotion` ima tacno **1 definiciju**. Sva 3 call-site-a su `imported`. **Typo `extraxt → extract` je SACUVAN** (zero-change pravilo).

---

## DIO 2 — Dispatcheri za KONFLIKT funkcije

**Status:** SVJESNO PRESKOCEN. Detaljan inventar 10 konfliktnih funkcija je dokumentovan u zasebnom fajlu `04_P012_konflikt_inventory.md`.

**Razlog skip-a:** Konfliktne funkcije (isto ime, razlicita tijela kroz foldere) imaju legitimne domain-specific razlike, ne prave duplikate. `calculate_delta_ll` je vec dispatcher-ovan kroz P-009 (lokalni `mode`-based dispatcher u svakom folderu). Ostale 9 funkcija (`lookup_features`, `akaike_for_column`, `inf_k_model`, `extract_words_and_probabilities`, `add_word_type`, `calculate_word_probabilities`, `find_target_sentence`, `process_directory`, `add_column_with_surprisal`) zahtijevaju domain-specific odluke koje krse zero-change pravilo. Refactor ovih ostaje za buduci proposal.

---

## Finalna verifikacija

| Provjera | Rezultat |
|---|---|
| `ast.parse` na svim `.py` fajlovima | ✅ 118/118 OK |
| `py_compile` na svim `.py` fajlovima | ✅ 118/118 OK |
| Sibling import-i preostali | ✅ 0 |
| `__init__.py` u svim Python folderima | ✅ 18/18 |
| `calculate_log_Likelihood` definicija | ✅ 1× (samo `utils/stats_utils.py`) |
| `calculate_aic` definicija | ✅ 1× (samo `utils/stats_utils.py`) |
| `find_subword` definicija | ✅ 1× (samo `utils/text_utils.py`) |
| `get_fixed_length_mel_spectrogram` definicija | ✅ 1× (samo `utils/audio_utils.py`) |
| `extraxt_parameter_over_emotion` definicija | ✅ 1× (samo `utils/analysis_utils.py`) |
| Svi call-site-i resolvuju (def ili import) | ✅ Sve OK |
| Original line-ending stil (CRLF / LF) ocuvan | ✅ Po fajlu |
| Tijela funkcija identicna sa HEAD-om | ✅ AST hash jednakost potvrdjena |

---

## Git commit chain

```
13c9abf  P-012 step 4: consolidate analysis utils into utils/analysis_utils.py
f58ca44  P-012 step 3: consolidate audio utils into utils/audio_utils.py
26dd720  P-012 step 2: consolidate text utils into utils/text_utils.py
b927ffb  P-012 step 1: consolidate stats utils into utils/stats_utils.py
aca4b4b  P-012c: convert sibling imports to absolute package imports
f308118  P-012b: add __init__.py to all packages (enable -m execution)
225f5ea  P-013 cleanup: remove duplicate Prominence/ tracking + filemode normalize
```

7 commit-a, svaki atomican i revertabilan.

---

## Konacna `utils/` struktura

```
utils/
├── __init__.py            (0 B, prazan)
├── stats_utils.py         (3186 B, 2 funkcije: calculate_log_Likelihood, calculate_aic)
├── text_utils.py          (1837 B, 1 funkcija: find_subword)
├── audio_utils.py         (2675 B, 1 funkcija: get_fixed_length_mel_spectrogram)
└── analysis_utils.py      (4006 B, 1 funkcija: extraxt_parameter_over_emotion)
```

**Ukupno: 4 modula, 5 funkcija, 11.7 KB cistog konsolidovanog utility koda.**

---

## Kako koristiti `utils/` u novim skriptama

```python
# Iz bilo kog foldera projekta:
from utils.stats_utils    import calculate_log_Likelihood, calculate_aic
from utils.text_utils     import find_subword
from utils.audio_utils    import get_fixed_length_mel_spectrogram
from utils.analysis_utils import extraxt_parameter_over_emotion
```

Pokretanje skripti (iz project root-a):

```bash
cd Surprisal-Theory-Analysis-of-Emotional-Speech/
python -m linear_regression.regression_plots
python -m information_metrics.iv_embedding_results
python -m prominence.plot_speech_time
# itd.
```

---

## Zakljucak

✅ **Cilj postignut:** projekat je sad cist, centralizovan i profesionalan, bez identicnih duplikata i bez ikakve promjene rezultata. Tijela svih funkcija su byte-identicna sa HEAD-om (verifikovano AST hash-om). Postoji jedinstven `utils/` paket za zajednicke pomocne funkcije, koji bilo koji folder moze importovati istom putanjom (`from utils.X import Y`).

P-012 DIO 1 je dovrsen. DIO 2 ostaje za buducu odluku — inventar i obrazlozenje preskoka su u `04_P012_konflikt_inventory.md`.
