# P-012 DIO 2 — KONFLIKT funkcije inventar (skip rationale)

> **Status: SKIPPED** (zero-change pravilo postuje se).
> Dokumentovan inventar 10 konfliktnih funkcija (isto ime, razlicita tijela
> kroz foldere) za buducu referencu. Ova faza se NE radi automatski jer
> svaka varijanta sluzi razlicitoj domen-specificnoj svrsi.

## Zasto skip

`calculate_delta_ll` je **vec** dispatcher-ovan kroz P-009 (lokalni
`mode`-based dispatcher u svakom folderu). Ostale 9 funkcija dijele
ime ali imaju legitimno razlicita tijela jer svaka folder-pipeline
radi nesto razlicito sa svojim podacima:

- **`lookup_features`** (7 varijanti) — svaki folder mapira razlicit
  set kolumna iz svojih CSV fajlova (prominence vs. surprisal vs.
  embedding features).
- **`akaike_for_column`** (4 varijante) — razlikuju se u baseline
  konvenciji i u nacinu agregacije po grupama.
- **`inf_k_model`** (3 varijante) — razlicite dimenzionalnosti i
  feature-set-ovi po folderu.
- **`extract_words_and_probabilities`** (2 varijante) — jedna za
  `information_metrics/` (radi sa per-token probabilites), druga za
  surprisal estimation modele (LLaMA / YugoGPT / N-gram).
- **`add_word_type`** (2 varijante) — razlicite POS-tag konvencije.
- **`add_column_with_surprisal`** (2 varijante) — razlicite mapping
  semantike izmedju surprisal modela.
- **`calculate_word_probabilities`** (2 varijante) — neuronski
  modeli vs. n-gram model (potpuno razlicit kod).
- **`find_target_sentence`** / **`process_directory`** — apsolutno
  razlicite funkcije koje samo dijele ime; nisu konflikt po
  semantici, samo po imenu.

Pretvaranje ovih u zajednicki dispatcher zahtijevalo bi:
- razumijevanje svake call-site signature i ocekivane semantike,
- preimenovanje varijanti (krsi zero-change pravilo),
- testiranje na realnim podacima (van P-012 obima).

## Inventar (puni hash report)

### lookup_features (9 defs, 7 variants)

| hash | folder | fajl |
|------|--------|------|
| `5c5539a3...` | additional_analysis | my_functions.py |
| `ff739c3b...` | duration_prediction | build_surprisal_datasets.py |
| `26837d9c...` | information_metrics | my_functions.py |
| `63fb6045...` | linear_regression | build_dataset.py |
| `f15655da...` | previous_surprisals | build_dataset.py |
| `3d4173e4...` | previous_surprisals | conjoint_data.py + correlation_coefficient.py |
| `ce9e2525...` | prominence | librosa_estimated_parameters.py + prominence_build_dataset.py |

### akaike_for_column (6 defs, 4 variants)

| hash | folderi |
|------|---------|
| `510a5b32...` | additional_analysis/my_functions.py |
| `d301b43b...` | duration_prediction/surprisal_results.py + linear_regression/regression_plots.py |
| `8f8b13b3...` | information_metrics/my_functions.py + split_over_effect/surprisal_results.py |
| `95647d30...` | linear_regression/regression_results_analysis.py |

### calculate_delta_ll (6 defs, 5 variants) — **vec dispatcher (P-009)**

| hash | folder/fajl |
|------|-------------|
| `c8e16089...` | additional_analysis/my_functions.py |
| `110608b5...` | duration_prediction/surprisal_results.py + linear_regression/regression_plots.py |
| `8cd8be5d...` | information_metrics/my_functions.py |
| `422f6f8a...` | linear_regression/regression_results_analysis.py |
| `8905beda...` | split_over_effect/surprisal_results.py |

### inf_k_model (5 defs, 3 variants)

| hash | folderi |
|------|---------|
| `1ecc643e...` | additional_analysis/my_functions.py |
| `8d4227eb...` | duration_prediction/surprisal_results.py + linear_regression/regression_plots.py |
| `b3edaa04...` | linear_regression/regression_results_analysis.py + linear_regression/residual_distribution.py |

### extract_words_and_probabilities (4 defs, 2 variants)

| hash | folderi |
|------|---------|
| `517b23c3...` | information_metrics/information_and_distance_functions.py |
| `d5014326...` | information_metrics/parameter_estimations/yugo_gpt_contextual_entropy.py + surprisal_estimation/llama.py + surprisal_estimation/yugo_gpt3_surprisal_estimation.py |

### add_word_type (3 defs, 2 variants)

| hash | folderi |
|------|---------|
| `131d9b53...` | additional_analysis/my_functions.py + linear_regression/build_dataset.py |
| `14258497...` | duration_prediction/transform_data_into_dataframe.py |

### calculate_word_probabilities (3 defs, 2 variants)

| hash | folderi |
|------|---------|
| `8af9d397...` | surprisal_estimation/llama.py + surprisal_estimation/yugo_gpt3_surprisal_estimation.py |
| `b15e6237...` | surprisal_estimation/ngram_surprisal_estimation/surprisal_estimation_n_gram_model.py |

### find_target_sentence (2 defs, 2 variants) — **isto ime, razlicita svrha**

| hash | folder |
|------|--------|
| `b6f2b3ff...` | feature_extraction/text_features_extraction.py |
| `69c2ed8e...` | transcript_correction/transcription_alignment.py |

### process_directory (2 defs, 2 variants) — **isto ime, razlicita svrha**

| hash | folder |
|------|--------|
| `32a1073c...` | feature_extraction/text_features_extraction.py |
| `38e7f7d0...` | prominence/convert_txt_to_lib.py |

### add_column_with_surprisal (2 defs, 2 variants)

| hash | folder |
|------|--------|
| `b6eda584...` | information_metrics/my_functions.py |
| `ae119e0a...` | split_over_effect/surprisal_results.py |

## Zakljucak

P-012 DIO 1 (consolidacija identicnih funkcija) je dovrsen u 4 koraka:
- `utils/stats_utils.py` (calculate_log_Likelihood + calculate_aic, 5 kopija → 1)
- `utils/text_utils.py` (find_subword, 3 kopije → 1)
- `utils/audio_utils.py` (get_fixed_length_mel_spectrogram, 2 kopije → 1)
- `utils/analysis_utils.py` (extraxt_parameter_over_emotion, 1 kopija premjestena)

DIO 2 (dispatcheri za KONFLIKT) ostaje kao buduca odluka. P-009 vec
ima dispatcher za `calculate_delta_ll`. Za ostale 9 funkcija, refactor
zahtijeva domain-specific odluke koje ne pripadaju zero-change ciklusu.
