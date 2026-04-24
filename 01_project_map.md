# 01 — Mapa projekta (Faza 1)

**Projekat:** Surprisal-Theory-Analysis-of-Emotional-Speech
**Datum:** 2026-04-24
**Režim:** read-only — nijedan kod nije mijenjan.
**Osnova:** `outputs/01_raw_scan.txt` (bulk skener: importi, `def`-ovi, prvih N linija koda, svi I/O pozivi za svaki fajl).

Dokument opisuje *šta je svaki fajl* (1–2 rečenice), kojoj **kategoriji** pripada, i na kraju daje zbirnu tabelu po kategorijama. Uparen je s `01_dependency_overview.md` (duplikati, import graf, top problemi, preporuka).

---

## 1. Kategorije

| Kategorija | Skraćeno | Značenje |
|---|---|---|
| **data_prep** | prep | Skripta gradi/objedinjuje CSV skup (čita sirove inpute, proizvodi training/prominence dataset) |
| **feature_extraction** | feat | Izvlači akustičke ili tekstualne parametre (prozodija, MFCC, POS, dužina riječi, fonetska obilježja) |
| **surprisal_estimation** | surp | Procjena surprisala / informacionih mjera upotrebom LM-ova (BERT, GPT, n-gram, Yugo, Llama, ALBERT, RoBERTa) |
| **modeling** | model | Obučavanje modela — linearna regresija ili neuralne mreže |
| **evaluation** | eval | Računanje metrika (log-likelihood, AIC, Δll, permutation test, baseline) nad već izgrađenim skupom |
| **visualization** | viz | Pravi grafike / prikaze rezultata |
| **utils** | util | Biblioteka funkcija koja se importuje iz drugih skripti |
| **script_orchestration** | orch | „Ljepilo" — premještanje fajlova, preimenovanje, organizacija foldera |
| **exploratory_notebook** | nb | Jupyter/Colab notebook (`.ipynb`); najčešće istraživački, nije dio reprodukcibilnog pipeline-a |
| **unclear** | ? | Nisam u stanju kategorisati bez dodatnog konteksta |

Napomena: mnogi fajlovi rade **dvije stvari istovremeno** (npr. grade dataset *i* računaju metriku *i* plotuju). U tim slučajevima dajem primarnu kategoriju + sekundarnu u opisu.

---

## 2. Fajlovi po folderima

### 2.1 `Additional files after recension/` (14 fajlova)

Posljednja iteracija (feb 2025) — vjerovatno revizija nakon recenzije rada.

| Fajl | Kat. | Opis |
|---|---|---|
| BERT_models_UniContext_surprisal_estimation.ipynb | nb + surp | Colab notebook — unigram kontekstualni surprisal preko BERT/RoBERTa/ALBERT. Piše u Google Drive; ne koristi se lokalnim putanjama. |
| RoBERTa and ALBERT surprisal estimation.ipynb | nb + surp | Colab notebook — surprisal za RoBERTa i ALBERT. Isto: samo Google Drive. |
| additional_functions_results.py | eval + viz | Iterira 5 kolona surprisala (GPT-2, BERT, BERTic, ngram-3, Yugo) i za svaku računa `inf_k_model` + `calculate_delta_ll_old`; plotuje završni grafik. |
| additional_models_results.py | eval + viz | Varijanta gornjeg za dodatne modele (konteksti uni/bi/tri, različiti k). Isti okvir: učitaj `training_data.csv`, prolazi kolone, plot. |
| analysis_accross_individual_speakers.py | eval + viz | Per-govornik analiza: `inf_k_model` + `akaike_for_column` za svakog govornika pojedinačno, plus seaborn-ski heatmap/boxplot. |
| analysis_accross_individual_words.py | eval + viz | Per-riječ analiza — slično, ali grupiše po riječi umjesto po govorniku. |
| baseline_model.py | eval | Linearna regresija bez surprisala nad `training_data.csv` (59 linija). **Identičan sadržaj kao `Linear regression/baseline_model.py`** (diff = 0). |
| baseline_model_prosody.py | eval | Baseline za prosody ciljnu varijablu (`energy_data.csv`); koristi `lookup_features` da doda log-prob kolonu. |
| build_dataset.py | prep | Gradi `training_data.csv`: iterira govornike, koristi `my_functions.lookup_features` i `add_word_type`, spaja sa surprisalima iz 5 LM-ova + POS tags + folds. **Važi uvjetno — ulazi žive u `../podaci/` siblingu.** |
| build_prominence_datasets.py | prep | Varijanta za prominence — čita `prosody 1 0 0` transkripte, spaja sa surprisalima. Ima lokalno definisan `find_subword`. |
| individual_speaker_graphs.py | viz + eval | Plotuje rezultate po govorniku (delta-ll preko svih k i surprisala). |
| my_functions.py | util | **Glavni utils za ovu iteraciju (204 linije).** Definiše: `inf_k_model`, `calculate_log_Likelihood`, `calculate_aic`, `akaike_for_column`, `calculate_delta_ll`, `calculate_delta_ll_old`, `lookup_features`, `add_word_type`, `most_similar_sentence_index` (+ ugniježđeno `common_chars`). |
| pos_tag_calculate.py | feat | Koristi `stanza` da doda POS tag za svaku riječ u `target_sentences.csv`. Izlaz: `pos_tags.csv`. |
| prosody_results_plots.py | viz + eval | Plotovi rezultata za prozodijske modele (koristi `my_functions.inf_k_model` + `calculate_delta_ll_old`). |

### 2.2 `Different information measurement parameters/` (29 fajlova)

Istraživanje alternativnih informacionih mjera (kontekstualna entropija, information value, adjusted surprisal) — sav sadržaj (nov 2024 – jun 2025).

| Fajl | Kat. | Opis |
|---|---|---|
| adjusted_surprisal_information_values_results.py | eval | Računa delta-ll i permutacioni test za adjusted_surprisal + information_value. |
| analize_po_govornicima.py | eval + viz | Po-govornik analiza (srpski naziv). Učitava različite surprisale (bertic/bert/gpt), akaike per column. |
| analysis_accross_individual_words.py | eval | Po-riječ delta-ll. **Isto ime kao u `Additional files after recension/`, drugi sadržaj.** |
| analysis_across_sentence_position.py | eval | Delta-ll po poziciji u rečenici. |
| baseline_results.py | eval | Dodaje baseline kolonu u dataset (36 linija, kratko). |
| build_dataset.py | prep | **BUG**: linija 8 → `from my_function import lookup_features` (nedostaje `s`; fajl se zove `my_functions.py`). Fajl kako je zapisan **ne može biti importovan**. Vidjeti `01_dependency_overview.md` → UNCLEAR-06. |
| build_dataset_for_different_model_embeddings.py | prep | Dataset za 4 modela (RoBERTa/ALBERT/BERTic/BERT) — kontekstualni i non-kontekstualni embeddingi. |
| build_dataset_for_embeddings.py | prep | Generička varijanta za embedding features (information_value ili adjusted_surprisal). |
| distribucija_gresaka.py | eval + viz | Prikaz raspodjele grešaka (residuala) po surprisal/model paru. |
| fonetic parameters.py | feat + eval | Računa `fonetic_model` nad fonetskim obilježjima + delta-ll. (Napomena: **ime fajla ima razmak**.) |
| individual_speaker_graphs.py | viz + eval | Plotovi po govorniku. Isto ime kao u `Additional files…` folderu — drugi sadržaj (koristi `add_column_with_surprisal`). |
| information_and_distance_functions.py | util | Biblioteka torch funkcija: `non_context_embedding`, `extract_words_and_embeddings`, `extract_words_and_probabilities`, `calculate_word_information_values`, `calculate_word_adjusted_surprisal`. |
| iv_embedding_results.py | eval + viz | Rezultati za Information Value sa embedding kontekstom/bez konteksta. |
| my_functions.py | util | **Drugi utils modul (288 linija, 22. nov 2024).** Definiše: `lookup_features`, `calculate_log_Likelihood`, `calculate_aic`, `akaike_for_column`, `calculate_delta_ll`, `paired_permutation_test`, `add_column_with_surprisal`, `fonetic_model`, `add_column`. **Neke funkcije imaju ISTO IME ali DRUGAČIJE POTPISE nego u `Additional files after recension/my_functions.py`.** |
| plot_results.py | viz | Grafik: Surprisal vs Contextual Entropy. |
| plot_results_for_different_models.py | viz + eval | Plot delta-ll + permutation test preko 4 modela. |
| surprisal_vs_entropy.py | eval | Direktno poređenje surprisala i entropije — delta-ll za oba. |
| `parameters estimations/` | *(subfolder)* | Sadržaj za procjenu parametara (embeddings, information value). |
| …/Adjusted Surprisals Embeddings.ipynb | nb + surp | Colab — adjusted surprisal sa embedding kontekstom (GPT-2). |
| …/Adjusted Surprisals.ipynb | nb + surp | Colab — bazna adjusted surprisal implementacija. |
| …/Cleaning vocabulary data.ipynb | nb + prep | Čišćenje `wordlist_classlawiki_sr_*.csv` (ćirilica→latinica, regex). |
| …/Fonetic features.ipynb | nb + feat | Izgradnja fonetskih obilježja (vokali, sonanti, konsonanti po mjestu tvorbe). |
| …/GPT_2_contextual_entropy.ipynb | nb + surp | Kontekstualna entropija GPT-2. |
| …/Information Value Embeddings BERT.ipynb | nb + surp | Information Value — BERT embeddings. |
| …/Information Value Embeddings BERTic.ipynb | nb + surp | Information Value — BERTic embeddings. |
| …/Information Value Embeddings RoBERTa.ipynb | nb + surp | Information Value — RoBERTa embeddings. |
| …/Information Value Embeddings.ipynb | nb + surp | Information Value — generička (GPT-2). |
| …/Information Value.ipynb | nb + surp | Bazna Information Value implementacija. |
| …/Yugo_GPT_contextual_entropy.py | surp | Kontekstualna entropija preko YugoGPT-a (147 linija). |
| …/embedding_information_value.py | surp | Information Value nad embeddingom (GPT-2). |

### 2.3 `Duration Prediction based on Surprisals/` (5 fajlova)

Predikcija trajanja izgovora riječi — glavni tok rada (avg 2024).

| Fajl | Kat. | Opis |
|---|---|---|
| baseline_model.py | eval | Baseline linearna regresija (60 linija). Sličan `Linear regression/baseline_model.py`, ali čita `general_data.csv`. |
| build_surprisal_datasets.py | prep | Kombinuje surprisale 5 LM-ova u jedan dataset. **Lokalno definiše `lookup_features`** (nije iz `my_functions`). |
| final_graphs.py | viz | Finalni grafici za rad — čita pred-izračunate `{surprisal}_results.csv`. |
| surprisal_results.py | eval | **Samodovoljan: lokalno definiše** `inf_k_model`, `calculate_log_Likelihood`, `calculate_aic`, `akaike_for_column`, `calculate_delta_ll`. Ne importuje `my_functions`. |
| transform_data_into_dataframe.py | prep | Objedinjuje sve govornike u jedan DataFrame, dodaje fold-ove i word_type. Lokalno definiše `add_word_type`, `lookup_freq`. |

### 2.4 `Emotion recognition/` (9 fajlova, bez obrisanih artifakata)

PyTorch CNN za prepoznavanje emocija iz audio spektrograma. Samostalan podprojekat.

| Fajl | Kat. | Opis |
|---|---|---|
| audiodataset.py | util + feat | `AudioDataset` (`torch.utils.data.Dataset`) + `create_dataloader`. Mel-spektogram se računa pri učitavanju. |
| create_csv_for_testing_synthetisized_data.py | prep | Generiše test-CSV za sintetičke TTS podatke (GPT-2 generisane rečenice). |
| create_datasets.py | prep | Train/val/test split iz `data_mono/` foldera (koristi `sklearn.model_selection.train_test_split`). |
| emotion_recognition_model.py | model | **Orkestrator treniranja** — uvozi `MyModel` i `create_dataloader`, radi tr/val petlju, snima `training_log.csv`, čuva checkpoint. (Napomena: **dupliciran sadržaj sa `training.py` u istom folderu**.) |
| loss_function.py | viz | Plot train/val loss krive iz `model/training_log.csv` (54 linije). |
| mymodel.py | model | Arhitektura: `LFLB` (local feature learning block) + `MyModel` (CNN nad mel spektogramom). |
| prosody_parameters_and_mfcc.py | feat | Poredi promjenu prozodijskih parametara i mel koeficijenata za različite emocije; koristi `pydub` za vremensku kompresiju/ekspanziju. |
| testing.py | eval + viz | Učita model checkpoint, evaluira na test setu, crta confusion matrix. |
| training.py | model | **Skoro identičan `emotion_recognition_model.py`** — tr/val petlja. Razlika: koristi `torch.cuda.is_available()` (drugi fajl to ne radi). |

### 2.5 `Fetures extraction/` *(tipfeler: „Features")* (1 fajl)

| Fajl | Kat. | Opis |
|---|---|---|
| text_features_extraction.py | feat + prep | Prolazi corrected transcripts foldere; za svaku riječ pridružuje: `word_position`, `word_length`, `speaker_gender`, `target_sentence` (preko fuzzywuzzy fuzz match). Proizvodi `data.csv` koji kasnije koristi cijeli pipeline. |

### 2.6 `Forced alignment/` (2 fajla + kredencijal koji je obrisan u Fazi 0)

Vremensko poravnavanje transkripta sa audiom preko Google Cloud Speech-to-Text API-ja.

| Fajl | Kat. | Opis |
|---|---|---|
| novosadska_baza_podataka.py | orch + feat | Šalje audio na Google Cloud Speech-to-Text (trebao je GCP service-account ključ, sada uklonjen). Upisuje `.txt` transkripte. |
| Resampling/resampling.py | orch | Resempluje wav fajl (`pydub`) na ciljni sample rate. Pokreće se za jedan specifičan snimak koji je imao dupli SR. |

### 2.7 `Generate graphs/` (8 fajlova)

Grafici iz rada — f0 (frekvencija), RMS (energija), speech rate, surprisal po rečenicama.

| Fajl | Kat. | Opis |
|---|---|---|
| audio_files_transcirpition_vizualization.py | viz | Prikaz jednog audio talasa sa segmentacijom i transkripcijom na nivou riječi (latinica→ćirilica mapping). |
| data_analysis.py | viz + prep | Statistike podataka korišćene u poglavlju „Analiza podataka" (+ generiše `folds` kolonu). |
| frequency_over_time.py | feat | Izvlači f0 sekvence po govorniku/emociji iz wav fajlova (librosa) → `f0_per_speaker_emotion.csv`. |
| frequency_over_time_plots.py | viz | Plot f0 serija usrednjenih po emociji. Dijeli `padding_sequence` kod sa ostalim `*_over_time*.py`. |
| rms_over_time.py | feat | Ista struktura kao `frequency_over_time.py` ali za RMS (energiju). |
| rms_over_time_plots.py | viz | Plot RMS serija. |
| speech_rate_over_time.py | feat + viz | Računa brzinu izgovaranja po vremenu (hash po govorniku/emociji) i plotuje. |
| surprisal_per_sentences.py | viz | Grafik surprisala (5 LM-ova) po target rečenicama + unigram frekvencija. |

### 2.8 `Linear regression/` (5 fajlova)

Glavni tok rada u radu za linearnu regresiju.

| Fajl | Kat. | Opis |
|---|---|---|
| baseline_model.py | eval | Baseline LR bez surprisala (59 linija). **Identično sa `Additional files after recension/baseline_model.py`.** |
| build_dataset.py | prep | Najopsežniji `build_dataset` (215 linija) — lokalno definiše `lookup_features` i `add_word_type`. Spaja Yugo, GPT, GPT-3, BERT, BERTic, ngram-2/3/4/5. |
| final_graphs.py | viz + eval | **Lokalno definiše** `inf_k_model`, `calculate_log_Likelihood`, `calculate_aic`, `akaike_for_column`, `calculate_delta_ll`. 320 linija; glavni grafici rada. |
| residual_distribution.py | viz + eval | Residual plot za predikciju trajanja — „neznatni rezultati, nisu korišćeni u radu" (citirano iz docstringa). |
| results.py | eval + viz | Računa rezultate i plotuje; **lokalno definiše iste 5 funkcija** kao `final_graphs.py` — ali sa **različitim potpisom** `akaike_for_column(column_name, model_name, …)` i `calculate_delta_ll(surprisal, k, emotion_data, std_data)`. |

### 2.9 `Mel coefficients and surprisals/` (6 fajlova)

Istražuje vezu surprisala i mel koeficijenata — predikcija MFCC-a iz teksta i surprisala (Tacotron-stil).

| Fajl | Kat. | Opis |
|---|---|---|
| Model_surprisal.ipynb | nb + model | Colab notebook — Tacotron-with-surprisal model (725 linija). Glavni istraživački fajl. |
| build_dataset.py | prep | Priprema podataka: iterira transkripte, pridružuje surprisale, čuva vocabulary (keras Tokenizer/pad_sequences), upisuje `general_data.csv`. |
| calculate_mel_spectrum.py | feat | Računa mel spektar za sve audio fajlove upotrebom librose; snima u isti DataFrame. |
| mel_spectrum_predict.py | orch | **Prazan/stub fajl (10 linija)** — samo docstring, nema koda. UNCLEAR-08. |
| model.py | model + eval | `CustomDataset` (torch) + `TacotronWithSurprisal` (encoder→decoder→mel). 306 linija. Čuva best checkpoint. |
| results.py | eval + viz | Prikaz rezultata — poređenje 3 eksperimenta („Results random", „Results keras", „Results surprisal"). |

### 2.10 `Pervious Surprisals/` *(tipfeler: „Previous")* (5 fajlova)

Efekat surprisala prethodnih riječi (lag-1, lag-2…) na prozodiju trenutne.

| Fajl | Kat. | Opis |
|---|---|---|
| build_dataset.py | prep | Gradi dataset gdje su uz svaki red dodati surprisali prethodnih N riječi (`calculate_former_surprisal`). Lokalno definiše `lookup_features` ali **sa drukčijim potpisom** (`surprisal_df` umjesto `freq_df`). |
| conjoint_data.py | prep | Spaja prominence dataset sa surprisal datasetom — koristi svoj lokalni `lookup_features`. |
| correlation_coefficient.py | eval | Računa Pearsonov koeficijent korelacije i R² između surprisal_prev i prominence varijabli. |
| plot_results.py | viz | Vizualizacija korelacionih rezultata. |
| prominence_build_dataset.py | prep | Sličan `Prominence/prominence_build_dataset.py` ali sa drugačijim prosody folderom (`prosody 1 0 0` za frekvenciju, `prosody` za energiju). |

### 2.11 `Prominence/` (10 fajlova)

Korelacija između wavelet-prozodije i surprisala.

| Fajl | Kat. | Opis |
|---|---|---|
| convert_txt_to_lib.py | orch | Konvertuje transkript `.txt` fajlove u `.lab` format za wavelet_gui aplikaciju. |
| correct_names.py | orch | Ispravlja nestandardne nazive `.prom` fajlova u finalnom folderu (`govornik_emocija_naziv.prom`). |
| correlation_results_representation.py | viz | Prikazuje korelacione rezultate wavelet transforma i surprisala različitih modela. |
| librosa_estimated_parameters.py | feat | Računa f0/energiju/duration preko librose i spaja sa 5 surprisala. Lokalno definiše `find_subword` i `lookup_features`. |
| move_data_to_final_folder.py | orch | Premješta sve fajlove iz raznih foldera u jedan (`all_files`) za wavelet_gui batch processing. |
| organize_data_for_wavelet_gui.py | orch | Priprema organizaciju wav + lib fajlova za wavelet_gui. |
| plot energy.py | viz + eval | Plotuje prozodijske parametre po emociji; ima linearnu regresiju surprisal→energija. (**ime sa razmakom**) |
| plot frequency.py | viz + eval | Isto za frekvenciju (f0). |
| plot speеch time.py | viz + eval | Isto za speech time. **Ime sadrži ćirilično „е" (U+0435)** — u Fazi 0 evidentirano kao untracked u gitu. |
| prominence_build_dataset.py | prep | Gradi `prominence_data.csv` — spaja wavelet prominence sa surprisalima 5 LM-ova. |

### 2.12 `Split-over effect/` (5 fajlova)

Analogno Duration Prediction-u, ali se ispituje efekat „pretakanja" prozodije iz prethodne riječi.

| Fajl | Kat. | Opis |
|---|---|---|
| baseline_model.py | eval | Baseline LR sa `add_column` helperom — **DRUGAČIJI** od ostalih `baseline_model.py` (79 linija, ima lokalnu `add_column`). |
| build_surprisal_datasets.py | prep | Iterira 5 surprisal CSV-ova, dodaje kolone shift-ovane za -1, -2 riječi, čuva u posebnom folderu. |
| final_graphs.py | viz | Finalni grafici za split-over; struktura identična `Duration Prediction based on Surprisals/final_graphs.py`. |
| surprisal_results.py | eval | Računa rezultate — **ponovo lokalno definiše** iste 5 funkcija (`calculate_log_Likelihood`, `calculate_aic`, `akaike_for_column`, `calculate_delta_ll`, `add_column_with_surprisal`). |
| transform_data_into_dataframe.py | prep | Kratka skripta (33 linije): `shift(1)` nad `length` i `log probability` kolonama; briše granice između rečenica. |

### 2.13 `Surprisal estimation/` (12 fajlova)

Primarna tačka za procjenu surprisala — većina je Google Colab `.ipynb` (pokretano na GPU-u na Colabu).

| Fajl | Kat. | Opis |
|---|---|---|
| BERT surprisal estimation.ipynb | nb + surp | Colab — BERT masked LM surprisal. |
| BERTic_surprisal_estimation.ipynb | nb + surp | Colab — BERTic (srpski BERT), 649 linija. |
| GPT-2 surprisal estimation.ipynb | nb + surp | Colab — GPT-2 (causal). |
| GPT-3 surprisal estimation.ipynb | nb + surp | Colab — `EleutherAI/gpt-neo-2.7B` (imenovano „GPT-3" iako je GPT-Neo). |
| Unigram frequency.ipynb | nb + feat + surp | Colab — klasla lematizacija + unigram frekvencija kao baseline „surprisal". |
| Yugo GPT-3 surprisal estimation.py | surp | Skripta (ne notebook) — surprisal preko `gordicaleksa/YugoGPT` (HuggingFace). 141 linija. (**ime sa razmakom**) |
| llama.py | surp | Analogno — `meta-llama/Llama-2-7b`. Struktura skoro identična `Yugo GPT-3 surprisal estimation.py` (duplicirani `extract_words_and_probabilities`, `calculate_word_probabilities`). |
| ngram surprisal estimation/lematization.py | prep | classla pipeline za lematizaciju skupa nad kojim se uči n-gram. |
| ngram surprisal estimation/lematization_target_sentences.py | prep | Lematizacija target rečenica. |
| ngram surprisal estimation/make_train_dataset.py | prep | Objedinjuje sve `.csv` fajlove iz `../../podaci/ngram datasets/` u jedan training set. |
| ngram surprisal estimation/surprisal_estimation_n_gram_model.py | surp | Računa n-gram surprisal (n=2,3,4,5) upotrebom `nltk.ngrams` + Laplace smoothing. 148 linija. |
| ngram surprisal estimation/word_frequency.py | viz | Histogram frekvencije riječi iz training seta (za prikaz u radu). |

### 2.14 `Transcript - correct/` (2 fajla)

Korekcija ASR transkripta pomoću fuzzy matching-a nad poznatim target rečenicama.

| Fajl | Kat. | Opis |
|---|---|---|
| list_of_uterrances.py | orch + prep | Iz svakog transkript fajla izdvaja prvu rečenicu → pravi `target_sentences.csv` i `wrong_transcription.csv`. |
| transcription_alignment.py | feat + prep | Najveći fajl u folderu (356 linija). 11 funkcija za fuzzy align: `align_words`, `pair_words_with_difference`, `align_endpoints`, `align_middlepoints`, `align_transcript`, `find_target_sentence`, `correct_sentence`, `correct_transcript`, `process_folder`. Koristi `fuzzywuzzy.fuzz` + `difflib.SequenceMatcher`. |

---

## 3. Zbirna tabela po kategorijama

| Kategorija | Broj fajlova | Komentar |
|---|---:|---|
| exploratory_notebook | 18 | Svi `.ipynb` — većina u `Surprisal estimation/` i `Different information measurement parameters/parameters estimations/`. Pokretani na Colabu. |
| data_prep | ~22 | `build_dataset*.py`, `transform_data*`, `lematization*`, transkript čišćenje. Puno duplikacije — vidjeti `01_dependency_overview.md`. |
| evaluation | ~20 | `*_results.py`, `baseline_model.py`, delta-ll / AIC / permutacioni testovi. Najveća koncentracija funkcijskih duplikata. |
| visualization | ~18 | `final_graphs.py`, `plot_*.py`, `data_analysis.py`, `*_over_time_plots.py`. |
| surprisal_estimation | ~9 | Python skripte i notebook-ovi koji zovu LM i računaju surprisal/information value. |
| feature_extraction | ~10 | Prozodija (librosa), mel spektar, fonetika, POS, dužina riječi. |
| modeling | ~5 | `mymodel.py`, `emotion_recognition_model.py`, `training.py`, `Mel coefficients/model.py`, `Model_surprisal.ipynb`. |
| script_orchestration | ~7 | `move_data_to_final_folder.py`, `correct_names.py`, `convert_txt_to_lib.py`, `Resampling/resampling.py`, itd. |
| utils | 3 | `Additional files…/my_functions.py` (204 lin), `Different info…/my_functions.py` (288 lin), `Different info…/information_and_distance_functions.py` (228 lin). |
| unclear | 1 | `Mel coefficients and surprisals/mel_spectrum_predict.py` — 10 linija, samo docstring. |

Napomena: zbir je ~113 jer su mnogi fajlovi u više kategorija (primarnu sam računao, sekundarnu nisam).

---

## 4. UNCERTAIN / UNCLEAR za ovu fazu

- **UNCLEAR-08** — `Mel coefficients and surprisals/mel_spectrum_predict.py` ima samo docstring, 10 linija. Je li ovo *planirana* skripta koja nikada nije napisana, ili je sadržaj negdje izgubljen?
- **UNCLEAR-09** — Da li je `training.py` u `Emotion recognition/` aktivna verzija (ima CUDA provjeru), a `emotion_recognition_model.py` stara verzija? Ili obratno? Za refaktorisanje bitno.
- **UNCLEAR-10** — `Additional files after recension/` sadrži iteraciju revizije. Da li to znači da je **taj folder** „tačka istine" za najnoviju verziju modela (i da su drugi folderi — npr. `Linear regression/`, `Duration Prediction based on Surprisals/` — zastarjeli), ili ga treba shvatiti kao *dodatak* postojećoj strukturi?
- **UNCLEAR-11** — `Pervious Surprisals/lookup_features` ima drugačiji potpis (`surprisal_df`) nego sve ostale verzije (`freq_df`). Da li su to semantički ista funkcija s različitim imenima argumenata ili **različite funkcije koje su slučajno dobile isto ime**? Kritično za odluku da li mogu biti spojene.

Za sva 4 pitanja preporučujem diskusiju prije Faze 2.

---

## 5. Šta slijedi

`01_dependency_overview.md` (prateći dokument) sadrži:
- graf importa + *kandidate za konsolidaciju* (ko koga importuje, šta se lokalno redefiniše),
- listu duplikata sa identifikovanim divergencijama potpisa,
- **top 3–5 najproblematičnijih dijelova** projekta,
- **preporuku odakle da počnemo refaktorisanje** (najveći impact/najmanji rizik),
- **konkretan prvi korak** — koji folder / koju funkciju prvu.
