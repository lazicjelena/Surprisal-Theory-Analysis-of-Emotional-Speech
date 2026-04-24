# 01 — Dependency overview + top problemi + preporuka (Faza 1)

**Projekat:** Surprisal-Theory-Analysis-of-Emotional-Speech
**Datum:** 2026-04-24
**Pratnja:** `01_project_map.md` (per-fajl opisi i kategorije)
**Režim:** i dalje read-only — **nijedan kod nije mijenjan, niti se predlaže njegova izmjena u ovom dokumentu**. Ovo je samo dijagnoza.

---

## 1. Struktura zavisnosti (uvodno)

Projekat ima **ravnu strukturu bez Python paketa**: nijedan folder nema `__init__.py`, imena foldera sadrže razmake (pa ne mogu biti modul-imena), i skripte se pokreću kao samostalne (Spyder/terminal). Rezultat:

- Import se dešava **samo iz fajlova u istom folderu** (`from my_functions import …`, `from mymodel import …`, `from audiodataset import …`).
- **Nema ni jednog importa preko granice foldera** u cijelom repozitorijumu. Svaka komunikacija između foldera ide preko **fajlova-artefakata u `../podaci/`**.
- Zato je zajednička logika kopirana (ne importovana) u svaki folder gdje je potrebna.

Zavisnosti među foldera teku preko ugovora o fajlovima (input → output CSV), ne preko koda. To je *činjenica stanja*, ne automatski problem — ali objašnjava zašto je kôd-duplikat brojan.

## 2. Lokalni importi (intra-folder)

Samo fajlovi koji zaista zovu nešto iz drugog `.py` u istom folderu:

### 2.1 `Additional files after recension/`

Centar je **`my_functions.py`** (204 lin). Importuju ga:

```
my_functions.py  ◄── additional_functions_results.py      (inf_k_model, calculate_delta_ll_old)
                ◄── additional_models_results.py          (inf_k_model, calculate_delta_ll_old)
                ◄── analysis_accross_individual_speakers.py  (inf_k_model, akaike_for_column)
                ◄── analysis_accross_individual_words.py  (inf_k_model, akaike_for_column)
                ◄── build_dataset.py                      (lookup_features, add_word_type)
                ◄── build_prominence_datasets.py          (lookup_features, most_similar_sentence_index)
                ◄── individual_speaker_graphs.py          (inf_k_model, calculate_delta_ll_old)
                ◄── prosody_results_plots.py              (inf_k_model, calculate_delta_ll_old)
```

8 fajlova importuje `my_functions`.

### 2.2 `Different information measurement parameters/`

Centar je **`my_functions.py`** (288 lin — *različit fajl*, vidi §4):

```
my_functions.py  ◄── adjusted_surprisal_information_values_results.py (add_column_with_surprisal, paired_permutation_test, calculate_delta_ll)
                ◄── analize_po_govornicima.py            (add_column_with_surprisal, akaike_for_column)
                ◄── analysis_accross_individual_words.py (add_column_with_surprisal, calculate_delta_ll)
                ◄── analysis_across_sentence_position.py (add_column_with_surprisal, calculate_delta_ll)
                ◄── baseline_results.py                  (add_column)
                ◄── build_dataset_for_different_model_embeddings.py (lookup_features, add_column, add_column_with_surprisal)
                ◄── build_dataset_for_embeddings.py      (lookup_features)
                ◄── distribucija_gresaka.py              (add_column_with_surprisal, akaike_for_column)
                ◄── fonetic parameters.py                (add_column, fonetic_model, paired_permutation_test, calculate_delta_ll)
                ◄── individual_speaker_graphs.py         (add_column_with_surprisal, calculate_delta_ll)
                ◄── iv_embedding_results.py              (add_column_with_surprisal, paired_permutation_test, calculate_delta_ll)
                ◄── plot_results_for_different_models.py (add_column_with_surprisal, paired_permutation_test, calculate_delta_ll)
                ◄── surprisal_vs_entropy.py              (add_column_with_surprisal, calculate_delta_ll)
```

13 fajlova. **Ali pažnja**: `build_dataset.py` u istom folderu ima **bag** na liniji 8:

```python
from my_function import lookup_features   # <-- "my_function", BEZ 's'
```

Fajl se zove `my_functions.py`, pa ovaj import pada s `ModuleNotFoundError`. **Kako je napisano, ovaj `build_dataset.py` ne može biti pokrenut.** (P-000 kandidat.)

`parameters estimations/embedding_information_value.py` importuje `information_and_distance_functions.calculate_word_information_values` — to je jedini intra-subfolder import u cijelom repou.

### 2.3 `Emotion recognition/`

```
mymodel.py       ◄── emotion_recognition_model.py  (MyModel)
                ◄── training.py                    (MyModel)
                ◄── testing.py                     (MyModel)

audiodataset.py  ◄── emotion_recognition_model.py  (create_dataloader)
                ◄── training.py                    (create_dataloader)
                ◄── testing.py                     (create_dataloader)
```

Urednije od drugih — tri fajla koriste dvije biblioteke. Ali **`emotion_recognition_model.py` i `training.py` su skoro identični** (§3.E).

### 2.4 Svi ostali folderi

`Duration Prediction based on Surprisals/`, `Linear regression/`, `Mel coefficients and surprisals/`, `Pervious Surprisals/`, `Prominence/`, `Split-over effect/`, `Generate graphs/`, `Surprisal estimation/`, `Transcript - correct/`, `Forced alignment/`, `Fetures extraction/`:

**Nijedan intra-folder import.** Svaki fajl je samodovoljan (uzima logiku koja mu treba lokalno).

---

## 3. Duplikati — potvrđeno čitanjem fajlova

Sva preklapanja dolje su **sigurna** (iz `01_raw_scan.txt`). Gdje postoji divergencija potpisa, to je eksplicitno naznačeno, jer znači da spajanje nije trivijalno.

### A. `lookup_features` — **8+ definicija, 2 različita semantička potpisa**

| Folder/fajl | Potpis | Komentar |
|---|---|---|
| Additional files after recension/my_functions.py:113 | `(data, freq_df, column_name)` | Kanonska verzija |
| Different information measurement parameters/my_functions.py:14 | `(data, freq_df, column_name)` | Isti potpis |
| Duration Prediction based on Surprisals/build_surprisal_datasets.py:12 | `(data, freq_df, column_name)` | Isti potpis, ali **lokalna kopija** umjesto importa |
| Linear regression/build_dataset.py:15 | `(data, freq_df, column_name)` | Isti potpis, lokalna kopija |
| Prominence/librosa_estimated_parameters.py:145 | `(data, freq_df, column_name)` | Isti potpis, lokalna kopija |
| Prominence/prominence_build_dataset.py:118 | `(data, freq_df, column_name)` | Isti potpis, lokalna kopija |
| **Pervious Surprisals/build_dataset.py:16** | `(data, surprisal_df, column_name)` | **DRUGI argument se zove `surprisal_df`** |
| **Pervious Surprisals/conjoint_data.py:12** | `(data, surprisal_df, column_name)` | Isti „drugi" potpis |
| **Pervious Surprisals/correlation_coefficient.py:18** | `(data, surprisal_df, column_name)` | Isti „drugi" potpis |

UNCLEAR-11: Varijante s `surprisal_df` — jesu li identične po logici kao one sa `freq_df`, samo sa drugim imenom argumenta, ili **rade drugačiji lookup** (npr. nad surprisal-kolonom umjesto frekvencijske)? Odgovor **mora postojati prije** nego što razmatramo spajanje.

### B. `calculate_log_Likelihood` — 6 identičnih kopija

Pojavljuje se u: `Additional files…/my_functions.py`, `Different info…/my_functions.py`, `Duration Prediction…/surprisal_results.py`, `Linear regression/final_graphs.py`, `Linear regression/results.py`, `Split-over effect/surprisal_results.py`.

Iz glava: svih 6 koristi `np.mean`, `np.std`, `norm.logpdf`. Po strukturi identične. (Formalna diff-provjera se može uraditi u Fazi 2 — **za sada smatrati identičnim, ali potvrditi prije spajanja**.)

### C. `calculate_aic` — 6 kopija

Iste lokacije kao B. Standardna AIC formula (3 linije). Po svemu sudeći identične. *Provjeriti diff u Fazi 2.*

### D. `akaike_for_column` — 6 kopija, **3 različita potpisa**

| Lokacija | Potpis |
|---|---|
| Additional files…/my_functions.py:72 | `(data, prominence, model_name, baseline_model='baseline')` |
| Different info…/my_functions.py:65 | `(data, model_name, baseline_model='baseline')` |
| Duration Prediction…/surprisal_results.py:62 | `(data, model_name, baseline_model='baseline')` |
| Linear regression/final_graphs.py:62 | `(data, model_name, baseline_model='baseline')` |
| **Linear regression/results.py:61** | `(column_name, model_name, baseline_model='baseline')` | **← prvi argument se zove `column_name`** |
| Split-over effect/surprisal_results.py:27 | `(data, model_name, baseline_model='baseline')` |

Tri varijante — versija sa `prominence` argumentom je najšira; `column_name` varijanta je najuža; ostale su srednje. **Nisu drop-in zamjenjive.**

### E. `calculate_delta_ll` — 6 kopija, **4 različita potpisa**

| Lokacija | Potpis |
|---|---|
| Additional files…/my_functions.py:82 | `(data, surprisal, k, emotion_data, std_data, prominence='time', function='power')` |
| Different info…/my_functions.py:74 | `(data, model_name, baseline="baseline -3")` |
| Duration Prediction…/surprisal_results.py:70 | `(data, surprisal_name, k)` |
| Linear regression/final_graphs.py:70 | `(data, surprisal_name, k)` |
| **Linear regression/results.py:74** | `(surprisal, k, emotion_data, std_data)` | bez `data` |
| Split-over effect/surprisal_results.py:36 | `(data, surprisal_name, k)` |

Najkompleksniji slučaj. „Isto ime, različita funkcija" — **spajanje NIJE moguće bez prethodnog razgovora o kom potpisu je „pravi"**.

### F. `inf_k_model` — 5 kopija, 2 potpisa

| Lokacija | Potpis |
|---|---|
| Additional files…/my_functions.py:14 | `(df, k, surprisal, prosody='time', function='power')` |
| Duration Prediction…/surprisal_results.py:17 | `(df, k, surprisal)` |
| Linear regression/final_graphs.py:17 | `(df, k, surprisal)` |
| Linear regression/results.py:17 | `(df, k, surprisal)` |
| Linear regression/residual_distribution.py:17 | `(df, k, surprisal)` |

Verzija sa 5 argumenata je nadskup verzije sa 3 — vjerovatno se može unifikovati sa default vrijednostima. *Ipak potvrditi da `function='power'` ne mijenja ponašanje osnovnog modela.*

### G. `baseline_model.py` — 4 kopije, 2 stvarne varijante

- Additional files…/baseline_model.py (59 lin)
- Linear regression/baseline_model.py (59 lin) — **identičan gornjem** (isti `head`, isti import set, iste I/O operacije)
- Duration Prediction…/baseline_model.py (60 lin) — skoro identičan; razlika je ulazni CSV (`training_data.csv` vs `general_data.csv`)
- **Split-over effect/baseline_model.py (79 lin)** — drugačiji: ima lokalnu `add_column(df, k=0)` funkciju, filtrira `warnings`, drugačija struktura

### H. `my_functions.py` u dva foldera — različit kod s istim imenom

| | Additional files after recension/my_functions.py | Different information measurement parameters/my_functions.py |
|---|---|---|
| Linije | 204 | 288 |
| Datum (iz docstringa) | 19. feb 2025 | 22. nov 2024 |
| Zajedničke funkcije (po imenu) | `lookup_features`, `calculate_log_Likelihood`, `calculate_aic`, `akaike_for_column`, `calculate_delta_ll` | (iste) |
| Samo ovdje | `inf_k_model`, `calculate_delta_ll_old`, `add_word_type`, `most_similar_sentence_index` | `paired_permutation_test`, `add_column_with_surprisal`, `fonetic_model`, `add_column` |
| Divergencija potpisa | `akaike_for_column(data, prominence, model_name, …)` | `akaike_for_column(data, model_name, …)` |
| Divergencija potpisa | `calculate_delta_ll(data, surprisal, k, emotion_data, std_data, …)` | `calculate_delta_ll(data, model_name, baseline="baseline -3")` |

**Ovo su dva suštinski različita modula** koji su slučajno oba nazvani `my_functions`. Ime ne odražava sadržaj.

### I. `padding_sequence` — 4 kopije u `Generate graphs/`

`frequency_over_time.py`, `frequency_over_time_plots.py`, `rms_over_time.py`, `rms_over_time_plots.py` — sve 4 počinju identičnom funkcijom `padding_sequence(f0_all_files)`. Najčistija prilika za izdvajanje zajedničkog koda u cijelom projektu.

### J. `get_fixed_length_mel_spectrogram` — 3 kopije

`Emotion recognition/audiodataset.py:16`, `Emotion recognition/prosody_parameters_and_mfcc.py:24`, `Mel coefficients and surprisals/calculate_mel_spectrum.py:34`. Kratka funkcija (~15 linija), ali bi se mogla izdvojiti.

### K. `find_subword` — 4 kopije

`Additional files…/build_prominence_datasets.py:77`, `Pervious Surprisals/prominence_build_dataset.py:84`, `Prominence/librosa_estimated_parameters.py:113`, `Prominence/prominence_build_dataset.py:86`.

### L. `extraxt_parameter_over_emotion` — **3 kopije, sa tipfelerom u imenu**

U `Prominence/plot energy.py`, `Prominence/plot frequency.py`, `Prominence/plot speеch time.py`. Ime je tipfeler („extraxt" umjesto „extract").

### M. „Dva fajla koja treniraju isti model"

`Emotion recognition/emotion_recognition_model.py` (122 lin) i `Emotion recognition/training.py` (119 lin) — ista petlja treniranja. Razlika: `training.py` provjerava `torch.cuda.is_available()`. UNCLEAR-09.

### N. `extract_words_and_probabilities` + `calculate_word_probabilities`

Duplirano doslovno između `Surprisal estimation/Yugo GPT-3 surprisal estimation.py` i `Surprisal estimation/llama.py`. Iste funkcije, razlika je samo u tome koji HuggingFace model se učitava.

---

## 4. Top 5 najproblematičnijih dijelova (rangirano po impactu)

### #1 — Dva konfliktujuća `my_functions.py` s istim imenom funkcija ali različitim potpisima

**Gdje:** `Additional files after recension/my_functions.py` (204 lin) i `Different information measurement parameters/my_functions.py` (288 lin).

**Zašto je problem:** Oba definišu `lookup_features`, `calculate_log_Likelihood`, `calculate_aic`, `akaike_for_column`, `calculate_delta_ll` — a **potpisi `akaike_for_column` i `calculate_delta_ll` nisu isti**. Što god čita `my_functions`, ponaša se drukčije u zavisnosti od foldera iz kog se pokreće. Ako se ikada fajlovi iz jednog foldera pomjere u drugi (ili neko pokuša da unifikuje u paket), rezultati tihih grešaka su praktično zagarantovani.

**Impact:** Visoki — utiče na ukupno 21 fajl u dva foldera.

**Rizik popravke:** Srednji — zahtijeva pažljivo razgraničenje koja verzija funkcije je prava.

### #2 — Lanac od 6 duplikata evaluacionih funkcija (`calculate_log_Likelihood`, `calculate_aic`, `akaike_for_column`, `calculate_delta_ll`, `inf_k_model`)

**Gdje:** Pojavljuju se u 6 foldera: `Additional files…`, `Different info…`, `Duration Prediction…`, `Linear regression` (×2 — `final_graphs.py` + `results.py`), `Split-over effect`.

**Zašto je problem:** Ovo su **funkcije koje proizvode brojeve koji idu u rad**. Kad god neko promijeni logiku u jednoj kopiji, ostalih 5 kopija tiho nastavljaju sa starom logikom. Trenutno imamo **4 različita potpisa za `calculate_delta_ll`** i **3 za `akaike_for_column`** — to znači da se jedan od njih ili 1) nikad ne poziva, ili 2) proizvodi drugačije brojeve od ostalih.

**Impact:** Kritičan za **naučnu valjanost rezultata**. Ako imaš u radu dva poglavlja koja oba koriste „AIC", a jedno koristi verziju iz `Linear regression/results.py:61` a drugo `Additional files…/my_functions.py:72`, onda si reportovala dvije različite metrike pod istim imenom.

**Rizik popravke:** Srednji — ali **popravka zahtijeva prvo eksperimentalno provjeriti** da li dva poziva iste funkcije sa istim ulazima daju iste izlaze (to nije dokazano bez testa).

### #3 — `build_dataset.py` pojavljuje se 7 puta, uz jednu (minimum) sigurnu bug-putanju

**Gdje:** `Additional files after recension/build_dataset.py` (140 lin), `Different information measurement parameters/build_dataset.py` (78 lin — **s bug-om `from my_function`**), `Duration Prediction based on Surprisals/build_surprisal_datasets.py` (86 lin), `Linear regression/build_dataset.py` (215 lin), `Mel coefficients and surprisals/build_dataset.py` (212 lin), `Pervious Surprisals/build_dataset.py` (167 lin), `Split-over effect/build_surprisal_datasets.py` (37 lin).

**Zašto je problem:** Svaka od ovih skripti sama „tumači" kako se spajaju govornici, surprisali, POS, folds. Ako je u jednoj varijanti fold-assignment drugačiji nego u drugoj, rezultati dva poglavlja nisu poredivi. Plus — `Different info…/build_dataset.py` ima sigurnu syntax-level grešku (`my_function` umjesto `my_functions`) koja ga čini **nepokretljivim** u trenutnom stanju.

**Impact:** Visok — to je tačka ulaza u svaki pipeline.

**Rizik popravke:** Visok — svaki od 7 fajlova ima specifične ulaze/izlaze koji se ne smiju promijeniti bez reprodukcione provjere.

### #4 — 4 kopije `padding_sequence` u `Generate graphs/` (naj-čišća prilika, najmanji rizik)

**Gdje:** `frequency_over_time.py`, `frequency_over_time_plots.py`, `rms_over_time.py`, `rms_over_time_plots.py`.

**Zašto je problem:** Ista funkcija, doslovno kopirana 4 puta. Očigledan poziv za ekstrakciju u `generate_graphs_utils.py` (ili sl.).

**Impact:** Nizak po obimu, ali **visok po „pedagoškom" efektu** — to je savršena prva vježba refaktorisanja jer se radi o samoj jedinoj zajedničkoj funkciji u već uređenom folderu.

**Rizik popravke:** **Najniži u cijelom projektu.** Funkcije su kratke, deterministične, bez zavisnosti.

### #5 — `Emotion recognition/`: dva fajla za isti training loop (`emotion_recognition_model.py` vs `training.py`)

**Gdje:** `Emotion recognition/emotion_recognition_model.py` (122 lin) i `Emotion recognition/training.py` (119 lin).

**Zašto je problem:** Oba fajla rade istu stvar — treniranje CNN-a za prepoznavanje emocija — sa malim razlikama u detaljima (CUDA check, gdje se snima checkpoint). Neko ko čita ovaj folder ne zna koji fajl je „pravi". Vjerovatno je jedan zastario a nije obrisan.

**Impact:** Srednji (isključivo organizacioni — ne utiče na brojeve u radu).

**Rizik popravke:** Nizak — **ali prvo treba pitati: koji fajl je korišćen da se generišu rezultati u radu?**

---

## 5. Preporuka odakle početi refaktorisanje

Kad postoji toliko duplikata sa divergentnim potpisima, **prva pobjeda mora biti nisko-rizična i vidljiva** da se povjerenje u proces izgradi prije nego što se krene u kompleksnije izmjene. Zato ne preporučujem da se krene od najvećeg problema (#1 i #2).

Predlažem sljedeći redoslijed:

### Faza 2-A (PRVI KORAK — najniži rizik): `Generate graphs/padding_sequence`

- **Jedan folder, 4 fajla, jedna funkcija.**
- Nema ulaza u `my_functions`, nema divergencije potpisa, nema naučne neizvjesnosti.
- **Proizvodnja brojeva se ne mijenja** (funkcija je deterministična i čisto tehnička — samo padduje listu do maksimalne dužine).
- Uspješna izvedba ove vježbe postavlja obrazac za ostatak: (a) napraviti `generate_graphs_utils.py` u istom folderu, (b) staviti `padding_sequence` tamo, (c) 4 fajla menjaju se minimalno (`from generate_graphs_utils import padding_sequence`), (d) verifikacija = **generisani grafici su piksel-za-piksel isti** kao prije.

### Faza 2-B (DRUGI KORAK): identifikacija „kanonskog" `my_functions.py`

Prije bilo kakve akcije na problemima #1 i #2 — **diskusija** o tome:

1. Koji folder iz §2 sadrži **posljednje rezultate koji su u radu**? („Additional files after recension" ime sugeriše — *after recension* = **poslije recenzije**, tj. najnovija iteracija — ali treba potvrditi.)
2. Da li su `my_functions.py` razlike u potpisu **namijerne** (različite metodologije za različite eksperimente), ili slučajne?
3. Koja od 6 kopija `calculate_delta_ll` je „prava"?

Na ova 3 pitanja mora odgovor dati autor (tj. ti), jer **ja iz koda samog ne mogu rekonstruisati naučnu namjeru**. Bez odgovora bilo kakvo spajanje može pokvariti rezultate.

### Faza 2-C (TREĆI KORAK): samodovoljni eval fajlovi

Nakon što znamo koja verzija `calculate_delta_ll` / `akaike_for_column` je kanonska, možemo predložiti **zamjenu lokalnih kopija u**:
- `Duration Prediction based on Surprisals/surprisal_results.py`
- `Linear regression/final_graphs.py`
- `Linear regression/results.py`
- `Split-over effect/surprisal_results.py`

sa importom iz zajedničkog modula. **Svaki od ovih koraka mora biti prijedlog (P-…) sa eksperimentalnom verifikacijom**: pokrenuti skriptu prije i poslije, uporediti čim je moguće (brojevi u output CSV-u).

### Faze 2-D i dalje

Ostatak (`build_dataset` duplikati, `lookup_features` za „surprisal_df" varijante u `Pervious Surprisals/`, dupli training fajlovi u `Emotion recognition/`, ostali slučajevi) — tek kada je prethodno uspješno završeno i kada imamo povjerenje u proces.

---

## 6. Konkretan prvi korak — prijedlog P-007

Format prati master-prompt (P-XXX, sa poljima Type/Files/Proposal/Rationale/Risk/Verification/Status).

> **[P-007] Ekstraktovati `padding_sequence` iz 4 fajla u `Generate graphs/` u zajednički modul**
>
> - **Type:** Refactor (move code, no logic change)
> - **Files:**
>   - nov: `Generate graphs/generate_graphs_utils.py` (sadrži samo `padding_sequence`)
>   - izmjene: `Generate graphs/frequency_over_time.py`, `…/frequency_over_time_plots.py`, `…/rms_over_time.py`, `…/rms_over_time_plots.py` → uklanjanje lokalne definicije i dodavanje `from generate_graphs_utils import padding_sequence`
> - **Current state:** 4 identična tijela funkcije kopirana kroz 4 fajla.
> - **Proposal:** Izdvojiti u `generate_graphs_utils.py`. **Ne mijenjati tijelo funkcije ni jedan bit** — samo prekopirati doslovno i ukloniti duplikate.
> - **Rationale:** Najmanji mogući refactor-korak koji (a) smanjuje duplikat, (b) uspostavlja obrazac „utils po folderu" koji ćemo koristiti i u narednim fazama, (c) ne dira naučne brojeve.
> - **Risk:** Vrlo nizak. Jedini način da se pokvari nešto je ako Python path ne nalazi novi modul (riješava se time što je u istom folderu — Python ga nalazi automatski kad se skripta pokrene iz tog foldera, kako je i sada slučaj).
> - **Verification:** Pokrenuti po jedan od 4 fajla (najlakše: `frequency_over_time_plots.py` pošto samo čita postojeći CSV i pravi plot). Plotovi poslije refaktora moraju biti piksel-identični. Ako imaš stari plot kao PNG, uporedi ih; ako ne, vizuelni pregled je dovoljan.
> - **Status:** predloženo — čekam odobrenje.

Ako ti ovo odgovara, reci „P-007 prihvaćeno" i napravit ću fajlove. Ako prije toga želiš da razgovaramo o koracima Faze 2-B (kanonska verzija `my_functions`), reci koji folder sadrži „glavnu" verziju koda ili me uputi kako da to sam provjerim.

---

## 7. UNCLEAR za ovu fazu

- **UNCLEAR-08** — `Mel coefficients and surprisals/mel_spectrum_predict.py`: 10 linija, samo docstring, bez koda. Namjerno prazna stub-skripta ili zagubljen sadržaj?
- **UNCLEAR-09** — Da li je „pravi" training fajl u `Emotion recognition/` → `training.py` (ima CUDA check) ili `emotion_recognition_model.py`? Drugi treba obrisati nakon odgovora.
- **UNCLEAR-10** — Folder `Additional files after recension/` — da li je to **tačka istine za rad poslije revizije** (tj. ostali folderi su zastarjeli), ili je *dodatak*?
- **UNCLEAR-11** — `Pervious Surprisals/lookup_features(data, surprisal_df, column_name)`: isti semantika kao kanonska `lookup_features(data, freq_df, column_name)`, samo s drugim imenom kolone, ili **različita funkcija**? Utiče na sve kasnije unifikacije.
- **UNCLEAR-12** — `Different information measurement parameters/build_dataset.py:8` → `from my_function import lookup_features` (typo). Ovaj fajl **trenutno ne može biti pokrenut**. Pitanje: je li korišćen u nekom obliku u proizvodnji rezultata (pa treba biti hitno popravljen jer imaš proces koji pada), ili je zastario i može biti obrisan?

Ovih 5 pitanja ne moraju biti odgovorena prije P-007 (koji je u `Generate graphs/`), ali moraju prije bilo čega iz §4 tačaka #1–#3.
