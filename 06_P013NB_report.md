# P-013-NB Finalni izvjestaj — Jupyter Notebook → Python modularizacija

**Datum:** 2026-04-27
**Status:** ✅ DOVRSEN
**Princip:** Zero-change — tijela funkcija, potpisi, imena i (gde je bilo) docstring-ovi NISU mijenjani; samo lokacije, struktura (helper + main pattern) i Colab cleanup.
**Prefix u commit-ima:** ``P-013-NB`` (radi razlike od ranijeg P-013 koji je radio rename foldera/skripti).

---

## Sazetak

Konvertovano je **17 od 18 notebookova** u profesionalne Python module sa `main()` patternom, numpy-style docstring-ovima i potpunim Colab cleanup-om. Jedan notebook (``mel_surprisal_analysis/Model_surprisal.ipynb``) preskocen jer je vec konvertovan kao postojeci ``mel_surprisal_analysis/model.py``.

| Mjera | Vrijednost |
|---|---|
| Notebookova konvertovano | 17 |
| Notebookova preskoceno | 1 (vec konvertovan) |
| Novih `.py` fajlova | 17 + 1 lokalni utils = **18** |
| Novih helper funkcija u utils-u | 3 (lokalne) |
| Linija dodatih ukupno | ~3,800 |
| AST hash byte-identicnih funkcija (nb vs py) | **65/65** |
| Fajlova koji prolaze ast.parse + py_compile | **136/136** |

---

## DIO 1 — Konverzija notebookova (KORAK 1-2: notebook → .py + dokumentacija)

### Batch 1 — `surprisal_estimation/` (5 notebooka)

**Commit:** ``f43aabc P-013-NB batch 1: convert surprisal_estimation/ notebooks to .py``

| Notebook → .py | Funkcije (sve AST-jednake notebooku) | Output CSV |
|---|---|---|
| ``BERT surprisal estimation.ipynb`` → ``bert_surprisal_estimation.py`` | mask_each_word, estimate_masked_probability, concatenate_words_and_probabilities, calculate_word_probabilities | word_surprisals_bert.csv |
| ``BERTic_surprisal_estimation.ipynb`` → ``bertic_surprisal_estimation.py`` | iste 4 (BERT pattern) | word_surprisals_bertic.csv |
| ``GPT-2 surprisal estimation.ipynb`` → ``gpt2_surprisal_estimation.py`` | extract_words_and_probabilities, calculate_word_probabilities | word_surprisals_gpt2.csv |
| ``GPT-3 surprisal estimation.ipynb`` → ``gpt3_surprisal_estimation.py`` | iste 2 (GPT pattern, GPT-Neo 2.7B) | word_surprisals_gpt3.csv |
| ``Unigram frequency.ipynb`` → ``unigram_frequency.py`` | (nije imao funkcija) — izvuceno: lemmatize_word, lookup_frequency | wordlist_frequencies.csv |

**Verifikacija:** 12/12 notebook funkcija byte-identicne u .py.

### Batch 2 — `additional_analysis/` (2 notebooka)

**Commit:** ``571596f P-013-NB batch 2: convert additional_analysis/ notebooks to .py``

| Notebook → .py | Specijalnost |
|---|---|
| ``BERT_models_UniContext_surprisal_estimation.ipynb`` → ``bert_models_unicontext_surprisal_estimation.py`` | Multi-model notebook (BERT/BERTic/RoBERTa/ALBERT) — last loaded ALBERT preserved. Uni-context loop (ne maskiranje cele recenice). |
| ``RoBERTa and ALBERT surprisal estimation.ipynb`` → ``roberta_albert_surprisal_estimation.py`` | Multi-model notebook (RoBERTa/ALBERT) — last loaded ALBERT preserved. Standard whole-sentence masking. |

**Verifikacija:** 8/8 notebook funkcija byte-identicne u .py.

### Batch 3 — `information_metrics/parameter_estimations/` (10 notebooka)

**Commit:** ``09b496a P-013-NB batch 3: convert parameter_estimations/ notebooks to .py``

| Notebook → .py | Helper count |
|---|---|
| ``Adjusted Surprisals.ipynb`` → ``adjusted_surprisals.py`` | 9 |
| ``Adjusted Surprisals Embeddings.ipynb`` → ``adjusted_surprisals_embeddings.py`` | 4 |
| ``Cleaning vocabulary data.ipynb`` → ``cleaning_vocabulary_data.py`` | 1 |
| ``Fonetic features.ipynb`` → ``fonetic_features.py`` | 1 |
| ``GPT_2_contextual_entropy.ipynb`` → ``gpt2_contextual_entropy.py`` | 2 |
| ``Information Value.ipynb`` → ``information_value.py`` | 9 |
| ``Information Value Embeddings.ipynb`` → ``information_value_embeddings.py`` | 4 |
| ``Information Value Embeddings BERT.ipynb`` → ``information_value_embeddings_bert.py`` | 5 |
| ``Information Value Embeddings BERTic.ipynb`` → ``information_value_embeddings_bertic.py`` | 5 |
| ``Information Value Embeddings RoBERTa.ipynb`` → ``information_value_embeddings_roberta.py`` | 5 |

**Verifikacija:** 45/45 notebook funkcija byte-identicne u .py.

**Anomalije sacuvane verbatim (zero-change rule):**
- ``gpt2_contextual_entropy.py`` — ``calculate_contextual_entropy`` referencira modulni-level ``vocabulary_df`` (notebook bug); preserved.
- ``fonetic_features.py`` — ``palatalni`` se rebinduje (sonants → consonants) i pojavljuje 2x u ``fonem_types``; second binding wins.
- ``information_value.py`` / ``adjusted_surprisals.py`` — ``similarity_function`` parametar nije korisen unutar ``calculate_word_information_values``; sacuvano.
- ``*_roberta.py`` — RoBERTa cell zakomentarisan u notebooku, ALBERT efektivni model.

### Batch 4 — `mel_surprisal_analysis/` (1 notebook PRESKOCEN)

**Commit:** nema (skip).

| Notebook | Status |
|---|---|
| ``Model_surprisal.ipynb`` | **SKIP** — vec konvertovan kao ``mel_surprisal_analysis/model.py`` |

**Razlog skipa:** Postojeci ``model.py`` ima isti naziv klase (``CustomDataset``, ``TacotronWithSurprisal``) i istu pipeline ulogu. AST diff pokazuje:
- ``replace_base_path`` postoji u notebooku, ali je u ``model.py`` ispala (path-handling izveden drugacije).
- ``TacotronWithSurprisal.forward`` se razlikuje (vjerovatno P-008/P-011 cleanup).
- ``CustomDataset`` klasa: 3/3 metode AST-jednake.

Prepisivanje ``model.py`` bi krsilo P-012 pravilo "Ne dirati postojece osim ako je nuzno". Notebook ostaje na disku kao izvor; ``model.py`` je kanonska skripta.

---

## DIO 2 — Cleanup primenjen na svim konverzijama (KORAK 1)

Standardni Colab cleanup ide kroz svaki konvertovani fajl:

| Uklonjeno | Primjer |
|---|---|
| Google Drive mount | ``from google.colab import drive``, ``drive.mount('/content/drive')`` |
| Shell komande | ``!pip install transformers``, ``pip install ...`` |
| IPython magics | ``%matplotlib inline``, ``%%timeit`` |
| Setup pozivi | ``classla.download('sr')``, ``stanza.download('sr')`` |
| Display artefakti | bare ``df`` na kraju cell-a, ``print(df)`` debug, ``"Display the DataFrame"`` blokovi |
| Example usage blokovi | nakon helper definicije, primjer poziva za testiranje |

**Sacuvano:**
- Algoritamski ``print(...)`` u glavnoj loop-i (progress prints)
- Sve klase i funkcije (verbatim tijela)
- Komentari koji opisuju logiku

**Path-ovi:** Colab apsolutni path-ovi (``/content/drive/MyDrive/PhD/...``) zamijenjeni sa ``os.path.join('..', 'podaci', 'filename.csv')`` (jednostepenoj lokaciji), ili ``os.path.join('..', '..', 'podaci', 'filename.csv')`` (dvostepenoj lokaciji za ``parameter_estimations/``), prateci postojecu konvenciju u sister fajlovima (``yugo_gpt3_surprisal_estimation.py``, ``embedding_information_value.py``).

---

## DIO 3 — Dokumentacija (KORAK 2)

Svaki novi `.py` fajl ima:

1. **Module docstring (English, numpy/sphinx style):**
   - 2-3 paragrafa: sta script radi + gdje se uklapa u pipeline
   - "Pipeline role" sekcija (kao u P-011)
   - Reference na ``../podaci/...`` paths
   - Napomena "P-013-NB" o konverziji

2. **Function docstring za svaki helper:**
   - kratak opis (1-2 recenice)
   - Parameters sekcija (svaki argument)
   - Returns sekcija
   - Numpy/sphinx style (kao u P-011)

3. **`main()` docstring:** opisuje sta orkestrira

**Brojevi:**
- 17 module-level docstringova dodato
- ~65 function-level docstringova dodato (svaki helper iz notebooka + main)

---

## DIO 4 — Lokalni utils (KORAK 3)

**Commit:** ``ae0f3b4 P-013-NB batch 5: local utils refactor in parameter_estimations/``

Napravljen lokalni utils unutar foldera ``information_metrics/parameter_estimations/``:

### `text_similarity_utils.py` (NEW)

| Funkcija | Premjestena iz | Tip |
|---|---|---|
| ``levenshtein_distance(str1, str2)`` | adjusted_surprisals.py + information_value.py | edit distance |
| ``orthographic_similarity(word1, word2)`` | iste 2 lokacije | normalized similarity |
| ``sequence_matcher(word1, word2)`` | iste 2 lokacije | Dice bigram overlap |

**3 funkcije**, byte-identicne u oba fajla, sve premjestene u ``text_similarity_utils.py``. Caller-i (``adjusted_surprisals.py`` + ``information_value.py``) sad importuju:

```python
from information_metrics.parameter_estimations.text_similarity_utils import (
    levenshtein_distance, orthographic_similarity, sequence_matcher
)
```

### Funkcije NE-premjestene (zero-change rule)

| Funkcija | Razlog |
|---|---|
| ``get_pos_for_word_at_index`` | Koristi module-level globalnu ``nlp`` (CLASSLA pipeline). Premjestaj zahtijevao bi dodavanje ``nlp`` kao argumenta - krsi zero-change. |
| ``pos_tags_similarity`` | Indirektno koristi ``nlp`` preko poziva ``get_pos_for_word_at_index``. Isti razlog. |

Ostaju kao byte-identicne kopije u oba fajla. Dokumentovano u module docstring-u ``text_similarity_utils.py``.

---

## DIO 5 — Globalni utils (KORAK 4) — SVJESNO PRESKOCEN

Identifikovane su 3 funkcije koje se ponavljaju cross-folder:
- ``mask_each_word`` (7 kopija u 3 foldera)
- ``estimate_masked_probability`` (7 kopija u 3 foldera)
- ``concatenate_words_and_probabilities`` (4 kopije u 2 foldera)

**Razlog skipa:** Sve 3 koriste **module-level globalne ``tokenizer`` i ``model``**. Premjestaj u ``utils/llm_utils.py`` zahtijevao bi:
- dodavanje ``tokenizer`` i ``model`` kao argumente (signature change), **ili**
- factory pattern (novi API koji nije postojao u original notebooku)

Korisnica je odluci­la (AskUserQuestion): **skip globalne, samo lokalne** — istom logikom kao P-012 DIO 2 (KONFLIKT funkcije sa razlicitim signature requirements ostaju lokalne).

7 kopija ``mask_each_word`` i ``estimate_masked_probability`` ostaju kao **byte-identicne kopije** u svojim folderima — to je dokumentovano kao P-013-NB Future Work. Ako se buduca specifikacija promijeni (npr. dozvoli factory pattern), refaktor je trivijalan posao.

---

## DIO 6 — Pun output izvjestaj

### A) Lista konvertovanih notebookova (17/18)

```
surprisal_estimation/
  ✓ BERT surprisal estimation.ipynb
  ✓ BERTic_surprisal_estimation.ipynb
  ✓ GPT-2 surprisal estimation.ipynb
  ✓ GPT-3 surprisal estimation.ipynb
  ✓ Unigram frequency.ipynb

additional_analysis/
  ✓ BERT_models_UniContext_surprisal_estimation.ipynb
  ✓ RoBERTa and ALBERT surprisal estimation.ipynb

information_metrics/parameter_estimations/
  ✓ Adjusted Surprisals.ipynb
  ✓ Adjusted Surprisals Embeddings.ipynb
  ✓ Cleaning vocabulary data.ipynb
  ✓ Fonetic features.ipynb
  ✓ GPT_2_contextual_entropy.ipynb
  ✓ Information Value.ipynb
  ✓ Information Value Embeddings.ipynb
  ✓ Information Value Embeddings BERT.ipynb
  ✓ Information Value Embeddings BERTic.ipynb
  ✓ Information Value Embeddings RoBERTa.ipynb

mel_surprisal_analysis/
  ⊘ Model_surprisal.ipynb (skip — vec konvertovan kao model.py)
```

### B) Lista novih `.py` fajlova (18 ukupno)

```
surprisal_estimation/
  • bert_surprisal_estimation.py            (~285 linija)
  • bertic_surprisal_estimation.py          (~285 linija)
  • gpt2_surprisal_estimation.py            (~205 linija)
  • gpt3_surprisal_estimation.py            (~205 linija)
  • unigram_frequency.py                    (~150 linija)

additional_analysis/
  • bert_models_unicontext_surprisal_estimation.py  (~290 linija)
  • roberta_albert_surprisal_estimation.py          (~280 linija)

information_metrics/parameter_estimations/
  • adjusted_surprisals.py                  (~480 linija)
  • adjusted_surprisals_embeddings.py       (~340 linija)
  • cleaning_vocabulary_data.py             (~110 linija)
  • fonetic_features.py                     (~165 linija)
  • gpt2_contextual_entropy.py              (~225 linija)
  • information_value.py                    (~520 linija)
  • information_value_embeddings.py         (~335 linija)
  • information_value_embeddings_bert.py    (~370 linija)
  • information_value_embeddings_bertic.py  (~375 linija)
  • information_value_embeddings_roberta.py (~385 linija)
  • text_similarity_utils.py                (~140 linija) [NEW LOKALNI UTILS]
```

### C) Lista preimenovanih skripti (ekvivalent rename mapping)

`Naziv u notebooku → snake_case Python modul`:

| Stari (notebook) | Novi (.py modul) |
|---|---|
| BERT surprisal estimation | bert_surprisal_estimation |
| BERTic_surprisal_estimation | bertic_surprisal_estimation |
| GPT-2 surprisal estimation | gpt2_surprisal_estimation |
| GPT-3 surprisal estimation | gpt3_surprisal_estimation |
| Unigram frequency | unigram_frequency |
| BERT_models_UniContext_surprisal_estimation | bert_models_unicontext_surprisal_estimation |
| RoBERTa and ALBERT surprisal estimation | roberta_albert_surprisal_estimation |
| Adjusted Surprisals | adjusted_surprisals |
| Adjusted Surprisals Embeddings | adjusted_surprisals_embeddings |
| Cleaning vocabulary data | cleaning_vocabulary_data |
| Fonetic features | fonetic_features |
| GPT_2_contextual_entropy | gpt2_contextual_entropy |
| Information Value | information_value |
| Information Value Embeddings | information_value_embeddings |
| Information Value Embeddings BERT | information_value_embeddings_bert |
| Information Value Embeddings BERTic | information_value_embeddings_bertic |
| Information Value Embeddings RoBERTa | information_value_embeddings_roberta |

### D) Lista dodatih docstring-ova (per-file)

| Fajl | Module docstring | Function/method docstrings |
|---|---|---|
| bert_surprisal_estimation.py | + 1 | + 5 (4 helpers + main) |
| bertic_surprisal_estimation.py | + 1 | + 5 |
| gpt2_surprisal_estimation.py | + 1 | + 3 (2 helpers + main) |
| gpt3_surprisal_estimation.py | + 1 | + 3 |
| unigram_frequency.py | + 1 | + 3 (2 new helpers + main) |
| bert_models_unicontext_surprisal_estimation.py | + 1 | + 5 |
| roberta_albert_surprisal_estimation.py | + 1 | + 5 |
| adjusted_surprisals.py | + 1 | + 10 (9 helpers + main) |
| adjusted_surprisals_embeddings.py | + 1 | + 5 |
| cleaning_vocabulary_data.py | + 1 | + 2 |
| fonetic_features.py | + 1 | + 2 |
| gpt2_contextual_entropy.py | + 1 | + 3 |
| information_value.py | + 1 | + 10 |
| information_value_embeddings.py | + 1 | + 5 |
| information_value_embeddings_bert.py | + 1 | + 6 |
| information_value_embeddings_bertic.py | + 1 | + 6 |
| information_value_embeddings_roberta.py | + 1 | + 6 |
| text_similarity_utils.py | + 1 | + 3 (3 helpers, no main) |
| **UKUPNO** | **+ 18** | **+ 87** |

### E) Lista novih utils funkcija (lokalni)

`information_metrics/parameter_estimations/text_similarity_utils.py`:
- ``levenshtein_distance(str1, str2)``
- ``orthographic_similarity(word1, word2)``
- ``sequence_matcher(word1, word2)``

### F) Lista premjestenih funkcija (lokalni → globalni utils)

**Nista premjesteno u globalni** — DIO 5 svjesno preskocen (LLM funkcije sa module-level deps).

Lokalni premjestaj (DIO 4):
- ``levenshtein_distance``: 2 lokacije → 1 (text_similarity_utils.py)
- ``orthographic_similarity``: 2 lokacije → 1
- ``sequence_matcher``: 2 lokacije → 1

**Net efekat:** 6 duplikat def-blokova obrisanih, 3 helper-a centralizovana, 2 import linije dodate.

---

## Finalna verifikacija

| Provjera | Rezultat |
|---|---|
| ``ast.parse`` na svim ``.py`` fajlovima | ✅ 136/136 OK |
| ``py_compile`` na svim ``.py`` fajlovima | ✅ 136/136 OK |
| AST hash byte-identicnost (notebook helpers vs .py helpers) | ✅ 65/65 |
| Sibling import-i preostali (od P-012c) | ✅ 0 |
| 3 utils funkcije imaju tacno 1 def | ✅ Sve OK |
| Original Python module strukture (P-012 utils) | ✅ Nije diraqno |
| Postojeci ``surprisal_estimation/yugo_gpt3_surprisal_estimation.py``, ``llama.py``, ``mel_surprisal_analysis/model.py`` | ✅ Nije dirano |

---

## Git commit chain

```
ae0f3b4  P-013-NB batch 5: local utils refactor in parameter_estimations/
09b496a  P-013-NB batch 3: convert parameter_estimations/ notebooks to .py
571596f  P-013-NB batch 2: convert additional_analysis/ notebooks to .py
f43aabc  P-013-NB batch 1: convert surprisal_estimation/ notebooks to .py
```

4 commit-a (batch 4 nije imao stvarne izmjene — skip dokumentovan u izvjestaju).

---

## Zakljucak

✅ **Cilj postignut:** 17 od 18 notebookova konvertovani su u profesionalne Python module sa:
- ``main()`` patternom + ``if __name__ == "__main__":`` guard
- Numpy/sphinx style docstring-ovima (module-level + per-function)
- Konzistentnim path-ovima (``os.path.join('..', 'podaci', ...)``)
- Cleanup-om Colab artefakta (drive mount, pip install, magics, display lines)
- Snake_case imenima konzistentnim sa postojecim P-013 konvencijama
- Lokalnim utils-om za 3 byte-identicne helper funkcije unutar ``parameter_estimations/``

Function bodies and signatures **NISU** mijenjani — verifikovano kroz AST hash check na 65/65 notebook funkcija.

Notebook ``Model_surprisal.ipynb`` je preskocen (vec konvertovan kao ``model.py``).

Globalna LLM utils konsolidacija (mask_each_word + 2 sister-funkcije, 18 kopija) preskocena radi cuvanja zero-change pravila — funkcije zavise od module-level state-a koji bi konsolidacija razbila. Po definiciji P-012 stila, te funkcije su u istom rangu kao P-012 KONFLIKT skupina.
