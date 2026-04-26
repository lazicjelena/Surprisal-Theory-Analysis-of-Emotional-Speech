# 00 — Inventura projekta (Faza 0)

**Projekat:** Surprisal-Theory-Analysis-of-Emotional-Speech
**Git remote:** `https://github.com/lazicjelena/Surprisal-Theory-Analysis-of-Emotional-Speech.git`
**Grana:** `main`
**Posljednji commit:** `8b975b9` — 2025-11-02 — „removed some tokens" (Jelena Lazic)
**Datum inventure:** 2026-04-24
**Režim:** read-only — nijedan fajl nije mijenjan niti brisan.

Napomena: `data/` folder (na korisnikovom računaru) nije priključen. Međutim, **unutar `Emotion recognition/` postoji podfolder `data/`** — o tome vidjeti „UNCLEAR-01" u sekciji Otvorena pitanja.

---

## 1. Osnovne veličine

| Metrika | Vrijednost |
|---|---|
| Ukupna veličina radnog sadržaja (bez `.git/`) | **79 MB** |
| Veličina `.git/` | **732 MB** |
| Ukupno foldera prvog nivoa | 14 |
| Ukupno fajlova (bez `.git/` i `.pyc`) | 158 |
| Tracked u git-u | 164 |
| Untracked na disku | 1 (`Prominence/plot speеch time.py` — sadrži ćirilično „е") |

## 2. Fajlovi po tipu (bez `.git/`)

| Tip | Broj | Komentar |
|---|---:|---|
| `.py` | 95 | glavni radni kod |
| `.pth` | 30 | PyTorch checkpoint-i (svi u `Emotion recognition/model/`, ~76 MB) |
| `.ipynb` | 18 | Jupyter notebook-ovi — tretiramo istim pravilima |
| `.csv` | 8 | 7 u `Emotion recognition/`, 1 kao training log |
| `.pyc` | 6 | **ne bi trebali biti u repozitorijumu** (nema `.gitignore`) |
| `.txt` | 4 | `installations.txt`, `results.txt`, prazan `requerements.txt` (tipfeler), `vocabulary_size.txt` (0 linija) |
| `.wav` | 1 | `Emotion recognition/modified_audio.wav` |
| `.json` | 1 | **`Forced alignment/mindful-server-…json`** — izgleda kao GCP service-account ključ (vidjeti bezbjednosni odjeljak) |
| `.md` | 1 | `README.md` |

## 3. Konfiguracioni/meta fajlovi — prisustvo

| Fajl | Status |
|---|---|
| `README.md` | postoji (726 B, kratak, bez detalja o pokretanju) |
| `installations.txt` | postoji na rootu — spisak `pip install` komandi, ne pravi `requirements.txt` |
| `requirements.txt` | **nedostaje** |
| `pyproject.toml` | nedostaje |
| `setup.py` / `setup.cfg` | nedostaje |
| `environment.yml` / `Pipfile` | nedostaje |
| `.gitignore` | **nedostaje** → posljedica: `__pycache__/`, `.pyc`, i 76 MB `.pth` težina su commit-ovani |
| `.python-version` / `tox.ini` | nedostaje |

## 4. Struktura foldera (top-level)

Broj fajlova po folderu (bez `.pyc`; uključuje ugniježđene):

| Folder | `.py` | `.ipynb` | `.pth` | `.csv` | ostalo | ukupno | veličina |
|---|---:|---:|---:|---:|---:|---:|---:|
| Additional files after recension | 12 | 2 | 0 | 0 | 0 | 14 | 220 K |
| Different information measurement parameters | 19 | 10 | 0 | 0 | 0 | 29 | 1.1 M |
| Duration Prediction based on Surprisals | 5 | 0 | 0 | 0 | 0 | 5 | 36 K |
| Emotion recognition | 9 | 0 | 30 | 8 | 2 | 49 | **77 M** |
| Fetures extraction *(tipfeler)* | 1 | 0 | 0 | 0 | 0 | 1 | 8 K |
| Forced alignment | 2 | 0 | 0 | 0 | 1 | 3 | 12 K |
| Generate graphs | 8 | 0 | 0 | 0 | 0 | 8 | 48 K |
| Linear regression | 5 | 0 | 0 | 0 | 0 | 5 | 52 K |
| Mel coefficients and surprisals | 5 | 1 | 0 | 0 | 2 | 8 | 60 K |
| Pervious Surprisals *(tipfeler)* | 5 | 0 | 0 | 0 | 0 | 5 | 40 K |
| Prominence | 10 | 0 | 0 | 0 | 0 | 10 | 64 K |
| Split-over effect | 5 | 0 | 0 | 0 | 0 | 5 | 32 K |
| Surprisal estimation | 7 | 5 | 0 | 0 | 0 | 12 | 308 K |
| Transcript - correct | 2 | 0 | 0 | 0 | 0 | 2 | 20 K |

## 5. Ugniježđeni podfolderi (nivoa 2)

| Putanja | Sadržaj | Komentar |
|---|---|---|
| `Emotion recognition/data/` | 6 CSV (700 K) | **UNCLEAR-01** — liči na podataka-podskup; korisnik je rekao da `data/` nije priključen |
| `Emotion recognition/model/` | 30 `.pth` + 1 CSV (76 M) | commit-ovani model checkpoint-i — glavni uzrok bubrenja repozitorijuma |
| `Different information measurement parameters/parameters estimations/` | 10 `.ipynb` + 2 `.py` (920 K) | logički je „podprojekat" unutar foldera |
| `Surprisal estimation/ngram surprisal estimation/` | 5 `.py` (24 K) | n-gram pipeline odvojen u podfolderu |
| `Forced alignment/Resampling/` | 1 `.py` (4 K) | jedna skripta u vlastitom podfolderu — vjerovatno kandidat za premještanje |

`__pycache__/` postoji u: `Additional files after recension/`, `Different information measurement parameters/`, `Emotion recognition/`. Sve bi trebalo ignorisati `.gitignore`-om (kandidat za Fazu 4, tek uz odobrenje).

## 6. Prva opservacija strukture (bez izmjena — samo za Fazu 1)

Elementi koji će biti predmet detaljne analize u sljedećoj fazi:

- Svi **folderi imaju razmak u imenu** (15 foldera s razmacima, 19 fajlova s razmacima). Posljedica: folderi nisu Python paketi (ne mogu biti `import`-ovani kao moduli) — skripte se vjerovatno pokreću kao samostalne. To je činjenica stanja, ne automatski problem.
- **Tipfeleri u imenima foldera:** `Fetures extraction` → vjerovatno `Features extraction`; `Pervious Surprisals` → vjerovatno `Previous Surprisals`. Kandidati za preimenovanje u Paketu B (Faza 4), ne sada.
- **Miješani jezik u imenima `.py` fajlova:** većina je engleski (`build_dataset.py`, `baseline_model.py`, …), ali postoje i srpski: `analize_po_govornicima.py`, `distribucija_gresaka.py`, `novosadska_baza_podataka.py`. Odluka o ujednačavanju ide u Fazu 3.
- **Mogući duplikati imena kroz folderi:** `build_dataset.py` postoji u najmanje 5 foldera; `baseline_model.py` u 4; `final_graphs.py` u 3; `my_functions.py` u 2; `plot_results.py` u 2; `analysis_accross_individual_words.py` u 2. Sadržaj još **nije čitan** — ne znamo da li su isti kod ili samo isto ime. Kandidat za Fazu 2 / potencijalno Paket C (izdvajanje zajedničkog koda).
- **Mogući „utils" modul već postoji:** `my_functions.py` se javlja u 2 foldera, a tu je i `information_and_distance_functions.py` — moguće da je dio zajednički. Provjeravamo u Fazi 2.
- **1 fajl s ćiriličnim karakterom u imenu:** `Prominence/plot speеch time.py` (slovo „е" je U+0435, ćirilično, umjesto Latin „e"). Zbog toga je git vidi kao drugu putanju nego što OS prikazuje, pa je **untracked** u trenutnoj verziji. Važno za Fazu 4.

## 7. Bezbjednosni i higijenski nalazi (za razgovor, ne za izmjenu)

Ovo nije dio refaktorisanja koda — ali je korisno da znaš prije nego što nastavimo, pogotovo ako je repo javan.

- **`Forced alignment/mindful-server-408912-9b007c83d67b.json`** — ime odgovara Google Cloud service-account JSON ključu. Ako jeste ključ i ako je javni repo, **ključ treba rotirati** u GCP konzoli (bez obzira da li ga uklanjamo iz repoa). `UNCLEAR-02`.
- Posljednji commit je „removed some tokens" — token-i su vjerovatno bili commit-ovani i i dalje postoje u **git istoriji**. `UNCLEAR-03`.
- `__pycache__/` i 76 MB `.pth` težina su u gitu zbog odsustva `.gitignore`. To je glavni razlog zašto je `.git/` 732 MB.

Ništa od ovoga se ne rješava u Fazi 0. Samo evidentiramo.

## 8. Otvorena pitanja (UNCLEAR)

- **UNCLEAR-01** — `Emotion recognition/data/` sadrži 6 CSV fajlova (700 K). Rekla si da `data/` nije priključen. Da li je ovaj `data/` poseban (npr. pred-obrađeni izvodi koji se smatraju dijelom repozitorijuma), ili je greškom prošao i treba da ga tretiram kao „pravi data folder" i da ga ne čitam?
- **UNCLEAR-02** — Da li `mindful-server-*.json` zaista jeste GCP service-account ključ? Ako jeste, preporučujem da se **odmah rotira** u GCP konzoli, nezavisno od refaktorisanja.
- **UNCLEAR-03** — Repo je javan i u istoriji je „removed some tokens". Da li si svjesna da token-i i dalje mogu biti izvučeni iz `git log` / `git show`? Ovo ne rješavamo sada, ali je odluka koju treba donijeti prije bilo kakve rewrite-history operacije.
- **UNCLEAR-04** — Da li radiš u virtualnom okruženju (venv/conda)? Odgovor će oblikovati preporuku za `requirements.txt` / `environment.yml` u Fazi 3.
- **UNCLEAR-05** — Koja Python verzija se koristi? U `__pycache__`-u vidim miks `cpython-311` i `cpython-312` — sugeriše da su skripte pokretane s različitim verzijama Python-a u različito vrijeme. Ciljna verzija za refaktorisanje?

## 9. Šta slijedi (Faza 1)

U Fazi 1 radim i dalje read-only, ali sada **čitam sadržaj** `.py` i `.ipynb` fajlova da bih napravio:

- `01_project_map.md` — per-fajl kratak opis (šta radi, koje ulaze čita, koje izlaze piše, koliko ima linija),
- `01_dependency_overview.md` — graf importa (ko koga importuje), spisak ulaznih tačaka, lista smell-ova (dugi fajlovi, dugačke funkcije, `from X import *`, hardkodirane putanje, globalne varijable).

Prije Faze 1 trebaju mi odgovori bar na **UNCLEAR-01** (tretiranje `Emotion recognition/data/`) i **UNCLEAR-05** (ciljna Python verzija). Ostalo može i poslije.
