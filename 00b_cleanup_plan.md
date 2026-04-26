# 00b — Plan čišćenja repozitorijuma (read-only analiza)

**Režim:** bez izmjena. Ovo je prijedlog, ništa nije obrisano niti izmijenjeno.
**Referenca:** dopuna inventuri iz `00_inventory.md`.

---

## 0. Napomena o trenutnom stanju radne kopije

Između prve i druge analize, `Emotion recognition/model/` folder je **već nestao s diska** (30 `.pth` + 1 CSV, 76 MB). Fajlovi su, međutim, **još uvijek praćeni u git indeksu** — nisu commit-ovani kao obrisani. To znači:

- Disk: `model/` ne postoji → kod koji čita iz `model/` će pući na runtime-u.
- Git: `git status` još uvijek pokazuje 31 fajl kao „deleted" u `Emotion recognition/model/`.
- Radnja: nakon commit-a koji ih stage-uje kao obrisane, biće uklonjeni iz **trenutnog** snapshot-a, ali **ostaju u istoriji** (vidjeti sekciju 7 dole).

Ostali ne-kod fajlovi (CSV data, WAV, JSON credential, pyc, txt) su i dalje na disku.

## 1. Inventar ne-kod fajlova — status i preporuka

Pregled u tri kategorije: **obrisati iz repoa**, **premjestiti van repoa**, **zadržati**.

### 1.1 Obrisati (reproducibilni artefakti, cache, ili beskorisno)

| # | Putanja | Veličina | Zašto bezbjedno obrisati |
|---|---|---:|---|
| D-01 | `Emotion recognition/data/train_dataset.csv` | 410 K | Output od `create_datasets.py`; reprodukcijsko, izvorni podaci u `../podaci/` |
| D-02 | `Emotion recognition/data/val_dataset.csv` | 126 K | isto |
| D-03 | `Emotion recognition/data/test_dataset.csv` | 95 K | isto |
| D-04 | `Emotion recognition/data/baseline_data.csv` | 34 K | Output od `create_csv_for_testing_synthetisized_data.py` |
| D-05 | `Emotion recognition/data/surprisal_data.csv` | 31 K | isto |
| D-06 | `Emotion recognition/data/google_speech_data.csv` | 5 K | isto |
| D-07 | `Emotion recognition/data/` (prazan folder poslije D-01…D-06) | — | Nakon brisanja gore — prazan |
| D-08 | `Emotion recognition/training_log.csv` | 1.7 K | Runtime log, piše ga `training.py`/`emotion_recognition_model.py` |
| D-09 | `Emotion recognition/modified_audio.wav` | 363 K | Runtime output od `prosody_parameters_and_mfcc.py` (piše kao „modified_audio.wav") |
| D-10 | `Emotion recognition/results.txt` | 190 B | Nijedna skripta ne čita niti piše — izgleda zaostalo |
| D-11 | `Emotion recognition/model/*.pth` (30 fajlova) | 76 M | Već obrisano s diska — samo treba commit-ovati brisanje |
| D-12 | `Emotion recognition/model/training_log.csv` | — | Već obrisano s diska — commit |
| D-13 | `Emotion recognition/model/` (prazan folder) | — | — |
| D-14 | `Mel coefficients and surprisals/vocabulary_size.txt` | 37 B | Runtime output (piše ga `build_dataset.py`), sada praktično prazan |
| D-15 | `Mel coefficients and surprisals/requerements.txt` | 0 B | Prazan + tipfeler. Potvrditi: je li namjeravao biti `requirements.txt`? *(UNCLEAR-06)* |
| D-16 | Svi `*.pyc` (6 fajlova) + tri `__pycache__/` foldera | 53 K | Python bytecode cache, regeneriše se automatski |
| D-17 | `installations.txt` | 205 B | *Ne obavezno* — ali sadržaj treba migrirati u `requirements.txt` (vidjeti 6.2) |

### 1.2 Premjestiti van repoa (osjetljivo ili veliki stalni artefakt)

| # | Putanja | Gdje | Zašto |
|---|---|---|---|
| M-01 | `Forced alignment/mindful-server-408912-9b007c83d67b.json` | Van repoa, npr. `~/.gcp-keys/mindful-server.json` | Google Cloud service-account ključ. Ne pripada repozitorijumu. Put ka ključu prosljediti preko `GOOGLE_APPLICATION_CREDENTIALS` env varijable (kod već čita env var, ali trenutno hardkodira ime fajla — Paket D) |

### 1.3 Zadržati (ne dirati)

| # | Putanja | Razlog |
|---|---|---|
| K-01 | `README.md` | Projektna dokumentacija |
| K-02 | svi `.py` i `.ipynb` fajlovi | Kod i notebook-ovi — predmet kasnijeg refaktorisanja |
| K-03 | `00_inventory.md`, `00b_cleanup_plan.md` | Radna dokumentacija refaktorisanja |

## 2. Prijedlozi (formalno, za odobravanje)

```
[P-001] Obrisati reproducibilne data CSV fajlove
Tip:           deduplicate / cleanup
Fajlovi:       Emotion recognition/data/*.csv (6 fajlova), zatim prazan folder data/
Trenutno:      Fajlovi su u repou; generišu ih `create_datasets.py` i `create_csv_for_testing_synthetisized_data.py` iz izvora u `../podaci/`
Predlog:       Obrisati ih iz radne kopije i stage-ovati kao deleted; dodati `Emotion recognition/data/` u `.gitignore`
Obrazloženje: Intermediate artefakti; zauzimaju 686 KB; regeneriše ih projektov sopstveni kod
Rizik:         `training.py`, `emotion_recognition_model.py`, `testing.py` (zakomentarisani), `Mel coefficients and surprisals/model.py` čitaju iz `data/` → biće potrebno pokrenuti `create_datasets.py` prije treninga
Verifikacija:  `git status` pokazuje 6 deleted CSV; `python create_datasets.py` regeneriše ih (pod uslovom da je `../podaci/` prisutan)
Status:        predlog
```

```
[P-002] Obrisati runtime artefakte u Emotion recognition/
Tip:           cleanup
Fajlovi:       training_log.csv, modified_audio.wav, results.txt
Trenutno:      Na disku; runtime izlazi iz treninga i obrade zvuka
Predlog:       Obrisati; dodati *.wav, training_log.csv u `.gitignore`
Obrazloženje: Nastaju automatski pri pokretanju pipeline-a; ne bi smjeli biti u repozitorijumu
Rizik:         Nikakav za kod. results.txt — **UNCLEAR-07**: da li sadrži rezultate bitne za disertaciju? (190 B, 7 linija)
Verifikacija:  `git status` pokazuje 3 deleted
Status:        predlog
```

```
[P-003] Commit-ovati brisanje Emotion recognition/model/ (već obrisano s diska)
Tip:           cleanup
Fajlovi:       Emotion recognition/model/*.pth (30), Emotion recognition/model/training_log.csv
Trenutno:      Fajlovi obrisani s diska, ali još praćeni u indeksu
Predlog:       `git rm -r "Emotion recognition/model"` (ili `git add -u`), zatim commit
Obrazloženje: Uklanja 76 MB iz trenutnog snapshot-a. (NE iz istorije — vidjeti sekciju 7.)
Rizik:         `testing.py` (linija 29) čita `model/model_epoch_30.pth` → puca bez modela. Lokalno rješenje: kopiju modela držati u `../modeli/` (vidjeti sekciju 6.3), a kod mijenjati u Paketu D Faze 4 (sada ne diramo kod)
Verifikacija:  `git log --stat -1` pokazuje -76 MB izmjene
Status:        predlog
```

```
[P-004] Obrisati __pycache__ iz repoa
Tip:           cleanup
Fajlovi:       3 __pycache__/ foldera, 6 .pyc fajlova
Trenutno:      Commit-ovani u git jer nema .gitignore
Predlog:       `git rm -r --cached ... __pycache__/` za sva tri foldera + obrisati s diska; dodati u .gitignore
Obrazloženje: Bytecode cache, automatski generisan, platformski zavisan
Rizik:         Nikakav
Verifikacija:  `git status` nakon — pycache nestao iz tracked fajlova
Status:        predlog
```

```
[P-005] Premjestiti GCP service-account ključ van repoa
Tip:           security / cleanup
Fajl:          Forced alignment/mindful-server-408912-9b007c83d67b.json
Trenutno:      U repou; `novosadska_baza_podataka.py` liniji 15 hardkodira ime fajla:
                 os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "mindful-server-408912-9b007c83d67b.json"
Predlog:       (a) premjestiti fajl van repoa (npr. %USERPROFILE%\.gcp-keys\mindful-server.json)
               (b) BEZBJEDNOSNO: **rotirati ključ u GCP konzoli** (jer je bio u javnom git-u)
               (c) kasnije (Paket D Faze 4) promijeniti kod da čita iz GOOGLE_APPLICATION_CREDENTIALS env var, bez hardkodiranja imena
Obrazloženje: Kredencijali ne pripadaju u git. Čak i ako je repo privatan, ključ treba rotirati jer je bio u istoriji.
Rizik:         Trenutno (pre izmjene koda) `novosadska_baza_podataka.py` **pada** jer ne nalazi `mindful-server-*.json` u radnoj kopiji.
               Privremeno rješenje: kopirati ključ u radni folder prije pokretanja te skripte, ili postaviti env var ručno.
Verifikacija:  Fajl više nije u `git ls-files`; kod i dalje radi ako je env var postavljen
Status:        predlog, čeka odluku o rotaciji ključa (UNCLEAR-02 iz inventure)
```

```
[P-006] Obrisati sitne zaostatke
Tip:           cleanup
Fajlovi:       Mel coefficients and surprisals/vocabulary_size.txt (37 B, 0 linija)
               Mel coefficients and surprisals/requerements.txt (0 B, tipfeler)
Trenutno:      Na disku; runtime artefakti ili greške
Predlog:       Obrisati oba
Obrazloženje: vocabulary_size.txt piše ga `build_dataset.py`; prazan je — nevažan.
               requerements.txt — prazan + tipfeler; potvrditi šta je tu trebalo biti (UNCLEAR-06).
Rizik:         Nikakav dok se sva runtime pisanja nastave raditi (pisanje kreira fajl)
Verifikacija:  Pokretanje `build_dataset.py` ponovo kreira vocabulary_size.txt ako je potrebno
Status:        predlog
```

## 3. Mapa hardkodiranih putanja u kodu

Nalazi iz grep-a preko svih `.py` i `.ipynb` fajlova.

### 3.1 Spoljni `../podaci/` (dobar znak — već je izvan repoa)

- **250 linija** u **80 fajlova** referencira `os.path.join('..','podaci', …)`. Kod već očekuje da `data/` živi **van repoa**, kao sibling folder.
- Primjeri putanja: `..\podaci\target_sentences.csv`, `..\podaci\surprisal values\word_surprisals_gpt2.csv`, `..\podaci\general_data.pkl`, `..\podaci\data_mono\…\*.wav`, itd.
- **Implikacija:** `Emotion recognition/data/` je lokalna anomalija — 250 ostalih referenci već izvan repoa.

### 3.2 Čita se iz `Emotion recognition/data/` (unutar repoa)

| Fajl : linija | Šta radi | Nakon brisanja |
|---|---|---|
| `Emotion recognition/training.py` : 21 | čita `data/train_dataset.csv` | **pada** dok se ne regeneriše |
| `Emotion recognition/training.py` : 22 | čita `data/val_dataset.csv` | **pada** |
| `Emotion recognition/emotion_recognition_model.py` : 19 | čita `data/train_dataset.csv` | **pada** |
| `Emotion recognition/emotion_recognition_model.py` : 20 | čita `data/val_dataset.csv` | **pada** |
| `Emotion recognition/testing.py` : 19–21 | **zakomentarisano** čitanje `data/*.csv` | nema uticaja |

### 3.3 Piše u `Emotion recognition/data/` (regeneriše sadržaj)

| Fajl : linija | Piše | Napomena |
|---|---|---|
| `Emotion recognition/create_datasets.py` : 49 | `data/train_dataset.csv` | Output |
| `Emotion recognition/create_datasets.py` : 51 | `data/val_dataset.csv` | Output |
| `Emotion recognition/create_datasets.py` : 53 | `data/test_dataset.csv` | Output |
| `Emotion recognition/create_csv_for_testing_synthetisized_data.py` : 31 | `data/surprisal_data.csv` | Output |
| `Emotion recognition/create_csv_for_testing_synthetisized_data.py` : 52 | `data/google_speech_data.csv` | Output |
| `Emotion recognition/create_csv_for_testing_synthetisized_data.py` : 76 | `data/baseline_data.csv` | Output |

**Zaključak:** CSV fajlovi u `data/` su izlaz dviju skripti → reproducibilni → bezbjedni za brisanje.

### 3.4 Čita/piše u `Emotion recognition/model/` (unutar repoa)

| Fajl : linija | Šta | Tip |
|---|---|---|
| `Emotion recognition/training.py` : 116–117 | `torch.save → model/model_epoch_{N}.pth` | Piše |
| `Emotion recognition/testing.py` : 29–31 | `torch.load → model/model_epoch_30.pth` | **Čita — puca bez modela** |
| `Emotion recognition/loss_function.py` : 13 | `pd.read_csv('model/training_log.csv')` | **Čita — puca bez foldera** |
| `Emotion recognition/emotion_recognition_model.py` : 118 | `/content/drive/MyDrive/PhD/…/model/model_epoch_{N}.pth` | Piše u Google Colab — nebitno lokalno |

`Mel coefficients and surprisals/model.py` linija 165: `save_dir = './best_model'` → runtime folder koji se kreira pokretanjem. Treba ga preventivno ignorisati (vidjeti `.gitignore`).

### 3.5 Čita/piše `.wav` i credential JSON

| Fajl : linija | Referenca |
|---|---|
| `Emotion recognition/prosody_parameters_and_mfcc.py` : 93 | `output_wav_file = 'modified_audio.wav'` (piše) |
| `Emotion recognition/prosody_parameters_and_mfcc.py` : 102 | čita WAV iz `..\podaci\data_mono\0001\0\…wav` (eksterni) |
| `Forced alignment/novosadska_baza_podataka.py` : 15 | `GOOGLE_APPLICATION_CREDENTIALS = "mindful-server-408912-9b007c83d67b.json"` — hardkodirano ime |

## 4. Šta prestaje da radi odmah nakon čišćenja

Bez ijedne promjene koda, ove skripte **neće se uspješno pokrenuti** dok se spoljni fajlovi ne pripreme:

1. `Emotion recognition/training.py` → treba `data/train_dataset.csv`, `data/val_dataset.csv` → pokrenuti prvo `create_datasets.py`.
2. `Emotion recognition/emotion_recognition_model.py` → isto.
3. `Emotion recognition/testing.py` → treba `model/model_epoch_30.pth` → vratiti checkpoint iz lokalne kopije.
4. `Emotion recognition/loss_function.py` → treba `model/training_log.csv` → dostupno tek nakon treninga ILI iz lokalne kopije.
5. `Forced alignment/novosadska_baza_podataka.py` → treba `mindful-server-*.json` u radnoj kopiji (ili env var do eksterne putanje).

Sve ostalo (80 fajlova koji referenciraju `..\podaci\`) nastavlja da radi **ako postoji `../podaci/` sibling folder** — a to je stanje koje ionako već imaš lokalno.

Ove posljedice ćemo sistemski otkloniti u **Paketu D (Faza 4)** uvođenjem config-layer-a (`paths.py` ili `.env`), ali to je tek nakon tvog eksplicitnog odobrenja i ne dira naučnu logiku.

## 5. Predlog `.gitignore`

```gitignore
# ---- Python ----
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# ---- Virtualna okruženja ----
venv/
env/
.venv/
.env
.env.*
!.env.example

# ---- Build / distribucija ----
build/
dist/
*.egg-info/
*.egg
pip-log.txt
pip-delete-this-directory.txt

# ---- Jupyter ----
.ipynb_checkpoints/
*.ipynb_checkpoints

# ---- IDE / editori ----
.vscode/
.idea/
*.swp
*.swo
*~
.project
.pydevproject

# ---- OS ----
.DS_Store
Thumbs.db
desktop.ini

# ---- Model težine i checkpoint-i (bilo gdje u projektu) ----
*.pth
*.pt
*.ckpt
*.safetensors
*.bin
*.onnx
*.tflite
*.h5
*.hdf5
*.pkl
*.joblib
best_model/
checkpoints/

# ---- Data formati koji su po pravilu veliki ----
*.parquet
*.feather
*.arrow
*.npy
*.npz
*.mat

# ---- Audio fajlovi (pripadaju u ../podaci/, ne u repo) ----
*.wav
*.mp3
*.flac
*.ogg
*.m4a

# ---- Specifično za ovaj projekat: intermediate artefakti ----
# CSV data-foldere u repou (podaci žive u ../podaci/ sibling folderu)
Emotion recognition/data/
Emotion recognition/model/

# Runtime training logovi u radnim folderima
**/training_log.csv
**/vocabulary_size.txt
**/results.txt

# ---- Google Cloud / API credentials ----
# Nikad ne commit-ovati kredencijale
*service_account*.json
*credentials*.json
mindful-server-*.json
*.pem
*.key
```

**Napomena o pravilima:**

- `*.pth`, `*.wav`, `*.pkl` ciljamo globalno — ako ikad u budućnosti dodaš model ili audio bilo gdje, automatski je zaštićen.
- `Emotion recognition/data/` i `Emotion recognition/model/` su specifični po putanji jer su lokalna anomalija (ostatak koda koristi `../podaci/`).
- `**/training_log.csv` ne isključuje CSV-ove generalno — samo taj specifičan runtime log.
- `!.env.example` dozvoljava šablon `.env.example` (ako ikad budemo pravili config primjer).
- `!data.csv` **nije** uključeno — CSV generalno **nisu** ignorisani (ponekad su dio koda kao mala referentna tabela). Ako se pojavi potreba, dodajemo specifične putanje.

## 6. Predlog organizacije van repoa

### 6.1 Preporučeni layout na disku

```
<root na Jelena desktopu>/
├── Surprisal-Theory-Analysis-of-Emotional-Speech/   ← ovaj repo (samo kod)
├── podaci/                                          ← izvor podataka (već postoji; ostaje van repoa)
│   ├── training_data.csv
│   ├── target_sentences.csv
│   ├── surprisal values/
│   │   ├── word_surprisals_gpt2.csv
│   │   └── …
│   ├── data_mono/       (audio fajlovi po govorniku/emociji)
│   └── …
└── modeli/                                          ← NOVI: checkpoint-i van repoa
    ├── emotion_recognition/
    │   ├── model_epoch_1.pth
    │   ├── …
    │   └── training_log.csv
    └── mel_coefficients/
        └── best_model_*.pth
```

- `podaci/` **već jeste** konvencija (250 referenci u kodu), pa to nije promjena — samo formalizacija.
- `modeli/` je novi sibling folder — tu prebaciš svoju lokalnu kopiju od 30 `.pth` fajlova.
- Putanje u kodu kasnije (Paket E Faza 4) mijenjamo s `'model/...'` na `os.path.join('..','modeli','emotion_recognition','...')`.

### 6.2 `installations.txt` → `requirements.txt`

Trenutni sadržaj `installations.txt`:

```
pip install fuzzywuzzy
pip install pydub
pip install google-cloud-speech
pip install librosa
pip install classla
pip install nltk
pip install pydub

classla.download('sr')
nltk.download('punkt')
```

Predlog — napraviti dva fajla (tek u Paketu ili Fazi 3, uz tvoje odobrenje):

- `requirements.txt` s čistom listom paketa (pinovati verzije poslije mapiranja svih biblioteka u Fazi 1):
  ```
  fuzzywuzzy
  pydub
  google-cloud-speech
  librosa
  classla
  nltk
  # + ostalo što se pojavi u Fazi 1 (torch, transformers, pandas, …)
  ```
- `scripts/setup_nltk_classla.py` ili `Makefile` target za `classla.download('sr')` i `nltk.download('punkt')`.

### 6.3 Credentials

- Fajl `mindful-server-*.json` → `%USERPROFILE%\.gcp-keys\mindful-server.json` (Windows) ili `~/.gcp-keys/` (Unix).
- Kod kasnije (Paket D Faze 4):
  ```python
  # umjesto: os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "mindful-server-...json"
  # radimo:
  import os
  if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
      raise RuntimeError(
          "Postavi GOOGLE_APPLICATION_CREDENTIALS env var na putanju do GCP ključa."
      )
  ```
- **Ne zaboraviti rotirati ključ u GCP konzoli** (UNCLEAR-02).

## 7. Istorija repoa (`.git/` je 732 MB)

Brisanje fajlova iz radne kopije **ne smanjuje** `.git/`. Težine modela i token-i iz starih commit-a ostaju u istoriji i mogu se izvući s `git log -p` / `git show <hash>`.

Dvije opcije:

**Opcija A — ne dirati istoriju (bezbjedno, ali repo ostaje 732 MB):**

- Jednostavno; ne mijenja hash-eve.
- `.git/` ostaje bubrelo; svaki novi klon tog repoa povlači 732 MB.
- Token-i iz prošlosti ostaju potencijalno dohvatljivi.

**Opcija B — rewrite istorije (`git filter-repo` ili BFG Repo-Cleaner):**

- Potpuno čisti fajlove iz svake istorijske tačke.
- Hash-evi commit-a se mijenjaju → svi klonovi i PR-ovi moraju se resync-ovati.
- Zahtijeva force-push; svi saradnici moraju re-klonirati.
- Treba prvo napraviti backup.

**Preporuka:** Krenuti s Opcijom A (commit brisanja + `.gitignore`) **sada**, a Opciju B raditi kasnije kao zasebnu odluku, tek kad završimo s refaktorisanjem. Tad će biti lakše raditi filter-repo operaciju jer će radni tree već biti čist.

**Ako odlučiš sada za Opciju B**, orijentaciono bi trebalo (NE izvršavaj automatski):
```powershell
# Prethodno: napravi backup kompletnog foldera (kopija .git-a uključeno)
# Zatim:
pip install git-filter-repo
cd Surprisal-Theory-Analysis-of-Emotional-Speech
git filter-repo --path "Emotion recognition/model" --invert-paths
git filter-repo --path-glob "*.pth" --invert-paths
git filter-repo --path "Forced alignment/mindful-server-408912-9b007c83d67b.json" --invert-paths
# force-push na GitHub:
git push --force origin main
```
**Ovo je destruktivno. Zahtijeva zasebno, eksplicitno „OK" tvoje.**

## 8. Predloženi redoslijed izvršenja čišćenja (kad daš OK)

Ja izvršavam komandu po komandu, ti vidiš diff posle svakog koraka. Ili ti izvršavaš ručno na Windowsu — kako god želiš.

1. **Kreirati `.gitignore`** (iz sekcije 5) — NEMA brisanja, samo novi fajl.
2. **Ukloniti praćenje pycache-a:** `git rm -r --cached "Additional files after recension/__pycache__" "Different information measurement parameters/__pycache__" "Emotion recognition/__pycache__"` → commit.
3. **Obrisati `Emotion recognition/data/`** (6 CSV fajlova + folder) → `git rm -r "Emotion recognition/data"` → commit.
4. **Obrisati runtime artefakte** u `Emotion recognition/`: `training_log.csv`, `modified_audio.wav`, `results.txt` → `git rm …` → commit.
5. **Stage-ovati brisanje `Emotion recognition/model/`** (već nestao s diska): `git add -u "Emotion recognition/model"` → commit.
6. **Obrisati sitne zaostatke:** `vocabulary_size.txt`, `requerements.txt` (tipfeler) → commit.
7. **Premjestiti GCP ključ van repoa** i stage-ovati brisanje → commit.
8. **(Odložiti) kreirati `requirements.txt` iz `installations.txt`** — ide u Fazu 3 (predlog ciljne strukture) ili odmah ako hoćeš.
9. **(Odložiti) rotirati GCP ključ** — uradiš u GCP konzoli, zasebno.
10. **(Odložiti) filter-repo** za čišćenje istorije — zasebno, nakon refaktorisanja.

Nakon koraka 1–7, `.git/` ostaje 732 MB, ali radna kopija i trenutni snapshot su čisti.

## 9. Otvorena pitanja (dodatak iz Faze 0)

- **UNCLEAR-06** — `Mel coefficients and surprisals/requerements.txt` je prazan + tipfeler („requerements"). Da li je trebalo da bude `requirements.txt` s paketima specifičnim za taj pod-pipeline, ili je slučajno ostao?
- **UNCLEAR-07** — `Emotion recognition/results.txt` (190 B, 7 linija) — sadrži li tekst rezultata koji treba sačuvati negdje drugo prije brisanja? (Nisam ga čitao — samo veličinu.)
- Potvrditi da je **lokalna kopija modela i podataka zaista netaknuta** prije nego što commit-ujemo brisanje.
- Potvrditi **Opciju A ili B** za istoriju repoa (ili odgoditi Opciju B za poslije).

Kada potvrdiš plan (ili tražiš izmjene), krećem s korakom 1 (kreiranje `.gitignore`). Ništa ne brišem dok se ne dogovorimo za svaki red tabele iz sekcije 1.1/1.2.
