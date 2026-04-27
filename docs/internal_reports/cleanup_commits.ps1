# cleanup_commits.ps1
# Generisano 2026-04-24 — pokreće se JEDNOM, iz root foldera projekta, u PowerShellu.
#
# Šta radi:
#   - Commit-uje fizička brisanja koja su već urađena u radnoj kopiji (P-001..P-006).
#   - Commit-uje novi .gitignore.
#   - Opciono commit-uje dva MD dokumenta iz Faze 0.
#
# Šta NE radi:
#   - Ne brise istoriju git-a (nije rewrite — OPCIJA A).
#   - Ne radi push. `git push origin main` radiš ručno nakon što provjeriš log.

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "== 0. Lokacija i git stanje ==" -ForegroundColor Cyan
Write-Host "   Radni folder: $PSScriptRoot"
git rev-parse --is-inside-work-tree | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "GRESKA: ovo nije git repo. Prekidam." -ForegroundColor Red
    exit 1
}

Write-Host "`n== 1. git status PRIJE commit-ovanja ==" -ForegroundColor Cyan
Write-Host "   Trebalo bi da vidiš:" -ForegroundColor Gray
Write-Host "   - ?? .gitignore, ?? 00_inventory.md, ?? 00b_cleanup_plan.md, ?? cleanup_commits.ps1" -ForegroundColor Gray
Write-Host "   - D  (obrisano) za 30 .pth + training_log.csv u model/" -ForegroundColor Gray
Write-Host "   - D  (obrisano) za 6 CSV u Emotion recognition/data/" -ForegroundColor Gray
Write-Host "   - D  (obrisano) za training_log.csv, modified_audio.wav, results.txt u Emotion recognition/" -ForegroundColor Gray
Write-Host "   - D  (obrisano) za 6 .pyc u 3 __pycache__ foldera" -ForegroundColor Gray
Write-Host "   - D  (obrisano) za vocabulary_size.txt, requerements.txt" -ForegroundColor Gray
Write-Host "   - D  (obrisano) za mindful-server-*.json" -ForegroundColor Gray
Write-Host "   (+ eventualno `?? Prominence/plot speech time.py` sa ćiriličnim e u imenu — ignoriši zasad)" -ForegroundColor Gray
Write-Host ""
git status --short
Write-Host ""
$ans = Read-Host "Nastavljamo s 7 commit-a? Ukucaj DA za nastavak, bilo šta drugo prekida"
if ($ans -ne "DA") {
    Write-Host "Prekinuto po tvom zahtjevu." -ForegroundColor Yellow
    exit 0
}

# -------------------------------------------------------------------
# Commit 1: .gitignore
# -------------------------------------------------------------------
Write-Host "`n== Commit 1/7: Add .gitignore ==" -ForegroundColor Cyan
git add -- .gitignore
git commit -m "Add .gitignore (Python, models, data, audio, credentials)"

# -------------------------------------------------------------------
# Commit 2: __pycache__
# -------------------------------------------------------------------
Write-Host "`n== Commit 2/7: Remove __pycache__ from tracked files ==" -ForegroundColor Cyan
git add -u -- "Additional files after recension/__pycache__"
git add -u -- "Different information measurement parameters/__pycache__"
git add -u -- "Emotion recognition/__pycache__"
git commit -m "Remove __pycache__ folders from tracked files"

# -------------------------------------------------------------------
# Commit 3: Emotion recognition/data/
# -------------------------------------------------------------------
Write-Host "`n== Commit 3/7: Remove Emotion recognition/data/ (reproducible artifacts) ==" -ForegroundColor Cyan
git add -u -- "Emotion recognition/data"
git commit -m "Remove Emotion recognition/data/ CSV artifacts (regenerable via create_datasets.py)"

# -------------------------------------------------------------------
# Commit 4: Runtime artifacts
# -------------------------------------------------------------------
Write-Host "`n== Commit 4/7: Remove runtime artifacts in Emotion recognition/ ==" -ForegroundColor Cyan
git add -u -- "Emotion recognition/training_log.csv"
git add -u -- "Emotion recognition/modified_audio.wav"
git add -u -- "Emotion recognition/results.txt"
git commit -m "Remove runtime artifacts (training_log.csv, modified_audio.wav, results.txt)"

# -------------------------------------------------------------------
# Commit 5: Emotion recognition/model/
# -------------------------------------------------------------------
Write-Host "`n== Commit 5/7: Remove Emotion recognition/model/ (checkpoints kept locally) ==" -ForegroundColor Cyan
git add -u -- "Emotion recognition/model"
git commit -m "Remove Emotion recognition/model/ PyTorch checkpoints (kept locally, outside repo)"

# -------------------------------------------------------------------
# Commit 6: Mali zaostaci
# -------------------------------------------------------------------
Write-Host "`n== Commit 6/7: Remove empty leftovers ==" -ForegroundColor Cyan
git add -u -- "Mel coefficients and surprisals/vocabulary_size.txt"
git add -u -- "Mel coefficients and surprisals/requerements.txt"
git commit -m "Remove empty leftovers (vocabulary_size.txt, requerements.txt typo)"

# -------------------------------------------------------------------
# Commit 7: GCP credential
# -------------------------------------------------------------------
Write-Host "`n== Commit 7/7: Remove GCP service-account credential ==" -ForegroundColor Cyan
git add -u -- "Forced alignment/mindful-server-408912-9b007c83d67b.json"
git commit -m "Remove GCP service-account credential from repo (rotate key in GCP console)"

# -------------------------------------------------------------------
# Opcioni Commit 8: refactoring dokumentacija
# -------------------------------------------------------------------
Write-Host "`n== Opcioni Commit 8: dodaj Phase 0 dokumentaciju? ==" -ForegroundColor Cyan
$addDocs = Read-Host "Commit-ovati 00_inventory.md i 00b_cleanup_plan.md u repo? (DA/NE)"
if ($addDocs -eq "DA") {
    if (Test-Path "00_inventory.md")        { git add -- "00_inventory.md" }
    if (Test-Path "00b_cleanup_plan.md")    { git add -- "00b_cleanup_plan.md" }
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git commit -m "Add Phase 0 refactoring documentation (inventory + cleanup plan)"
    } else {
        Write-Host "   Nema šta da se commit-uje (fajlovi već trackirani ili ne postoje)." -ForegroundColor Gray
    }
}

# -------------------------------------------------------------------
# Kraj
# -------------------------------------------------------------------
Write-Host "`n== ZAVRSENO ==" -ForegroundColor Green
Write-Host "Posljednjih 10 commit-a:" -ForegroundColor Gray
git log --oneline -10
Write-Host ""
Write-Host "Dalje:" -ForegroundColor Yellow
Write-Host "  1. Provjeri `git status` (trebalo bi da bude prazno ili samo Prominence sa ćiriličnim 'e')"
Write-Host "  2. Ako je sve OK:  git push origin main"
Write-Host "  3. Bezbjednost: rotiraj GCP ključ mindful-server-408912 u GCP konzoli"
Write-Host "  4. Ako ti budu potrebni podaci lokalno: stavi ih u sibling folder ..\podaci\ i ..\modeli\"
