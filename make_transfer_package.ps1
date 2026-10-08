<#
.SYNOPSIS
  Assembles the gitignored artifacts the laptop needs into transfer_to_laptop/.

.DESCRIPTION
  One-off helper for the 3060 -> laptop move described in
  TRANSFER_TO_LAPTOP.md. Everything this copies is gitignored by design,
  so git cannot carry it.

  Tier 1 (~865 MB)  every runs/ folder WITHOUT its *.pt weights (the audit
                    trail, CLAUDE.md rule 4), both recogniser caches, the
                    four checkpoints that are actually still in use, and
                    data/modernizer_vocab.json.
  Tier 2 (~3.1 GB)  + data/s2, data/s2_extra, data/raw_corpus text corpora.
  Tier 3 (~15 GB)   + data/s1. Only if you intend to retrain the CRNN on a
                    4 GB card, which TRANSFER_TO_LAPTOP.md argues against.

  Writes MANIFEST.sha256 in `sha256sum -c` format so the laptop can verify
  the copy. Reads only; nothing in the repo is modified or deleted.

.EXAMPLE
  .\make_transfer_package.ps1 -Tier 1
.EXAMPLE
  .\make_transfer_package.ps1 -Tier 2 -SkipHash
#>
[CmdletBinding()]
param(
    [ValidateSet(1, 2, 3)] [int] $Tier = 1,
    [string] $Dest = "transfer_to_laptop",
    [switch] $SkipHash,
    [switch] $Force
)

$ErrorActionPreference = "Stop"

# Checkpoints still in use. Everything else under runs/ is superseded and
# its weights are left behind on purpose -- the results.json files still
# travel, so every number in STATUS.md stays auditable.
$Checkpoints = @(
    "runs/20261006T120718Z_crnn_train_s1/best_model.pt",
    "runs/20260930T052116Z_modernizer_pretrain_s3/best_model.pt",
    "runs/20261007T144601Z_modernizer_finetune_s2_b3/best_model.pt",
    "runs/20261007T131859Z_modernizer_finetune_s2_b4_T1.5/best_model.pt"
)

# Incomplete runs (no results.json), per STATUS.md section 6.
$SkipRuns = @(
    "20261004T111145Z_crnn_train_s1",
    "20261004T134542Z_crnn_train_s1",
    "20261007T031640Z_modernizer_finetune_s2_b3"
)

if (-not (Test-Path "CLAUDE.md")) {
    throw "Run this from the repo root (CLAUDE.md not found here)."
}
if ((Test-Path $Dest) -and (-not $Force)) {
    throw "$Dest already exists. Delete it, or pass -Force to add to it."
}
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

function Copy-Tree {
    param([string] $From, [string] $To, [string[]] $ExcludeFiles, [string[]] $ExcludeDirs)

    if (-not (Test-Path $From)) {
        Write-Host "  SKIP (missing): $From" -ForegroundColor Yellow
        return
    }
    $rc = @($From, $To, "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NP", "/R:2", "/W:2")
    if ($ExcludeFiles) { $rc += "/XF"; $rc += $ExcludeFiles }
    if ($ExcludeDirs)  { $rc += "/XD"; $rc += $ExcludeDirs }

    robocopy @rc | Out-Null
    # robocopy: 0-7 are success, 8+ are real failures. Reset afterwards --
    # a success code of 1 ("files copied") would otherwise leak out as this
    # script's own non-zero exit status.
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE): $From" }
    $global:LASTEXITCODE = 0
    Write-Host "  copied: $From" -ForegroundColor Green
}

function Copy-One {
    param([string] $RelPath)

    if (-not (Test-Path $RelPath)) {
        Write-Host "  SKIP (missing): $RelPath" -ForegroundColor Yellow
        return
    }
    $target = Join-Path $Dest $RelPath
    New-Item -ItemType Directory -Force -Path (Split-Path $target -Parent) | Out-Null
    Copy-Item -Path $RelPath -Destination $target -Force
    $mb = [math]::Round((Get-Item $RelPath).Length / 1MB, 1)
    Write-Host "  copied: $RelPath ($mb MB)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Building tier-$Tier package in $Dest/" -ForegroundColor Cyan
Write-Host ""

# --- Tier 1 -----------------------------------------------------------
Write-Host "Tier 1: run records (no weights), caches, live checkpoints, vocab"

# Every run folder minus weights. The two cache folders come along here
# too -- logprobs.npy is .npy, not .pt, so it is not excluded.
Copy-Tree -From "runs" -To (Join-Path $Dest "runs") `
          -ExcludeFiles @("*.pt", "*.pth", "*.ckpt") -ExcludeDirs $SkipRuns

foreach ($ckpt in $Checkpoints) { Copy-One $ckpt }
Copy-One "data/modernizer_vocab.json"

# --- Tier 2 -----------------------------------------------------------
if ($Tier -ge 2) {
    Write-Host ""
    Write-Host "Tier 2: S2 renders and the built text corpora"
    Copy-Tree -From "data\s2"       -To (Join-Path $Dest "data\s2")
    Copy-Tree -From "data\s2_extra" -To (Join-Path $Dest "data\s2_extra")

    # Text corpora only -- not extracted/ (re-extractable from the zip) and
    # not the 35 MB zip itself (the laptop already downloaded it).
    Get-ChildItem "data\raw_corpus" -File |
        Where-Object { $_.Extension -in @(".txt", ".jsonl") } |
        ForEach-Object { Copy-One ("data/raw_corpus/" + $_.Name) }
}

# --- Tier 3 -----------------------------------------------------------
if ($Tier -ge 3) {
    Write-Host ""
    Write-Host "Tier 3: S1 (12 GB, 23k files -- this is the slow one)"
    Copy-Tree -From "data\s1" -To (Join-Path $Dest "data\s1")
}

# --- Manifest ---------------------------------------------------------
$files = Get-ChildItem $Dest -Recurse -File |
         Where-Object { $_.Name -ne "MANIFEST.sha256" }
$totalGb = [math]::Round(($files | Measure-Object -Property Length -Sum).Sum / 1GB, 2)

Write-Host ""
if ($SkipHash) {
    Write-Host "Skipping checksums (-SkipHash). Verify the copy some other way." -ForegroundColor Yellow
} else {
    Write-Host "Hashing $($files.Count) files (~10 s/GB)..." -ForegroundColor Cyan
    $destFull = (Resolve-Path $Dest).Path
    $lines = foreach ($f in $files) {
        $rel = $f.FullName.Substring($destFull.Length + 1).Replace("\", "/")
        $hash = (Get-FileHash -Algorithm SHA256 -Path $f.FullName).Hash.ToLower()
        "$hash  $rel"      # two spaces: sha256sum -c format
    }
    # ASCII with no BOM and LF-only endings. Both matter: sha256sum chokes
    # on a UTF-8 BOM, and WriteAllLines' default CRLF makes it read the \r
    # as part of every filename ("No such file or directory" on all lines).
    $body = ($lines -join "`n") + "`n"
    [System.IO.File]::WriteAllText(
        (Join-Path $destFull "MANIFEST.sha256"), $body, [System.Text.ASCIIEncoding]::new())
    Write-Host "  wrote $Dest/MANIFEST.sha256" -ForegroundColor Green
}

Write-Host ""
Write-Host "Done. $($files.Count) files, $totalGb GB in $Dest/" -ForegroundColor Cyan
Write-Host "Next: copy its CONTENTS over the laptop repo root (merge, do not replace),"
Write-Host "then 'sha256sum -c MANIFEST.sha256' and the Part 5 smoke test."
Write-Host ""

exit 0
