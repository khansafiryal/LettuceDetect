# Reproduksi paper LettuceDetect (arXiv 2502.17125) di RAGTruth.
# Jalankan dari root repo di terminal VS Code (PowerShell). Jika muncul error
# "running scripts is disabled", awali dengan: powershell -ExecutionPolicy Bypass -File reproduce.ps1
#   ./reproduce.ps1                     # semua tahap, model base
#   ./reproduce.ps1 -Model large          # ModernBERT-large
#   ./reproduce.ps1 -BatchSize 8 -GradAccum 1 -Bf16:$false   # setting asli paper (butuh GPU besar, mis. A100)
#   ./reproduce.ps1 -SkipDownload -SkipPreprocess   # langsung training
# Semua output tercatat di output/logs/<model>_<timestamp>/

param(
    [ValidateSet("base", "large")] [string]$Model = "base",
    # Default: batch 2 x akumulasi 4 = batch efektif 8 (sama dengan paper), bf16 agar muat di GPU 8 GB
    [int]$BatchSize = 2,
    [int]$GradAccum = 4,
    [switch]$Bf16 = $true,
    [int]$Epochs = 6,
    [double]$LearningRate = 1e-5,
    [switch]$SkipDownload,
    [switch]$SkipPreprocess,
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"  # agar print() langsung masuk ke log, bukan tertahan di buffer

$DataDir = "data/ragtruth"
$DataJson = "$DataDir/ragtruth_data.json"
$ModelName = "answerdotai/ModernBERT-$Model"
$OutputDir = "output/lettucedetect-$Model"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = "output/logs/${Model}_$Stamp"
New-Item -ItemType Directory -Force $LogDir | Out-Null

function Run-Step([string]$Name, [string]$LogFile, [scriptblock]$Block) {
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    $start = Get-Date
    "# $Name`n# mulai: $start" | Out-File -Encoding utf8 "$LogDir/$LogFile"
    # stderr (progress bar tqdm) jangan dianggap error fatal oleh PowerShell 5.1
    $ErrorActionPreference = "Continue"
    & $Block 2>&1 | ForEach-Object {
        $line = "$_"
        Write-Host $line
        Add-Content -Encoding utf8 -Path "$LogDir/$LogFile" -Value $line
    }
    $ErrorActionPreference = "Stop"
    if ($LASTEXITCODE -ne 0) { throw "$Name gagal (exit code $LASTEXITCODE). Lihat $LogDir/$LogFile" }
    $dur = (Get-Date) - $start
    "# selesai: $(Get-Date)  durasi: $dur" | Out-File -Encoding utf8 -Append "$LogDir/$LogFile"
    Write-Host "$Name selesai dalam $dur" -ForegroundColor Green
}

# 0. Catat environment
Run-Step "0. Environment" "00_environment.log" {
    "commit: $(git rev-parse HEAD)"
    "model: $ModelName  batch: $BatchSize  grad_accum: $GradAccum (efektif $($BatchSize * $GradAccum))  bf16: $Bf16  epochs: $Epochs  lr: $LearningRate"
    python --version
    python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); print('transformers', transformers.__version__)"
}

# 1. Download RAGTruth
if (-not $SkipDownload) {
    Run-Step "1. Download RAGTruth" "01_download.log" {
        New-Item -ItemType Directory -Force $DataDir | Out-Null
        foreach ($f in "response.jsonl", "source_info.jsonl") {
            curl.exe -fsSL -o "$DataDir/$f" "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/$f"
            "$f : $((Get-Item "$DataDir/$f").Length) bytes"
        }
    }
}

# 2. Preprocessing -> ragtruth_data.json
if (-not $SkipPreprocess) {
    Run-Step "2. Preprocessing" "02_preprocess.log" {
        python lettucedetect/preprocess/preprocess_ragtruth.py --input_dir $DataDir --output_dir $DataDir
        python -c "import json, collections; d = json.load(open('$DataJson', encoding='utf-8')); s = d['samples'] if isinstance(d, dict) else d; print('total', len(s), dict(collections.Counter(x['split'] for x in s)))"
    }
}

# 3. Training
if (-not $SkipTraining) {
    Run-Step "3. Training" "03_train.log" {
        $extra = @()
        if ($Bf16) { $extra += "--bf16" }
        python scripts/train.py --ragtruth-path $DataJson --model-name $ModelName --output-dir $OutputDir --batch-size $BatchSize --grad-accum $GradAccum --epochs $Epochs --learning-rate $LearningRate @extra
    }
}

# 4. Evaluasi di split test
foreach ($level in "example_level", "span_level") {
    Run-Step "4. Evaluasi ($level)" "04_eval_$level.log" {
        python scripts/evaluate.py --model_path $OutputDir --data_path $DataJson --evaluation_type $level
    }
}

Write-Host "`nSelesai. Semua log ada di $LogDir" -ForegroundColor Green
