#!/usr/bin/env bash
# =============================================================================
#  run_pipeline.sh — Full IR Challenge Pipeline (nohup-safe)
# =============================================================================
#
#  Usage (runs in foreground, logs to ./logs/):
#    bash run_pipeline.sh
#
#  Usage (detached, fully background):
#    nohup bash run_pipeline.sh > logs/master.log 2>&1 &
#    echo "PID: $!"               # save this to kill if needed
#    tail -f logs/master.log      # watch progress from another terminal
#
#  Skip steps already done:
#    SKIP_DOWNLOAD=1   bash run_pipeline.sh   # skip kaggle download
#    SKIP_SUB1=1       bash run_pipeline.sh   # skip submission 1
#    START_AT=3        bash run_pipeline.sh   # start from submission 3
# IF HAVE DATA - SKIP_DOWNLOAD=1 nohup bash run_pipeline.sh > logs/master.log 2>&1 &
#
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR" data submissions models

# ── Env vars with defaults ────────────────────────────────────────────────────
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
START_AT="${START_AT:-1}"          # start at submission N (1–5)
PYTHON="${PYTHON:-python3}"

# ── Timestamp helper ──────────────────────────────────────────────────────────
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# ── Logging helper ────────────────────────────────────────────────────────────
log() {
    local level="$1"; shift
    echo "[$(ts)] [$level] $*"
}

# ── Step runner: logs stdout+stderr to file, shows live, records timing ───────
run_step() {
    local step_name="$1"
    local log_file="$LOG_DIR/${step_name}.log"
    shift
    local cmd=("$@")

    log "START" "===== $step_name ====="
    log "INFO"  "Command: ${cmd[*]}"
    log "INFO"  "Log file: $log_file"

    local start_ts
    start_ts=$(date +%s)

    # tee: writes to log file AND stdout (visible in master.log / terminal)
    if "${cmd[@]}" 2>&1 | tee "$log_file"; then
        local elapsed=$(( $(date +%s) - start_ts ))
        log "OK"    "===== $step_name DONE in ${elapsed}s ====="
        echo ""
    else
        local exit_code=$?
        local elapsed=$(( $(date +%s) - start_ts ))
        log "ERROR" "===== $step_name FAILED (exit $exit_code) after ${elapsed}s ====="
        log "ERROR" "Check $log_file for details."
        exit $exit_code
    fi
}

# ── Check Python ──────────────────────────────────────────────────────────────
check_python() {
    if ! command -v "$PYTHON" &>/dev/null; then
        log "ERROR" "Python not found: $PYTHON"
        log "ERROR" "Set PYTHON=python or PYTHON=python3.11 etc."
        exit 1
    fi
    log "INFO" "Python: $($PYTHON --version)"
}

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

log "INFO" "=============================================="
log "INFO" "  IR Challenge Pipeline Starting"
log "INFO" "  Working dir: $SCRIPT_DIR"
log "INFO" "  START_AT:    $START_AT"
log "INFO" "  SKIP_DOWNLOAD: $SKIP_DOWNLOAD"
log "INFO" "=============================================="
echo ""

check_python

# ── Step 0: Install dependencies ─────────────────────────────────────────────
if [[ "$START_AT" -le 0 ]] || [[ ! -f "$LOG_DIR/00_setup.log" ]]; then
    run_step "00_setup" bash setup.sh
else
    log "SKIP" "Step 0 (setup) — already done (delete logs/00_setup.log to redo)"
fi

# ── Step 1: Download data ─────────────────────────────────────────────────────
if [[ "$SKIP_DOWNLOAD" -eq 1 ]]; then
    log "SKIP" "Step 1 (download) — SKIP_DOWNLOAD=1"
elif [[ -f "data/corpus.jsonl" && -f "data/test_queries.csv" ]]; then
    log "SKIP" "Step 1 (download) — data files already in ./data/"
else
    run_step "01_download" "$PYTHON" download_data.py
fi

# Verify data files exist before proceeding
for f in data/corpus.jsonl data/test_queries.csv data/train_queries.csv data/train_qrels.csv; do
    if [[ ! -f "$f" ]]; then
        log "ERROR" "Required file missing: $f"
        log "ERROR" "Run: python download_data.py  (and make sure Kaggle credentials are set)"
        exit 1
    fi
done
log "INFO" "All data files present."
echo ""

# ── Step 2: Submission 1 — BM25 Baseline ─────────────────────────────────────
if [[ "$START_AT" -le 1 ]]; then
    if [[ -f "submissions/submission_01_bm25_baseline.csv" ]] && [[ "${SKIP_SUB1:-0}" -eq 1 ]]; then
        log "SKIP" "Submission 1 — SKIP_SUB1=1"
    else
        run_step "sub1_bm25_baseline" "$PYTHON" src/01_bm25_baseline.py
    fi
else
    log "SKIP" "Submission 1 (START_AT=$START_AT)"
fi

# ── Step 3: Submission 2 — Query Expansion ────────────────────────────────────
if [[ "$START_AT" -le 2 ]]; then
    if [[ -f "submissions/submission_02_bm25_qe.csv" ]] && [[ "${SKIP_SUB2:-0}" -eq 1 ]]; then
        log "SKIP" "Submission 2 — SKIP_SUB2=1"
    else
        run_step "sub2_query_expansion" "$PYTHON" src/02_query_expansion.py
    fi
else
    log "SKIP" "Submission 2 (START_AT=$START_AT)"
fi

# ── Step 4: Submission 3 — Hybrid Dense + BM25 ───────────────────────────────
if [[ "$START_AT" -le 3 ]]; then
    if [[ -f "submissions/submission_03_hybrid_rrf.csv" ]] && [[ "${SKIP_SUB3:-0}" -eq 1 ]]; then
        log "SKIP" "Submission 3 — SKIP_SUB3=1"
    else
        run_step "sub3_hybrid_dense" "$PYTHON" src/03_dense_hybrid.py
    fi
else
    log "SKIP" "Submission 3 (START_AT=$START_AT)"
fi

# ── Step 5: Submission 4 — LTR Reranker ──────────────────────────────────────
if [[ "$START_AT" -le 4 ]]; then
    if [[ -f "submissions/submission_04_ltr.csv" ]] && [[ "${SKIP_SUB4:-0}" -eq 1 ]]; then
        log "SKIP" "Submission 4 — SKIP_SUB4=1"
    else
        run_step "sub4_ltr_reranker" "$PYTHON" src/04_ltr_reranker.py
    fi
else
    log "SKIP" "Submission 4 (START_AT=$START_AT)"
fi

# ── Step 6: Submission 5 — Full Ensemble ─────────────────────────────────────
if [[ "$START_AT" -le 5 ]]; then
    if [[ -f "submissions/submission_05_full_ensemble.csv" ]] && [[ "${SKIP_SUB5:-0}" -eq 1 ]]; then
        log "SKIP" "Submission 5 — SKIP_SUB5=1"
    else
        run_step "sub5_full_ensemble" "$PYTHON" src/05_ensemble.py
    fi
else
    log "SKIP" "Submission 5 (START_AT=$START_AT)"
fi

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
log "INFO" "=============================================="
log "INFO" "  PIPELINE COMPLETE"
log "INFO" "=============================================="
echo ""
log "INFO" "Submissions generated:"
for f in submissions/submission_*.csv; do
    if [[ -f "$f" ]]; then
        lines=$(wc -l < "$f")
        log "INFO" "  ✓  $f  ($lines lines)"
    fi
done

echo ""
log "INFO" "Local nDCG@100 evaluation (training queries):"
"$PYTHON" src/evaluator.py submissions/submission_*.csv 2>&1 | tee "$LOG_DIR/final_evaluation.log"

echo ""
log "INFO" "Upload submissions to Kaggle:"
log "INFO" "  https://www.kaggle.com/c/ir-20261-rc/submissions"
log "INFO" "  Start with submission_01, work up to submission_05."
log "INFO" "  Daily limit: 20 submissions."
echo ""
log "INFO" "All logs in: $LOG_DIR/"