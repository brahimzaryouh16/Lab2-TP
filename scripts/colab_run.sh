#!/usr/bin/env bash
# Lab 2 on Colab: MNIST CNN, ViT, Faster R-CNN, VGG16/AlexNet.
#
# One model per `colab exec` and the JSON report is downloaded immediately,
# so a session that dies loses only the stage in flight. Re-running skips
# stages whose report already exists locally.
#
# Usage (from WSL):
#   wsl -d Ubuntu -- bash /mnt/c/Users/ibrahm/Desktop/lab2/scripts/colab_run.sh
#   STAGES="cnn vit" REUSE=1 ... colab_run.sh run
#
# Stages: setup | run | all (default). Knobs: SESSION, GPU, REUSE=1,
# STAGES, KEEP_SESSION=1, RETRIES.
set -euo pipefail

COLAB="${COLAB:-$HOME/.venvs/colab-cli/bin/colab}"
REPO_WSL="${REPO_WSL:-/mnt/c/Users/ibrahm/Desktop/lab2}"
SESSION="${SESSION:-lab2}"
GPU="${GPU:-T4}"
REMOTE="${REMOTE:-/content/lab2}"
REUSE="${REUSE:-0}"
RETRIES="${RETRIES:-3}"
STAGE="${1:-all}"
STAGES="${STAGES:-cnn vit frcnn vgg16 alexnet}"
LOG_DIR="$REPO_WSL/logs"
OUT_LOCAL="$REPO_WSL/results"
CODE_TGZ="/tmp/lab2_code.tgz"

log() { echo -e "\n== $* =="; }
die() { echo "error: $*" >&2; exit 1; }

retry() {
  local i=1
  until "$@"; do
    if (( i >= RETRIES )); then return 1; fi
    echo "  (network retry $i/$RETRIES: $*)" >&2
    i=$((i + 1)); sleep 15
  done
}

exec_code() {  # code [timeout]
  printf '%s\n' "$1" | "$COLAB" exec -s "$SESSION" --timeout "${2:-300}"
}
retry_code() {  # code [timeout]
  local code="$1" timeout="${2:-300}" i=1
  until printf '%s\n' "$code" | "$COLAB" exec -s "$SESSION" \
      --timeout "$timeout"; do
    if (( i >= RETRIES )); then return 1; fi
    echo "  (network retry $i/$RETRIES: exec)" >&2
    i=$((i + 1)); sleep 15
  done
}

cleanup() {
  local rc=$?
  if [[ $rc -ne 0 && "${KEEP_SESSION:-0}" != "1" ]]; then
    echo "error: run failed (rc=$rc); stopping session '$SESSION'" >&2
    "$COLAB" stop -s "$SESSION" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

[[ -x "$COLAB" ]] || die "colab CLI not found at $COLAB"
mkdir -p "$LOG_DIR" "$OUT_LOCAL"

if [[ "$STAGE" == "all" || "$STAGE" == "setup" ]]; then
  if [[ "$REUSE" == "1" ]]; then
    log "reusing session '$SESSION' (skipping new/install)"
  else
    log "replacing any existing session '$SESSION'"
    "$COLAB" stop -s "$SESSION" >/dev/null 2>&1 || true

    log "requesting $GPU runtime ('$SESSION')"
    "$COLAB" new -s "$SESSION" --gpu "$GPU" || \
      die "could not create the session; authenticate: $COLAB sessions"
  fi

  log "GPU / torchvision sanity check"
  retry_code "import torch
try:
    import torchvision
    tv = torchvision.__version__
except ImportError:
    tv = None
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('torch:', torch.__version__, '| torchvision:', tv)
assert torch.cuda.is_available(), 'no GPU in this runtime'" 300 || \
    die "kernel unusable; rerun with REUSE=0"

  log "packing Lab 2 code"
  tar --exclude='__pycache__' -czf "$CODE_TGZ" -C "$REPO_WSL" \
    src scripts requirements.txt
  ls -lh "$CODE_TGZ"

  log "preparing remote workspace"
  retry_code "import os
root = '$REMOTE'
os.makedirs(os.path.join(root, 'results'), exist_ok=True)
print('remote dirs:', sorted(os.listdir(root)))" 120

  retry "$COLAB" upload -s "$SESSION" "$CODE_TGZ" "$REMOTE/code.tgz"

  retry_code "import os, tarfile
root = '$REMOTE'
with tarfile.open(os.path.join(root, 'code.tgz')) as tf:
    tf.extractall(root)
print('extracted:', sorted(os.listdir(root)))" 300 || die "code extraction failed"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "run" ]]; then
  for stage in $STAGES; do
    if [[ -f "$OUT_LOCAL/$stage.json" ]]; then
      log "skip $stage: report already downloaded"
      continue
    fi
    case "$stage" in
      cnn)     timeout=1800 ;;
      vit)     timeout=3000 ;;
      frcnn)   timeout=5400 ;;
      vgg16)   timeout=5400 ;;
      alexnet) timeout=3000 ;;
      *)       die "unknown stage '$stage'" ;;
    esac
    log "stage $stage (timeout ${timeout}s)"
    logf="$LOG_DIR/colab_$stage.log"
    set +e
    "$COLAB" exec -s "$SESSION" --timeout "$timeout" \
      --env REMOTE="$REMOTE" --env STAGE="$stage" \
      -f "$REPO_WSL/scripts/colab_runner.py" 2>&1 | tee "$logf"
    rc=${PIPESTATUS[0]}
    set -e
    grep -q "\[done\] $stage" "$logf" || \
      die "stage $stage failed (rc=$rc); see $logf"
    retry "$COLAB" download -s "$SESSION" "$REMOTE/results/$stage.json" \
      "$OUT_LOCAL/$stage.json" || die "could not download $stage report"
  done
fi

log "exporting session log"
"$COLAB" log -s "$SESSION" -o "$LOG_DIR/colab_session.md" || true

if [[ "$STAGE" == "all" || "$STAGE" == "run" ]]; then
  log "stopping session"
  "$COLAB" stop -s "$SESSION" || true
else
  log "leaving session '$SESSION' running for the next stage"
fi

log "done"
for stage in $STAGES; do
  if [[ -f "$OUT_LOCAL/$stage.json" ]]; then
    echo "report: $OUT_LOCAL/$stage.json"
  fi
done
exit 0
