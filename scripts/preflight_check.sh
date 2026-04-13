#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "== Preflight de Entrega (Lab 07) =="
echo

python3 - <<'PY'
from pathlib import Path

for f in [Path("data/generate_dataset.py"), Path("train/finetune.py")]:
    source = f.read_text(encoding="utf-8")
    compile(source, str(f), "exec")
PY
echo "[OK] Sintaxe Python validada."

if [[ ! -f data/train.jsonl || ! -f data/test.jsonl ]]; then
  echo "[ERRO] data/train.jsonl e/ou data/test.jsonl não encontrados."
  echo "       Rode: python3 data/generate_dataset.py"
  exit 1
fi

TRAIN_LINES="$(wc -l < data/train.jsonl | tr -d ' ')"
TEST_LINES="$(wc -l < data/test.jsonl | tr -d ' ')"
TOTAL_LINES="$((TRAIN_LINES + TEST_LINES))"

if (( TOTAL_LINES < 50 )); then
  echo "[ERRO] Dataset com menos de 50 amostras (total atual: ${TOTAL_LINES})."
  exit 1
fi

echo "[OK] Dataset encontrado: treino=${TRAIN_LINES}, teste=${TEST_LINES}, total=${TOTAL_LINES}."

if [[ -d OpenAI ]]; then
  cp data/train.jsonl OpenAI/train.jsonl
  cp data/test.jsonl OpenAI/test.jsonl
  echo "[OK] Cópia em OpenAI/ atualizada."
fi

echo
echo "Pendências manuais antes da entrega:"
echo "1) Executar treinamento em GPU: python3 train/finetune.py"
echo "2) Commitar arquivos e criar tag v1.0"
echo
echo "Preflight concluído."
