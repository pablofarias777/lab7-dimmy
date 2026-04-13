#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERRO] OPENAI_API_KEY não definido."
  echo "Exemplo: export OPENAI_API_KEY='sua-chave-aqui'"
  exit 1
fi

echo "== 1) Gerando dataset sintético =="
python3 data/generate_dataset.py

echo
echo "== 2) Validando checklist local =="
./scripts/preflight_check.sh

if [[ "${RUN_TRAIN:-0}" == "1" ]]; then
  echo
  echo "== 3) Executando fine-tuning =="
  python3 train/finetune.py
else
  echo
  echo "Treinamento pulado (defina RUN_TRAIN=1 para executar o fine-tuning neste script)."
fi

echo
echo "Próximos comandos para fechar no GitHub:"
echo "  git add ."
echo "  git commit -m 'Entrega Lab 07 - LoRA/QLoRA'"
echo "  git push origin main"
echo "  git tag -a v1.0 -m 'Entrega final Lab 07'"
echo "  git push origin v1.0"
