from __future__ import annotations

import json
import os
import random
import sys
import time

from openai import OpenAI

TOTAL_SAMPLES = 55
MIN_SAMPLES = 50
TRAIN_RATIO = 0.90
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, ".."))
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.jsonl")
TEST_PATH = os.path.join(OUTPUT_DIR, "test.jsonl")
LEGACY_OUTPUT_DIR = os.path.join(ROOT_DIR, "OpenAI")
LEGACY_TRAIN_PATH = os.path.join(LEGACY_OUTPUT_DIR, "train.jsonl")
LEGACY_TEST_PATH = os.path.join(LEGACY_OUTPUT_DIR, "test.jsonl")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

TOPICS = [
    "variáveis e tipos de dados",
    "estruturas condicionais (if/elif/else)",
    "laços de repetição (for e while)",
    "funções e escopo",
    "listas e operações com listas",
    "dicionários em Python",
    "compreensão de listas (list comprehension)",
    "manipulação de strings",
    "tratamento de exceções (try/except)",
    "leitura e escrita de arquivos",
    "módulos e imports",
    "programação orientada a objetos (classes e objetos)",
    "herança e polimorfismo",
    "funções lambda e map/filter",
    "decoradores (decorators)",
    "geradores (generators)",
    "bibliotecas padrão: os, sys, math, datetime",
    "numpy básico",
    "pandas básico",
    "debug e boas práticas de código",
]

SYSTEM_PROMPT = """Você é um professor de Python. 
Sua tarefa é criar um par de instrução e resposta para treinar um modelo de linguagem.
Responda SOMENTE com um JSON válido no formato abaixo, sem qualquer texto fora do JSON:

{
  "prompt": "<pergunta ou instrução clara sobre Python>",
  "response": "<resposta completa, didática e com exemplos de código quando pertinente>"
}"""


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _exit_if_auth_error(err: BaseException) -> None:
    status = getattr(err, "status_code", None)
    body = str(err).lower()
    if status == 401 or "invalid_api_key" in body or "incorrect api key" in body:
        print(
            "\nErro de autenticação OpenAI (401): a chave não foi aceite.\n"
            "• Gera uma chave nova em https://platform.openai.com/account/api-keys\n"
            "• Cola a chave completa (costuma começar por sk-proj- ou sk- e é bem longa).\n"
            "• No terminal: export OPENAI_API_KEY='a_tua_chave_aqui'\n"
            "• Não uses o texto literal \"sk-...\" dos exemplos — isso não é uma chave válida.\n"
            "• Confirma que tens crédito / billing ativo na conta OpenAI.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    if status == 429 and "insufficient_quota" in body:
        print(
            "\nErro de quota OpenAI (429 insufficient_quota): a chave está válida, mas sem crédito/cota.\n"
            "• Verifique billing em https://platform.openai.com/settings/organization/billing\n"
            "• Depois tente novamente: python data/generate_dataset.py\n",
            file=sys.stderr,
        )
        sys.exit(1)


def generate_sample(client: OpenAI, topic: str) -> dict | None:
    user_msg = (
        f"Crie um par de instrução e resposta sobre o seguinte tópico de Python: {topic}. "
        "A pergunta deve ser algo que um aluno iniciante ou intermediário faria. "
        "A resposta deve ser detalhada e incluir exemplos de código Python."
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
            max_tokens=600,
        )
        raw = completion.choices[0].message.content or ""
        data = json.loads(_strip_json_fence(raw))

        if data.get("prompt") and data.get("response"):
            return {"prompt": data["prompt"].strip(), "response": data["response"].strip()}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  [AVISO] Resposta inválida para '{topic}': {e}")
    except Exception as e:
        _exit_if_auth_error(e)
        print(f"  [ERRO] Chamada à API falhou para '{topic}': {e}")

    return None


def main():
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print("Defina OPENAI_API_KEY no ambiente antes de executar.", file=sys.stderr)
        sys.exit(1)
    if api_key in {"sk-...", "sk-...."} or (api_key.startswith("sk-") and len(api_key) < 20):
        print(
            "OPENAI_API_KEY parece incompleta ou é um placeholder (ex.: sk-...).\n"
            "Usa a chave real copiada de https://platform.openai.com/account/api-keys",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    print(f"Gerando {TOTAL_SAMPLES} amostras no domínio de Programação Python...")
    samples = []

    topic_pool = (TOPICS * ((TOTAL_SAMPLES // len(TOPICS)) + 2))[:TOTAL_SAMPLES]
    random.shuffle(topic_pool)

    for i, topic in enumerate(topic_pool, start=1):
        print(f"  [{i:02d}/{TOTAL_SAMPLES}] tópico: {topic}")
        sample = generate_sample(client, topic)

        if sample:
            samples.append(sample)
        else:
            print(f"  [!] Amostra {i} descartada, tentando novamente...")
            fallback = random.choice(TOPICS)
            sample2 = generate_sample(client, fallback)
            if sample2:
                samples.append(sample2)

        time.sleep(0.5)

    print(f"\nTotal de amostras coletadas: {len(samples)}")
    if len(samples) < MIN_SAMPLES:
        print(
            f"Erro: são necessárias pelo menos {MIN_SAMPLES} amostras (obtidas: {len(samples)}).",
            file=sys.stderr,
        )
        sys.exit(1)

    random.shuffle(samples)

    split_idx = int(len(samples) * TRAIN_RATIO)
    train_data = samples[:split_idx]
    test_data = samples[split_idx:]

    def save_jsonl(data: list[dict], path: str):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Salvo: {path}  ({len(data)} amostras)")

    save_jsonl(train_data, TRAIN_PATH)
    save_jsonl(test_data, TEST_PATH)

    if os.path.isdir(LEGACY_OUTPUT_DIR):
        save_jsonl(train_data, LEGACY_TRAIN_PATH)
        save_jsonl(test_data, LEGACY_TEST_PATH)
        print("  (Cópia adicional também salva em ./OpenAI para compatibilidade.)")

    print("\nDataset gerado com sucesso!")
    print(f"  Treino : {len(train_data)} amostras → {TRAIN_PATH}")
    print(f"  Teste  : {len(test_data)}  amostras → {TEST_PATH}")


if __name__ == "__main__":
    main()
