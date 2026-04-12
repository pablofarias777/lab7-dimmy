"""
Passo 1 – Engenharia de Dados Sintéticos

Gera um dataset de instrução no domínio de Programação Python usando a API da OpenAI.
Produz pelo menos 50 pares (prompt, response) e salva em formato .jsonl
dividido em treino (90%) e teste (10%).

Uso:
    export OPENAI_API_KEY="sua-chave"
    python data/generate_dataset.py
"""

import os
import json
import random
import time
from openai import OpenAI

# ── configurações ──────────────────────────────────────────────────────────────
TOTAL_SAMPLES   = 55        # gera um pouco acima do mínimo de 50
TRAIN_RATIO     = 0.90
OUTPUT_DIR      = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH      = os.path.join(OUTPUT_DIR, "train.jsonl")
TEST_PATH       = os.path.join(OUTPUT_DIR, "test.jsonl")
MODEL           = "gpt-3.5-turbo"
# ───────────────────────────────────────────────────────────────────────────────

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Tópicos variados para forçar diversidade no dataset
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


def generate_sample(topic: str) -> dict | None:
    """Chama a API da OpenAI e retorna um par (prompt, response)."""
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
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.8,
            max_tokens=600,
        )
        raw = completion.choices[0].message.content.strip()
        data = json.loads(raw)

        # valida que os campos existem e não estão vazios
        if data.get("prompt") and data.get("response"):
            return {"prompt": data["prompt"].strip(), "response": data["response"].strip()}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  [AVISO] Resposta inválida para '{topic}': {e}")
    except Exception as e:
        print(f"  [ERRO] Chamada à API falhou para '{topic}': {e}")

    return None


def main():
    print(f"Gerando {TOTAL_SAMPLES} amostras no domínio de Programação Python...")
    samples = []

    # embaralha os tópicos e repete ciclicamente se precisar de mais amostras
    topic_pool = (TOPICS * ((TOTAL_SAMPLES // len(TOPICS)) + 2))[:TOTAL_SAMPLES]
    random.shuffle(topic_pool)

    for i, topic in enumerate(topic_pool, start=1):
        print(f"  [{i:02d}/{TOTAL_SAMPLES}] tópico: {topic}")
        sample = generate_sample(topic)

        if sample:
            samples.append(sample)
        else:
            print(f"  [!] Amostra {i} descartada, tentando novamente...")
            # uma segunda tentativa com tópico diferente
            fallback = random.choice(TOPICS)
            sample2 = generate_sample(fallback)
            if sample2:
                samples.append(sample2)

        # respeita o rate-limit da OpenAI (max ~60 req/min no tier gratuito)
        time.sleep(0.5)

    print(f"\nTotal de amostras coletadas: {len(samples)}")

    # embaralha antes de dividir para evitar viés de ordenação
    random.shuffle(samples)

    split_idx    = int(len(samples) * TRAIN_RATIO)
    train_data   = samples[:split_idx]
    test_data    = samples[split_idx:]

    def save_jsonl(data: list[dict], path: str):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Salvo: {path}  ({len(data)} amostras)")

    save_jsonl(train_data, TRAIN_PATH)
    save_jsonl(test_data,  TEST_PATH)

    print("\nDataset gerado com sucesso!")
    print(f"  Treino : {len(train_data)} amostras → {TRAIN_PATH}")
    print(f"  Teste  : {len(test_data)}  amostras → {TEST_PATH}")


if __name__ == "__main__":
    main()
