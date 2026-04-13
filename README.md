# Laboratório 07 — Especialização de LLMs com LoRA e QLoRA

> **Disciplina:** Inteligência Artificial Aplicada  
> **Instituição:** Instituto iCEV  
> **Aluno:** Pablo Ferreira de Andrade Farias  
> **Orientador:** Prof. Dimmy  
> **Entrega:** versão `v1.0`

---

> **Nota de Integridade Acadêmica:**  
> *"Partes geradas/complementadas com IA, revisadas por Pablo Ferreira de Andrade Farias"*

> **Uso de IA:**  
> Ferramentas de IA generativa foram usadas como apoio na estruturação e documentação do projeto. Todo o conteúdo foi revisado criticamente e validado pelo aluno antes da submissão.

---

## Objetivo

Este laboratório implementa um **pipeline completo de fine-tuning** de um LLM para o domínio de **Programação Python**, utilizando técnicas de eficiência de memória:

| Técnica | O que faz | Biblioteca |
|---------|-----------|------------|
| **LoRA** (Low-Rank Adaptation) | Treina apenas uma fração dos parâmetros | `peft` |
| **QLoRA** (Quantized LoRA) | Carrega o modelo em 4-bits para reduzir VRAM | `bitsandbytes` |

O pipeline cobre:
- geração de dataset sintético com OpenAI API;
- divisão treino/teste em `.jsonl`;
- treinamento com `SFTTrainer`;
- salvamento do adaptador LoRA treinado.

---

## Estrutura do Projeto

```text
lab7-dimmy/
├── README.md
├── requirements.txt
├── .gitignore
├── OpenAI/
│   ├── train.jsonl
│   └── test.jsonl
├── data/
│   ├── generate_dataset.py
│   ├── train.jsonl
│   └── test.jsonl
├── train/
│   └── finetune.py
└── scripts/
    ├── preflight_check.sh
    └── finish_delivery.sh
```

---

## Como Executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente

```bash
export OPENAI_API_KEY="sua-chave-aqui"
export OPENAI_MODEL="gpt-4o-mini"  # opcional
```

Para treino com modelos do Hugging Face (quando necessário):

```bash
export HF_TOKEN="hf_..."
# opcional: export HF_MODEL_ID="meta-llama/Llama-2-7b-hf"
```

### 3. Gerar dataset sintético

```bash
python data/generate_dataset.py
```

Resultado esperado:
- pelo menos 50 pares `prompt/response`;
- split aproximado 90/10;
- arquivos em `data/train.jsonl` e `data/test.jsonl`.

Se a pasta `OpenAI/` existir, o script também grava cópia em `OpenAI/train.jsonl` e `OpenAI/test.jsonl`.

### 4. Validar preflight local

```bash
./scripts/preflight_check.sh
```

Esse script valida:
- sintaxe Python dos scripts principais;
- existência dos `.jsonl` de treino e teste;
- mínimo de 50 amostras no total.

### 5. Executar fine-tuning

```bash
python train/finetune.py
```

O adaptador LoRA será salvo em:

```text
./llama2-python-lora/
```

### 6. Fluxo automatizado (opcional)

```bash
./scripts/finish_delivery.sh
# com treino no mesmo fluxo:
# RUN_TRAIN=1 ./scripts/finish_delivery.sh
```

---

## Explicação Técnica

### Passo 1 — Engenharia de Dados Sintéticos

O dataset é gerado via API da OpenAI com pares no formato:

```json
{"prompt":"...","response":"..."}
```

O script:
- cria amostras em tópicos de Programação Python;
- tenta recuperar falhas com fallback de tópico;
- exige mínimo de 50 exemplos válidos antes de salvar.

### Passo 2 — Quantização (QLoRA)

No treinamento, o modelo base é carregado em 4-bit via `BitsAndBytesConfig`:

- `load_in_4bit=True`
- `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_compute_dtype=torch.float16`

Isso reduz consumo de memória e permite treinar em hardware mais limitado.

### Passo 3 — Configuração LoRA

No `LoraConfig` são usados os hiperparâmetros obrigatórios do laboratório:

- `r=64`
- `lora_alpha=16`
- `lora_dropout=0.1`
- `task_type=TaskType.CAUSAL_LM`

### Passo 4 — Pipeline de Treinamento e Otimização

Com `TrainingArguments`, o projeto usa:

- `optim="paged_adamw_32bit"`
- `lr_scheduler_type="cosine"`
- `warmup_ratio=0.03`

Ao final, o adaptador é salvo com `trainer.model.save_pretrained(...)`.

---

## Dependências Principais

| Biblioteca | Versão |
|------------|--------|
| `openai` | `>=1.0.0` |
| `torch` | `>=2.1.0` |
| `transformers` | `>=4.36.0` |
| `peft` | `>=0.7.0` |
| `bitsandbytes` | `>=0.41.0` |
| `accelerate` | `>=0.25.0` |
| `trl` | `>=0.7.0` |
| `datasets` | `>=2.14.0` |

---

## Checklist de Entrega

- [ ] `data/train.jsonl` e `data/test.jsonl` gerados (mínimo 50 pares).
- [ ] Treinamento executado sem erro no ambiente com GPU.
- [ ] Adaptador salvo em `llama2-python-lora/`.
- [ ] Código e datasets enviados no GitHub.
- [ ] Tag/release final marcada como `v1.0`.

Comandos de fechamento:

```bash
git add .
git commit -m "Entrega Lab 07 - LoRA/QLoRA"
git push origin main
git tag -a v1.0 -m "Entrega final Lab 07"
git push origin v1.0
```

---

## Referências

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PEFT Docs](https://huggingface.co/docs/peft)
- [TRL SFTTrainer Docs](https://huggingface.co/docs/trl/sft_trainer)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
