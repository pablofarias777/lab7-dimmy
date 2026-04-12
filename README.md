# Lab 07 – Especialização de LLMs com LoRA e QLoRA

Fine-tuning de um modelo de linguagem fundacional (Llama 2 7B) utilizando PEFT/LoRA e quantização QLoRA, viabilizando o treinamento em hardware limitado (GPU com ~16GB de VRAM ou menos).

## Domínio escolhido

**Programação Python** — o modelo é especializado para responder perguntas e resolver exercícios de Python para iniciantes e intermediários.

## Estrutura do projeto

```
lab07-lora-qlora/
├── README.md
├── requirements.txt
├── data/
│   ├── generate_dataset.py     # Passo 1: geração do dataset sintético via OpenAI
│   ├── train.jsonl             # Split de treino (90%)
│   └── test.jsonl              # Split de teste (10%)
└── train/
    └── finetune.py             # Passos 2, 3 e 4: QLoRA + LoRA + SFTTrainer
```

## Como reproduzir

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Gerar o dataset sintético

Configure sua chave da OpenAI:

```bash
export OPENAI_API_KEY="sua-chave-aqui"
```

Execute o script de geração:

```bash
python data/generate_dataset.py
```

Isso vai gerar os arquivos `data/train.jsonl` e `data/test.jsonl`.

### 3. Executar o fine-tuning

> **Requisito:** GPU com suporte a CUDA e pelo menos 16GB de VRAM (ex: T4, A10, A100). Recomenda-se usar Google Colab Pro ou Kaggle com GPU T4.

```bash
python train/finetune.py
```

O modelo adaptador será salvo em `./llama2-python-lora/`.

## Hiperparâmetros principais

| Parâmetro | Valor |
|---|---|
| Modelo base | `meta-llama/Llama-2-7b-hf` |
| Quantização | 4-bit NF4 (QLoRA) |
| LoRA Rank (r) | 64 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.1 |
| Otimizador | `paged_adamw_32bit` |
| LR Scheduler | `cosine` |
| Warmup Ratio | 0.03 |

## Nota de uso de IA

Partes geradas/complementadas com IA, revisadas por [Pablo Farias].
