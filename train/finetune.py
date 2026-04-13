from __future__ import annotations

import os

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from trl import SFTTrainer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_TRAIN = os.path.join(ROOT, "data", "train.jsonl")
OUTPUT_ADAPTER = os.path.join(ROOT, "llama2-python-lora")
BASE_MODEL = os.getenv("HF_MODEL_ID", "meta-llama/Llama-2-7b-hf")
HF_TOKEN = os.getenv("HF_TOKEN")


def make_formatting_func(tok: PreTrainedTokenizerBase):
    eos = tok.eos_token or ""

    def _fmt(example: dict) -> str:
        prompt = example["prompt"].strip()
        response = example["response"].strip()
        return f"### Instrução:\n{prompt}\n\n### Resposta:\n{response}{eos}"

    return _fmt


def main() -> None:
    if not os.path.isfile(DATA_TRAIN):
        raise SystemExit(
            f"Ficheiro de treino em falta: {DATA_TRAIN}\n"
            "Gera o dataset com: export OPENAI_API_KEY=... && python data/generate_dataset.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    dataset = load_dataset("json", data_files={"train": DATA_TRAIN})["train"]

    training_args = TrainingArguments(
        output_dir=os.path.join(ROOT, "outputs"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        formatting_func=make_formatting_func(tokenizer),
        max_seq_length=512,
        packing=False,
    )

    trainer.train()
    os.makedirs(OUTPUT_ADAPTER, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_ADAPTER)
    tokenizer.save_pretrained(OUTPUT_ADAPTER)
    print(f"Adaptador guardado em: {OUTPUT_ADAPTER}")


if __name__ == "__main__":
    main()
