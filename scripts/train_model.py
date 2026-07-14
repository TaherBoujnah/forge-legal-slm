"""
Fine-tune Llama-3.2-3B-Instruct on legal contract extraction data.

Native Hugging Face pipeline (transformers + peft + bitsandbytes + trl) --
no Unsloth, no xformers. Tuned for a single 12GB-class GPU (e.g. RTX 5070)
using 4-bit QLoRA.

NEW: a thermal-safety callback that monitors GPU temperature during training
and PAUSES (not aborts) if it gets too hot, resuming once it cools down.
This can't replace proper laptop cooling/airflow, but it stops training from
being the thing that pushes the GPU into a hardware shutdown -- it backs off
before that point instead of running flat-out for hours straight.

Setup (run once):
    pip uninstall -y unsloth unsloth-zoo torchao xformers
    pip install transformers datasets peft trl bitsandbytes accelerate pynvml

Run:
    python scripts/train_model.py
"""

import inspect
import json
import os
import random
import time

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"  # ungated mirror, no HF login needed
DATASET_PATH = "data/synthetic/training_dataset_cleaned.jsonl"  
                                                          
                                                          
OUTPUT_DIR = "outputs_hf_native"
ADAPTER_DIR = "models/legal_qlora_adapter"
EVAL_FRACTION = 0.15  # held-out slice, never trained on
SEED = 42
MAX_SEQ_LENGTH = 1536  # checked against dataset below; raise if truncation warnings appear

# --- Thermal safety settings ---
GPU_TEMP_PAUSE_CELSIUS = 83   # pause training when GPU hits this temperature
GPU_TEMP_RESUME_CELSIUS = 75  # resume once it cools back down to this
TEMP_CHECK_INTERVAL_STEPS = 5  # how often (in training steps) to check temperature
PAUSE_POLL_SECONDS = 15        # how often to re-check temp while paused


# ------------------------------------------------------------------
# Thermal safety callback
# ------------------------------------------------------------------

class ThermalSafetyCallback(TrainerCallback):
    """Monitors GPU temperature via pynvml and pauses training (sleeping in
    a loop, not killing the process) whenever the GPU crosses
    GPU_TEMP_PAUSE_CELSIUS, resuming once it drops back to
    GPU_TEMP_RESUME_CELSIUS. This is a software-level safety margin, NOT a
    substitute for proper laptop airflow/cooling -- it just stops training
    from being the specific thing that pushes the GPU into a hardware
    thermal shutdown during multi-hour runs.

    Requires: pip install pynvml
    If pynvml or an NVIDIA GPU isn't available, this callback disables
    itself with a warning rather than crashing training.
    """

    def __init__(self, pause_temp=GPU_TEMP_PAUSE_CELSIUS, resume_temp=GPU_TEMP_RESUME_CELSIUS,
                 check_every_n_steps=TEMP_CHECK_INTERVAL_STEPS, poll_seconds=PAUSE_POLL_SECONDS):
        self.pause_temp = pause_temp
        self.resume_temp = resume_temp
        self.check_every_n_steps = check_every_n_steps
        self.poll_seconds = poll_seconds
        self.enabled = True
        self.handle = None
        self.total_pause_seconds = 0.0
        self.pause_events = 0

        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            print(f"[ThermalSafety] Monitoring enabled -- will pause training above "
                  f"{self.pause_temp}°C, resume below {self.resume_temp}°C.")
        except Exception as e:
            print(f"[ThermalSafety] Could not initialize GPU temperature monitoring "
                  f"({e}). Thermal safety is DISABLED for this run -- install pynvml "
                  f"('pip install pynvml') and ensure an NVIDIA GPU is present to enable it.")
            self.enabled = False

    def _get_temp(self):
        try:
            return self.pynvml.nvmlDeviceGetTemperature(
                self.handle, self.pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            return None

    def on_step_end(self, args, state, control, **kwargs):
        if not self.enabled:
            return control

        if state.global_step % self.check_every_n_steps != 0:
            return control

        temp = self._get_temp()
        if temp is None:
            return control

        if temp >= self.pause_temp:
            print(f"\n[ThermalSafety] GPU at {temp}°C (>= {self.pause_temp}°C threshold). "
                  f"Pausing training to let it cool down...")
            pause_start = time.time()
            self.pause_events += 1

            while True:
                time.sleep(self.poll_seconds)
                current_temp = self._get_temp()
                if current_temp is None:
                    print("[ThermalSafety] Lost temperature reading while paused -- "
                          "resuming training to avoid getting stuck.")
                    break
                print(f"[ThermalSafety] ...still cooling, currently {current_temp}°C "
                      f"(need <= {self.resume_temp}°C to resume)")
                if current_temp <= self.resume_temp:
                    break

            paused_for = time.time() - pause_start
            self.total_pause_seconds += paused_for
            print(f"[ThermalSafety] Resuming training after {paused_for:.0f}s pause "
                  f"(total paused so far this run: {self.total_pause_seconds:.0f}s "
                  f"across {self.pause_events} pause event(s)).\n")

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.enabled and self.pause_events > 0:
            print(f"\n[ThermalSafety] Training finished. Paused {self.pause_events} "
                  f"time(s) for a total of {self.total_pause_seconds:.0f}s "
                  f"({self.total_pause_seconds / 60:.1f} min) due to GPU temperature. "
                  f"If this happened often, consider improving laptop airflow/cooling "
                  f"or lowering the GPU power limit (e.g. via MSI Afterburner or "
                  f"'nvidia-smi -pl <watts>') for future runs.")


def load_and_split_dataset(path: str, eval_fraction: float, seed: int):
    """Load the ChatML JSONL dataset and carve out a real held-out eval split.

    NOTE: this is a random split of the *same* synthetic generator's output --
    it tells you if the model overfits within-distribution, but it is NOT a
    substitute for testing on real/unseen contract text. Keep a separate
    manual eval set (see scripts/evaluate_model.py) for the metric that
    actually matters.
    """
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))

    random.Random(seed).shuffle(examples)
    n_eval = max(1, int(len(examples) * eval_fraction))
    eval_examples = examples[:n_eval]
    train_examples = examples[n_eval:]

    print(f"Loaded {len(examples)} total examples -> "
          f"{len(train_examples)} train / {len(eval_examples)} eval")

    return (
        Dataset.from_list(train_examples),
        Dataset.from_list(eval_examples),
    )


def check_max_length(dataset: Dataset, tokenizer, max_length: int):
    """Sanity check: warn if training examples are being silently truncated.

    Truncation here doesn't just cut context -- it can cut off part of the
    assistant's JSON answer, which quietly corrupts the training signal.
    """
    lengths = []
    for ex in dataset:
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        lengths.append(len(tokenizer(text)["input_ids"]))

    lengths.sort()
    p50 = lengths[len(lengths) // 2]
    p95 = lengths[int(len(lengths) * 0.95)]
    over_limit = sum(1 for l in lengths if l > max_length)

    print(f"Token lengths -- median: {p50}, p95: {p95}, max: {max(lengths)}")
    if over_limit:
        print(f"WARNING: {over_limit}/{len(lengths)} examples exceed "
              f"max_length={max_length} and will be truncated. "
              f"Consider raising MAX_SEQ_LENGTH.")


def build_sft_config(**kwargs):
    """Builds an SFTConfig while tolerating parameter renames across trl
    versions -- max_seq_length/max_length and eval_strategy/
    evaluation_strategy have both changed names in different trl releases,
    and nightly/bleeding-edge installs are exactly where this bites.
    Only passes a given key if the installed SFTConfig actually accepts it,
    trying known alternate names before giving up on that key entirely.
    """
    accepted_params = set(inspect.signature(SFTConfig.__init__).parameters)

    RENAME_ALTERNATES = {
        "max_seq_length": ["max_seq_length", "max_length"],
        "eval_strategy": ["eval_strategy", "evaluation_strategy"],
    }

    resolved_kwargs = {}
    for key, value in kwargs.items():
        candidates = RENAME_ALTERNATES.get(key, [key])
        matched = next((c for c in candidates if c in accepted_params), None)
        if matched:
            resolved_kwargs[matched] = value
        else:
            print(f"[build_sft_config] WARNING: none of {candidates} are accepted by "
                  f"the installed SFTConfig -- skipping '{key}'={value!r}. Check your "
                  f"trl version if this setting matters to you.")

    return SFTConfig(**resolved_kwargs)


def build_trainer(model, tokenizer, sft_config, train_dataset, eval_dataset, formatting_func, callbacks):
    """Instantiate SFTTrainer while tolerating the tokenizer/processing_class
    rename across trl versions -- relevant since nightly builds are exactly
    where this kind of breaking change shows up without warning.
    """
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    tokenizer_kwarg = "processing_class" if "processing_class" in trainer_params else "tokenizer"

    return SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        callbacks=callbacks,
        **{tokenizer_kwarg: tokenizer},
    )


def main():
    print("Starting training: native HF pipeline, RTX 5070 compatibility mode...")

    if not torch.cuda.is_available():
        print("CRITICAL ERROR: CUDA not detected. Check driver/torch install.")
        return
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Run "
            f"scripts/synthetic-data-generator.py first, or point DATASET_PATH "
            f"at your cleaned dataset."
        )

    compute_dtype = torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading {MODEL_ID} in 4-bit ({compute_dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset, eval_dataset = load_and_split_dataset(DATASET_PATH, EVAL_FRACTION, SEED)
    check_max_length(train_dataset, tokenizer, MAX_SEQ_LENGTH)

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    sft_config = build_sft_config(
        output_dir=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    thermal_callback = ThermalSafetyCallback()

    trainer = build_trainer(
        model, tokenizer, sft_config, train_dataset, eval_dataset, formatting_func,
        callbacks=[thermal_callback],
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving best adapter to {ADAPTER_DIR}...")
    trainer.save_model(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print("Done. Next: run scripts/evaluate_model.py against a real held-out "
          "set before trusting these weights.")


if __name__ == "__main__":
    main()