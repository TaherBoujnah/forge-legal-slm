import sys
import os
import json
import traceback

print("🚀 Script launched successfully! Loading massive AI libraries...", flush=True)

print("📦 Checking core dependencies (NumPy)...", flush=True)
try:
    import numpy as np
    # The NumPy 2.0 ABI change causes silent C-level segfaults in Transformers on Windows.
    if int(np.__version__.split('.')[0]) >= 2:
        print(f"\n❌ FATAL ERROR: NumPy {np.__version__} detected!")
        print("NumPy 2.x is fundamentally incompatible with the compiled Rust binaries in Transformers on Windows.")
        print("This is what causes the script to silently crash and vanish.")
        print("👉 FIX: Run this command in your terminal: pip install \"numpy<2\"")
        sys.exit(1)
except ImportError:
    pass # If numpy isn't installed at all, let transformers handle the error

print("📦 Importing PyTorch...", flush=True)
try:
    import torch
    from torch.utils.data import Dataset
except Exception as e:
    print(f"\n❌ PyTorch Import Error:\n{traceback.format_exc()}", flush=True)
    sys.exit(1)

print("📦 Importing Transformers (and underlying Rust binaries)...", flush=True)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
except Exception as e:
    print(f"\n❌ Transformers Import Error:\n{traceback.format_exc()}", flush=True)
    sys.exit(1)

print("📦 Importing PEFT...", flush=True)
try:
    from peft import LoraConfig, get_peft_model
except Exception as e:
    print(f"\n❌ PEFT Import Error:\n{traceback.format_exc()}", flush=True)
    sys.exit(1)


# -------------------------------------------------------------------
# CUSTOM DATASET: We built this to completely bypass the crashing 
# 'datasets' library. It uses pure Python to read the JSON lines.
# -------------------------------------------------------------------
class LegalDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=1024):
        self.inputs = []
        print(f"📂 Loading dataset manually from {filepath} (Bypassing 'datasets' library)...", flush=True)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                # Convert ChatML dictionary into flat text
                text = tokenizer.apply_chat_template(data["messages"], tokenize=False, add_generation_prompt=False)
                
                # Tokenize the text natively
                tokenized = tokenizer(
                    text, 
                    truncation=True, 
                    max_length=max_length, 
                    padding="max_length"
                )
                
                self.inputs.append(tokenized)
        print(f"✅ Successfully loaded {len(self.inputs)} examples directly into memory!", flush=True)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = self.inputs[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["input_ids"]) # For Causal LM, the labels are the inputs shifted
        }


def main():
    print("\n✅ All libraries loaded successfully! Starting the engine...", flush=True)

    # Use an ungated model so you don't have to deal with Hugging Face logins
    model_id = "unsloth/Llama-3.2-3B-Instruct" 
    
    print(f"📥 Downloading/Loading Model '{model_id}'... (This may take a while)", flush=True)

    # Safest float type for your modern GPU
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load natively without bitsandbytes compression!
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=compute_dtype,
        device_map="auto"
    )
    
    print("✅ Model loaded! Loading Tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Critical token fixes for standard HF training
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("⚙️ Applying LoRA Adapters...", flush=True)
    
    # Enable gradient checkpointing to save VRAM natively
    model.gradient_checkpointing_enable()
    
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    dataset_path = "data/synthetic/training_dataset.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}. Run the generator first.")
        return

    # Initialize our crash-proof manual dataset
    dataset = LegalDataset(dataset_path, tokenizer)
    
    # Use standard PyTorch data collator to batch the texts
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("🔥 Initializing the standard PyTorch Trainer...", flush=True)
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=data_collator,
        args=TrainingArguments(
            per_device_train_batch_size=1, # Lowered to 1 to ensure it fits in VRAM uncompressed
            gradient_accumulation_steps=8, # Adjusted to keep effective batch size identical
            warmup_steps=5,
            max_steps=50, 
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            output_dir="outputs_stable",
            optim="adamw_torch", # Standard stable PyTorch optimizer
            gradient_checkpointing=True # Double-checking VRAM savings
        ),
    )

    print("🚀 STARTING TRAINING LOOP!", flush=True)
    trainer.train()
    
    print("💾 Saving fine-tuned weights...", flush=True)
    model.save_pretrained("models/legal_standard_adapter")
    tokenizer.save_pretrained("models/legal_standard_adapter")
    print("🎉 Training complete. Weights saved to models/legal_standard_adapter")

if __name__ == "__main__":
    main()