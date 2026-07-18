import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_DIR = "models/legal_qlora_adapter"
MERGED_DIR = "models/legal_slm_merged_custom"

def main():
    print(f"1. Loading base model: {BASE_MODEL}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"2. Fusing your trained adapter from {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    merged_model = model.merge_and_unload()

    print(f"3. Saving the fully merged model to {MERGED_DIR}...")
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    
    print("✅ Merge complete! Your model is ready to be converted to GGUF.")

if __name__ == "__main__":
    main()