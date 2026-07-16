"""
100% NATIVE PYTHON GGUF EXPORTER (NO UNSLOTH REQUIRED)
This script merges your LoRA adapters using core Hugging Face libraries on the CPU
to completely avoid Windows CUDA errors, then downloads and runs the official 
llama.cpp Python converter.
"""
import os
import sys
import subprocess
import urllib.request

def run_cmd(cmd):
    subprocess.check_call(cmd)

def main():
    print("🚀 Step 1: Fixing dependencies (NumPy and TorchAO)...")
    
    # 1. Uninstall torchao to fix the 'torch.int1' attribute error crash
    print("Uninstalling torchao to prevent PyTorch conflicts...")
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
    
    # 2. Added "numpy<2" to automatically downgrade to a compatible version
    run_cmd([sys.executable, "-m", "pip", "install", "numpy<2", "gguf", "protobuf", "sentencepiece"])

    # Delay importing these until AFTER dependencies are fixed above!
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_path = "models/legal_qlora_adapter"
    merged_dir = "models/legal_slm_merged"
    
    # We keep the exact same output name so your Terraform and Lambda scripts don't break!
    gguf_name = "legal_slm-unsloth.Q4_K_M.gguf" 
    
    if not os.path.exists(adapter_path):
        print(f"❌ Error: Could not find {adapter_path}. Did you finish training?")
        return

    print("\n🧠 Step 2: Loading Base Model in Native PyTorch (CPU Mode)...")
    # Doing this on CPU avoids ANY Windows CUDA / PyTorch conflicts
    base_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-3B-Instruct",
        device_map="cpu", 
        torch_dtype=torch.float16 # Keeps RAM usage around 6GB
    )
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")

    print("⚙️ Step 3: Applying your LoRA Adapters...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("🔄 Step 4: Merging weights (Permanently baking in legal knowledge)...")
    merged_model = model.merge_and_unload()

    print(f"💾 Step 5: Saving merged model to {merged_dir}...")
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print("\n📥 Step 6: Downloading the official llama.cpp python converter...")
    script_url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
    script_path = "convert_hf_to_gguf.py"
    if not os.path.exists(script_path):
        urllib.request.urlretrieve(script_url, script_path)

    print(f"\n📦 Step 7: Converting to GGUF format...")
    print("(Using Q8_0 format since we bypassed the C++ compilers. It will be ~3.2GB).")
    run_cmd([
        sys.executable, script_path, 
        merged_dir, 
        "--outfile", gguf_name, 
        "--outtype", "q8_0"
    ])
    
    print(f"\n✅ SUCCESS! Your final deployable file is ready: {gguf_name}")

if __name__ == "__main__":
    main()