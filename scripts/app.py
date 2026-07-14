import gradio as gr
import fitz  # PyMuPDF
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==============================================================================
# 🚀 FORGE: HUGGING FACE SPACE DEMO (Gradio Web App)
# ==============================================================================

MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_DIR = "models/legal_qlora_adapter" # The folder you trained

print("Loading Inference Engine...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # HF Free Tier doesn't have a GPU, so we load to CPU in standard precision
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="cpu", 
        torch_dtype=torch.float32
    )
    
    # Inject your custom trained weights!
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    model_loaded = True
    print("✅ Model successfully loaded!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model_loaded = False

def extract_text_from_pdf(pdf_path):
    """Parses text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text[:3000] # Truncate to fit context window for demo
    except Exception as e:
        return f"Error reading PDF: {e}"

def process_document(input_type, text_input, pdf_file):
    """Main routing function for the Gradio UI."""
    if not model_loaded:
        return {"error": "Model failed to load. Check logs."}

    raw_text = ""
    if input_type == "Paste Text" and text_input:
        raw_text = text_input[:3000]
    elif input_type == "Upload PDF" and pdf_file is not None:
        raw_text = extract_text_from_pdf(pdf_file)
    else:
        return {"error": "Please provide either text or a PDF."}

    messages = [
        {"role": "system", "content": "Extract JSON from legal text."},
        {"role": "user", "content": raw_text}
    ]
    
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        
        cleaned = generated.strip().strip("`").replace("json\n", "", 1)
        return json.loads(cleaned)
        
    except json.JSONDecodeError:
        return {"error": "Model failed to output valid JSON.", "raw_output": generated}
    except Exception as e:
        return {"error": str(e)}

# Build the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🔨 Forge: Enterprise Legal SLM Extraction")
    gr.Markdown("Upload a legal contract (NDA, MSA, Lease) or paste raw text. This custom-trained 3B parameter model will deterministically extract the entities, dates, and liability caps directly into structured JSON.")
    gr.Markdown("*(Note: Running on Hugging Face Free CPU Tier. Inference may take 30-60 seconds).*")
    
    with gr.Row():
        with gr.Column():
            input_type = gr.Radio(["Paste Text", "Upload PDF"], label="Input Method", value="Paste Text")
            text_input = gr.Textbox(label="Raw Legal Text", lines=10, placeholder="Paste NDA or contract text here...")
            pdf_input = gr.File(label="Upload PDF Contract", file_types=[".pdf"], visible=False)
            submit_btn = gr.Button("Extract Structured Data", variant="primary")
            
        with gr.Column():
            json_output = gr.JSON(label="Extracted Payload")
            
    def toggle_inputs(choice):
        if choice == "Paste Text":
            return gr.update(visible=True), gr.update(visible=False)
        return gr.update(visible=False), gr.update(visible=True)
        
    input_type.change(fn=toggle_inputs, inputs=input_type, outputs=[text_input, pdf_input])
    
    submit_btn.click(
        fn=process_document,
        inputs=[input_type, text_input, pdf_input],
        outputs=json_output
    )

if __name__ == "__main__":
    demo.launch()