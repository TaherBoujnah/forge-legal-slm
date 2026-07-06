import gradio as gr
import fitz  # PyMuPDF
import json
import os
from llama_cpp import Llama

# ==============================================================================
# 🚀 FORGE: HUGGING FACE SPACE DEMO
# This script turns your fine-tuned GGUF into a public, sharable web app.
# ==============================================================================

# 1. Load the model globally so it stays in RAM
MODEL_PATH = "legal_slm-unsloth.Q4_K_M.gguf"

print("Loading Inference Engine...")
try:
    # We use a small n_ctx for free Hugging Face Spaces to avoid memory crashes
    llm = Llama(model_path=MODEL_PATH, n_ctx=1024, n_threads=2)
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

def extract_text_from_pdf(pdf_path):
    """Parses text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        # Truncate text to fit into the context window for the demo
        return text[:3000] 
    except Exception as e:
        return f"Error reading PDF: {e}"

def process_document(input_type, text_input, pdf_file):
    """Main routing function for the Gradio UI."""
    if not model_loaded:
        return {"error": "Model not loaded. Ensure the .gguf file is in the root directory."}

    # Handle input types
    raw_text = ""
    if input_type == "Paste Text" and text_input:
        raw_text = text_input[:3000]
    elif input_type == "Upload PDF" and pdf_file is not None:
        raw_text = extract_text_from_pdf(pdf_file)
    else:
        return {"error": "Please provide either text or a PDF."}

    # Format the ChatML prompt for our specific model
    prompt = f"""<|im_start|>system
Extract JSON from legal text.<|im_end|>
<|im_start|>user
{raw_text}<|im_end|>
<|im_start|>assistant
"""
    
    try:
        # Run inference
        response = llm(
            prompt,
            max_tokens=512,
            stop=["<|im_end|>"],
            temperature=0.0 # Deterministic
        )
        
        output = response['choices'][0]['text'].strip()
        cleaned = output.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
        
    except json.JSONDecodeError:
        return {"error": "Model failed to output valid JSON.", "raw_output": output}
    except Exception as e:
        return {"error": str(e)}

# 2. Build the Professional Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🔨 Forge: Enterprise Legal SLM Extraction Engine")
    gr.Markdown("Upload a legal contract (NDA, MSA, Lease) or paste raw text. The custom-trained 3B parameter model will deterministically extract the entities, dates, and liability caps directly into structured JSON.")
    
    with gr.Row():
        with gr.Column():
            input_type = gr.Radio(["Paste Text", "Upload PDF"], label="Input Method", value="Paste Text")
            text_input = gr.Textbox(label="Raw Legal Text", lines=10, placeholder="Paste NDA or contract text here...")
            pdf_input = gr.File(label="Upload PDF Contract", file_types=[".pdf"], visible=False)
            
            submit_btn = gr.Button("Extract Structured Data", variant="primary")
            
        with gr.Column():
            json_output = gr.JSON(label="Extracted Payload")
            
    # UI Logic to toggle between text and file upload
    def toggle_inputs(choice):
        if choice == "Paste Text":
            return gr.update(visible=True), gr.update(visible=False)
        return gr.update(visible=False), gr.update(visible=True)
        
    input_type.change(fn=toggle_inputs, inputs=input_type, outputs=[text_input, pdf_input])
    
    # Wire up the button
    submit_btn.click(
        fn=process_document,
        inputs=[input_type, text_input, pdf_input],
        outputs=json_output
    )

if __name__ == "__main__":
    demo.launch()