import json
import os
import base64
import boto3
import fitz  # PyMuPDF
from llama_cpp import Llama

# ------------------------------------------------------
# COLD START INITIALIZATION
# ------------------------------------------------------
S3_BUCKET = os.environ.get("MODEL_BUCKET", "forge-legal-slm-artifacts-c547855a")
MODEL_KEY = "models/base/legal_slm_custom_q3.gguf" 
LOCAL_MODEL_PATH = "/tmp/model.gguf"

s3_client = boto3.client('s3')

print("Initializing Serverless Inference Engine...")

if not os.path.exists(LOCAL_MODEL_PATH):
    print(f"Downloading model from s3://{S3_BUCKET}/{MODEL_KEY}...")
    s3_client.download_file(S3_BUCKET, MODEL_KEY, LOCAL_MODEL_PATH)
    print("Download complete.")

llm = Llama(
    model_path=LOCAL_MODEL_PATH,
    n_ctx=1024,
    n_threads=4,
    verbose=False
)
print("Model loaded into memory successfully.")

# ------------------------------------------------------
# INFERENCE LOGIC
# ------------------------------------------------------
PAGES_PER_CHUNK = 3

def _build_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,POST",
            "Content-Type": "application/json"
        },
        "body": json.dumps(body) if isinstance(body, dict) else body
    }

def extract_json_from_chunk(text_chunk: str):
    prompt = f"""<|im_start|>system
Extract JSON from legal text.<|im_end|>
<|im_start|>user
{text_chunk}<|im_end|>
<|im_start|>assistant
"""
    try:
        response = llm(prompt, max_tokens=512, stop=["<|im_end|>"], temperature=0.0)
        output_text = response['choices'][0]['text'].strip()
        cleaned = output_text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except Exception as e:
        print(f"Error parsing chunk: {e}")
        return {}

def lambda_handler(event, context):
    print(f"REQUEST RECEIVED: {event.get('httpMethod')} {event.get('path')}")
    
    # 1. Health Check
    if event.get('httpMethod') == 'GET' or event.get('path') == '/health':
        return _build_response(200, {"status": "alive"})

    # 2. CORS Preflight
    if event.get('httpMethod') == 'OPTIONS':
        return _build_response(200, "OK")

    try:
        # 3. Decode Body
        body = event.get('body', '')
        is_base64 = event.get('isBase64Encoded', False)
        body_bytes = base64.b64decode(body) if is_base64 else body.encode('utf-8')
        
        # 4. Extract PDF
        pdf_start = body_bytes.find(b'%PDF-')
        if pdf_start == -1:
            return _build_response(400, {"error": "No valid PDF found."})
        
        pdf_end = body_bytes.rfind(b'%%EOF')
        pdf_clean_bytes = body_bytes[pdf_start:(pdf_end + 5) if pdf_end != -1 else len(body_bytes)]

        # 5. Process PDF
        doc = fitz.open(stream=pdf_clean_bytes, filetype="pdf")
        chunks = []
        current_chunk = ""
        for i in range(len(doc)):
            current_chunk += doc.load_page(i).get_text("text") + "\n\n"
            if (i + 1) % PAGES_PER_CHUNK == 0 or i == len(doc) - 1:
                chunks.append(current_chunk.strip())
                current_chunk = ""

        # 6. Extraction & Aggregation
        chunk_results = [extract_json_from_chunk(c) for c in chunks]
        master_json = {"parties": [], "effective_date": "Not Specified", "liability_cap": "Not Specified", "termination_conditions": "Not Specified"}
        
        for result in chunk_results:
            master_json["parties"].extend([p for p in result.get("parties", []) if p not in master_json["parties"] and p != "Not Specified"])
            for key in ["effective_date", "liability_cap", "termination_conditions"]:
                if master_json[key] == "Not Specified": master_json[key] = result.get(key, "Not Specified")

        return _build_response(200, {"status": "success", "extracted_data": master_json})

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return _build_response(500, {"error": str(e)})