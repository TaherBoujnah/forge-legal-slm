import json
import os
import base64
import boto3
import fitz  # PyMuPDF
from llama_cpp import Llama

# ------------------------------------------------------
# COLD START INITIALIZATION
# ------------------------------------------------------
# This runs exactly once when the Lambda container spins up.
# It downloads the weights from S3 to the Lambda's /tmp/ storage.

S3_BUCKET = os.environ.get("MODEL_BUCKET", "forge-legal-slm-artifacts-c547855a")
# Ensure this matches the exact path in your S3 bucket!
MODEL_KEY = "models/base/legal_slm-unsloth.Q4_K_M.gguf" 
LOCAL_MODEL_PATH = "/tmp/model.gguf"

s3_client = boto3.client('s3')

print("Initializing Serverless Inference Engine...")

# Download model from S3 if it's not already cached in /tmp/
if not os.path.exists(LOCAL_MODEL_PATH):
    print(f"Downloading model from s3://{S3_BUCKET}/{MODEL_KEY}...")
    s3_client.download_file(S3_BUCKET, MODEL_KEY, LOCAL_MODEL_PATH)
    print("Download complete.")

# Initialize the llama.cpp engine
# Using context size 2048 and 4 threads for optimal AWS Lambda CPU performance
llm = Llama(
    model_path=LOCAL_MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)
print("Model loaded into memory successfully.")

# ------------------------------------------------------
# INFERENCE LOGIC
# ------------------------------------------------------
PAGES_PER_CHUNK = 3

def extract_json_from_chunk(text_chunk: str):
    prompt = f"""<|im_start|>system
Extract JSON from legal text.<|im_end|>
<|im_start|>user
{text_chunk}<|im_end|>
<|im_start|>assistant
"""
    try:
        response = llm(
            prompt,
            max_tokens=512,
            stop=["<|im_end|>"],
            temperature=0.0
        )
        output_text = response['choices'][0]['text'].strip()
        cleaned = output_text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except Exception as e:
        print(f"Error parsing chunk: {e}")
        return {}

def lambda_handler(event, context):
    """
    Main entry point for AWS Lambda / API Gateway.
    """
    print("Incoming request received.")
    
    # 1. CORS Preflight check
    if event.get('httpMethod') == 'OPTIONS':
        return _build_response(200, "OK")

    try:
        # 2. Extract the PDF bytes from the API Gateway event
        body = event.get('body', '')
        is_base64_encoded = event.get('isBase64Encoded', False)
        
        if is_base64_encoded:
            body_bytes = base64.b64decode(body)
        else:
            body_bytes = body.encode('utf-8')
            
        # VERY basic multipart/form-data boundary parsing to extract the PDF bytes
        pdf_start = body_bytes.find(b'%PDF-')
        if pdf_start == -1:
            return _build_response(400, {"error": "No valid PDF file found in request payload."})
        
        pdf_end = body_bytes.rfind(b'%%EOF') + 5
        pdf_clean_bytes = body_bytes[pdf_start:pdf_end]

        # 3. Read the PDF via PyMuPDF
        doc = fitz.open(stream=pdf_clean_bytes, filetype="pdf")
        total_pages = len(doc)
        
        chunks = []
        current_chunk = ""
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            current_chunk += page.get_text("text") + "\n\n"
            if (page_num + 1) % PAGES_PER_CHUNK == 0 or page_num == total_pages - 1:
                chunks.append(current_chunk.strip())
                current_chunk = ""

        # 4. Process Chunks
        chunk_results = [extract_json_from_chunk(c) for c in chunks]

        # 5. Smart Aggregation
        master_json = {
            "parties": [],
            "effective_date": "Not Specified",
            "liability_cap": "Not Specified",
            "termination_conditions": "Not Specified"
        }

        for result in chunk_results:
            for party in result.get("parties", []):
                if party not in master_json["parties"] and party not in ["Not Specified", "N/A"]:
                    master_json["parties"].append(party)
            for key in ["effective_date", "liability_cap", "termination_conditions"]:
                val = result.get(key, "Not Specified")
                if val and val not in ["Not Specified", "N/A", ""] and master_json[key] == "Not Specified":
                    master_json[key] = val

        if not master_json["parties"]: 
            master_json["parties"] = ["Not Specified"]

        print("Extraction complete.")
        
        # 6. Return response to API Gateway
        return _build_response(200, {
            "status": "success",
            "total_pages": total_pages,
            "extracted_data": master_json
        })

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return _build_response(500, {"error": "Internal Server Error during PDF processing."})


def _build_response(status_code, body):
    """Helper to format API Gateway HTTP responses with proper CORS headers."""
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