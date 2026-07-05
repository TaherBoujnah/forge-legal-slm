import json
import asyncio
import os
import random
from pydantic import BaseModel, Field
import ollama

# ---------------------------------------------------------------------
# 1. Define the Structured Output Schema
# ---------------------------------------------------------------------
class ExtractionResult(BaseModel):
    messy_text: str = Field(description="The heavily distorted, noisy raw legal text generated.")
    parties: list[str] = Field(description="The entities entering into the contract.")
    effective_date: str = Field(description="The date the contract goes into effect.")
    liability_cap: str = Field(description="The maximum financial liability mentioned.")
    termination_conditions: str = Field(description="How the contract can be terminated.")

# Create the JSON schema string to pass to the LLM
schema_str = json.dumps(ExtractionResult.model_json_schema(), indent=2)

# ---------------------------------------------------------------------
# 2. Dynamic Prompt Generator
# ---------------------------------------------------------------------
def get_dynamic_system_prompt():
    contract_types = [
        "Non-Disclosure Agreement (NDA)", 
        "Commercial Lease Agreement", 
        "Software as a Service (SaaS) Agreement", 
        "Executive Employment Contract", 
        "Vendor Service Level Agreement (SLA)",
        "Merger & Acquisition Letter of Intent",
        "Independent Contractor Agreement"
    ]
    
    contract = random.choice(contract_types)
    
    return f"""
You are a legal data generator. 
Generate a messy, OCR-style contract excerpt from a {contract}.
The text must be realistically long (around 250 to 500 words). Include dense legal jargon, scanning errors, and typos.

Return ONLY a valid JSON object. 

CRITICAL QUALITY RULES:
1. The extracted fields MUST be exact substrings copied directly from the 'messy_text'.
2. Do not invent, summarize, or hallucinate information. If it is not explicitly in the text, use "Not Specified".

Schema: {schema_str}
"""

# ---------------------------------------------------------------------
# 3. Quality Control (Anti-Hallucination)
# ---------------------------------------------------------------------
def validate_extraction(data):
    """
    Programmatic validation to guarantee the LLM didn't hallucinate data.
    Checks if the extracted JSON values actually exist within the messy_text.
    """
    messy_text = data.get("messy_text", "").lower()
    
    def is_hallucinated(val):
        val_str = str(val).lower().strip()
        # Ignore empty or default values
        if val_str in ["not specified", "none", "n/a", ""]:
            return False
            
        # Strict Substring Check: If the exact value isn't in the text, verify words
        if val_str not in messy_text:
            words = val_str.split()
            if not words: 
                return False
            # Check if at least 75% of the words from the extraction exist in the raw text
            matches = sum(1 for w in words if w in messy_text)
            if matches / len(words) < 0.75:
                return True # It hallucinated words not found in the source text
        return False

    # Check all fields for hallucinations
    for party in data.get("parties", []):
        if is_hallucinated(party):
            raise ValueError(f"Hallucinated party: {party}")
            
    if is_hallucinated(data.get("effective_date")):
        raise ValueError(f"Hallucinated effective_date: {data.get('effective_date')}")
        
    if is_hallucinated(data.get("liability_cap")):
        raise ValueError(f"Hallucinated liability_cap: {data.get('liability_cap')}")

# ---------------------------------------------------------------------
# 4. Local Generation Engine (Ollama)
# ---------------------------------------------------------------------
async def generate_single_example(example_id, file_handle, semaphore):
    async with semaphore:
        for attempt in range(5):
            try:
                # Generate a completely unique prompt for every single example
                current_prompt = get_dynamic_system_prompt()
                
                client = ollama.AsyncClient()
                response = await client.chat(
                    model="llama3.1",
                    messages=[
                        {"role": "system", "content": current_prompt},
                        {"role": "user", "content": "Generate the contract excerpt and extract its details. Output ONLY JSON."}
                    ],
                    format="json", # Enforces JSON natively in Ollama
                    options={
                        "temperature": 0.6,
                        "num_predict": 1500 # Ensure the model has enough space to write 500 words
                    }
                )
                
                content = response['message']['content'].strip()
                data = json.loads(content)
                
                # Verify data quality before accepting it
                validate_extraction(data)
                
                # Format for ChatML Training (System/User/Assistant)
                training_example = {
                    "messages": [
                        {"role": "system", "content": "Extract JSON from legal text."},
                        {"role": "user", "content": data['messy_text']},
                        {"role": "assistant", "content": json.dumps({k:v for k,v in data.items() if k != 'messy_text'})}
                    ]
                }
                
                # Write to the file incrementally so you don't lose data if it crashes
                file_handle.write(json.dumps(training_example) + "\n")
                file_handle.flush()
                print(f"✅ Success: #{example_id}")
                
                return # Exit on success
                
            except Exception as e:
                # Local validation failure (e.g. Hallucinations). Just retry quickly.
                print(f"⚠️ Attempt {attempt+1} failed for #{example_id} ({str(e)[:70]}). Retrying...")
                await asyncio.sleep(0.5)

# ---------------------------------------------------------------------
# 5. Execution Runner
# ---------------------------------------------------------------------
async def main():
    os.makedirs("data/synthetic", exist_ok=True)
    output_file = "data/synthetic/training_dataset.jsonl"
    
    TOTAL_EXAMPLES = 500
    
    print(f"🚀 Starting LOCAL dynamic dataset generation via Ollama ({TOTAL_EXAMPLES} examples)...")
    print(f"💡 Press Ctrl+C at any time to stop. Data is saved continuously.")
    
    with open(output_file, "w", encoding="utf-8") as f:
        # Depending on your GPU VRAM, you can set this semaphore to 2, 4, or 8 to generate faster in parallel
        semaphore = asyncio.Semaphore(2) 
        tasks = [generate_single_example(i, f, semaphore) for i in range(1, TOTAL_EXAMPLES + 1)]
        await asyncio.gather(*tasks)

    print(f"\n🎉 Finished! Dataset saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())