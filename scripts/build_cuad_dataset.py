import json
import os
from datasets import load_dataset

def main():
    print("🚀 Downloading CUAD (Contract Understanding Atticus Dataset)...")
    print("This contains real contracts labeled by legal experts.")
    
    # 🚨 SECURITY BYPASS: Load directly from the Parquet files to avoid the "Dataset scripts are no longer supported" error
    try:
        dataset = load_dataset(
            "theatticusproject/cuad-qa", 
            name="default", # Force default config to bypass custom script
            split="train", 
        )
    except Exception:
        # Ultimate Fallback: Direct read of the raw data files
        print("Fallback: Reading raw parquet data directly from HF...")
        dataset = load_dataset("parquet", data_files="hf://datasets/theatticusproject/cuad-qa/data/train-*.parquet", split="train")
    
    print(f"✅ Downloaded {len(dataset)} question-answer pairs.")
    print("🧠 Processing and grouping into Forge JSON Schema...")
    
    # CUAD is in SQuAD format (1 row = 1 question). We need to group by the contract text.
    grouped_contracts = {}
    
    for row in dataset:
        context = row['context']
        question = row['question'].lower()
        answers = row['answers']['text']
        
        # We only want chunks of text that actually fit in a small LLM context window
        if len(context) > 3500:
            context = context[:3500] + "..."
            
        if context not in grouped_contracts:
            grouped_contracts[context] = {
                "parties": [],
                "effective_date": "Not Specified",
                "liability_cap": "Not Specified",
                "termination_conditions": "Not Specified"
            }
            
        ans_text = answers[0].strip() if len(answers) > 0 else "Not Specified"
        
        # Map CUAD's legal questions to our specific JSON schema
        if "parties" in question and ans_text != "Not Specified":
            if ans_text not in grouped_contracts[context]["parties"]:
                grouped_contracts[context]["parties"].append(ans_text)
                
        elif "effective date" in question and ans_text != "Not Specified":
            grouped_contracts[context]["effective_date"] = ans_text
            
        elif "liability cap" in question and ans_text != "Not Specified":
            grouped_contracts[context]["liability_cap"] = ans_text
            
        elif "terminate" in question and ans_text != "Not Specified":
            grouped_contracts[context]["termination_conditions"] = ans_text

    # Filter out contracts that have absolutely no useful data
    valid_contracts = []
    for context, data in grouped_contracts.items():
        if not data["parties"]:
            data["parties"] = ["Not Specified"]
            
        # If it found at least one piece of real data, keep it
        if data["parties"] != ["Not Specified"] or data["effective_date"] != "Not Specified":
            valid_contracts.append({
                "context": context,
                "data": data
            })
            
    print(f"✅ Successfully mapped {len(valid_contracts)} real-world contract chunks.")
    
    # Merge with existing synthetic data
    output_file = "data/synthetic/training_dataset.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"💾 Appending real data to {output_file}...")
    
    with open(output_file, "a", encoding="utf-8") as f:
        for contract in valid_contracts:
            training_example = {
                "messages": [
                    {"role": "system", "content": "Extract JSON from legal text."},
                    {"role": "user", "content": contract["context"]},
                    {"role": "assistant", "content": json.dumps(contract["data"])}
                ]
            }
            f.write(json.dumps(training_example) + "\n")
            
    print("🎉 Dataset mixing complete! You now have a hybrid Synthetic + Real dataset.")

if __name__ == "__main__":
    main()