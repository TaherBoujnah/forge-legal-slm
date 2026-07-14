"""
Diagnose the fine-tuned model's worst-scoring predictions.

Prints prediction vs. ground truth vs. source text for the lowest-F1
examples in the held-out set, so you can see real failure modes
(formatting mismatch vs. genuine content error) before deciding what to fix.

Run:
    python scripts/diagnose_errors.py
"""

import json
import re
import textwrap
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from train_model import load_and_split_dataset, DATASET_PATH, EVAL_FRACTION, SEED, MODEL_ID

ADAPTER_DIR = "models/legal_qlora_adapter"
FIELDS = ["parties", "effective_date", "liability_cap", "termination_conditions"]
MAX_NEW_TOKENS = 500  # bumped from 300 -- see note in evaluate_model.py
NUM_WORST_TO_SHOW = 15


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())


def calculate_f1(prediction: str, truth: str) -> float:
    pred_tokens = set(normalize_text(prediction).split())
    truth_tokens = set(normalize_text(truth).split())
    if not pred_tokens or not truth_tokens:
        return 1.0 if pred_tokens == truth_tokens else 0.0
    common = pred_tokens.intersection(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)


def extract_user_content(messages: List[Dict[str, str]]) -> str:
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    return ""


def extract_ground_truth(messages: List[Dict[str, str]]) -> Dict:
    for m in messages:
        if m["role"] == "assistant":
            try:
                return json.loads(m["content"])
            except json.JSONDecodeError:
                return {}
    return {}


def load_finetuned_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    return model


def wrap(text: str, width: int = 100) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width)) or "(empty)"


def main():
    print("Loading held-out eval split...")
    _, eval_dataset = load_and_split_dataset(DATASET_PATH, EVAL_FRACTION, SEED)
    print(f"Running inference on {len(eval_dataset)} examples...\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = load_finetuned_model()

    scored_examples = []

    for i, example in enumerate(eval_dataset):
        messages = example["messages"]
        source_text = extract_user_content(messages)
        truth = extract_ground_truth(messages)

        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        parse_failed = False
        try:
            cleaned = generated.strip().strip("`").replace("json\n", "", 1)
            pred = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            pred = {f: "" for f in FIELDS}
            parse_failed = True

        field_f1s = {}
        for field in FIELDS:
            p_val = str(pred.get(field, ""))
            t_val = str(truth.get(field, ""))
            field_f1s[field] = calculate_f1(p_val, t_val)

        avg_f1 = sum(field_f1s.values()) / len(FIELDS)

        scored_examples.append({
            "index": i,
            "avg_f1": avg_f1,
            "field_f1s": field_f1s,
            "prediction": pred,
            "ground_truth": truth,
            "source_text": source_text,
            "raw_generation": generated,
            "parse_failed": parse_failed,
        })

        if (i + 1) % 20 == 0:
            print(f"  ...processed {i + 1}/{len(eval_dataset)}")

    # Sort worst-first
    scored_examples.sort(key=lambda x: x["avg_f1"])

    print(f"\n{'='*100}")
    print(f"WORST {NUM_WORST_TO_SHOW} EXAMPLES (sorted lowest F1 first)")
    print(f"{'='*100}\n")

    for ex in scored_examples[:NUM_WORST_TO_SHOW]:
        print(f"--- Example #{ex['index']}  |  avg F1: {ex['avg_f1']:.2f}  |  "
              f"parse_failed: {ex['parse_failed']} ---\n")

        print("SOURCE TEXT:")
        print(wrap(ex["source_text"][:600]))
        print()

        if ex["parse_failed"]:
            print("RAW GENERATION (JSON parse failed):")
            print(wrap(ex["raw_generation"][:500]))
            print()
        else:
            for field in FIELDS:
                print(f"[{field}]  (F1: {ex['field_f1s'][field]:.2f})")
                print(f"  Predicted: {wrap(ex['prediction'].get(field, ''), 90)}")
                print(f"  Truth:     {wrap(ex['ground_truth'].get(field, ''), 90)}")
                print()

        print(f"{'-'*100}\n")

    # Quick tally of likely failure modes
    parse_failures = sum(1 for ex in scored_examples if ex["parse_failed"])
    low_f1_but_parsed = sum(
        1 for ex in scored_examples if not ex["parse_failed"] and ex["avg_f1"] < 0.5
    )
    print(f"\nSUMMARY:")
    print(f"  Total examples: {len(scored_examples)}")
    print(f"  JSON parse failures: {parse_failures}")
    print(f"  Parsed OK but avg F1 < 0.5: {low_f1_but_parsed}")
    print(f"  (Parse failures point to truncation/format issues. Low F1 with")
    print(f"   successful parsing points to genuine content/wording mismatches --")
    print(f"   look at the printed examples above to tell which is which.)")


if __name__ == "__main__":
    main()  