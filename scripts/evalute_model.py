"""
Evaluate the fine-tuned legal extraction adapter against the held-out set.

Changes from the previous (broken) version:
  - REVERTED repetition_penalty / no_repeat_ngram_size during generation.
    These suppressed the model's ability to legitimately repeat JSON
    syntax ("," ":" '"' etc.) and caused it to substitute wrong/hallucinated
    tokens instead -- overall_f1 dropped from 66.91% to 25.44% and
    hallucination rate jumped to 25%. Confirmed bad, do not use.
  - ADDED post-hoc repetition detection: after generation, we scan for a
    long substring repeated 3+ times in a row (the actual observed failure
    mode -- the model looping on one clause) and truncate at the first
    repeat, then attempt to close the JSON. This fixes the same problem
    without touching how the model generates valid syntax.

Kept from before:
  - Data-quality flagging for likely-mislabeled ground truth
  - Document-type split (templated/synthetic vs real-world/SEC-style)

Run:
    python scripts/evaluate_model.py
"""

import json
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from train_model import load_and_split_dataset, DATASET_PATH, EVAL_FRACTION, SEED, MODEL_ID

ADAPTER_DIR = "models/legal_qlora_adapter"
FIELDS = ["parties", "effective_date", "liability_cap", "termination_conditions"]
MAX_NEW_TOKENS = 500

# Plain greedy decoding -- matches the run that scored 66.91% F1.
# No repetition_penalty, no no_repeat_ngram_size: both were tested and
# made things dramatically worse (see note above).
GENERATION_KWARGS = dict(
    do_sample=False,
)


# ------------------------------------------------------------------
# Post-hoc degenerate-repetition detection and truncation
# ------------------------------------------------------------------

def detect_and_truncate_repetition(text: str, min_repeat_len: int = 25, min_repeats: int = 3) -> Tuple[str, bool]:
    """Looks for a substring of at least `min_repeat_len` characters that
    repeats `min_repeats` or more times back-to-back (the observed failure
    mode: the model looping on one clause). If found, truncates the text
    at the point just before the repetition starts.

    Returns (possibly_truncated_text, was_truncated).
    """
    # Look for a repeated chunk using a sliding window -- cheap and good
    # enough for this use case; we don't need this to be fast, just correct.
    n = len(text)
    for chunk_len in range(min_repeat_len, min(200, n // min_repeats) + 1, 5):
        for start in range(0, n - chunk_len * min_repeats):
            chunk = text[start:start + chunk_len]
            if not chunk.strip():
                continue
            repeats = 1
            pos = start + chunk_len
            while text[pos:pos + chunk_len] == chunk:
                repeats += 1
                pos += chunk_len
                if repeats >= min_repeats:
                    break
            if repeats >= min_repeats:
                return text[:start], True
    return text, False


def try_close_json(truncated_text: str) -> Optional[dict]:
    """Given text truncated mid-JSON, try to close it into valid JSON by
    trimming to the last complete key-value pair and closing brackets.
    Returns a parsed dict, or None if it still can't be salvaged."""
    text = truncated_text.rstrip()
    # Trim back to the last comma or opening brace so we don't leave a
    # dangling half-written value.
    last_safe = max(text.rfind(","), text.rfind("{"))
    if last_safe == -1:
        return None
    candidate = text[:last_safe] if text[last_safe] == "," else text[:last_safe + 1]
    # Close any open string, then close the object.
    if candidate.count('"') % 2 != 0:
        candidate += '"'
    candidate = candidate.rstrip().rstrip(",")
    candidate += "}"
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None


def parse_model_output(generated: str) -> Tuple[Dict[str, Any], bool, bool]:
    """Attempts to parse the model's raw output as JSON. If that fails,
    checks for degenerate repetition, truncates, and retries.

    Returns (parsed_dict, parse_failed, was_truncated_for_repetition).
    """
    cleaned = generated.strip().strip("`")
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].lstrip("\n")

    try:
        return json.loads(cleaned), False, False
    except (json.JSONDecodeError, ValueError):
        pass

    truncated, was_truncated = detect_and_truncate_repetition(cleaned)
    if was_truncated:
        salvaged = try_close_json(truncated)
        if salvaged is not None:
            return salvaged, False, True

    return {f: "" for f in FIELDS}, True, was_truncated


# ------------------------------------------------------------------
# Scoring helpers (unchanged from previous version)
# ------------------------------------------------------------------

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())


DATE_FORMATS_TO_TRY = [
    "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
    "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%d %B %Y", "%d %b %Y",
]


def normalize_date(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    cleaned = text.strip()
    for fmt in DATE_FORMATS_TO_TRY:
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return normalize_text(text)


def split_parties(text: str) -> set:
    if not isinstance(text, str) or not text.strip():
        return set()
    parts = re.split(r',|;|\band\b', text, flags=re.IGNORECASE)
    return {normalize_text(p) for p in parts if normalize_text(p)}


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


def calculate_set_f1(prediction: str, truth: str, splitter) -> float:
    pred_set = splitter(prediction)
    truth_set = splitter(truth)
    if not pred_set or not truth_set:
        return 1.0 if pred_set == truth_set else 0.0
    common = pred_set.intersection(truth_set)
    if not common:
        return 0.0
    precision = len(common) / len(pred_set)
    recall = len(common) / len(truth_set)
    return 2 * (precision * recall) / (precision + recall)


def calculate_date_match(prediction: str, truth: str) -> float:
    p_norm = normalize_date(prediction)
    t_norm = normalize_date(truth)
    if not p_norm and not t_norm:
        return 1.0
    return 1.0 if p_norm == t_norm else 0.0


def score_field(field: str, prediction: str, truth: str) -> float:
    if field == "parties":
        return calculate_set_f1(prediction, truth, split_parties)
    if field == "effective_date":
        return calculate_date_match(prediction, truth)
    return calculate_f1(prediction, truth)


def detect_hallucination(prediction: str, source_text: str) -> bool:
    if not isinstance(prediction, str) or prediction.strip().lower() in (
        "not specified", "n/a", ""
    ):
        return False
    pred_norm = normalize_text(prediction)
    source_norm = normalize_text(source_text)
    if pred_norm in source_norm:
        return False
    pred_words = pred_norm.split()
    if not pred_words:
        return False
    matched_words = sum(1 for w in pred_words if w in source_norm)
    return (matched_words / len(pred_words)) < 0.75


# ------------------------------------------------------------------
# Data quality flagging (unchanged from previous version)
# ------------------------------------------------------------------

SUSPICIOUS_DATE_KEYWORDS = [
    "terminate", "termination", "renew", "notice", "shall be effective on",
    "in accordance with", "expiration",
]
SUSPICIOUS_PARTY_PREFIXES = [
    "tenant:", "landlord:", "address:", "hereinafter", "collectively",
    "individually", "shall have", "subject to",
]


def flag_suspicious_ground_truth(truth: Dict[str, Any]) -> List[str]:
    flags = []
    date_val = normalize_text(str(truth.get("effective_date", "")))
    if any(kw in date_val for kw in SUSPICIOUS_DATE_KEYWORDS):
        flags.append("effective_date")
    parties_val = str(truth.get("parties", "")).lower()
    if any(prefix in parties_val for prefix in SUSPICIOUS_PARTY_PREFIXES):
        flags.append("parties")
    return flags


# ------------------------------------------------------------------
# Document-type classification (unchanged from previous version)
# ------------------------------------------------------------------

REAL_DOCUMENT_MARKERS = [
    "exhibit 10", "securities and exchange commission", "confidential treatment",
    "sec ", "8-k", "10-k", "10-q", "certain identified information",
]


def classify_document(source_text: str) -> str:
    text_lower = source_text.lower()
    if any(marker in text_lower for marker in REAL_DOCUMENT_MARKERS):
        return "real_world_sec_style"
    return "templated_synthetic"


# ------------------------------------------------------------------
# Aggregate evaluation
# ------------------------------------------------------------------

def evaluate_dataset(predictions: List[Dict], ground_truths: List[Dict], source_texts: List[str]):
    results = {
        "overall_f1": 0.0,
        "exact_match_rate": 0.0,
        "hallucination_rate": 0.0,
        "hallucination_rate_on_populated_fields": 0.0,
        "json_parse_failure_rate": 0.0,
        "salvaged_via_truncation_rate": 0.0,
        "flagged_ground_truth_rate": 0.0,
        "field_f1": {f: 0.0 for f in FIELDS},
    }

    total_examples = len(predictions)
    if total_examples == 0:
        return results

    total_f1 = 0
    total_exact = 0
    total_hallucinations = 0
    total_entities = 0
    total_populated_truth_fields = 0
    total_hallucinations_on_populated = 0
    parse_failures = sum(1 for p in predictions if p.get("_parse_failed"))
    salvaged = sum(1 for p in predictions if p.get("_salvaged_via_truncation"))
    flagged_examples = 0

    for pred, truth, source in zip(predictions, ground_truths, source_texts):
        example_score = 0
        exact_matches = 0

        if flag_suspicious_ground_truth(truth):
            flagged_examples += 1

        for field in FIELDS:
            p_val = str(pred.get(field, ""))
            t_val = str(truth.get(field, ""))

            score = score_field(field, p_val, t_val)
            example_score += score
            results["field_f1"][field] += score
            if score == 1.0:
                exact_matches += 1

            is_hallucination = detect_hallucination(p_val, source)
            if is_hallucination:
                total_hallucinations += 1
            total_entities += 1

            truth_is_populated = t_val.strip().lower() not in ("not specified", "n/a", "")
            if truth_is_populated:
                total_populated_truth_fields += 1
                if is_hallucination:
                    total_hallucinations_on_populated += 1

        total_f1 += (example_score / len(FIELDS))
        if exact_matches == len(FIELDS):
            total_exact += 1

    results["overall_f1"] = round((total_f1 / total_examples) * 100, 2)
    results["exact_match_rate"] = round((total_exact / total_examples) * 100, 2)
    results["hallucination_rate"] = round((total_hallucinations / total_entities) * 100, 2)
    results["json_parse_failure_rate"] = round((parse_failures / total_examples) * 100, 2)
    results["salvaged_via_truncation_rate"] = round((salvaged / total_examples) * 100, 2)
    results["flagged_ground_truth_rate"] = round((flagged_examples / total_examples) * 100, 2)
    if total_populated_truth_fields:
        results["hallucination_rate_on_populated_fields"] = round(
            (total_hallucinations_on_populated / total_populated_truth_fields) * 100, 2
        )
    for f in FIELDS:
        results["field_f1"][f] = round((results["field_f1"][f] / total_examples) * 100, 2)

    return results


def print_markdown_table(title: str, results_by_model: Dict[str, dict]):
    names = list(results_by_model.keys())
    n = list(results_by_model.values())[0].get('n', '?')
    print(f"\n### {title} (n={n})")
    print("| Metric | " + " | ".join(names) + " |")
    print("|--------|" + "|".join(["---"] * len(names)) + "|")
    for metric_key, label in [
        ("overall_f1", "**Overall F1 Score**"),
        ("exact_match_rate", "**Exact Match Rate**"),
        ("hallucination_rate", "**Hallucination Rate (all fields)**"),
        ("hallucination_rate_on_populated_fields", "**Hallucination Rate (populated fields only)**"),
        ("json_parse_failure_rate", "**JSON Parse Failure Rate**"),
        ("salvaged_via_truncation_rate", "**Salvaged via repetition-truncation**"),
        ("flagged_ground_truth_rate", "**Flagged (possibly mislabeled) ground truth**"),
    ]:
        row = f"| {label} | " + " | ".join(f"{results_by_model[n_][metric_key]}%" for n_ in names) + " |"
        print(row)

    print("\n#### Field-Level F1 Breakdown")
    print("| Field | " + " | ".join(names) + " |")
    print("|-------|" + "|".join(["---"] * len(names)) + "|")
    for f in FIELDS:
        row = f"| `{f}` | " + " | ".join(f"{results_by_model[n_]['field_f1'][f]}%" for n_ in names) + " |"
        print(row)


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

def load_model(use_adapter: bool):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    if use_adapter:
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    return model


def extract_user_content(messages: List[Dict[str, str]]) -> str:
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    return ""


def extract_ground_truth(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    for m in messages:
        if m["role"] == "assistant":
            try:
                return json.loads(m["content"])
            except json.JSONDecodeError:
                return {}
    return {}


def run_inference(model, tokenizer, eval_dataset) -> Tuple[List[Dict], List[Dict], List[str]]:
    predictions, ground_truths, source_texts = [], [], []

    for i, example in enumerate(eval_dataset):
        messages = example["messages"]
        user_content = extract_user_content(messages)
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
                pad_token_id=tokenizer.eos_token_id,
                **GENERATION_KWARGS,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        parsed, parse_failed, was_salvaged = parse_model_output(generated)
        parsed["_parse_failed"] = parse_failed
        parsed["_salvaged_via_truncation"] = was_salvaged

        predictions.append(parsed)
        ground_truths.append(truth)
        source_texts.append(user_content)

        if (i + 1) % 20 == 0:
            print(f"  ...processed {i + 1}/{len(eval_dataset)} examples")

    return predictions, ground_truths, source_texts


def split_by_document_type(predictions, ground_truths, source_texts):
    groups = {"templated_synthetic": ([], [], []), "real_world_sec_style": ([], [], [])}
    for pred, truth, source in zip(predictions, ground_truths, source_texts):
        doc_type = classify_document(source)
        groups[doc_type][0].append(pred)
        groups[doc_type][1].append(truth)
        groups[doc_type][2].append(source)
    return groups


def main():
    print("Loading held-out eval split (same seed/fraction as training -- "
          "these examples were never trained on)...")
    _, eval_dataset = load_and_split_dataset(DATASET_PATH, EVAL_FRACTION, SEED)
    print(f"Evaluating on {len(eval_dataset)} held-out examples.\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    results_by_model = {}
    split_results_by_model = {"templated_synthetic": {}, "real_world_sec_style": {}}
    doc_type_counts = {}

    for label, use_adapter in [
        ("Llama-3.2-3B (Base)", False),
        ("Forge SLM (Fine-Tuned)", True),
    ]:
        print(f"Running {label}...")
        model = load_model(use_adapter=use_adapter)
        preds, truths, sources = run_inference(model, tokenizer, eval_dataset)

        overall = evaluate_dataset(preds, truths, sources)
        overall["n"] = len(eval_dataset)
        results_by_model[label] = overall

        groups = split_by_document_type(preds, truths, sources)
        for doc_type, (g_preds, g_truths, g_sources) in groups.items():
            group_results = evaluate_dataset(g_preds, g_truths, g_sources)
            group_results["n"] = len(g_preds)
            split_results_by_model[doc_type][label] = group_results
            doc_type_counts[doc_type] = len(g_preds)

        del model
        torch.cuda.empty_cache()

    print_markdown_table("Overall Model Evaluation Results (blended)", results_by_model)
    print_markdown_table(
        "Results on templated/synthetic documents only",
        split_results_by_model["templated_synthetic"],
    )
    print_markdown_table(
        "Results on real-world/SEC-filing-style documents only",
        split_results_by_model["real_world_sec_style"],
    )

    print(f"\nDocument type breakdown in held-out set: {doc_type_counts}")

    with open("eval_results.json", "w") as f:
        json.dump({
            "overall": results_by_model,
            "by_document_type": split_results_by_model,
        }, f, indent=2)
    print("\nRaw results saved to eval_results.json")
    print("\nNOTE: 'flagged_ground_truth_rate' marks examples whose LABELS look")
    print("suspicious. These drag down exact_match/F1 regardless of model")
    print("quality. The real-world subset had ~57% flagged last run --")
    print("that's a data cleanup task, not something this script can fix.")


if __name__ == "__main__":
    main()