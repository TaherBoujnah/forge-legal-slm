"""
Clean the legal extraction training dataset BEFORE training.

This targets the root cause found during manual inspection: the "parties"
field in a meaningful fraction of examples contains entire clause
fragments (e.g. "Premier shall have the right to terminate this
Agreement...") instead of just entity names -- and the model faithfully
learned to reproduce that pattern. Similarly, "effective_date" sometimes
contains termination/renewal language instead of an actual date.

This script does NOT modify training_dataset.jsonl in place. It writes:
  - data/synthetic/training_dataset_cleaned.jsonl  (the cleaned dataset)
  - data/synthetic/cleaning_report.jsonl           (every change, for review)

Nothing here is fully automatic -- read the report before trusting the
cleaned file blindly. Heuristics can have false positives/negatives.

Run:
    python scripts/clean_dataset.py
"""

import json
import re
from typing import Any, Dict, List, Tuple

INPUT_PATH = "data/synthetic/training_dataset.jsonl"
OUTPUT_PATH = "data/synthetic/training_dataset_cleaned.jsonl"
REPORT_PATH = "data/synthetic/cleaning_report.jsonl"

FIELDS = ["parties", "effective_date", "liability_cap", "termination_conditions"]


# ------------------------------------------------------------------
# "parties" field cleaning -- drop entries that are clause fragments,
# not entity names.
# ------------------------------------------------------------------

# Boilerplate wrapper markers -- text AFTER one of these, in a party
# entry, is legal-boilerplate wrapping, not part of the name itself.
# We truncate here and KEEP the name that precedes it, rather than
# discarding the whole entry.
BOILERPLATE_MARKERS = [
    r'\(hereinafter\b', r',\s*a\s+\w+\s+corporation\b', r',\s*a\s+\w+\s+limited liability company\b',
    r',\s*an?\s+individual\b', r',\s*a\s+company\b', r',\s*whose address\b',
    r'\bwith (its|his|her) principal place of business\b', r'\bresiding at\b',
    r'\ban entity with\b',
]
BOILERPLATE_TRUNCATE_RE = re.compile("|".join(BOILERPLATE_MARKERS), re.IGNORECASE)

# Signals that an entry (or what's LEFT after truncating boilerplate) is
# a pure clause fragment with no real name in it at all -- these get
# fully dropped, not truncated, because there's nothing worth keeping.
PURE_CLAUSE_SIGNALS = [
    r'\bshall\b', r'\bsubject to\b', r'\bpursuant to\b',
    r'\bin accordance with\b', r'\bmay terminate\b', r'\bagrees? to\b',
    r'\bwarrants?\b', r'\brepresents?\b', r'\bthe right to\b',
]
PURE_CLAUSE_RE = re.compile("|".join(PURE_CLAUSE_SIGNALS), re.IGNORECASE)

MAX_REASONABLE_PARTY_NAME_WORDS = 12  # legit entity names are rarely longer than this
MIN_NAME_LENGTH_AFTER_TRUNCATION = 2  # chars -- avoid keeping empty/near-empty fragments

# Label prefixes like "TENANT:", "LANDLORD:" that should be stripped from
# the front of an entry, keeping the actual name that follows.
LABEL_PREFIX_RE = re.compile(
    r'^\s*(TENANT|LANDLORD|FIRST PARTY|SECOND PARTY|CLIENT|VENDOR|BUYER|SELLER|COMPANY NAME)\s*:\s*',
    re.IGNORECASE,
)


def clean_single_entry(entry: str) -> Tuple[str, str]:
    """Cleans one 'parties' entry. Returns (cleaned_entry_or_empty, reason)
    where reason is '' if kept as-is, 'truncated' if boilerplate was
    stripped, or 'dropped' if nothing usable remained."""
    if not isinstance(entry, str) or not entry.strip():
        return "", "dropped"

    working = entry.strip()

    # Strip label prefixes like "TENANT:" first, keeping what follows.
    working = LABEL_PREFIX_RE.sub("", working).strip()

    # If the WHOLE entry (before any truncation) is a pure obligation
    # clause with no name-like content, drop it entirely -- e.g.
    # "Premier shall have the right to terminate this Agreement..."
    # These typically don't start with a capitalized name token, or the
    # clause language appears very early in the string.
    match = PURE_CLAUSE_RE.search(working)
    if match and match.start() < 15:
        # Clause language appears almost immediately -- no real name prefix.
        return "", "dropped"

    # Truncate at the first boilerplate marker, keep what precedes it.
    boilerplate_match = BOILERPLATE_TRUNCATE_RE.search(working)
    if boilerplate_match:
        name_part = working[:boilerplate_match.start()].strip().rstrip(",").strip()
        if len(name_part) >= MIN_NAME_LENGTH_AFTER_TRUNCATION:
            was_truncated = name_part != working
            return name_part, ("truncated" if was_truncated else "")
        return "", "dropped"

    # No boilerplate marker found. If it's still a pure clause fragment
    # (obligation language anywhere, and long), drop it.
    if PURE_CLAUSE_RE.search(working) and len(working.split()) > MAX_REASONABLE_PARTY_NAME_WORDS:
        return "", "dropped"

    # If it's just unusually long but has no obligation-clause language,
    # it's likely a verbose-but-legitimate name (e.g. a long official
    # entity name) -- keep it as-is rather than guessing.
    return working, ""


def clean_parties_field(raw_value: Any) -> Tuple[Any, List[str], List[Tuple[str, str]]]:
    """Cleans a 'parties' field. Handles both list and string representations.
    Returns (cleaned_value, list_of_fully_dropped_entries, list_of_(original, truncated)_pairs)."""
    dropped = []
    truncated_pairs = []

    if isinstance(raw_value, list):
        entries = raw_value
    elif isinstance(raw_value, str):
        entries = [p.strip() for p in re.split(r',|;', raw_value) if p.strip()]
    else:
        return raw_value, dropped, truncated_pairs

    kept = []
    for entry in entries:
        cleaned, reason = clean_single_entry(entry)
        if reason == "dropped":
            dropped.append(entry)
        elif reason == "truncated":
            truncated_pairs.append((entry, cleaned))
            kept.append(cleaned)
        else:
            kept.append(cleaned)

    cleaned_value = kept if isinstance(raw_value, list) else ", ".join(kept)
    return cleaned_value, dropped, truncated_pairs


# ------------------------------------------------------------------
# "effective_date" field cleaning -- flag (don't silently rewrite --
# there's no safe automatic replacement for a wrong date) entries that
# contain termination/renewal language instead of a date.
# ------------------------------------------------------------------

SUSPICIOUS_DATE_KEYWORDS = [
    "terminate", "termination", "renew", "notice", "shall be effective on",
    "in accordance with", "expiration",
]


def effective_date_is_suspicious(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return any(kw in lowered for kw in SUSPICIOUS_DATE_KEYWORDS)


# ------------------------------------------------------------------
# Main cleaning pass
# ------------------------------------------------------------------

def extract_assistant_json(messages: List[Dict[str, str]]) -> Tuple[int, Dict[str, Any]]:
    """Finds the assistant message and parses its JSON content.
    Returns (message_index, parsed_dict) or (-1, {}) if not found/parseable."""
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            try:
                return i, json.loads(m["content"])
            except json.JSONDecodeError:
                return i, {}
    return -1, {}


def clean_example(example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Cleans a single training example. Returns (cleaned_example, change_report)."""
    report = {
        "parties_dropped": [],
        "parties_truncated": [],  # list of {"original": ..., "cleaned": ...}
        "effective_date_flagged": False,
        "modified": False,
    }

    messages = example.get("messages", [])
    assistant_idx, truth = extract_assistant_json(messages)
    if assistant_idx == -1 or not truth:
        return example, report

    # Clean parties
    if "parties" in truth:
        cleaned_parties, dropped, truncated_pairs = clean_parties_field(truth["parties"])
        if dropped or truncated_pairs:
            truth["parties"] = cleaned_parties
            report["parties_dropped"] = dropped
            report["parties_truncated"] = [
                {"original": orig, "cleaned": clean} for orig, clean in truncated_pairs
            ]
            report["modified"] = True

    # Flag (not auto-fix) suspicious effective_date
    if "effective_date" in truth and effective_date_is_suspicious(truth["effective_date"]):
        report["effective_date_flagged"] = True
        # Not modifying automatically -- no safe substitute exists.
        # This example should be manually reviewed or excluded.

    if report["modified"]:
        new_messages = list(messages)
        new_messages[assistant_idx] = dict(new_messages[assistant_idx])
        new_messages[assistant_idx]["content"] = json.dumps(truth)
        cleaned_example = dict(example)
        cleaned_example["messages"] = new_messages
        return cleaned_example, report

    return example, report


def main():
    print(f"Loading {INPUT_PATH}...")
    examples = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples.\n")

    # ------------------------------------------------------------
    # Exclude examples with a completely empty assistant target ({}).
    # Found during manual verification: 42/900 examples in this dataset
    # have NO fields at all in the label. Training on these would teach
    # the model that it's sometimes fine to return nothing for a real
    # contract -- actively harmful, not just a missed opportunity.
    # ------------------------------------------------------------
    filtered_examples = []
    n_empty_excluded = 0
    for ex in examples:
        _, truth = extract_assistant_json(ex.get("messages", []))
        if not truth:
            n_empty_excluded += 1
            continue
        filtered_examples.append(ex)

    print(f"Excluded {n_empty_excluded} examples with a completely empty "
          f"assistant target ('{{}}') -- these are pre-existing data defects, "
          f"not something cleaning can fix.")
    print(f"Proceeding with {len(filtered_examples)} examples.\n")
    examples = filtered_examples

    cleaned_examples = []
    reports = []
    n_parties_modified = 0
    n_dates_flagged = 0
    total_dropped = 0
    total_truncated = 0

    for i, example in enumerate(examples):
        cleaned, report = clean_example(example)
        report["example_index"] = i
        cleaned_examples.append(cleaned)

        if report["parties_dropped"] or report["parties_truncated"]:
            n_parties_modified += 1
            total_dropped += len(report["parties_dropped"])
            total_truncated += len(report["parties_truncated"])
        if report["effective_date_flagged"]:
            n_dates_flagged += 1

        if report["parties_dropped"] or report["parties_truncated"] or report["effective_date_flagged"]:
            reports.append(report)

    print(f"Examples with 'parties' modified: {n_parties_modified}")
    print(f"  - Entries fully DROPPED (pure clause fragments, no name present): {total_dropped}")
    print(f"  - Entries TRUNCATED (boilerplate stripped, real name KEPT): {total_truncated}")
    print(f"Examples with 'effective_date' flagged as suspicious (NOT auto-fixed, "
          f"needs manual review): {n_dates_flagged}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in cleaned_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"\nCleaned dataset written to {OUTPUT_PATH}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        for r in reports:
            f.write(json.dumps(r) + "\n")
    print(f"Change/flag report written to {REPORT_PATH}")

    print("\nNEXT STEPS:")
    print("  1. Skim cleaning_report.jsonl -- check a sample of removed 'parties'")
    print("     entries to confirm the heuristic isn't discarding real entity names")
    print("     with unusual formatting (e.g. very long official company names).")
    print("  2. Manually review the effective_date_flagged examples -- these need")
    print("     a human to either correct the date or drop the example. No safe")
    print("     automatic fix exists for a wrong date.")
    print("  3. Once satisfied, point train_model.py's DATASET_PATH at")
    print(f"     '{OUTPUT_PATH}' instead of the original file, and retrain.")


if __name__ == "__main__":
    main()