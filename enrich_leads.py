"""
enrich_leads.py
---------------
Reads a leads CSV and enriches each row with:
  - Score (1–10)
  - Justification (1 sentence)
  - Outbound Angle (≤8 words)
  - Outbound Message (2 sentences, ≤40 words)

Sends all rows in a single API call and parses the JSON response.
Falls back to per-row retries for any rows that fail to parse.

Usage:
    python enrich_leads.py                        # uses defaults
    python enrich_leads.py --input my_leads.csv   # custom input
    python enrich_leads.py --output results.csv   # custom output

Requirements:
    pip install anthropic pandas
    export ANTHROPIC_API_KEY=sk-...
"""

import argparse
import json
import os
import sys

import anthropic
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 4096
INPUT_FILE = "job leads.csv"
OUTPUT_FILE = "job leads enriched.csv"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert outbound sales assistant helping a GTM engineer write targeted cold outreach.

Scoring rules:
- Prefer founder/C-suite/executive titles
- Prefer smaller companies (10–100 employees is strongest)
- Prefer clear GTM or growth engineering hiring signal
- Penalize large companies (500+) and non-GTM roles (designer, recruiter, etc.)

Outbound Message tone:
- Operator-to-operator, not salesy
- Reference the GTM hiring signal naturally
- Incorporate the outbound angle
- No buzzwords (no "revenue infrastructure", "intent-driven", etc.)

Respond ONLY with a valid JSON array — no markdown, no explanation.
Each element must have exactly these keys:
  "score"           — integer 1–10
  "justification"   — 1 sentence, under 20 words
  "outbound_angle"  — max 8 words, a concrete practical offer
  "outbound_message"— 2 sentences, under 40 words total
"""

BATCH_PROMPT_HEADER = "Enrich the following leads. Return one JSON object per lead in the same order.\n\n"

ROW_TEMPLATE = """\
Lead {index}:
  Name: {full_name}
  Title: {title}
  Company: {company}
  Employees: {employees}
  Industry: {industry}
"""

REQUIRED_KEYS = {"score", "justification", "outbound_angle", "outbound_message"}

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def format_row(index: int, row: dict) -> str:
    full_name = f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip()
    return ROW_TEMPLATE.format(
        index=index,
        full_name=full_name,
        title=row.get("Title", ""),
        company=row.get("Company Name", ""),
        employees=row.get("# Employees", ""),
        industry=row.get("Industry", ""),
    )


def build_batch_prompt(rows: list[dict]) -> str:
    parts = [BATCH_PROMPT_HEADER]
    for i, row in enumerate(rows, start=1):
        parts.append(format_row(i, row))
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def call_api(client: anthropic.Anthropic, prompt: str) -> str:
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def parse_batch_response(text: str) -> list[dict]:
    # Strip markdown code fences if the model wraps the JSON anyway
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def validate_result(result: dict) -> bool:
    return REQUIRED_KEYS.issubset(result.keys())

# ---------------------------------------------------------------------------
# Core enrichment
# ---------------------------------------------------------------------------

def enrich_batch(client: anthropic.Anthropic, rows: list[dict]) -> list[dict | None]:
    prompt = build_batch_prompt(rows)
    print(f"Sending {len(rows)} rows in a single API call...")
    raw = call_api(client, prompt)

    try:
        results = parse_batch_response(raw)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error on batch response: {e}")
        return [None] * len(rows)

    if len(results) != len(rows):
        print(f"  Warning: expected {len(rows)} results, got {len(results)}")

    validated = []
    for i, result in enumerate(results):
        if validate_result(result):
            validated.append(result)
        else:
            print(f"  Row {i + 1} missing keys — will retry individually")
            validated.append(None)

    return validated


def enrich_single(client: anthropic.Anthropic, index: int, row: dict) -> dict | None:
    prompt = BATCH_PROMPT_HEADER + format_row(1, row) + "\nReturn a JSON array with one object."
    try:
        raw = call_api(client, prompt)
        results = parse_batch_response(raw)
        result = results[0] if results else {}
        return result if validate_result(result) else None
    except Exception as e:
        print(f"  Row {index} retry failed: {e}")
        return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def enrich_csv(input_path: str, output_path: str) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path!r}")

    rows = df.to_dict(orient="records")
    results = enrich_batch(client, rows)

    # Retry any rows that failed
    for i, result in enumerate(results):
        if result is None:
            name = f"{rows[i].get('First Name', '')} {rows[i].get('Last Name', '')}".strip()
            print(f"  Retrying row {i + 1} ({name}) individually...")
            results[i] = enrich_single(client, i + 1, rows[i])

    # Write results back to dataframe
    for col in ["Score", "Justification", "Outbound Angle", "Outbound Message"]:
        df[col] = ""

    for i, result in enumerate(results):
        if result:
            df.at[i, "Score"] = result.get("score", "")
            df.at[i, "Justification"] = result.get("justification", "")
            df.at[i, "Outbound Angle"] = result.get("outbound_angle", "")
            df.at[i, "Outbound Message"] = result.get("outbound_message", "")
        else:
            print(f"  Row {i + 1} could not be enriched — leaving blank.")

    df.to_csv(output_path, index=False)
    print(f"Saved enriched CSV to {output_path!r}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Enrich a leads CSV with AI-generated outbound fields.")
    parser.add_argument("--input", default=INPUT_FILE, help="Path to input CSV")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Path to output CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    enrich_csv(args.input, args.output)
