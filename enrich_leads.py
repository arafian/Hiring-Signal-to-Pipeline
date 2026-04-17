"""
enrich_leads.py
---------------
Reads a leads CSV and enriches each row with:
  - Score (1–10)
  - Justification (1 sentence)
  - Outbound Angle (≤8 words)
  - Outbound Message (2 sentences, ≤40 words)

Usage:
    python enrich_leads.py                        # uses defaults
    python enrich_leads.py --input my_leads.csv   # custom input
    python enrich_leads.py --output results.csv   # custom output

Requirements:
    pip install anthropic pandas
    export ANTHROPIC_API_KEY=sk-...
"""

import argparse
import csv
import os
import sys
import time

import anthropic
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 300
INPUT_FILE = "job leads.csv"
OUTPUT_FILE = "job leads enriched.csv"
DELAY_BETWEEN_REQUESTS = 0.3  # seconds

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert outbound sales assistant helping a GTM engineer write targeted cold outreach.
Respond ONLY with raw CSV values — no markdown, no labels, no extra text.
Output exactly 4 lines in this order:
1. Score (integer 1–10)
2. Justification (1 sentence, under 20 words)
3. Outbound Angle (max 8 words, a concrete practical offer)
4. Outbound Message (2 sentences, under 40 words total)

Scoring rules:
- Prefer founder/C-suite/executive titles
- Prefer smaller companies (10–100 employees is strongest)
- Prefer clear GTM or growth engineering hiring signal
- Penalize large companies (500+) and non-GTM roles (designer, recruiter, etc.)

Tone for Outbound Message:
- Operator-to-operator, not salesy
- Reference the GTM hiring signal naturally
- Incorporate the outbound angle
- No buzzwords (no "revenue infrastructure", "intent-driven", etc.)
"""

USER_PROMPT_TEMPLATE = """\
Lead details:
- Name: {full_name}
- Title: {title}
- Company: {company}
- Employees: {employees}
- Industry: {industry}

Generate Score, Justification, Outbound Angle, and Outbound Message for this lead.
"""

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_prompt(row: dict) -> str:
    full_name = f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip()
    return USER_PROMPT_TEMPLATE.format(
        full_name=full_name,
        title=row.get("Title", ""),
        company=row.get("Company Name", ""),
        employees=row.get("# Employees", ""),
        industry=row.get("Industry", ""),
    )


def parse_response(text: str) -> dict:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return {
        "Score": lines[0] if len(lines) > 0 else "",
        "Justification": lines[1] if len(lines) > 1 else "",
        "Outbound Angle": lines[2] if len(lines) > 2 else "",
        "Outbound Message": lines[3] if len(lines) > 3 else "",
    }


def enrich_row(client: anthropic.Anthropic, row: dict) -> dict:
    prompt = build_prompt(row)
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text
    return parse_response(raw)


def enrich_csv(input_path: str, output_path: str) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path!r}")

    enriched_cols = ["Score", "Justification", "Outbound Angle", "Outbound Message"]
    for col in enriched_cols:
        df[col] = ""

    for i, row in df.iterrows():
        name = f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip()
        company = row.get("Company Name", "")
        print(f"  [{i + 1}/{len(df)}] {name} — {company} ...", end=" ", flush=True)

        try:
            result = enrich_row(client, row.to_dict())
            for col in enriched_cols:
                df.at[i, col] = result.get(col, "")
            print(f"score={result.get('Score', '?')}")
        except Exception as e:
            print(f"ERROR: {e}")

        if i < len(df) - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    df.to_csv(output_path, index=False)
    print(f"\nSaved enriched CSV to {output_path!r}")


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
