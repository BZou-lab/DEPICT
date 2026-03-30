#!/usr/bin/env python3
"""
Interactive helper functions for generating YAML “augmented text” for a
compound table.

– Define the utilities once, then call them in a notebook or REPL on a
  small test DataFrame before scaling up.

Requirements:
    pip install pandas openai tqdm   # tqdm only for progress bar

Environment:
    export OPENAI_API_KEY="sk-…"    # or set openai.api_key directly
"""

from pathlib import Path
from typing import Any
import json
import ast

import pandas as pd
import openai

# ──── CONFIG (EDIT TO TASTE) ───────────────────────────────────────────────
CSV_PATH            = Path("./RawData/compounds_df_wTarMoA_part2.csv")          # full dataset
SYSTEM_PROMPT_PATH  = Path("./RawData/prompt.txt")

MODEL_NAME   = "gpt-4o"
TEMPERATURE  = 0.2
MAX_TOKENS   = 512

# If you prefer in‑code auth, uncomment:
openai.api_key = '' # please fill in your API key

# ──── CONSTANT USER TEMPLATE ───────────────────────────────────────────────
USER_TEMPLATE = """
Here is the compound record. Fill in the YAML template.

```json
{json_block}
```
"""

def _maybe(val: Any) -> Any:
    """Return None for NaN/empty/whitespace, else the original value."""
    if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
        return None
    return val


def row_to_json_block(row: pd.Series) -> str:
    """Convert one DataFrame row into the JSON payload shown to the LLM."""
    # Pipe-delimited helper fields (unchanged) --------------
    targets = _maybe(row.get("target"))
    if targets is not None:
        targets = [t.strip() for t in str(targets).split("|") if t.strip()]

    indications = _maybe(row.get("indication"))
    if indications is not None:
        indications = [i.strip() for i in str(indications).split("|") if i.strip()]

    # NEW: just forward the lists as-is ---------------------
    payload = {
        "compound_name"   : row["pert_iname"],
        "canonical_smiles": row["canonical_smiles_list"],      # ← list of str
        "pubchem_cid"     : [str(cid) for cid in row["pubchem_cid_list"]],
        "targets"         : targets if targets is not None else [],
        "moa"             : _maybe(row["moa"]),
        "disease_area"    : _maybe(row["disease_area"]),
        "indication"      : indications if indications is not None else None,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def generate_yaml_for_row(row: pd.Series,
                          system_prompt: str | None = None) -> str:
    """Return the YAML description for *one* compound row."""
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT_PATH.read_text()

    user_msg = USER_TEMPLATE.format(json_block=row_to_json_block(row))
    response = openai.chat.completions.create(
        model       = MODEL_NAME,
        messages    = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg}
        ],
        temperature = TEMPERATURE,
        max_tokens  = MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()

# ──── LOAD DATA ────────────────────────────────────────────────────────────
# Use converters to turn the *stringified* lists in the CSV into real lists
converters = {
    "pubchem_cid_list":      lambda x: ast.literal_eval(x) if pd.notna(x) else [],
    "canonical_smiles_list": lambda x: ast.literal_eval(x) if pd.notna(x) else [],
}
compounds_info_part2 = pd.read_csv(
    CSV_PATH,
    index_col=0,
    converters=converters
)

system_prompt_text = SYSTEM_PROMPT_PATH.read_text()

# Optionally display progress if tqdm is present
try:
    from tqdm.auto import tqdm
    iterator = tqdm(compounds_info_part2.itertuples(index=False), total=len(compounds_info_part2))
except ImportError:
    iterator = compounds_info_part2.itertuples(index=False)

yaml_list: list[str] = []
for row in iterator:
    s = pd.Series(row._asdict())
    yaml_list.append(generate_yaml_for_row(s, system_prompt=system_prompt_text))



compounds_info_part2["augmented_yaml"] = yaml_list

compounds_info_part2.to_csv("./RawData/compounds_with_yaml_part2.csv", index=False)
