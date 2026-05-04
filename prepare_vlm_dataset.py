"""
Prepare VLM Fine-Tuning Dataset for Bilingual Indian Financial Document Extractor
==================================================================================

Combines both bank statement datasets into a unified VLM training format:
  - Converts multi-page PDFs to individual page images
  - Normalizes JSON labels into a unified extraction schema
  - Creates instruction-tuning samples: (image, prompt, response_json)
  - Splits into train / val / test
  - Outputs a JSONL file ready for Qwen2.5-VL fine-tuning
"""

import json
import os
import random
import shutil
import sys
from pathlib import Path
from pdf2image import convert_from_path

# ── Config ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DS1_DIR = ROOT / "bank_statements_dataset"
DS2_DIR = ROOT / "synthetic_bank_statement_dataset_100"
DS3_DIR = ROOT / "bank_statements_dataset_1000_real"
OUT_DIR = ROOT / "vlm_training_dataset"
IMAGES_DIR = OUT_DIR / "images"
DPI = 150  # good balance between quality and size for VLM training
SEED = 42

# ── Unified target schema ──────────────────────────────────────────────────
# This is what the VLM should learn to output given a bank statement image.
# We define a simplified "header" extraction (per-document) and keep
# transactions as a list.

SYSTEM_PROMPT = (
    "You are a financial document extraction AI. Given an image of an Indian "
    "bank statement page, extract all visible structured information and return "
    "it as JSON. The document may contain text in English, Hindi (Devanagari), "
    "or Gujarati script. Extract exactly what you see — do not infer missing fields."
)

EXTRACTION_PROMPT = (
    "Extract all structured information from this Indian bank statement image. "
    "Return a JSON object with these fields where visible:\n"
    "- bank_name, branch\n"
    "- account_holder_name, account_number, ifsc, account_type\n"
    "- statement_period (from, to)\n"
    "- opening_balance, closing_balance\n"
    "- currency (default INR)\n"
    "- transactions: list of {date, description, debit, credit, balance}\n\n"
    "Use null for fields not visible on this page. "
    "Preserve original script (Hindi/Gujarati/English) in descriptions."
)


def normalize_ds1_label(label: dict) -> dict:
    """Normalize bank_statements_dataset JSON to unified schema."""
    ah = label.get("account_holder", {})
    summary = label.get("summary", {})
    period = label.get("statement_period", {})

    transactions = []
    for t in label.get("transactions", []):
        txn = {
            "date": t.get("date"),
            "description": t.get("description"),
            "debit": t["amount"] if t.get("dr_or_cr") == "DR" else None,
            "credit": t["amount"] if t.get("dr_or_cr") == "CR" else None,
            "balance": t.get("balance_after"),
        }
        transactions.append(txn)

    return {
        "bank_name": label.get("bank_name"),
        "branch": ah.get("branch"),
        "account_holder_name": ah.get("name"),
        "account_number": ah.get("account_no"),
        "ifsc": ah.get("ifsc"),
        "account_type": ah.get("account_type"),
        "statement_period": {
            "from": period.get("from"),
            "to": period.get("to"),
        },
        "opening_balance": summary.get("opening_balance"),
        "closing_balance": summary.get("closing_balance"),
        "currency": "INR",
        "transactions": transactions,
    }


def normalize_ds2_label(label: dict) -> dict:
    """Normalize synthetic_bank_statement_dataset_100 JSON to unified schema."""
    cust = label.get("customer", {})
    acct = label.get("account", {})

    # Parse statement_period string like "01-01-2026 to 31-01-2026"
    period_str = acct.get("statement_period", "")
    period_parts = period_str.split(" to ")
    period = {
        "from": period_parts[0].strip() if len(period_parts) == 2 else None,
        "to": period_parts[1].strip() if len(period_parts) == 2 else None,
    }

    transactions = []
    for t in label.get("transactions", []):
        txn = {
            "date": t.get("txn_date"),
            "description": t.get("description"),
            "debit": t.get("debit"),
            "credit": t.get("credit"),
            "balance": t.get("balance"),
        }
        transactions.append(txn)

    return {
        "bank_name": acct.get("bank_name"),
        "branch": acct.get("branch_name"),
        "account_holder_name": cust.get("name"),
        "account_number": acct.get("account_no"),
        "ifsc": acct.get("ifsc"),
        "account_type": acct.get("account_type"),
        "statement_period": period,
        "opening_balance": acct.get("opening_balance"),
        "closing_balance": acct.get("closing_balance"),
        "currency": acct.get("currency", "INR"),
        "transactions": transactions,
    }


def pdf_to_images(pdf_path: Path, output_prefix: str) -> list[Path]:
    """Convert a PDF to page images, return list of image paths."""
    pages = convert_from_path(str(pdf_path), dpi=DPI)
    image_paths = []
    for i, page in enumerate(pages):
        img_name = f"{output_prefix}_page{i+1}.png"
        img_path = IMAGES_DIR / img_name
        page.save(str(img_path), "PNG")
        image_paths.append(img_path)
    return image_paths


def create_page_samples(
    image_paths: list[Path],
    full_label: dict,
    source_id: str,
) -> list[dict]:
    """
    Create VLM training samples.

    Strategy:
    - Page 1 always gets the full header + first chunk of transactions
    - Subsequent pages get only transactions visible on that page
    - For simplicity in training, we split transactions roughly evenly across pages
      and always include header info on page 1
    """
    num_pages = len(image_paths)
    all_txns = full_label.get("transactions", [])
    txns_per_page = max(1, len(all_txns) // num_pages) if num_pages > 0 else len(all_txns)

    samples = []
    for page_idx, img_path in enumerate(image_paths):
        # Split transactions for this page
        start = page_idx * txns_per_page
        if page_idx == num_pages - 1:
            page_txns = all_txns[start:]  # last page gets remainder
        else:
            page_txns = all_txns[start : start + txns_per_page]

        if page_idx == 0:
            # First page: full header + transactions
            response = {
                "bank_name": full_label["bank_name"],
                "branch": full_label["branch"],
                "account_holder_name": full_label["account_holder_name"],
                "account_number": full_label["account_number"],
                "ifsc": full_label["ifsc"],
                "account_type": full_label["account_type"],
                "statement_period": full_label["statement_period"],
                "opening_balance": full_label["opening_balance"],
                "closing_balance": full_label["closing_balance"]
                if num_pages == 1
                else None,
                "currency": full_label["currency"],
                "transactions": page_txns,
            }
        else:
            # Continuation pages: only transactions
            response = {
                "bank_name": full_label["bank_name"],
                "branch": None,
                "account_holder_name": None,
                "account_number": full_label["account_number"],
                "ifsc": None,
                "account_type": None,
                "statement_period": None,
                "opening_balance": None,
                "closing_balance": full_label["closing_balance"]
                if page_idx == num_pages - 1
                else None,
                "currency": "INR",
                "transactions": page_txns,
            }

        sample = {
            "id": f"{source_id}_p{page_idx+1}",
            "source": source_id,
            "page": page_idx + 1,
            "total_pages": num_pages,
            "image": str(img_path.relative_to(OUT_DIR)),
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"<image>\n{EXTRACTION_PROMPT}",
                },
                {
                    "role": "assistant",
                    "content": json.dumps(response, ensure_ascii=False, indent=2),
                },
            ],
        }
        samples.append(sample)

    return samples


def process_dataset_1() -> list[dict]:
    """Process bank_statements_dataset (DS1)."""
    samples = []
    pdf_dir = DS1_DIR / "pdfs"
    label_dir = DS1_DIR / "labels"

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    print(f"[DS1] Processing {len(pdf_files)} PDFs...")

    for pdf_path in pdf_files:
        stem = pdf_path.stem
        label_path = label_dir / f"{stem}.json"
        if not label_path.exists():
            print(f"  SKIP {stem}: no label found")
            continue

        label = json.loads(label_path.read_text(encoding="utf-8"))
        unified_label = normalize_ds1_label(label)
        source_id = f"ds1_{stem}"

        image_paths = pdf_to_images(pdf_path, source_id)
        page_samples = create_page_samples(image_paths, unified_label, source_id)
        samples.extend(page_samples)
        print(f"  {stem}: {len(image_paths)} pages → {len(page_samples)} samples")

    return samples


def process_dataset_2() -> list[dict]:
    """Process synthetic_bank_statement_dataset_100 (DS2)."""
    samples = []
    pdf_dir = DS2_DIR / "pdfs"
    json_dir = DS2_DIR / "json"

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    print(f"\n[DS2] Processing {len(pdf_files)} PDFs...")

    for pdf_path in pdf_files:
        stem = pdf_path.stem
        label_path = json_dir / f"{stem}.json"
        if not label_path.exists():
            print(f"  SKIP {stem}: no label found")
            continue

        label = json.loads(label_path.read_text(encoding="utf-8"))
        unified_label = normalize_ds2_label(label)
        source_id = f"ds2_{label.get('statement_id', stem)}"

        image_paths = pdf_to_images(pdf_path, source_id)
        page_samples = create_page_samples(image_paths, unified_label, source_id)
        samples.extend(page_samples)
        print(f"  {stem}: {len(image_paths)} pages → {len(page_samples)} samples")

    return samples


def process_dataset_3() -> list[dict]:
    """Process bank_statements_dataset_1000_real (DS3) — real bank names, same schema as DS1."""
    samples = []
    pdf_dir = DS3_DIR / "pdfs"
    label_dir = DS3_DIR / "labels"

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    print(f"\n[DS3] Processing {len(pdf_files)} PDFs (real bank names)...")

    for pdf_path in pdf_files:
        stem = pdf_path.stem
        label_path = label_dir / f"{stem}.json"
        if not label_path.exists():
            print(f"  SKIP {stem}: no label found")
            continue

        label = json.loads(label_path.read_text(encoding="utf-8"))
        unified_label = normalize_ds1_label(label)
        source_id = f"ds3_{stem}"

        image_paths = pdf_to_images(pdf_path, source_id)
        page_samples = create_page_samples(image_paths, unified_label, source_id)
        samples.extend(page_samples)

        # Print progress every 100 PDFs
        idx = pdf_files.index(pdf_path)
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(pdf_files)}] {stem}: {len(image_paths)} pages")

    print(f"  Done: {len(samples)} page-samples from {len(pdf_files)} PDFs")
    return samples


def split_by_document(samples: list[dict], train_ratio=0.8, val_ratio=0.1):
    """
    Split samples by SOURCE DOCUMENT (not by page) to prevent data leakage.
    All pages of the same document go into the same split.
    """
    # Group by source document
    doc_samples = {}
    for s in samples:
        doc_id = s["source"]
        doc_samples.setdefault(doc_id, []).append(s)

    doc_ids = sorted(doc_samples.keys())
    random.seed(SEED)
    random.shuffle(doc_ids)

    n = len(doc_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_docs = doc_ids[:n_train]
    val_docs = doc_ids[n_train : n_train + n_val]
    test_docs = doc_ids[n_train + n_val :]

    train = [s for doc in train_docs for s in doc_samples[doc]]
    val = [s for doc in val_docs for s in doc_samples[doc]]
    test = [s for doc in test_docs for s in doc_samples[doc]]

    return train, val, test


def write_jsonl(samples: list[dict], path: Path):
    """Write samples as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main():
    # Create output directories
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Preflight: datasets are not shipped in this repo — verify they're present
    missing = [d for d in [DS1_DIR, DS2_DIR, DS3_DIR] if not d.exists()]
    if missing:
        print("ERROR: Missing dataset folder(s):")
        for d in missing:
            print(f"  - {d.name}")
        print(
            "\nDownload the datasets from:\n"
            "  https://drive.google.com/file/d/1fNrKiVHW6aS4RjD1j9jw5IYhST922FzJ/view"
        )
        print(f"\nUnzip into: {ROOT}/")
        sys.exit(1)

    # Process all datasets
    ds1_samples = process_dataset_1()
    ds2_samples = process_dataset_2()
    ds3_samples = process_dataset_3()
    all_samples = ds1_samples + ds2_samples + ds3_samples

    print(f"\n{'='*60}")
    print(f"Total samples: {len(all_samples)}")
    print(f"  DS1 (bilingual, fictional banks): {len(ds1_samples)} page-samples")
    print(f"  DS2 (synthetic, English blend):   {len(ds2_samples)} page-samples")
    print(f"  DS3 (real bank names, 1000 docs): {len(ds3_samples)} page-samples")

    # Split by document
    train, val, test = split_by_document(all_samples)
    print(f"\nSplit (by document, no leakage):")
    print(f"  Train: {len(train)} samples")
    print(f"  Val:   {len(val)} samples")
    print(f"  Test:  {len(test)} samples")

    # Write JSONL files
    write_jsonl(train, OUT_DIR / "train.jsonl")
    write_jsonl(val, OUT_DIR / "val.jsonl")
    write_jsonl(test, OUT_DIR / "test.jsonl")
    write_jsonl(all_samples, OUT_DIR / "all.jsonl")

    # Write unified schema reference
    schema = {
        "description": "Unified VLM training dataset for Indian bank statement extraction",
        "total_samples": len(all_samples),
        "splits": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "sources": {
            "ds1_bank_statements_dataset": {
                "samples": len(ds1_samples),
                "features": "Bilingual (Hindi+Gujarati+English), 4 fictional banks, 4 formats, 1-5 pages",
            },
            "ds2_synthetic_bank_statement_dataset": {
                "samples": len(ds2_samples),
                "features": "English with Hindi-blend terms, 4 formats, 1-2 pages",
            },
            "ds3_bank_statements_1000_real": {
                "samples": len(ds3_samples),
                "features": "Bilingual (Hindi+Gujarati+English), 12 real Indian bank names (SBI/HDFC/ICICI/etc), 4 formats, 1-5 pages",
            },
        },
        "target_schema": {
            "bank_name": "string",
            "branch": "string | null",
            "account_holder_name": "string | null",
            "account_number": "string",
            "ifsc": "string | null",
            "account_type": "string | null",
            "statement_period": {"from": "string", "to": "string"},
            "opening_balance": "number | null",
            "closing_balance": "number | null",
            "currency": "INR",
            "transactions": [
                {
                    "date": "string",
                    "description": "string",
                    "debit": "number | null",
                    "credit": "number | null",
                    "balance": "number",
                }
            ],
        },
        "conversation_format": "Qwen2.5-VL compatible (system/user/assistant)",
        "image_dpi": DPI,
    }
    (OUT_DIR / "dataset_info.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nOutput written to: {OUT_DIR}")
    print(f"  images/     → {len(list(IMAGES_DIR.glob('*.png')))} page images")
    print(f"  train.jsonl → {len(train)} samples")
    print(f"  val.jsonl   → {len(val)} samples")
    print(f"  test.jsonl  → {len(test)} samples")
    print(f"  all.jsonl   → {len(all_samples)} samples")
    print(f"  dataset_info.json → schema + stats")
    print("\nDataset is ready for Qwen2.5-VL fine-tuning!")


if __name__ == "__main__":
    main()
