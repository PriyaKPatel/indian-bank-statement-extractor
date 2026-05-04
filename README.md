# Bilingual Indian Financial Document Extractor

Fine-tuned **Qwen2.5-VL-3B** for structured JSON extraction from Indian bank statements in Hindi, Gujarati, and English.

---

## Approach: Iterative Data Scaling

We followed a 3-step iterative approach — starting small to validate the pipeline, then scaling data to push accuracy.

---

### Step 1: Baseline (Zero-Shot)

Evaluated the base Qwen2.5-VL-3B-Instruct without any fine-tuning on Indian bank statements.

| Metric | Score |
|---|---|
| JSON Parse Rate | 95.0% |
| Header Fields (avg) | 70.8% |
| Opening Balance | 41.4% |
| Txn Date Accuracy | 0.0% |
| Txn Amount Accuracy | 41.6% |
| Indic Script Gap | 5.4pp |

**Observation:** The base model can output JSON but fails at numerical extraction (dates, balances) and treats Indic scripts as secondary.

---

### Step 2: Fine-Tune on Small Dataset (322 samples)

First fine-tuning run using 200 documents (100 fictional bilingual + 100 English-blend synthetic) producing 322 page-level training samples.

| Metric | Baseline | Step 2 FT | Delta |
|---|---|---|---|
| JSON Parse Rate | 95.0% | 95.0% | — |
| Header Fields (avg) | 70.7% | **73.7%** | +3.0pp |
| Opening Balance | 45.7% | **95.0%** | +49.3pp |
| Txn Date Accuracy | 22.2% | **69.8%** | +47.6pp |
| Txn Amount Accuracy | 32.8% | **62.0%** | +29.2pp |
| Indic Script Gap | 23.7pp | 25.4pp | -1.7pp (worse) |

**Observation:** Numerical extraction massively improved. But account numbers stayed low (35%), and the Indic gap slightly worsened — the model overfit to English-heavy patterns in the small dataset.

---

### Step 3: Scale to Full Dataset (2,355 samples)

Added 1,000 more documents with **real Indian bank names** (SBI, HDFC, ICICI, etc.) and heavy Hindi+Gujarati content. Total: 1,300 documents → 2,355 training samples.

| Metric | Baseline | Step 2 (322) | Step 3 (2,355) |
|---|---|---|---|
| JSON Parse Rate | 95.0% | 95.0% | **100.0%** |
| Bank Name | 90-95% | 90% | **100.0%** |
| Account Holder | 64.8-89.5% | 94.3% | **100.0%** |
| IFSC | 79.8-95% | 95% | **100.0%** |
| Opening Balance | 41.4-45.7% | 95% | **100.0%** |
| Closing Balance | 37.9-46.3% | 65% | **75.0%** |
| Txn Date Accuracy | 0-22.2% | 69.8% | **75.2%** |
| Txn Amount Accuracy | 32.8-41.6% | 62.0% | 57.7% |
| Txn Description Sim. | 44.1-48.4% | 60.9% | **59.9%** |
| Txn Count Accuracy | 84.8% | 84.0% | **92.2%** |
| **Indic Script Gap** | 5.4-23.7pp | 25.4pp | **0.0pp** |

---

### Key Achievements (Step 3 Final)

- **100% JSON parse rate** — production-ready structured output every time
- **5 out of 7 header fields at 100%** — bank name, account holder, IFSC, account type, opening balance
- **Indic script gap eliminated** — Hindi/Gujarati descriptions extracted with identical accuracy to English
- **75% transaction date extraction** — from 0% baseline (model couldn't read dates at all)
- **92% transaction count accuracy** — model captures nearly all rows from tables

---

## Model & Training

| | |
|---|---|
| Base Model | Qwen2.5-VL-3B-Instruct |
| Method | Unsloth LoRA (r=128, rsLoRA, all-linear) |
| Precision | bf16, 16-bit base |
| Training Data | 2,355 page images |
| Validation | 283 page images |
| Test | 275 page images |
| Epochs | 3 |
| Effective Batch | 16 |
| LR | 5e-5, cosine schedule |
| GPU | NVIDIA RTX PRO 6000 Blackwell (102 GB) |

---

## Dataset

**Total: 1,300 synthetic Indian bank statement PDFs → 2,913 page-level training samples**

**Download processed dataset:** [vlm_training_dataset.zip (Google Drive)](https://drive.google.com/file/d/1fNrKiVHW6aS4RjD1j9jw5IYhST922FzJ/view?usp=sharing)

| Source | Documents | Pages | Features |
|---|---|---|---|
| `bank_statements_dataset/` | 100 | 267 | 4 fictional banks, bilingual (Hi+Gu+En), 4 formats |
| `synthetic_bank_statement_dataset_100/` | 100 | 135 | 4 English-blend formats (classic/ledger/modern/cooperative) |
| `bank_statements_dataset_1000_real/` | 1,000 | 2,511 | 12 real Indian bank names (SBI, HDFC, ICICI, Axis, PNB, BOB, Kotak, Canara, UBI, BOI, IndusInd, MUCB), bilingual |

### Languages
- English
- Hindi (Devanagari script)
- Gujarati script
- Code-mixed narrations (e.g., UPI descriptions mixing all three)

### Document Formats
- FORMAT-A through FORMAT-D (4 distinct layouts)
- 1-5 pages per statement
- Average 68 transactions per statement

---

## Repository Contents

This repo contains the **code, schemas, and result reports** for the project. The raw datasets (PDFs + labels) and the trained LoRA adapter are too large to ship in git — they are downloaded separately (see [External Downloads](#external-downloads)).

```
indian_bank_statement_extractor/
├── README.md                              ← This file
├── LICENSE                                ← MIT
├── requirements.txt                       ← Python dependencies
├── .gitignore
├── prepare_vlm_dataset.py                 ← PDF→image + label normalization script
├── indian_financial_doc_extractor.ipynb   ← Main training + evaluation notebook (Colab-ready)
├── indian_financial_doc_extractor_v2_100.ipynb   ← V2 run on 322-sample dataset (with outputs)
├── indian_financial_doc_extractor_v2_1000.ipynb  ← V2 run on 2,355-sample dataset (with outputs)
├── final_report.json                      ← Final results (2,355 training samples)
└── final_report_run1_322samples.json      ← First run results (322 samples)
```

### External Downloads

The following are **not included in this repo**. After cloning, place them at the repo root before running `prepare_vlm_dataset.py`:

| Folder / File | Description | Source |
|---|---|---|
| `bank_statements_dataset/` | DS1: 100 bilingual statements (fictional banks) | [Drive](https://drive.google.com/file/d/1fNrKiVHW6aS4RjD1j9jw5IYhST922FzJ/view?usp=sharing) |
| `synthetic_bank_statement_dataset_100/` | DS2: 100 English-blend statements | same |
| `bank_statements_dataset_1000_real/` | DS3: 1000 statements (real bank names) | same |
| `vlm_training_dataset/` (processed) | Output of `prepare_vlm_dataset.py` — 2,913 images + JSONL splits | regenerate locally |

---

## Setup

```bash
pip install -r requirements.txt

# pdf2image needs a system-level Poppler install:
#   macOS:  brew install poppler
#   Linux:  apt-get install poppler-utils
#   Colab:  already installed
```

---

## How to Use

### 1. Prepare Dataset (already done)

```bash
python prepare_vlm_dataset.py
```

Converts all PDFs to page images, normalizes labels, creates train/val/test JSONL splits.

### 2. Train on Google Colab

1. Upload `vlm_training_dataset.zip` to Colab (or Google Drive)
2. Open `indian_financial_doc_extractor.ipynb`
3. Run all cells — GPU auto-detection adapts parameters to your hardware

### 3. Inference

```python
from unsloth import FastVisionModel
from PIL import Image

model, processor = FastVisionModel.from_pretrained("./qwen_vlm_finetuned/lora_adapter")
FastVisionModel.for_inference(model)

image = Image.open("bank_statement.png")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "Extract all structured information from this Indian bank statement image..."},
]}]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                                        tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
result = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

### Output Format

```json
{
  "bank_name": "HDFC BANK LIMITED",
  "branch": "AHMEDABAD BRANCH",
  "account_holder_name": "REHAAN MUTTI",
  "account_number": "56192900428738",
  "ifsc": "HDFC0849278",
  "account_type": "SALARY",
  "statement_period": {"from": "2026-04-06", "to": "2026-05-21"},
  "opening_balance": 18421.9,
  "closing_balance": 47383.59,
  "currency": "INR",
  "transactions": [
    {"date": "2026-04-06", "description": "UPI-CR/રાજેશ પટેલ/ઉધાર પાછું/191424506178", "debit": null, "credit": 6991.0, "balance": 25412.9},
    {"date": "2026-04-06", "description": "GPAY-UPI/HP PETROL PUMP@oksbi/Bills", "debit": 112.98, "credit": null, "balance": 48063.2}
  ]
}
```

---

## Why This Project Matters

- **India-specific problem**: Indian bank statements mix English, Hindi, and Gujarati on the same page. Global models treat Indic scripts as secondary.
- **Real-world relevance**: Indian financial institutions (banks, insurers, NBFCs) process millions of such documents daily — accurate multilingual extraction is critical.
- **Indic gap eliminated**: After fine-tuning, Hindi/Gujarati descriptions are extracted with the same accuracy as English — proving the model learned script-agnostic extraction.
- **Production-ready**: 100% JSON parse rate means the output can be directly consumed by downstream systems without error handling.

---

## Technical Highlights

- **Unsloth** for 2x training speedup over vanilla HuggingFace
- **Rank-stabilized LoRA (rsLoRA)** at r=128 for high-capacity adaptation
- **Document-level train/test split** preventing data leakage across pages
- **Multi-page handling**: Page 1 gets full header + transactions; continuation pages get transaction-only extraction
- **Auto-adaptive GPU config**: Notebook detects GPU tier and adjusts batch size, LoRA rank, and precision automatically

---

## Limitations & Future Work

- **Account numbers (44.8%)**: Long digit strings remain challenging at 150 DPI. Higher resolution or a two-pass zoom approach could help.
- **Transaction amounts (57.7%)**: Multi-column numerical extraction from tables has room for improvement with more epochs or larger model.
- **Real-world scans**: All data is synthetic PDFs. Phone-camera captures with skew, blur, and lighting would require additional augmentation.
- **More languages**: Currently Hindi + Gujarati + English. Extending to Tamil, Bengali, Marathi would increase applicability.

---

## Author

**Priya Patel** — May 2026

Demonstrating India-specific document AI capabilities with native Gujarati language expertise.
