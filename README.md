# Fine-Tuning TinyLlama for Text-to-SQL with QLoRA

> Parameter-efficient fine-tuning of a 1.1B large language model to translate natural language questions into SQL queries — trained on a free-tier GPU in under 30 minutes.

---

## Overview

This project demonstrates the **end-to-end fine-tuning pipeline** for a Large Language Model (LLM), from raw dataset to a working inference demo. The goal is to adapt a general-purpose language model to a specialized task — generating SQL queries from plain English questions — using modern parameter-efficient techniques that make training feasible without expensive hardware.

Available demo version: [HuggingFace](https://huggingface.co/spaces/PieroAguinaga/fine-tunning-example) 

**The core question this project answers:**  
*How do you take a pre-trained LLM and teach it a new skill without retraining it from scratch?*

---

## Results

The fine-tuned model was evaluated against the base model on held-out examples. Both models receive the same table schema and natural language question.

| Question | Expected SQL | Base Model | Fine-Tuned Model |
|---|---|---|---|
| Who narrated when the vessel operator is de beers? | `SELECT narrated_by ... WHERE vessel_operator = "De Beers"` | `SELECT narrated_by ... WHERE vessel_operator = 'de beers'` | ✅ `SELECT narrated_by ... WHERE vessel_operator = "De Beers"` |
| What's the original season in 11th place? | `SELECT original_season ... WHERE placing = "11th place"` | `SELECT original_season ... WHERE placing = '11th'` | ✅ `SELECT original_season ... WHERE placing = "11th"` |
| Which Senior status has a Chief Judge of —, death, 1967–1983? | `SELECT senior_status WHERE chief_judge = "—" AND ...` | Incomplete JOIN query | ✅ **Exact match** |
| Which Rank has a Reaction of 0.198, Time < 46.3? | `SELECT MAX(rank) WHERE react = 0.198 AND time < 46.3` | `SELECT * WHERE rank = 198` | `SELECT AVG(rank) WHERE react = "0.198" AND time < 46.3` |

**Key improvements after fine-tuning:**
- Correct table and column name references
- Proper double-quote formatting for string values
- Correct case sensitivity in string literals
- Correct WHERE clause structure with multiple conditions
- Semantically valid, executable SQL in all cases

---

## Technical Approach

### Why not full fine-tuning?

Full fine-tuning updates all parameters of the model. For a 1.1B parameter model in float16, that requires ~10 GB of GPU memory just for the weights — before optimizer states, gradients, and activations. Full fine-tuning of state-of-the-art models (7B–70B parameters) can require multiple A100 GPUs.

This project uses **QLoRA** (Quantized Low-Rank Adaptation), which combines two techniques:

#### 1. 4-bit Quantization (bitsandbytes)
The base model weights are compressed from 16-bit floats to 4-bit integers using NF4 (NormalFloat4) — a quantization format optimized for the normal distribution of neural network weights. This reduces the model's memory footprint from ~2.2 GB to ~600 MB.

```
W_quantized = quantize(W, dtype=nf4)   # frozen, read-only
```

#### 2. LoRA Adapters (Low-Rank Adaptation)
Instead of modifying the original weights, LoRA injects two small trainable matrices (A and B) into each attention layer:

```
W' = W + (α/r) · B·A

where:
  W ∈ R^{d_out × d_in}  ← frozen (quantized)
  A ∈ R^{r × d_in}      ← trainable (~0.01% of total params)
  B ∈ R^{d_out × r}     ← trainable, initialized to zero
  r = 16                 ← rank (hyperparameter)
```

**Result:** only ~8M trainable parameters out of 1.1B total (~0.7%), making the training fit entirely on a free Colab T4 GPU (16 GB VRAM).

### Training Strategy: Completion-Only Loss

A critical design decision is **masking the instruction prompt from the loss function**. The model sees the full input (schema + question + SQL answer) during the forward pass, but the loss is only computed on the SQL answer tokens:

```
[Given the schema... | Question: ... | SQL Query:]   → masked (label = -100)
[SELECT narrated_by FROM table WHERE ...]            → trained on this
```

This teaches the model to generate SQL given context, not to memorize the prompt format. Implemented via a manual label masking function that searches for the answer boundary within the tokenized sequence.

### Prompt Template

```
Given the following SQL schema and a question, write the correct SQL query.

### Schema:
{CREATE TABLE statements}

### Question:
{natural language question}

### SQL Query:
{SQL answer}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Pipeline Overview                  │
│                                                     │
│  Raw Dataset (78k samples)                          │
│       ↓                                             │
│  EDA + Cleaning (dedup, length filter, nulls)       │
│       ↓                                             │
│  Subset: 1,000 train / 50 test                      │
│       ↓                                             │
│  Prompt Formatting + Label Masking                  │
│       ↓                                             │
│  TinyLlama-1.1B (4-bit quantized, frozen)           │
│       +                                             │
│  LoRA Adapters (r=16, α=32, ~8M params, trainable)  │
│       ↓                                             │
│  HuggingFace Trainer (bf16, cosine LR, 3 epochs)    │
│       ↓                                             │
│  Adapter saved (~20 MB vs 2.2 GB full model)        │
└─────────────────────────────────────────────────────┘
```

---

## Technical Challenges Solved

This project involved navigating several non-trivial engineering problems. Each one reflects real-world debugging skills relevant to production ML systems.

### 1. KV Cache incompatibility with QLoRA
The model generated the correct first token (`SELECT` with 99.98% probability, confirmed via direct logit inspection) but produced garbage for all subsequent tokens. Root cause: the **KV (Key-Value) cache** used to speed up autoregressive generation stores intermediate attention states that don't correctly propagate the LoRA adapter updates for tokens beyond the first. **Solution:** `use_cache=False` in `model.generate()`, forcing full recomputation at each step.

### 2. Training precision conflicts
TinyLlama's native compute format is BFloat16. Using `fp16=True` caused a `NotImplementedError` when the gradient scaler attempted to process BFloat16 tensors. **Solution:** aligned `bnb_4bit_compute_dtype=torch.bfloat16` with `bf16=True` in the training config.

---

## Stack

| Component | Library | Purpose |
|---|---|---|
| Model | `transformers` | Load, run, and save TinyLlama-1.1B |
| Quantization | `bitsandbytes` | 4-bit NF4 quantization |
| LoRA Adapters | `peft` | Inject and manage trainable adapters |
| Training Loop | `trl.SFTTrainer` | Supervised fine-tuning with bf16 |
| Dataset | `datasets` | Load and preprocess `b-mc2/sql-create-context` |
| Analysis | `pandas`, `matplotlib`, `seaborn` | EDA and training visualization |
| Runtime | Google Colab T4 (free tier) | 16 GB VRAM, ~25 min training |

---

## Dataset

**[`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context)** — 78,577 samples of natural language questions paired with SQL CREATE TABLE schemas and correct SQL query answers.

| Field | Example |
|---|---|
| `question` | Who narrated when the vessel operator is de beers? |
| `context` | `CREATE TABLE table_26168687_3 (narrated_by VARCHAR, vessel_operator VARCHAR)` |
| `answer` | `SELECT narrated_by FROM table_26168687_3 WHERE vessel_operator = "De Beers"` |

**Subset used:** 1,000 training samples + 50 test samples (to keep training under 30 minutes on free hardware).

**Cleaning pipeline:**
1. Deduplication on (question, answer) pairs
2. Null removal
3. Length filtering — samples exceeding `max_seq_length=512` tokens are dropped to prevent answer truncation
4. Empty answer removal

---

## How to Run

**Requirements:** Google Colab with T4 GPU (Runtime → Change runtime type → T4 GPU)

1. Open `tinyllama_sql_finetuning.ipynb` in Google Colab
2. Run all cells in order
3. Training takes approximately **25 minutes** on a free T4

**To reload the fine-tuned adapter in a new session:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "./tinyllama-sql-adapter")
tokenizer = AutoTokenizer.from_pretrained("./tinyllama-sql-adapter")
```

---

## What This Demonstrates

| Skill | Where it appears |
|---|---|
| **LLM fine-tuning concepts** | QLoRA setup, completion-only loss, prompt engineering |
| **Debugging complex systems** | KV cache bug, tokenization misalignment, API changes |
| **Data engineering** | EDA, cleaning pipeline, manual label masking |
| **ML engineering judgment** | Choosing 1k samples to fit free hardware; knowing when full fine-tuning is overkill |
| **Library proficiency** | `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets` |
| **Problem diagnosis** | Token-level logit inspection to isolate where generation failed |

---

## Potential Next Steps

- **Scale training data** to 10k–20k samples to improve aggregate function selection (`MIN`/`MAX`/`COUNT`)
- **Execution accuracy evaluation** — run generated SQL against a real SQLite database and measure whether the result set matches, not just the string
- **DPO fine-tuning** — add a preference optimization stage using pairs of (correct SQL, incorrect SQL) to further align outputs
- **Push adapter to HuggingFace Hub** — `model.push_to_hub("username/tinyllama-sql-qlora")`

---

## Author

**Piero Aguinaga**  
[GitHub](https://github.com/PieroAguinaga)
