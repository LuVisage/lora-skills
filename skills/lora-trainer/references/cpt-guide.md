# CPT (Continual Pre-Training) Complete Guide

CPT and SFT are fundamentally different: CPT teaches the model new domain knowledge (e.g., medicine, law), SFT teaches the model to respond in a specific format.

## CPT vs SFT Parameter Differences

```
Parameter      SFT (Instruction Tuning)      CPT (Continual Pre-Training)
-------        -------------------------      ----------------------------
rank           r=4-32 (data-driven)          r=16-64 (stronger knowledge injection needed)
alpha          2 x r                         2 x r (standard ratio unchanged)
target_mods    [q,v] or [q,k,v,o]            [q,k,v,o,up,down,gate] (all layers)
lr             1e-4 to 3e-4                  5e-5 to 1e-4 (more conservative)
epochs         1-5                           1-3 (usually more data)
dropout        0.05-0.15                     0.05 (CPT has more data, less dropout needed)
data format    instruction/output pairs       Plain text paragraphs (JSONL, one "text" field per line)
template       Needs instruction template     No template, concatenate text directly
packing        Optional                       Strongly recommended (short texts -> long sequences)
target_batch   16                             32-64 (larger batch for stable training)
```

## CPT Data Format

```
{"text": "Chapter 1: Introduction to Cell Biology\n\nCells are the basic unit of life..."}
{"text": "Deep learning applications in medical imaging are becoming increasingly widespread..."}
```

## CPT Execution Flow

1. Confirm user intent is CPT, not SFT (key question: does the user want to inject domain knowledge?)
2. Data check: verify each line has a `text` field; estimate total token count (CPT recommends > 10M tokens)
3. VRAM calculation: target_modules count is high (7), optimizer VRAM is 3-4x higher than SFT
4. Parameter recommendation: use CPT-specific rules (table above), enable packing
5. Script generation: use plain text concatenation template, do NOT add instruction prefix

## Using train_qlora.py for CPT

Configure the script with these changes:
- `TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]`
- `neftune_noise_alpha = 0` in TrainingArguments
- Set appropriate `MAX_SEQ_LENGTH` for your data (typically 4096 or 8192 for long documents)

## When Packing Helps for CPT

CPT data often comes as mixed-length documents. Packing combines short documents into max-length sequences, improving GPU utilization 3-10x.

See `packing-guide.md` for implementation details.
