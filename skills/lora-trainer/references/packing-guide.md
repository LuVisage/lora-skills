# Packing (Sequence Packing) Guide

Packing combines multiple short training samples into one long sequence, improving GPU utilization.

## When to Use

```
Suitable:
  [OK] CPT (text passages typically < max_length)
  [OK] Short QA (avg < 500 tokens, max_length=4096)

Not Suitable:
  [FAIL] Long text (avg > 2048 tokens, wastes tokens on concatenation)
  [FAIL] Scenarios requiring strict attention masks (packing interferes with mask)
```

## Effect

- Short-data scenarios: throughput improvement 3-10x
- Long-data scenarios: minimal benefit or even slower

## Cost

- Implementation complexity increases (need correct position_ids and attention_mask setup)
- Cross-sample attention can cause minor quality loss if not handled properly

## Implementation

### Via PEFT/SFTTrainer

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    packing=True,
    max_seq_length=4096,
)
```

### Via ConstantLengthDataset

```python
from trl import ConstantLengthDataset

dataset = ConstantLengthDataset(
    tokenizer=tokenizer,
    dataset=raw_dataset,
    seq_length=4096,
    infinite=False,
)
```

### Manual Implementation Notes

If implementing manually:
1. Concatenate samples with EOS token separator
2. Compute position_ids per-sample (reset at each boundary)
3. Use block-diagonal attention mask
4. Cross-sample attention must be masked out to prevent leakage
