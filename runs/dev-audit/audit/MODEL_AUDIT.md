# Model audit

- Base: `/home/nxq/data/KEPT/models/Qwen2.5-7B`
- Reasoning: `/home/nxq/data/KEPT/models/DeepSeek-R1-Distill-Qwen-7B`
- Config used: R

## Config differences
```json
{
  "rope_theta": {
    "B": 1000000.0,
    "R": 10000
  },
  "sliding_window": {
    "B": 131072,
    "R": 4096
  },
  "transformers_version": {
    "B": "4.40.1",
    "R": "4.44.0"
  }
}
```

## Parameter compatibility
- B tensors: 339; R tensors: 339
- Missing in B: 0; extra in B: 0
- Shape mismatches: 0; dtype mismatches: 0

## Tokenizer
- vocab: B=151665, R=151665
- same token different id: 0 (sample)
- only in B: 7 sample; only in R: 7 sample

See model_audit.json for full detail and hashes.
