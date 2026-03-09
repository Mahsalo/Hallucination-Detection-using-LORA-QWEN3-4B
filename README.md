# LLM Hallucination Detection with LoRA Fine-Tuning

A parameter-efficient approach to detecting hallucinations in Large Language Model outputs using LoRA and QLoRA fine-tuning on Qwen3-4B.

## Overview

This project builds a hallucination detection classifier that determines whether an LLM-generated answer is factually grounded or hallucinated. The key insight is that models trained on a single domain (e.g., QA) fail to generalize to other domains (e.g., summarization), and multi-domain training is essential for robust performance.

## Key Results

| Model | QA (F1) | Summarization (F1) | Trainable Params |
|-------|---------|-------------------|------------------|
| Baseline (zero-shot) | 0.82 | — | 0% |
| LoRA (QA-only) | 0.99 | 0.28 | 0.5% |
| QLoRA (QA-only) | 0.97 | — | 0.3% |
| **LoRA (Combined)** | **0.98** | **0.65** | 0.5% |

### Key Findings

1. **LoRA achieves near-perfect performance** on in-domain data (0.99 F1 on QA)
2. **Severe generalization gap**: QA-trained model drops to 0.28 F1 on summarization
3. **Multi-domain training recovers performance**: Combined training achieves 0.98 F1 (QA) and 0.65 F1 (summarization)
4. **Parameter efficiency**: Only 0.3-0.5% of model parameters are trained

## Methodology

### Phase 1: Single-Domain Training (QA)

- **Dataset**: HaluEval QA benchmark (10K samples, 50/50 balanced)
- **Model**: Qwen3-4B with LoRA adapters on Q, K, V, O projections
- **Config**: rank=16, alpha=32, dropout=0.05
- **Result**: 0.99 Macro F1 on QA test set

### Phase 2: Generalization Gap Analysis

Evaluated the QA-trained model on summarization hallucinations (WikiBio GPT-3 dataset):
- **Result**: F1 dropped from 0.99 → 0.28
- **Insight**: Hallucination patterns are domain-specific; QA contradictions differ from summarization fabrications

### Phase 3: Multi-Domain Training

Combined QA + Summarization datasets (~4K samples total):
- **QA performance**: 0.98 F1 (minimal regression from 0.99)
- **Summarization performance**: 0.65 F1 (recovered from 0.28)
- **Insight**: Multi-domain training enables cross-domain generalization

## Technical Details

### LoRA Configuration

```python
LoraConfig(
    r=16,                    # Low-rank dimension
    lora_alpha=32,           # Scaling factor (2x rank)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)
```

### QLoRA Configuration

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

### Training

- **Epochs**: 3
- **Batch size**: 2 (with gradient accumulation = 8)
- **Learning rate**: 1e-4 with cosine scheduler
- **Max sequence length**: 512 tokens

## Datasets

| Dataset | Domain | Samples | Source |
|---------|--------|---------|--------|
| HaluEval | QA | 10,000 | `pminervini/HaluEval` |
| WikiBio GPT-3 | Summarization | 238 | `potsawee/wiki_bio_gpt3_hallucination` |

## Prompt Format

```
Given the context: {knowledge}
and the question: {question}
is the answer: {answer}
hallucinated or not? Answer yes or no:
```

## Requirements

```
torch>=2.0
transformers>=4.40
peft>=0.10
trl>=0.8
bitsandbytes>=0.43
datasets
scikit-learn
```

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load fine-tuned model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
model = PeftModel.from_pretrained(base_model, "./lora_combined_output")

# Predict
prompt = """Given the context: {context}
and the question: {question}
is the answer: {answer}
hallucinated or not? Answer yes or no:"""

# Model outputs "yes" (hallucinated) or "no" (not hallucinated)
```

## Conclusions

1. **Parameter-efficient fine-tuning works**: LoRA achieves comparable results to full fine-tuning with <1% trainable parameters
2. **Domain matters**: Single-domain training creates brittle models that fail on distribution shift
3. **Multi-domain training is essential**: Combining diverse hallucination types improves generalization
4. **QLoRA enables accessibility**: 4-bit quantization reduces GPU memory from ~14GB to ~6GB with minimal quality loss

## References

- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*
- Li et al. (2023). *HaluEval: A Large-Scale Hallucination Evaluation Benchmark*

## License

MIT
