<p align="center">
  <img src="https://zenlm.org/logo.png" width="200"/>
</p>

<h1 align="center">Zen Guard Gen</h1>

<p align="center">
  <strong>8B Generative Safety Moderation Model</strong>
</p>

<p align="center">
  🌐 <a href="https://zenlm.org">Website</a> •
  🤗 <a href="https://huggingface.co/zenlm/zen-guard-gen">Hugging Face</a> •
  📄 <a href="https://zenlm.org/papers/zen-guard.pdf">Paper</a> •
  📖 <a href="https://docs.zenlm.org">Documentation</a>
</p>

---

## Introduction

**Zen Guard Gen** is an 8B parameter generative safety classification model for comprehensive prompt and response moderation. It's the larger variant of the Zen Guard family, providing highest accuracy for batch processing scenarios.

## Features

- 🛡️ **8B Parameters**: Maximum accuracy for safety classification
- 🌍 **119 Languages**: Multilingual safety moderation
- 🚦 **Three-Tier Classification**: Safe, Controversial, Unsafe
- 📊 **9 Safety Categories**: Comprehensive content analysis
- ⚡ **120ms Latency**: Optimized for batch processing

## Model Specifications

| Specification | Value |
|---------------|-------|
| Parameters | 8B |
| Type | Generative |
| Base Model | Qwen3-8B |
| Context Length | 32,768 tokens |
| Languages | 119 |
| Latency | ~120ms |
| VRAM (FP16) | 16GB |
| VRAM (INT8) | 8GB |

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "zenlm/zen-guard-gen"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Prompt moderation
prompt = "How do I learn programming?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128)
result = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(result)
# Output: Safety: Safe
#         Categories: None

# Response moderation
response = "Here's a Python tutorial..."
messages = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": response}
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
result = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(result)
# Output: Safety: Safe
#         Categories: None
#         Refusal: No
```

## Deployment

```bash
# SGLang
python -m sglang.launch_server --model-path zenlm/zen-guard-gen --port 30000

# vLLM
vllm serve zenlm/zen-guard-gen --port 8000 --max-model-len 32768
```

## Performance

| Metric | Zen Guard Gen |
|--------|---------------|
| Accuracy | 96.8% |
| F1 Score | 94.2% |
| False Positive | 2.1% |

## Related Models

- [zen-guard](https://huggingface.co/zenlm/zen-guard) - 4B base model
- [zen-guard-stream](https://huggingface.co/zenlm/zen-guard-stream) - 4B streaming model

## License

Apache 2.0

## Citation

```bibtex
@misc{zenguardgen2025,
    title={Zen Guard Gen: 8B Generative Safety Moderation},
    author={Hanzo AI and Zoo Labs Foundation},
    year={2025},
    publisher={HuggingFace},
    howpublished={\url{https://huggingface.co/zenlm/zen-guard-gen}}
}
```

## Based On

Built upon [Qwen3Guard-Gen-8B](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B).

---

<p align="center">
  <strong>Zen AI</strong> - Clarity Through Intelligence<br>
  <a href="https://zenlm.org">zenlm.org</a>
</p>
