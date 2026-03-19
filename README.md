# AD-Copilot

This is the official repository for the paper:

> **AD-Copilot**
> [[Paper]](https://arxiv.org/abs/2603.13779v1) &nbsp;|&nbsp; [[Model]](https://huggingface.co/jiang-cc/AD-Copilot)

---

## Paper

If you find this work useful, please cite our paper:

```bibtex
@article{adcopilot2026,
  title     = {AD-Copilot},
  author    = {},
  journal   = {arXiv preprint arXiv:2603.13779},
  year      = {2026},
  url       = {https://arxiv.org/abs/2603.13779v1}
}
```

## Model

The pre-trained model is available on Hugging Face:

- **Model**: [jiang-cc/AD-Copilot](https://huggingface.co/jiang-cc/AD-Copilot)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("jiang-cc/AD-Copilot")
tokenizer = AutoTokenizer.from_pretrained("jiang-cc/AD-Copilot")
```

## Links

| Resource | Link |
|----------|------|
| Paper (arXiv) | https://arxiv.org/abs/2603.13779v1 |
| Model (Hugging Face) | https://huggingface.co/jiang-cc/AD-Copilot |
