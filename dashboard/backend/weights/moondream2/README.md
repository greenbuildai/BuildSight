---
license: apache-2.0
---

moondream2 is a small vision language model designed to run efficiently on edge devices. Check out the [GitHub repository](https://github.com/vikhyat/moondream) for details, or try it out on the [Hugging Face Space](https://huggingface.co/spaces/vikhyatk/moondream2)!

**Benchmarks**

| Release | VQAv2 | GQA | TextVQA | POPE | TallyQA |
| --- | --- | --- | --- | --- | --- |
| **2024-03-04** (latest) | 74.2 | 58.5 | 36.4 | (coming soon) | (coming soon) |

**Usage**

```bash
pip install transformers timm einops
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model_id = "vikhyatk/moondream2"
revision = "2024-03-05"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

image = Image.open('<IMAGE_PATH>')
enc_image = model.encode_image(image)
print(model.answer_question(enc_image, "Describe this image.", tokenizer))
```

The model is updated regularly, so we recommend pinning the model version to a
specific release as shown above.