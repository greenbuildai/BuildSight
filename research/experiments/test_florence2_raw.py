"""
Florence-2 RAW output diagnostic — BuildSight
Shows exactly what the model generates before any filtering,
so we can tune prompts and the echo-rejection logic.
"""

import sys
import time
import torch
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKEND = PROJECT_ROOT / "dashboard" / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(PROJECT_ROOT))

import geoai_vlm_util as vlm

IMG = PROJECT_ROOT / "data/annotated_results/ensemble/normal.jpg"

# Task variants to test
TASKS = [
    ("<CAPTION>",               ""),
    ("<DETAILED_CAPTION>",      ""),
    ("<MORE_DETAILED_CAPTION>", ""),
    ("<VQA>",  "Are the workers wearing helmets and safety vests?"),
    ("<VQA>",  "How many workers are visible on site?"),
    ("<VQA>",  "What activity is happening on the construction site?"),
    ("<VQA>",  "Describe the safety equipment worn by workers."),
    ("<VQA>",  "What are the workers doing?"),
]

SEP = "-" * 70

def run_raw(task_token: str, question: str, img: Image.Image):
    prompt = task_token + question
    inputs = vlm._VLM_PROCESSOR(text=prompt, images=img, return_tensors="pt")

    model_device = next(vlm._VLM_MODEL.parameters()).device
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model_device, dtype=vlm._VLM_DTYPE) if v.is_floating_point() else v.to(model_device)

    t0 = time.time()
    with torch.inference_mode():
        generated_ids = vlm._VLM_MODEL.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=128,
            num_beams=3,
            do_sample=False,
            early_stopping=True,
        )
    elapsed = round(time.time() - t0, 2)

    raw_text = vlm._VLM_PROCESSOR.batch_decode(generated_ids, skip_special_tokens=False)[0]
    try:
        parsed = vlm._VLM_PROCESSOR.post_process_generation(
            raw_text, task=task_token, image_size=(img.width, img.height)
        )
        parsed_val = parsed.get(task_token, raw_text)
    except Exception as e:
        parsed_val = f"[post_process error: {e}]"

    return raw_text, parsed_val, elapsed


def main():
    print(SEP)
    print("  Florence-2 RAW Output Diagnostic")
    print(SEP)

    print("\n[1] Loading model ...")
    ok = vlm._try_load_vlm()
    if not ok:
        print("  FAIL — model did not load")
        sys.exit(1)
    print(f"  Loaded: {vlm._VLM_MODEL_ID_LOADED}  device={vlm._VLM_DEVICE}")

    print(f"\n[2] Loading test image: {IMG}")
    img = Image.open(IMG).convert("RGB")
    print(f"  Size: {img.size}")

    print(f"\n[3] Running {len(TASKS)} task variants ...\n")

    for task_token, question in TASKS:
        prompt_display = f"{task_token}{question}" if question else task_token
        print(f"\n  PROMPT : {prompt_display}")
        raw, parsed, elapsed = run_raw(task_token, question, img)
        print(f"  RAW    : {repr(raw[:200])}")
        print(f"  PARSED : {repr(str(parsed)[:200])}")
        print(f"  TIME   : {elapsed}s")

    print(f"\n{SEP}")
    print("  Diagnostic complete.")
    print(SEP)


if __name__ == "__main__":
    main()
