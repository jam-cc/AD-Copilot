"""
AD-Copilot evaluation on MMAD benchmark.

Usage:
  # Run inference + evaluation
  python scripts/evaluate_mmad.py \
      --model_path jiang-cc/AD-Copilot \
      --data_root /path/to/MMAD/dataset \
      --output_dir results/

  # Evaluate existing predictions
  python scripts/evaluate_mmad.py \
      --eval_only \
      --predictions results/predictions.json

Requires the MMAD dataset. See https://github.com/jam-cc/MMAD for details.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_model(model_path: str):
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    print(f"Attention: {attn_impl}")

    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=64 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    return model, processor


@torch.inference_mode()
def run_inference(model, processor, good_image_path, test_image_path, prompt,
                  max_new_tokens=128, image_max_edge=512):
    good_img = Image.open(good_image_path).convert("RGB")
    test_img = Image.open(test_image_path).convert("RGB")
    if image_max_edge > 0:
        good_img.thumbnail((image_max_edge, image_max_edge), Image.Resampling.LANCZOS)
        test_img.thumbnail((image_max_edge, image_max_edge), Image.Resampling.LANCZOS)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": good_img},
                {"type": "image", "image": test_img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def evaluate_predictions(predictions: list[dict]) -> dict:
    """Compute accuracy metrics from predictions."""
    correct = sum(1 for p in predictions if p.get("correct", False))
    total = len(predictions)
    accuracy = correct / total * 100 if total > 0 else 0

    # Per-task breakdown
    task_metrics = {}
    for p in predictions:
        task = p.get("task", "unknown")
        if task not in task_metrics:
            task_metrics[task] = {"correct": 0, "total": 0}
        task_metrics[task]["total"] += 1
        if p.get("correct", False):
            task_metrics[task]["correct"] += 1

    for task, m in task_metrics.items():
        m["accuracy"] = m["correct"] / m["total"] * 100 if m["total"] > 0 else 0

    return {
        "overall_accuracy": accuracy,
        "total": total,
        "correct": correct,
        "per_task": task_metrics,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AD-Copilot on MMAD")
    p.add_argument("--model_path", type=str, default="jiang-cc/AD-Copilot")
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--output_dir", type=str, default="results/")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--image_max_edge", type=int, default=512)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--predictions", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()

    if args.eval_only:
        assert args.predictions, "--predictions required with --eval_only"
        with open(args.predictions) as f:
            predictions = json.load(f)
        metrics = evaluate_predictions(predictions)
        print(json.dumps(metrics, indent=2))
        return

    assert args.data_root, "--data_root required for inference"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    model, processor = load_model(args.model_path)

    # Load MMAD questions
    qa_path = os.path.join(args.data_root, "mmad_questions.json")
    if not os.path.exists(qa_path):
        print(f"MMAD question file not found at {qa_path}")
        print("Please download from https://github.com/jam-cc/MMAD")
        return

    with open(qa_path) as f:
        questions = json.load(f)

    print(f"Running inference on {len(questions)} questions...")
    predictions = []
    for q in tqdm(questions):
        good_path = os.path.join(args.data_root, q["good_image"])
        test_path = os.path.join(args.data_root, q["test_image"])
        prompt = q["prompt"]

        output = run_inference(
            model, processor, good_path, test_path, prompt,
            args.max_new_tokens, args.image_max_edge,
        )

        pred = {**q, "prediction": output}
        if "answer" in q:
            pred["correct"] = output.strip().lower().startswith(
                q["answer"].strip().lower()
            )
        predictions.append(pred)

    # Save predictions
    pred_path = os.path.join(args.output_dir, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to {pred_path}")

    # Evaluate
    metrics = evaluate_predictions(predictions)
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults:")
    print(f"  Overall accuracy: {metrics['overall_accuracy']:.2f}%")
    for task, m in metrics["per_task"].items():
        print(f"  {task}: {m['accuracy']:.2f}% ({m['correct']}/{m['total']})")


if __name__ == "__main__":
    main()
