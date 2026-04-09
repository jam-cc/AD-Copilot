"""
AD-Copilot inference script.

Usage:
  python scripts/inference.py \
      --model_path jiang-cc/AD-Copilot \
      --good_image path/to/good.png \
      --bad_image path/to/test.png

  python scripts/inference.py \
      --model_path jiang-cc/AD-Copilot \
      --good_image path/to/good.png \
      --bad_image path/to/test.png \
      --prompt "Describe the anomaly in the second image." \
      --max_new_tokens 256

  python scripts/inference.py \
      --model_path jiang-cc/AD-Copilot \
      --good_image path/to/good.png \
      --bad_image path/to/test.png \
      --benchmark
"""

from __future__ import annotations

import argparse
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_image(path: str, max_edge: int = 0) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max_edge > 0:
        img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return img


def build_inputs(processor, good_image, bad_image, prompt, device, max_edge=512):
    good_img = load_image(good_image, max_edge)
    bad_img = load_image(bad_image, max_edge)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": good_img},
                {"type": "image", "image": bad_img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(device)


@torch.inference_mode()
def generate(model, inputs, max_new_tokens=128):
    return model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)


def decode(processor, inputs, generated_ids):
    trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def benchmark(model, processor, args):
    print("\n--- Benchmark ---")

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total / 1e9:.2f}B (trainable: {trainable / 1e9:.2f}B)")

    inputs = build_inputs(
        processor, args.good_image, args.bad_image,
        args.prompt, args.device, args.image_max_edge,
    )

    # Warmup
    print("Warming up...")
    for _ in range(args.warmup):
        generate(model, inputs, max_new_tokens=8)
    torch.cuda.synchronize()

    # Latency
    times = []
    for i in range(args.runs):
        inputs = build_inputs(
            processor, args.good_image, args.bad_image,
            args.prompt, args.device, args.image_max_edge,
        )
        torch.cuda.synchronize()
        t0 = time.time()
        out = generate(model, inputs, max_new_tokens=args.max_new_tokens)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        n_tokens = out.shape[1] - inputs.input_ids.shape[1]
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s ({n_tokens} tokens, {n_tokens/elapsed:.1f} tok/s)")

    avg = sum(times) / len(times)
    times_sorted = sorted(times)
    p90 = times_sorted[int(len(times_sorted) * 0.9)]
    print(f"  Mean: {avg:.3f}s | P90: {p90:.3f}s")

    # Memory
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak GPU memory: {peak:.2f} GB")


def parse_args():
    p = argparse.ArgumentParser(description="AD-Copilot Inference")
    p.add_argument("--model_path", type=str, default="jiang-cc/AD-Copilot")
    p.add_argument("--good_image", type=str, required=True)
    p.add_argument("--bad_image", type=str, required=True)
    p.add_argument(
        "--prompt", type=str,
        default="The first image is a normal sample. Is there any anomaly "
                "in the second image? A. Yes B. No. Please answer the letter only.",
    )
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--image_max_edge", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Loading model: {args.model_path}")
    try:
        attn_impl = "flash_attention_2"
        import flash_attn  # noqa: F401
    except ImportError:
        attn_impl = "sdpa"
    print(f"Attention: {attn_impl}")

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=64 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        trust_remote_code=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    # Single inference
    inputs = build_inputs(
        processor, args.good_image, args.bad_image,
        args.prompt, args.device, args.image_max_edge,
    )
    generated_ids = generate(model, inputs, args.max_new_tokens)
    output = decode(processor, inputs, generated_ids)

    print(f"\nPrompt: {args.prompt}")
    print(f"Output: {output}")

    if args.benchmark:
        benchmark(model, processor, args)


if __name__ == "__main__":
    main()
