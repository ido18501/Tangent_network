from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


SUBJECTS = [
    "a single maple leaf",
    "a single oak leaf",
    "a glass bottle",
    "a ceramic mug",
    "a wine glass",
    "a spoon",
    "a pair of scissors",
    "a seashell",
    "a pear",
    "an apple",
    "a banana",
    "a fish in side profile",
    "a bird in side profile",
    "a vase",
    "a desk lamp",
    "a shoe in side view",
    "a chair with a clear silhouette",
    "a plier tool",
    "a hammer",
    "a key",
]

STYLES = [
    "studio product photo",
    "minimalist catalog photo",
    "high-contrast object photo",
    "clean isolated subject photo",
]

BACKGROUNDS = [
    "on a plain white background",
    "on a plain light gray background",
    "isolated on a seamless studio background",
]

CAMERA_HINTS = [
    "centered composition",
    "single dominant object",
    "sharp edges",
    "clean outline",
    "high resolution",
    "no clutter",
    "no extra objects",
    "no text",
    "no watermark",
    "no shadow clutter",
]

NEGATIVE_PROMPT = (
    "busy background, multiple objects, text, watermark, logo, frame, border, "
    "cropped object, blur, motion blur, low contrast, clutter, occlusion, "
    "high texture background, extra предметs, duplicate object"
)


def build_prompt(rng: random.Random) -> str:
    subject = rng.choice(SUBJECTS)
    style = rng.choice(STYLES)
    bg = rng.choice(BACKGROUNDS)
    hints = ", ".join(CAMERA_HINTS)
    return f"{style}, {subject}, {bg}, {hints}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True, help="number of images to generate")
    parser.add_argument("--out_dir", type=str, default="generated_curve_images")
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe = pipe.to(args.device)

    for i in range(args.k):
        prompt = build_prompt(rng)
        gen = torch.Generator(device=args.device).manual_seed(args.seed + i)

        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            generator=gen,
        ).images[0]

        out_path = out_dir / f"img_{i:05d}.png"
        txt_path = out_dir / f"img_{i:05d}.txt"

        image.save(out_path)
        txt_path.write_text(prompt, encoding="utf-8")

        print(f"[{i+1}/{args.k}] saved {out_path.name}")
        print(f"  prompt: {prompt}")


if __name__ == "__main__":
    main()
