"""
Med-Flamingo Multimodal Inference

Runs multimodal few-shot inference using Med-Flamingo.
All inputs (paths, prompts, outputs) are provided via CLI.
"""

import argparse
import os
import sys

import torch
from accelerate import Accelerator
from einops import repeat
from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.append("..")

from open_flamingo.open_flamingo import create_model_and_transforms
from src.utils import FlamingoProcessor


# --------------------------------------------------
# Argument Parser
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Med-Flamingo Multimodal Inference"
    )

    parser.add_argument("--llama-path", required=True,
                        help="Path to LLaMA model weights")
    parser.add_argument("--image-list", required=True,
                        help="Text file with image paths (one per line)")
    parser.add_argument("--prompt-file", required=True,
                        help="Text file containing the prompt")
    parser.add_argument("--output-file", required=True,
                        help="File to save the generated output")
    parser.add_argument("--max-new-tokens", type=int, default=150,
                        help="Maximum number of tokens to generate")

    return parser.parse_args()


# --------------------------------------------------
# Model Initialization
# --------------------------------------------------

def initialize_model(llama_path, accelerator):
    if not os.path.exists(llama_path):
        raise ValueError(
            "LLaMA model not found. Please check README for setup instructions."
        )

    print("Loading Med-Flamingo model...")

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=llama_path,
        tokenizer_path=llama_path,
        cross_attn_every_n_layers=4,
    )

    checkpoint_path = hf_hub_download(
        repo_id="med-flamingo/med-flamingo",
        filename="model.pt",
    )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=accelerator.device),
        strict=False,
    )

    processor = FlamingoProcessor(tokenizer, image_processor)

    model = accelerator.prepare(model)
    model.eval()

    return model, processor


# --------------------------------------------------
# Data Loading
# --------------------------------------------------

def load_images(image_list_file):
    with open(image_list_file, "r") as f:
        paths = [line.strip() for line in f if line.strip()]

    return [Image.open(p).convert("RGB") for p in paths]


def load_prompt(prompt_file):
    with open(prompt_file, "r") as f:
        return f.read()


# --------------------------------------------------
# Inference
# --------------------------------------------------

def run_inference(model, processor, images, prompt, accelerator, max_new_tokens):
    device = accelerator.device

    pixels = processor.preprocess_images(images)
    pixels = repeat(pixels, "N c h w -> b N T c h w", b=1, T=1)

    tokenized = processor.encode_text(prompt)

    with torch.no_grad():
        generated = model.generate(
            vision_x=pixels.to(device),
            lang_x=tokenized["input_ids"].to(device),
            attention_mask=tokenized["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
        )

    return processor.tokenizer.decode(generated[0])


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    args = parse_args()
    accelerator = Accelerator()

    model, processor = initialize_model(args.llama_path, accelerator)
    images = load_images(args.image_list)
    prompt = load_prompt(args.prompt_file)

    response = run_inference(
        model,
        processor,
        images,
        prompt,
        accelerator,
        args.max_new_tokens,
    )

    with open(args.output_file, "w") as f:
        f.write(response)

    print("Inference completed successfully.")
    print(response)


if __name__ == "__main__":
    main()
