"""
Stage I Inference Script for Rad-Flamingo

This script runs inference over the full dataset to generate
patient-centric explanations using prompt templates as specified in the paper.
All paths and runtime options are provided via CLI arguments.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from xraygpt.common.config import Config
from xraygpt.common.dist_utils import get_rank
from xraygpt.common.registry import registry
from xraygpt.conversation.conversation import Chat

# Required registrations
from xraygpt.datasets.builders import *
from xraygpt.models import *
from xraygpt.processors import *
from xraygpt.runners import *
from xraygpt.tasks import *


# --------------------------------------------------
# Argument Parser
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage I Inference for Rad-Flamingo"
    )

    # Model / runtime
    parser.add_argument("--cfg-path", required=True,
                        help="Path to model configuration file")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU id to use")

    # Data
    parser.add_argument("--projections-csv", required=True,
                        help="CSV file with image projections")
    parser.add_argument("--reports-xlsx", required=True,
                        help="Excel file with findings and impressions")
    parser.add_argument("--image-root", required=True,
                        help="Root directory containing images")

    # Prompting
    parser.add_argument("--few-shot-file", required=True,
                        help="Text file containing few-shot examples")
    parser.add_argument("--instruction", required=True,
                        help="Instruction string")

    # Output
    parser.add_argument("--output-file", required=True,
                        help="Path to save outputs")

    return parser.parse_args()


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def setup_seeds(cfg):
    seed = cfg.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# --------------------------------------------------
# Model Initialization
# --------------------------------------------------

def initialize_model(args):
    print("Initializing model...")

    cfg = Config(args)
    setup_seeds(cfg)

    model_cfg = cfg.model_cfg
    model_cfg.device_8bit = args.gpu_id

    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg).to(f"cuda:{args.gpu_id}")

    vis_cfg = cfg.datasets_cfg.openi.vis_processor.train
    vis_processor = registry.get_processor_class(
        vis_cfg.name
    ).from_config(vis_cfg)

    chat = Chat(model, vis_processor, device=f"cuda:{args.gpu_id}")

    print("Model initialization complete.")
    return chat, vis_processor


# --------------------------------------------------
# Inference
# --------------------------------------------------

def run_inference(chat, vis_processor, args):
    df_proj = pd.read_csv(args.projections_csv)
    df_report = pd.read_excel(args.reports_xlsx)

    with open(args.few_shot_file, "r") as f:
        few_shot = f.read()

    with open(args.output_file, "w") as fout:
        for idx, uid in enumerate(df_proj["uid"], start=1):
            row_img = df_proj[df_proj["uid"] == uid]
            row_rep = df_report[df_report["uid"] == uid]

            if row_rep.empty:
                continue

            filename = "CXR" + row_img["filename"].values[0].split(".")[0]
            image_path = os.path.join(args.image_root, f"{filename}.png")

            if not os.path.exists(image_path):
                continue

            # Prepare text
            findings = str(row_rep["findings"].values[0]).replace("XXXX", "")
            impression = str(row_rep["impression"].values[0])

            findings = f"Findings: {findings}"
            impression = f"Impressions: {impression}"

            # Load image
            image = Image.open(image_path).convert("RGB")
            image_tensor = vis_processor(image)

            # Build prompt
            prompt = (
                few_shot
                + findings
                + impression
                + args.instruction
            )

            # Inference
            response, _ = chat.answer(image_tensor, prompt)

            fout.write(f"{uid}\t{response}\n")

            if idx % 50 == 0:
                print(f"Processed {idx} samples")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    chat, vis_processor = initialize_model(args)
    run_inference(chat, vis_processor, args)


