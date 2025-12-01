import os
import sys
import shutil
import tempfile

import gradio as gr
from huggingface_hub import snapshot_download


TAG = "hf"
CKPT_ROOT = "checkpoints"
CKPT_DIR = "/data/sam3d/checkpoints"
PIPELINE_CONFIG = os.path.join(CKPT_DIR, "pipeline.yaml")


def ensure_checkpoints():
    token = os.environ.get("HF_TOKEN")

    snapshot_download(
        repo_id="facebook/sam-3d-objects",
        repo_type="model",
        local_dir=CKPT_DIR,
        allow_patterns=["checkpoints/*"],
        token=token,
    )

def load_inference():
    ensure_checkpoints()
    os.environ["LIDRA_SKIP_INIT"] = "true"
    from pathlib import Path
    import subprocess

    from sam3d_objects.pipeline.inference_pipeline import Inference
    inference = Inference(PIPELINE_CONFIG, compile=False)
    return inference


inference = load_inference()

def run_sam3d(image, mask, seed=42):
    from PIL import Image
    import numpy as np

    if image is None or mask is None:
        raise gr.Error("Please provide both an image and a mask.")

    # Ensure RGBA for the image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGBA")
    image_np = np.array(image)

    # Convert mask to a binary 0/1 array
    if not isinstance(mask, Image.Image):
        mask = Image.fromarray(mask)
    mask = mask.convert("L")
    mask_np = (np.array(mask) > 0).astype("uint8")

    # Run the model
    output = inference(image_np, mask_np, seed=int(seed))

    # Save Gaussian splat to a temporary .ply
    tmp_dir = tempfile.mkdtemp()
    out_path = os.path.join(tmp_dir, "splat.ply")
    output["gs"].save_ply(out_path)

    return out_path


title = "SAM 3D Objects – Single Object 3D Reconstruction"
description = (
    "Upload an RGB image and a corresponding mask (white object on black background). "
    "The app runs Meta's SAM 3D Objects model and outputs a Gaussian splat (.ply)."
)

demo = gr.Interface(
    fn=run_sam3d,
    inputs=[
        gr.Image(label="Input image", type="pil"),
        gr.Image(label="Object mask (binary / grayscale)", type="pil"),
        gr.Slider(0, 100000, value=42, step=1, label="Seed"),
    ],
    outputs=gr.File(label="Output .ply (Gaussian splat)"),
    title=title,
    description=description,
)

if __name__ == "__main__":
    import gradio as gr
    demo = build_ui(inference)  # whatever your UI builder is
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
    demo.launch()