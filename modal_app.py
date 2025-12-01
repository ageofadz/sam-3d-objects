import base64
import io
import os
from typing import Dict

import modal

# ---------- Modal app ----------

app = modal.App("sam3d-objects")

CHECKPOINT_VOLUME_NAME = "sam3d-checkpoints"
volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)

# Base image: Debian + Python + build tools + Torch + PyTorch3D
sam3d_image = (
    modal.Image.debian_slim()  # Python 3.11 on Debian slim
    .apt_install(
        "git",
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libgl1",
        "cmake",
        "ninja-build",
        "build-essential",
    )
    # 1) Torch stack EXACTLY like gist (cu128 channel)
    .pip_install(
        "torch==2.8.0+cu128",
        "torchvision==0.23.0+cu128",
        "torchaudio==2.8.0+cu128",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    # 2) Core SAM-3D deps (mirroring their requirements.txt)
    .pip_install(
        # geometry / render
        "pytorch3d==0.7.8",
        "kaolin==0.18.0",
        "open3d==0.18.0",
        "opencv-python==4.9.0.80",
        "trimesh",
        "plyfile",
        "pyvista",

        # sparse conv
        "spconv-cu120==2.3.6",   # matches gist’s spconv-cu120

        # 3D Gaussian splats
        "gsplat==1.5.3",

        # meta’s helper libs
        "moge==1.0.0",           # provides utils3d, so you DON’T need manual git clone
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "loguru==0.7.2",
        "lightning==2.5.6",
        "pytorch-lightning==2.5.6",

        # web / API glue
        "fastapi==0.121.3",
        "starlette==0.50.0",
        "uvicorn",
        "gradio==5.49.0",

        # misc bits they rely on
        "easydict==1.13",
        "pyquaternion==0.9.9",
        "pymeshfix==0.17.0",
        "imageio",
        "scipy",
        "pandas",
        "timm==0.9.16",
        "matplotlib==3.10.7",
        "seaborn==0.13.2",
        "scikit-learn",
    )
    # 3) Your local code (cloned sam-3d-objects repo, with sam3d_objects/ package)
    .add_local_python_source("sam3d_objects")
)

hf_secret = modal.Secret.from_name("hf-token")

# ---------- Inference loader (with pytorch3d) ----------

_inference = None  # process-global singleton


def _ensure_checkpoints():
    """
    Download checkpoints to /data/sam3d/checkpoints once, using HF_TOKEN.
    """
    from huggingface_hub import snapshot_download

    ckpt_dir = "/data/sam3d/checkpoints"
    pipeline_cfg = os.path.join(ckpt_dir, "pipeline.yaml")

    if os.path.exists(pipeline_cfg):
        return pipeline_cfg

    os.makedirs(ckpt_dir, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret is not set in Modal")

    snapshot_download(
        repo_id="facebook/sam-3d-objects",
        repo_type="model",
        local_dir=ckpt_dir,
        allow_patterns=["checkpoints/*", "pipeline.yaml"],
        token=token,
    )
    return pipeline_cfg


def _ensure_inference():
    """
    Lazily load sam3d_objects.pipeline.Inference once per container.
    This import path expects your repo layout to match facebookresearch/sam-3d-objects.
    """
    global _inference
    if _inference is not None:
        return _inference

    os.environ["LIDRA_SKIP_INIT"] = "true"  # skip sam3d_objects.init side-effects

    pipeline_cfg = _ensure_checkpoints()

    from sam3d_objects.pipeline.inference_pipeline import Inference

    _inference = Inference(pipeline_cfg, compile=False)
    return _inference


def _decode_b64_field(item: Dict, *keys: str) -> bytes:
    """
    Try multiple keys in order, accept:
    - raw base64
    - data:...;base64,<data>
    - strings with whitespace/newlines
    - missing '=' padding
    """
    s = None
    for key in keys:
        if key and key in item and item[key]:
            s = item[key]
            break

    if s is None:
        raise ValueError(f"Missing required field (one of {keys})")

    # data URL
    if s.startswith("data:"):
        s = s.split(",", 1)[1]

    s = s.strip().replace(" ", "").replace("\n", "").replace("\r", "")
    missing = len(s) % 4
    if missing:
        s += "=" * (4 - missing)

    return base64.b64decode(s)


def _run_sam3d_on_bytes(image_bytes: bytes, mask_bytes: bytes, seed: int) -> bytes:
    """
    Core inference: bytes -> numpy -> Inference -> .ply bytes
    Uses pytorch3d under the hood as required by sam3d_objects.
    """
    from PIL import Image
    import numpy as np
    import tempfile

    inference = _ensure_inference()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

    image_np = np.array(img)
    mask_np = (np.array(mask) > 0).astype("uint8")

    output = inference(image_np, mask_np, seed=seed)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
    tmp.close()
    output["gs"].save_ply(tmp.name)

    with open(tmp.name, "rb") as f:
        ply_bytes = f.read()

    os.unlink(tmp.name)
    return ply_bytes


# ---------- HTTP endpoint ----------

@app.function(
    image=sam3d_image,
    secrets=[hf_secret],
    volumes={"/data": volume},
    timeout=60 * 30,
)
@modal.fastapi_endpoint(method="POST")
def api_run_sam3d(item: dict):
    """
    POST body examples:

    {
      "image_b64": "<base64>",
      "mask_b64": "<base64>",
      "seed": 42
    }

    or

    {
      "image": "data:image/jpeg;base64,...",
      "mask": "data:image/png;base64,..."
    }
    """
    image_bytes = _decode_b64_field(item, "image_b64", "image")
    mask_bytes = _decode_b64_field(item, "mask_b64", "mask")
    seed = int(item.get("seed", 42))

    ply_bytes = _run_sam3d_on_bytes(image_bytes, mask_bytes, seed)


    try:
        from fastapi.responses import Response
    except ImportError:
        raise ImportError("fastapi is not installed. Please install it with 'pip install fastapi'.")

    return Response(
        content=ply_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="splat.ply"'},
    )