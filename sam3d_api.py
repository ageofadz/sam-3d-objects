# sam3d_api.py
import base64
import io
import os
import tempfile
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image
import numpy as np
from huggingface_hub import snapshot_download

# ----- HF checkpoints -----

TAG = "hf"
CKPT_ROOT = "/data/sam3d"
CKPT_DIR = os.path.join(CKPT_ROOT, "checkpoints")
PIPELINE_CONFIG = os.path.join(CKPT_DIR, "pipeline.yaml")

_inference = None  # singleton per process


def ensure_checkpoints() -> str:
    """
    Downloads checkpoints into /data/sam3d/checkpoints if not present.
    Returns the path to pipeline.yaml
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    if os.path.exists(PIPELINE_CONFIG):
        return PIPELINE_CONFIG

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN env var must be set on the Vast instance")

    snapshot_download(
        repo_id="facebook/sam-3d-objects",
        repo_type="model",
        local_dir=CKPT_DIR,
        allow_patterns=["checkpoints/*", "pipeline.yaml"],
        local_dir_use_symlinks=False,
        token=token,
    )
    if not os.path.exists(PIPELINE_CONFIG):
        raise RuntimeError(f"pipeline.yaml not found in {CKPT_DIR}")

    return PIPELINE_CONFIG


def get_inference():
    """
    Lazy-load Inference from sam3d_objects.
    """
    global _inference
    if _inference is not None:
        return _inference

    os.environ["LIDRA_SKIP_INIT"] = "true"  # skip heavy init side effects

    cfg_path = ensure_checkpoints()
    from sam3d_objects.pipeline.inference_pipeline import Inference

    _inference = Inference(cfg_path, compile=False)
    return _inference


# ----- FastAPI models -----

class Sam3DRequest(BaseModel):
    # can be either raw base64 or data URL, same for mask
    image: str
    mask: str
    seed: int | None = 42


app = FastAPI(title="SAM-3D Objects API")


def _decode_b64(s: str) -> bytes:
    """Supports raw base64 or data:...;base64,..."""
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    s = s.strip().replace(" ", "").replace("\n", "").replace("\r", "")
    missing = len(s) % 4
    if missing:
        s += "=" * (4 - missing)
    return base64.b64decode(s)


@app.post("/run_sam3d")
def run_sam3d(body: Sam3DRequest):
    try:
        image_bytes = _decode_b64(body.image)
        mask_bytes = _decode_b64(body.mask)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    inference = get_inference()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

    image_np = np.array(img)
    mask_np = (np.array(mask) > 0).astype("uint8")

    output = inference(image_np, mask_np, seed=int(body.seed or 42))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
    tmp.close()
    output["gs"].save_ply(tmp.name)

    with open(tmp.name, "rb") as f:
        ply_bytes = f.read()

    os.unlink(tmp.name)

    return Response(
        content=ply_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename=\"splat.ply\"'},
    )