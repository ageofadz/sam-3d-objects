import base64
import io
import os
import tempfile
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image
from huggingface_hub import snapshot_download

# ---------- Config ----------

HF_REPO_ID = "facebook/sam-3d-objects"
CKPT_DIR = "/data/sam3d/checkpoints"  # adjust if you prefer another path
PIPELINE_CONFIG = os.path.join(CKPT_DIR, "pipeline.yaml")

# Global singleton for the model
_inference = None

app = FastAPI(
    title="SAM-3D Objects API",
    description="Simple FastAPI wrapper around facebookresearch/sam-3d-objects",
    version="0.1.0",
)


# ---------- Pydantic request model ----------

class Sam3DRequest(BaseModel):
    # You can send either image_b64 / mask_b64, or data URLs in image / mask.
    image: Optional[str] = None
    image_b64: Optional[str] = None
    mask: Optional[str] = None
    mask_b64: Optional[str] = None
    seed: int = 42


# ---------- Helpers ----------

def _ensure_checkpoints() -> str:
    """
    Download checkpoints + pipeline.yaml once into CKPT_DIR.
    Returns the path to pipeline.yaml.
    """
    os.makedirs(CKPT_DIR, exist_ok=True)

    if os.path.exists(PIPELINE_CONFIG):
        return PIPELINE_CONFIG

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable is not set")

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="model",
        local_dir=CKPT_DIR,
        allow_patterns=["checkpoints/*", "pipeline.yaml"],
        token=hf_token,
    )
    if not os.path.exists(PIPELINE_CONFIG):
        raise RuntimeError(f"pipeline.yaml not found after download in {CKPT_DIR}")

    return PIPELINE_CONFIG


def _ensure_inference():
    """
    Lazily import and construct sam3d_objects.pipeline.Inference exactly once.
    """
    global _inference
    if _inference is not None:
        return _inference

    # Avoid side-effect heavy init logic from sam3d_objects/__init__.py
    os.environ["LIDRA_SKIP_INIT"] = "true"

    cfg_path = _ensure_checkpoints()

    from sam3d_objects.pipeline.inference_pipeline import Inference

    _inference = Inference(cfg_path, compile=False)
    return _inference


def _decode_b64_field(obj: Sam3DRequest, *field_names: str) -> bytes:
    """
    Take the first non-empty field from field_names on the request.
    Accepts:
      - raw base64
      - data:...;base64,<data>
      - whitespace / newlines
      - missing '=' padding
    """
    s: Optional[str] = None
    for name in field_names:
        value = getattr(obj, name, None)
        if value:
            s = value
            break

    if not s:
        raise HTTPException(
            status_code=400,
            detail=f"Missing field; provide one of: {', '.join(field_names)}",
        )

    # Handle data URL form
    if s.startswith("data:"):
        s = s.split(",", 1)[1]

    s = s.strip().replace(" ", "").replace("\n", "").replace("\r", "")
    missing = len(s) % 4
    if missing:
        s += "=" * (4 - missing)

    try:
        return base64.b64decode(s)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")


def _run_sam3d(image_bytes: bytes, mask_bytes: bytes, seed: int) -> bytes:
    """
    Core pipeline:
      bytes -> PIL -> numpy -> Inference -> .ply bytes
    """
    inference = _ensure_inference()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

    image_np = np.array(img)
    mask_np = (np.array(mask) > 0).astype("uint8")

    output = inference(image_np, mask_np, seed=int(seed))

    # Save Gaussian splat (ply) to a temp file, then read back to bytes
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
    tmp.close()
    output["gs"].save_ply(tmp.name)

    with open(tmp.name, "rb") as f:
        ply_bytes = f.read()

    os.unlink(tmp.name)
    return ply_bytes


# ---------- FastAPI routes ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def run_sam3d_endpoint(req: Sam3DRequest):
    """
    POST /run
    Body example:

    {
      "image": "data:image/jpeg;base64,...",
      "mask": "data:image/png;base64,...",
      "seed": 42
    }

    or:

    {
      "image_b64": "<base64>",
      "mask_b64": "<base64>"
    }
    """
    # Decode inputs
    image_bytes = _decode_b64_field(req, "image_b64", "image")
    mask_bytes = _decode_b64_field(req, "mask_b64", "mask")

    ply_bytes = _run_sam3d(image_bytes, mask_bytes, req.seed)

    return Response(
        content=ply_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="splat.ply"'},
    )