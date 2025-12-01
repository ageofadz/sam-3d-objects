import os
import io
import base64
import tempfile

import runpod
from huggingface_hub import snapshot_download
from PIL import Image
import numpy as np

# Adjust if you want a different directory in the container
CKPT_DIR = os.environ.get("CKPT_DIR", "/workspace/checkpoints")
PIPELINE_CONFIG = os.path.join(CKPT_DIR, "pipeline.yaml")

# Singleton inference object per pod
_inference = None


def ensure_checkpoints() -> str:
    """
    Download Meta's SAM-3D Objects checkpoints once per pod.
    Assumes HF_TOKEN is provided in the environment (RunPod secret or env var).
    """
    os.makedirs(CKPT_DIR, exist_ok=True)

    if os.path.exists(PIPELINE_CONFIG):
        return PIPELINE_CONFIG

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not set in environment")

    snapshot_download(
        repo_id="facebook/sam-3d-objects",
        repo_type="model",
        local_dir=CKPT_DIR,
        allow_patterns=["checkpoints/*", "pipeline.yaml"],
        token=hf_token,
        local_dir_use_symlinks=False,
    )

    if not os.path.exists(PIPELINE_CONFIG):
        raise RuntimeError(f"pipeline.yaml not found in {CKPT_DIR}")

    return PIPELINE_CONFIG


def ensure_inference():
    """
    Lazily import and construct the Inference pipeline.
    This expects the Meta repo to be importable as `sam3d_objects`.
    """
    global _inference
    if _inference is not None:
        return _inference

    # Avoid running sam3d_objects.init side-effects
    os.environ["LIDRA_SKIP_INIT"] = "true"

    cfg_path = ensure_checkpoints()

    # Import after deps are ready
    from sam3d_objects.pipeline.inference_pipeline import Inference  # type: ignore

    _inference = Inference(cfg_path, compile=False)
    return _inference


def decode_b64_field(obj, *keys) -> bytes:
    """
    Accept multiple key names and handle:

    - raw base64
    - data:...;base64,<data>
    - missing padding
    - stray whitespace/newlines
    """
    s = None
    for k in keys:
        if k and k in obj and obj[k]:
            s = obj[k]
            break

    if s is None:
        raise ValueError(f"Missing required field; tried keys: {keys}")

    # Strip data URL header
    if s.startswith("data:"):
        s = s.split(",", 1)[1]

    s = s.strip().replace("\n", "").replace(" ", "")
    missing = len(s) % 4
    if missing:
        s += "=" * (4 - missing)

    return base64.b64decode(s)


def run_sam3d_on_bytes(image_bytes: bytes, mask_bytes: bytes, seed: int) -> bytes:
    """
    Core pipeline: bytes -> numpy -> Inference -> .ply bytes
    """
    inference = ensure_inference()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

    img_np = np.array(img)
    mask_np = (np.array(mask) > 0).astype("uint8")

    out = inference(img_np, mask_np, seed=seed)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
    tmp.close()

    out["gs"].save_ply(tmp.name)

    with open(tmp.name, "rb") as f:
        ply_bytes = f.read()

    os.unlink(tmp.name)
    return ply_bytes


def handler(event):
    """
    RunPod Serverless entrypoint.

    Expected event format:

    {
      "input": {
        "image_b64": "<base64 image>",   // or "image": "data:image/png;base64,..."
        "mask_b64":  "<base64 mask>",    // or "mask": "data:image/png;base64,..."
        "seed": 42                        // optional
      }
    }

    Returns:
    {
      "output": {
        "ply_b64": "<base64-encoded .ply>",
        "filename": "splat.ply",
        "content_type": "application/octet-stream"
      }
    }
    """
    try:
        body = event.get("input") or {}

        image_bytes = decode_b64_field(body, "image_b64", "image")
        mask_bytes = decode_b64_field(body, "mask_b64", "mask")
        seed = int(body.get("seed", 42))

        ply_bytes = run_sam3d_on_bytes(image_bytes, mask_bytes, seed)

        ply_b64 = base64.b64encode(ply_bytes).decode("ascii")

        return {
            "output": {
                "ply_b64": ply_b64,
                "filename": "splat.ply",
                "content_type": "application/octet-stream",
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    # Start RunPod serverless loop
    runpod.serverless.start({"handler": handler})