# vast_manager.py
import os
import time
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---- Config via env vars ----

VAST_API_KEY = os.environ.get("VAST_API_KEY")  # create in Vast
if not VAST_API_KEY:
    raise RuntimeError("Set VAST_API_KEY env var.")

# You can either hardcode a specific machine id OR a template
# that you've already set up in Vast with your Docker image / disk.
VAST_API_BASE = "https://vast.ai/api/v0"

# For simplicity, we'll assume:
# - You know the 'offer_id' or 'machine_id' to buy from Vast.
# - You know which port your sam3d_api is on (8000).
SAM3D_MACHINE_ID = os.environ.get("SAM3D_MACHINE_ID")  # e.g. "123456"
SAM3D_PORT = int(os.environ.get("SAM3D_PORT", "8000"))

STATE_PATH = Path(__file__).with_name("vast_state.json")


class EnsureResponse(BaseModel):
    backend_url: str
    instance_id: str


app = FastAPI(title="SAM-3D Vast manager")


def _load_state() -> dict:
    if STATE_PATH.exists():
        import json

        return json.loads(STATE_PATH.read_text())
    return {}


def _save_state(state: dict) -> None:
    import json

    STATE_PATH.write_text(json.dumps(state))


def _vast_get(path: str, params: dict | None = None) -> dict:
    params = dict(params or {})
    params["api_key"] = VAST_API_KEY
    r = requests.get(f"{VAST_API_BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _vast_post(path: str, payload: dict | None = None) -> dict:
    params = {"api_key": VAST_API_KEY}
    r = requests.post(f"{VAST_API_BASE}{path}", json=payload or {}, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _vast_put(path: str, payload: dict | None = None) -> dict:
    params = {"api_key": VAST_API_KEY}
    r = requests.put(f"{VAST_API_BASE}{path}", json=payload or {}, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _get_instance_info(instance_id: str) -> dict:
    # Adjust path according to Vast docs if needed
    return _vast_get(f"/instances/{instance_id}")


def _kill_instance(instance_id: str) -> None:
    # Adjust path according to Vast docs if needed
    _vast_put(f"/instances/{instance_id}/kill", {})


def _buy_instance() -> dict:
    """
    Minimal 'buy' call. You will need to align this JSON with Vast's
    current API fields (machine_id / image / args / ports...).
    This is the one place you'll probably tweak after reading Vast docs.
    """

    if not SAM3D_MACHINE_ID:
        raise RuntimeError("Set SAM3D_MACHINE_ID env var to your chosen Vast machine id")

    payload = {
        # Example structure – check Vast docs for exact keys:
        "machine_id": int(SAM3D_MACHINE_ID),
        "image": "nvidia/cuda:12.1.0-runtime-ubuntu22.04",  # or your own docker image
        "disk": 50,
        "onstart_cmd": (
            "cd /root/sam3d && "
            "export HF_TOKEN=$HF_TOKEN && "
            "export PYTHONPATH=$PYTHONPATH:$(pwd) && "
            "uvicorn sam3d_api:app --host 0.0.0.0 --port 8000"
        ),
        # Any extra env/ports config you need goes here.
    }

    # In Vast docs this is something like /instances/buy or /buy
    resp = _vast_post("/instances/buy", payload)
    # Adjust field names based on Vast response
    return resp


def _wait_for_running(instance_id: str, timeout_s: int = 600) -> dict:
    start = time.time()
    while True:
        info = _get_instance_info(instance_id)
        state = info.get("state") or info.get("status")
        if state in ("running", "RUNNING"):  # adjust to whatever Vast returns
            return info
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Instance {instance_id} did not become running in time.")
        time.sleep(5)


def _build_backend_url(info: dict) -> str:
    """
    Convert Vast instance info -> accessible URL for sam3d_api.
    You might need to adjust which field contains the public IP/host.
    """
    # These keys are placeholders; adjust to Vast's actual response schema.
    host = info.get("public_ip") or info.get("ssh_host") or info.get("hostname")
    if not host:
        raise RuntimeError(f"Cannot find public host in instance info: {info}")
    return f"http://{host}:{SAM3D_PORT}"


def _ensure_instance() -> EnsureResponse:
    state = _load_state()
    instance_id: Optional[str] = state.get("instance_id")
    if instance_id:
        try:
            info = _get_instance_info(instance_id)
            status = info.get("state") or info.get("status")
            if status in ("running", "RUNNING"):
                url = _build_backend_url(info)
                return EnsureResponse(backend_url=url, instance_id=instance_id)
        except Exception:
            # treat as dead / invalid; fall through to create new
            instance_id = None

    # No good instance: buy a new one
    resp = _buy_instance()
    # Adjust to how Vast returns the instance id; this is a placeholder.
    new_id = str(resp.get("new_contract") or resp.get("id") or resp.get("instance_id"))
    if not new_id:
        raise RuntimeError(f"Could not parse instance id from buy response: {resp}")

    state["instance_id"] = new_id
    _save_state(state)

    info = _wait_for_running(new_id)
    url = _build_backend_url(info)
    return EnsureResponse(backend_url=url, instance_id=new_id)


# ----- FastAPI endpoints -----


@app.post("/ensure-sam3d", response_model=EnsureResponse)
def ensure_sam3d():
    """
    POST /ensure-sam3d
    Returns: { backend_url, instance_id }
    """
    try:
        return _ensure_instance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class StopRequest(BaseModel):
    instance_id: Optional[str] = None


@app.post("/stop-sam3d")
def stop_sam3d(body: StopRequest):
    """
    POST /stop-sam3d
    Body: { "instance_id": "..." } or omit to use stored one.
    """
    state = _load_state()
    inst_id = body.instance_id or state.get("instance_id")
    if not inst_id:
        raise HTTPException(status_code=400, detail="No instance id provided or stored")

    try:
        _kill_instance(inst_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error killing instance: {e}")

    if state.get("instance_id") == inst_id:
        state.pop("instance_id", None)
        _save_state(state)

    return {"ok": True, "instance_id": inst_id}