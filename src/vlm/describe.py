"""
Generate natural-language defect descriptions with LLaVA (via Ollama, local & free).

This fulfils the PROMPT ENGINEERING technique requirement. The grade here comes
from SYSTEMATIC prompt design, not just calling the model. See prompts.py.

Model: llava:7b (runs on modest hardware; use 13b only if a GPU is free).
Time-boxed: if not working by end of Week 4, drop it (3 techniques still meet brief).

Setup (one-time, separate from pip):
    install Ollama, then:  ollama pull llava:7b   (and `ollama serve` if not running)

Usage (called by backend after classification):
    text = describe_defect(image, defect_label)            # image: path or PIL.Image

Owner: Member 4 (Product & Integration Lead) -- that's you
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from src.vlm.prompts import build_messages

DEFAULT_MODEL = "llava:7b"
DEFAULT_TIMEOUT = 120  # seconds; vision models can be slow on CPU


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def image_to_b64(image) -> str:
    """Encode an image (path/str/Path or PIL.Image) to base64 for Ollama."""
    if isinstance(image, (str, Path)):
        return base64.b64encode(Path(image).read_bytes()).decode("ascii")
    # Assume a PIL.Image-like object with .save().
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def describe_defect(image, defect_label=None, prompt_variant="cot_fewshot",
                    model=DEFAULT_MODEL, host=None, timeout=DEFAULT_TIMEOUT):
    """Return an inspector-facing description string for a flagged defect.

    Args:
        image: path (str/Path) or PIL.Image of the part.
        defect_label: optional classifier label ("good"/"defect") for context.
        prompt_variant: one of prompts.PROMPTS (default the expected-best combo).
        model: Ollama model tag.
        host: Ollama base URL (default $OLLAMA_HOST or http://localhost:11434).
        timeout: request timeout in seconds.

    Returns:
        The model's description text (stripped).

    Raises:
        RuntimeError: if Ollama is unreachable or returns an error. The backend
        wraps this call and degrades to no-description, so /predict never breaks.
    """
    host = (host or _ollama_host()).rstrip("/")
    image_b64 = image_to_b64(image)
    messages = build_messages(prompt_variant, image_b64, defect_label=defect_label)

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(
            f"Ollama returned HTTP {exc.code}: {detail[:200]}. Is the model "
            f"'{model}' pulled? Try `ollama pull {model}`."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"could not reach Ollama at {host} ({exc.reason}). Start it with "
            f"`ollama serve` and pull the model: `ollama pull {model}`."
        ) from exc

    content = (body.get("message") or {}).get("content", "")
    if not content:
        raise RuntimeError(f"Ollama response had no content: {str(body)[:200]}")
    return content.strip()


def check_ollama(model=DEFAULT_MODEL, host=None, timeout=5) -> dict:
    """Quick connectivity probe. Returns {reachable, model_available, models}."""
    host = (host or _ollama_host()).rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError) as exc:
        return {"reachable": False, "model_available": False, "models": [],
                "error": str(getattr(exc, "reason", exc))}
    names = [m.get("name", "") for m in tags.get("models", [])]
    available = any(n == model or n.startswith(model.split(":")[0]) for n in names)
    return {"reachable": True, "model_available": available, "models": names}
