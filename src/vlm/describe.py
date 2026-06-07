"""
Generate natural-language defect descriptions with LLaVA (via Ollama, local & free).

This fulfils the PROMPT ENGINEERING technique requirement. The grade here comes
from SYSTEMATIC prompt design, not just calling the model. See prompts.py.

Model: llava:7b (runs on modest hardware; use 13b only if a GPU is free).
Time-boxed: if not working by end of Week 4, drop it (3 techniques still meet brief).

Usage (called by backend after classification):
    text = describe_defect(image_path, defect_label)

Owner: Member 4 (Product & Integration Lead) -- that's you
"""


def describe_defect(image_path, defect_label, prompt_variant="cot_fewshot"):
    """Return an inspector-facing description string for a flagged defect."""
    raise NotImplementedError("Member 4: call Ollama /api/generate with the chosen prompt")
