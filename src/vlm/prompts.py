"""
Prompt engineering: variants + evaluation harness.

The REPORT needs a 'Prompt Engineering Methodology' subsection. To fill it,
implement 3-4 prompt variants below, run them on ~20 sample images, score the
outputs (human eval: accuracy of defect type / location / severity, 1-5), and
pick the best. Document the comparison -- that IS the methodology section.

Techniques to demonstrate (all required by the brief):
    - zero_shot       : plain instruction (baseline prompt)
    - cot             : chain-of-thought (identify -> localize -> severity -> action)
    - fewshot         : in-context examples (2-3 image+description pairs)
    - cot_fewshot     : combination (expected best)

Owner: Member 4
"""

from __future__ import annotations

# A consistent persona/format instruction shared by every variant — this is the
# single biggest lever on output quality, so it lives outside the variants.
SYSTEM_PROMPT = (
    "You are a meticulous manufacturing quality-control inspector. You examine "
    "product photographs and report defects concisely and factually. Never "
    "invent defects that are not visible. Keep answers under 80 words."
)

PROMPTS = {
    "zero_shot": (
        "Describe any manufacturing defect visible in this image."
    ),
    "cot": (
        "Inspect this manufactured part step by step. "
        "First, identify the defect type. "
        "Second, describe where on the part it is located. "
        "Third, estimate its severity (minor/moderate/severe). "
        "Finally, recommend an inspector action."
    ),
    "fewshot": (
        # Worked examples are injected as prior chat turns (see build_messages);
        # this is the instruction attached to the new image.
        "Given the examples above, describe the defect in this new image "
        "in the same structured format."
    ),
    "cot_fewshot": (
        # Combination: few-shot examples + the step-by-step instruction.
        "Following the examples above, inspect this part step by step: "
        "defect type, location, severity, recommended action."
    ),
}

# Few-shot exemplars. Populate with a handful of curated (image, gold answer)
# pairs to enable the `fewshot` / `cot_fewshot` variants. Entries whose image
# file does not exist are skipped, so this can stay empty until you pick them.
#   FEWSHOT_EXAMPLES = [
#       {"image": "data/raw/bottle/test/broken_large/000.png",
#        "answer": "Defect type: crack. Location: bottle neck, upper-right. "
#                  "Severity: severe. Action: reject unit."},
#       ...
#   ]
FEWSHOT_EXAMPLES: list[dict] = []

# Variants that should receive the few-shot exemplars.
_USES_FEWSHOT = {"fewshot", "cot_fewshot"}


def _encode_example_images(examples):
    """Read+encode each example image; skip entries whose file is missing."""
    from pathlib import Path

    from src.vlm.describe import image_to_b64  # local import avoids a cycle

    encoded = []
    for ex in examples:
        img = ex.get("image")
        if img and Path(img).exists():
            encoded.append({"b64": image_to_b64(img), "answer": ex.get("answer", "")})
    return encoded


def build_messages(variant, image_b64, defect_label=None, fewshot_examples=None):
    """Build an Ollama /api/chat ``messages`` list for one image + variant.

    Args:
        variant: key into PROMPTS.
        image_b64: base64-encoded image to describe.
        defect_label: optional classifier label to give the model context.
        fewshot_examples: override the module FEWSHOT_EXAMPLES (mainly for tests).

    Returns:
        list of role/content(/images) dicts ready to POST to Ollama.
    """
    if variant not in PROMPTS:
        raise ValueError(f"unknown prompt variant {variant!r}; choices: {sorted(PROMPTS)}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if variant in _USES_FEWSHOT:
        examples = fewshot_examples if fewshot_examples is not None else FEWSHOT_EXAMPLES
        for ex in _encode_example_images(examples):
            messages.append({
                "role": "user",
                "content": "Inspect this part and describe the defect.",
                "images": [ex["b64"]],
            })
            messages.append({"role": "assistant", "content": ex["answer"]})

    content = PROMPTS[variant]
    if defect_label:
        content = (f"A classifier has flagged this part as '{defect_label}'. "
                   + content)

    messages.append({"role": "user", "content": content, "images": [image_b64]})
    return messages


def evaluate_prompts(sample_images, variants=None, model="llava:7b",
                     out_path="experiments/results/prompt_eval.csv", **kwargs):
    """Run each variant over sample_images and write a human-scoring sheet.

    For every (image, variant) pair this calls the model and records the output;
    failures (e.g. Ollama down for one call) are captured per-cell rather than
    aborting the sweep. The CSV has empty score columns to fill in by hand:
    type/location/severity accuracy + an overall 1-5 — that table becomes the
    report's prompt-engineering methodology.

    Returns the list of row dicts.
    """
    import csv
    from pathlib import Path

    from src.vlm.describe import describe_defect  # local import avoids a cycle

    variants = variants or list(PROMPTS.keys())
    rows = []
    for img in sample_images:
        for variant in variants:
            try:
                output = describe_defect(img, defect_label=None,
                                         prompt_variant=variant, model=model, **kwargs)
                err = ""
            except Exception as exc:  # keep the sweep going on a single failure
                output, err = "", str(exc)
            rows.append({
                "image": str(img), "variant": variant, "output": output, "error": err,
                "type_accuracy_1to5": "", "location_accuracy_1to5": "",
                "severity_accuracy_1to5": "", "overall_1to5": "", "notes": "",
            })

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)

    # Companion markdown: variant outputs grouped per image, for eyeballing and
    # for pasting examples into the report.
    md = out.with_suffix(".md")
    by_img: dict[str, list] = {}
    for r in rows:
        by_img.setdefault(r["image"], []).append(r)
    lines = ["# Prompt-variant outputs\n"]
    for img, rs in by_img.items():
        lines.append(f"\n## {Path(img).name}\n")
        for r in rs:
            text = r["output"] or f"_(error: {r['error']})_"
            lines.append(f"- **{r['variant']}**: {text}")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out} and {md} ({len(rows)} rows: {len(sample_images)} images x "
          f"{len(variants)} variants). Fill in the score columns in the CSV for the report.")
    return rows
