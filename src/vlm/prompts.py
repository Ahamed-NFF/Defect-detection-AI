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
        # Member 4: prepend 2-3 worked image+answer examples in the message payload
        "Given the examples above, describe the defect in this new image "
        "in the same structured format."
    ),
    "cot_fewshot": (
        # Combination: few-shot examples + the step-by-step instruction
        "Following the examples above, inspect this part step by step: "
        "defect type, location, severity, recommended action."
    ),
}


def evaluate_prompts(sample_images):
    """Run each variant over sample_images, collect outputs for human scoring."""
    raise NotImplementedError("Member 4: loop variants, save outputs to a scoring sheet")
