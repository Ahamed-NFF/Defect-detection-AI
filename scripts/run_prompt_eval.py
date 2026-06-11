"""
Run the prompt-engineering evaluation for the report.

For each category this picks real test-defect images, uses the first few as
few-shot exemplars (with a templated gold answer derived from the defect-type
folder name), then sweeps all four prompt variants (zero_shot, cot, fewshot,
cot_fewshot) over the remaining images via LLaVA and writes a scoring sheet
(CSV) + a readable transcript (MD) to experiments/results/.

Prerequisite: Ollama running with the model pulled:
    ollama pull llava:7b           # then `ollama serve` if not already running

Run (lightweight — no torch needed; just needs Ollama reachable):
    python scripts/run_prompt_eval.py --categories bottle hazelnut carpet \
        --shots 2 --num 5 --model llava:7b

Owner: Member 4 (prompt engineering)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm import prompts as P
from src.vlm.describe import check_ollama

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def defect_images(data_root: Path, category: str) -> list[Path]:
    """Real test-defect images for a category (test/<defect_type>/*, not good)."""
    test_dir = data_root / category / "test"
    imgs: list[Path] = []
    if test_dir.is_dir():
        for sub in sorted(test_dir.iterdir()):
            if sub.is_dir() and sub.name != "good":
                imgs += sorted(p for p in sub.rglob("*")
                               if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return imgs


def gold_answer(defect_type: str) -> str:
    """A templated reference answer for a few-shot exemplar (format demonstrator)."""
    nice = defect_type.replace("_", " ")
    return (f"Defect type: {nice}. Location: visible on the part surface. "
            f"Severity: moderate to severe. Recommended action: reject the unit "
            f"and flag the batch for inspection.")


def main(argv=None):
    p = argparse.ArgumentParser(description="Sweep LLaVA prompt variants over defect images.")
    p.add_argument("--categories", nargs="+", default=["bottle", "hazelnut", "carpet"])
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--shots", type=int, default=2, help="few-shot exemplars per category")
    p.add_argument("--num", type=int, default=5, help="eval images per category")
    p.add_argument("--model", default="llava:7b")
    p.add_argument("--out-dir", default="experiments/results")
    args = p.parse_args(argv)

    status = check_ollama(model=args.model)
    if not status["reachable"]:
        print(f"WARNING: Ollama not reachable ({status.get('error')}). "
              f"Start it (`ollama serve`) and pull the model (`ollama pull {args.model}`). "
              f"Running anyway — outputs will record connection errors.")
    elif not status["model_available"]:
        print(f"WARNING: model '{args.model}' not found in Ollama ({status['models']}). "
              f"Pull it: `ollama pull {args.model}`.")

    data_root = Path(args.data_root)
    for cat in args.categories:
        imgs = defect_images(data_root, cat)
        if len(imgs) < args.shots + 1:
            print(f"[{cat}] skip — only {len(imgs)} defect image(s) found under {data_root/cat}")
            continue

        shots = imgs[: args.shots]
        evals = imgs[args.shots: args.shots + args.num]
        # Inject category-specific few-shot exemplars for this run.
        P.FEWSHOT_EXAMPLES = [
            {"image": str(s), "answer": gold_answer(s.parent.name)} for s in shots
        ]
        print(f"\n[{cat}] {len(shots)} few-shot exemplar(s), {len(evals)} eval image(s)")
        P.evaluate_prompts(
            [str(e) for e in evals],
            model=args.model,
            out_path=str(Path(args.out_dir) / f"prompt_eval_{cat}.csv"),
        )

    print("\nDone. Score the CSVs by hand (type/location/severity 1-5) for the report; "
          "the .md files hold the raw transcripts.")


if __name__ == "__main__":
    main()
