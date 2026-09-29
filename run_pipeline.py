"""Run the whole project: download → clean → sentiment → features → notebooks.

    python run_pipeline.py                      # uses the shipped RoBERTa scores: no GPU or PyTorch needed
    python run_pipeline.py --recompute-roberta  # score the reviews yourself (see README for hardware)

Steps whose output already exists are skipped; once a step runs, every later step runs too.
Pass --force to redo everything.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from config import ANALYSIS, META, REVIEWS, SENT_LEXICON, SENT_ROBERTA  # noqa: E402

NOTEBOOKS = sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb"))
ENV = {**os.environ, "PYTHONUTF8": "1"}  # consistent text encoding on every OS


def run(*cmd):
    subprocess.run([sys.executable, *map(str, cmd)], check=True, cwd=ROOT, env=ENV)


def check_kernel(name):
    """Warn when the Jupyter kernel would run a different Python than this one."""
    from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel
    try:
        exe = KernelSpecManager().get_kernel_spec(name).argv[0]
    except NoSuchKernel:
        raise SystemExit(f"No Jupyter kernel named {name!r}. Install one for this environment:\n"
                         f"    python -m ipykernel install --user --name {name}")
    if Path(exe).is_absolute() and Path(exe).resolve() != Path(sys.executable).resolve():
        print(f"WARNING: Jupyter kernel {name!r} runs {exe}, not this Python ({sys.executable}).\n"
              "If a notebook fails with ModuleNotFoundError, register this environment and retry:\n"
              "    python -m ipykernel install --user --name rating-sentiment\n"
              "    python run_pipeline.py --kernel rating-sentiment")


def main():
    if sys.version_info < (3, 10):
        raise SystemExit(f"Python 3.10 or newer is required (this is {sys.version.split()[0]}).")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute-roberta", action="store_true",
                    help="score reviews with RoBERTa yourself instead of using the shipped scores")
    ap.add_argument("--device", default="auto",
                    help="with --recompute-roberta: auto, cuda, xpu, mps, openvino-gpu, openvino-cpu or cpu")
    ap.add_argument("--skip-notebooks", action="store_true", help="stop after building the analysis table")
    ap.add_argument("--kernel", default="python3", help="Jupyter kernel used to execute the notebooks")
    ap.add_argument("--force", action="store_true", help="redo every step")
    ap.add_argument("--no-verify", action="store_true",
                    help="accept raw files whose checksum differs")
    args = ap.parse_args()
    

    # (title, outputs that mark the step as done or None to always run, action, invalidates later steps)
    roberta = (("Score sentiment: RoBERTa (recomputing)", None,
                lambda: run(SRC / "sentiment_roberta.py", "--device", args.device), True)
               if args.recompute_roberta else
               ("Score sentiment: RoBERTa (shipped scores)", [SENT_ROBERTA],
                lambda: run(SRC / "precomputed.py", "install"), True))
    steps = [
        # download.py skips files it already has, so running it never invalidates later steps
        ("Download the data", None,
         lambda: run(SRC / "download.py", *(["--no-verify"] if args.no_verify else [])), False),
        ("Clean the reviews", [REVIEWS, META], lambda: run(SRC / "prepare.py"), True),
        ("Score sentiment: VADER + TextBlob", [SENT_LEXICON], lambda: run(SRC / "sentiment_lexicon.py"), True),
        roberta,
        ("Build the analysis table", [ANALYSIS], lambda: run(SRC / "build_features.py"), True),
    ]
    rerun_rest = args.force
    for i, (title, outputs, action, invalidates) in enumerate(steps, 1):
        if outputs is not None and not rerun_rest and all(p.exists() for p in outputs):
            print(f"[{i}/{len(steps) + 1}] {title}: already done")
            continue
        print(f"\n[{i}/{len(steps) + 1}] {title}", flush=True)
        t = time.time()
        action()
        print(f"    done in {time.time() - t:.0f}s", flush=True)
        rerun_rest = rerun_rest or invalidates

    if args.skip_notebooks:
        print("\nSkipped the notebooks (--skip-notebooks). The analysis table is in data/processed/.")
        return
    check_kernel(args.kernel)
    print(f"\n[{len(steps) + 1}/{len(steps) + 1}] Run the notebooks", flush=True)
    for nb in NOTEBOOKS:
        t = time.time()
        print(f"    {nb.name} ...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
             f"--ExecutePreprocessor.kernel_name={args.kernel}", "--ExecutePreprocessor.timeout=3600",
             str(nb)], cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            print("FAILED\n" + result.stderr[-4000:])
            raise SystemExit(f"{nb.name} failed; the error is above. Open it in Jupyter to investigate.")
        print(f"{time.time() - t:.0f}s", flush=True)

    print("\nAll done. Results:\n"
          "  reports/research_questions.md   short answers to the research questions\n"
          "  reports/report.md               full write-up\n"
          "  notebooks/*.ipynb               analysis code with fresh outputs (open with: jupyter lab)\n"
          "  reports/figures/                every chart")


if __name__ == "__main__":
    main()
