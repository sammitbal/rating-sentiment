"""Score every review with a RoBERTa sentiment classifier, on whatever hardware is available.

Most users never need this: run_pipeline.py installs the shipped scores from data/precomputed/.
Run it to recompute them, e.g. after changing the cleaning code.

--device auto picks, in order:
  cuda          NVIDIA GPU, or AMD GPU with a ROCm build of PyTorch (fp16)
  xpu           Intel GPU with an XPU build of PyTorch
  mps           Apple-silicon GPU
  openvino-gpu  Intel GPU through OpenVINO (pip install openvino)
  cpu           everything else
openvino-cpu can also be chosen by hand; it is often faster than plain PyTorch on x86 CPUs.

Reviews are processed in length-sorted chunks checkpointed to disk, so an interrupted run resumes
where it stopped. Reviews longer than MAX_TOKENS keep their first HEAD tokens and their last
(MAX_TOKENS - 2 - HEAD) tokens: openings and closings carry most of a review's verdict (head+tail
truncation, Sun et al. 2019).

Usage:  python src/sentiment_roberta.py [--device auto] [--limit N]
"""
import argparse
import contextlib
import math
import re
import time

import numpy as np
import pandas as pd

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging
except ImportError:
    raise SystemExit("Recomputing the RoBERTa scores needs PyTorch and transformers:\n"
                     "    pip install -r requirements-roberta.txt\n"
                     "(See the README for GPU builds of PyTorch.) Without them, run_pipeline.py "
                     "uses the shipped scores instead.")

from config import REVIEWS, ROBERTA_CHUNKS, ROBERTA_MODEL, SENT_ROBERTA

MAX_TOKENS = 256
HEAD = 128
MAX_BATCH = 128
CHARS_PER_MAX_INPUT = 1100  # ~MAX_TOKENS of English text; used only for the ETA
DEVICES = ["auto", "cuda", "xpu", "mps", "openvino-gpu", "openvino-cpu", "cpu"]
# Padding widths to a multiple keeps the number of distinct tensor shapes small, which matters
# for backends that compile a kernel per shape.
PAD_MULTIPLE = {"cpu": 1, "cuda": 8}
DEFAULT_PAD_MULTIPLE = 16


def openvino_devices():
    try:
        import openvino as ov
    except ImportError:
        return []
    return ov.Core().available_devices


def resolve_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    if "GPU" in openvino_devices():
        return "openvino-gpu"
    return "cpu"


def preprocess(text):
    # The normalisation the cardiffnlp models were trained with.
    text = re.sub(r"(?<!\w)@\w+", "@user", text)
    return re.sub(r"http\S+", "http", text)


def encode(tok, texts):
    body = MAX_TOKENS - 2
    bos, eos = tok("", add_special_tokens=True)["input_ids"]  # <s>, </s> for RoBERTa
    raw = tok(texts, add_special_tokens=False, truncation=False)["input_ids"]
    ids = [[bos] + (x if len(x) <= body else x[:HEAD] + x[-(body - HEAD):]) + [eos] for x in raw]
    n_tokens = np.array([len(x) for x in raw])
    return ids, n_tokens, n_tokens > body


def batches(widths, token_budget):
    """Group indices into width-sorted batches of at most token_budget padded tokens."""
    batch = []
    for i in np.argsort(widths, kind="stable"):
        # Sorted ascending, so the current item is the widest in its batch.
        if batch and (len(batch) == MAX_BATCH or widths[i] * (len(batch) + 1) > token_budget):
            yield batch
            batch = []
        batch.append(i)
    if batch:
        yield batch


def torch_runner(model, device, threads):
    if threads:
        torch.set_num_threads(threads)
    model.to(device)
    # fp16 on CUDA/ROCm GPUs; fp32 elsewhere (differences are at the 1e-3 level in probability).
    precision = (lambda: torch.autocast(device_type="cuda", dtype=torch.float16)) if device == "cuda" \
        else contextlib.nullcontext

    @torch.inference_mode()
    def run(ids, mask):
        with precision():
            logits = model(input_ids=torch.from_numpy(ids).to(device),
                           attention_mask=torch.from_numpy(mask).to(device)).logits
        return logits.float().cpu().numpy()
    return run


def openvino_runner(model, device):
    try:
        import openvino as ov
    except ImportError:
        raise SystemExit("--device openvino-* needs OpenVINO:  pip install openvino")
    example = {k: torch.ones(2, 16, dtype=torch.long) for k in ("input_ids", "attention_mask")}
    ov_model = ov.convert_model(model, example_input=example)
    ov_model.reshape({inp.get_any_name(): ov.PartialShape([-1, -1]) for inp in ov_model.inputs})
    compiled = ov.Core().compile_model(ov_model, device.split("-")[1].upper())
    return lambda ids, mask: compiled({"input_ids": ids, "attention_mask": mask})[0]


def score(run, input_ids, pad_id, n_labels, pad_multiple, token_budget):
    lengths = np.array([len(x) for x in input_ids])
    widths = -(-lengths // pad_multiple) * pad_multiple
    probs = np.zeros((len(input_ids), n_labels), dtype=np.float32)
    for b in batches(widths, token_budget):
        ids = np.full((len(b), widths[b].max()), pad_id, dtype=np.int64)
        mask = np.zeros_like(ids)
        for row, i in enumerate(b):
            ids[row, :lengths[i]] = input_ids[i]
            mask[row, :lengths[i]] = 1
        logits = run(ids, mask)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs[b] = e / e.sum(axis=1, keepdims=True)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=DEVICES, default="auto")
    ap.add_argument("--limit", type=int, help="score a random sample of N reviews (for testing)")
    ap.add_argument("--chunk-size", type=int, default=20000)
    ap.add_argument("--token-budget", type=int,
                    help="padded tokens per forward pass; lower it if the GPU runs out of memory "
                         "(default: 16384 on CPU, 32768 on GPUs)")
    ap.add_argument("--threads", type=int, help="PyTorch CPU threads (default: all physical cores)")
    args = ap.parse_args()
    hf_logging.set_verbosity_error()

    device = resolve_device(args.device)
    token_budget = args.token_budget or (16384 if device.endswith("cpu") else 32768)
    df = pd.read_parquet(REVIEWS, columns=["review_id", "full_text"])
    if args.limit:
        df = df.sample(args.limit, random_state=0)
    tag = f"limit{args.limit}" if args.limit else "full"
    out_path = SENT_ROBERTA if not args.limit else SENT_ROBERTA.with_name(
        f"sentiment_roberta_{tag}_{device}.parquet")
    chunk_dir = ROBERTA_CHUNKS / f"{tag}_{device}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Length-sorted chunks keep padding (wasted compute) low inside each batch.
    df = (df.assign(_len=df.full_text.str.len())
            .sort_values(["_len", "review_id"], kind="stable").reset_index(drop=True))
    cost = np.minimum(df._len.to_numpy(), CHARS_PER_MAX_INPUT) + 50

    tok = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL).eval()
    n_labels = model.config.num_labels
    cols = [f"rb_{model.config.id2label[i].lower()[:3]}" for i in range(n_labels)]
    run = (openvino_runner(model, device) if device.startswith("openvino")
           else torch_runner(model, device, args.threads))
    print(f"{len(df):,} reviews, labels {cols}, device {device}", flush=True)

    n_chunks = math.ceil(len(df) / args.chunk_size)
    t0, cost_done = time.time(), 0
    for c in range(n_chunks):
        path = chunk_dir / f"chunk_{c:04d}.parquet"
        lo, hi = c * args.chunk_size, min((c + 1) * args.chunk_size, len(df))
        if path.exists():
            continue
        t = time.time()
        part = df.iloc[lo:hi]
        ids, n_tokens, truncated = encode(tok, [preprocess(x) for x in part.full_text])
        probs = score(run, ids, tok.pad_token_id, n_labels,
                      PAD_MULTIPLE.get(device, DEFAULT_PAD_MULTIPLE), token_budget)
        out = pd.DataFrame(probs, columns=cols)
        out.insert(0, "review_id", part.review_id.to_numpy())
        out["n_tokens"] = n_tokens
        out["truncated"] = truncated
        out.to_parquet(path, index=False)

        cost_done += cost[lo:hi].sum()
        eta = cost[hi:].sum() / (cost_done / (time.time() - t0)) / 60
        print(f"chunk {c + 1}/{n_chunks}: {hi - lo:,} reviews in {time.time() - t:.0f}s, "
              f"ETA {eta:.0f} min", flush=True)

    parts = [pd.read_parquet(chunk_dir / f"chunk_{c:04d}.parquet") for c in range(n_chunks)]
    pd.concat(parts, ignore_index=True).sort_values("review_id").to_parquet(out_path, index=False)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
