# Rating–sentiment consistency in Amazon reviews

How often does a reviewer's star rating disagree with what they wrote? What predicts those
disagreements? Are some reviewers systematically inconsistent? And are inconsistent reviews more or
less helpful?

The project answers these questions for the **All_Beauty** category of **Amazon Reviews '23**
(Hou et al., 2024): 694,252 reviews after cleaning. It extends **Kung, Orlando & Kirimlioglu (2024)**,
*Were You Helpful: Predicting Helpful Votes from Amazon Reviews* (arXiv:2412.02884), which studied
helpful votes in the same category.

**Results**
- [reports/research_questions.md](reports/research_questions.md): short answers to the four questions
- [reports/report.md](reports/report.md): the full write-up, with methods, figures and limitations
- [notebooks/](notebooks/): all the analysis code, saved with its outputs

---

## Quick start

The project runs on Windows, macOS and Linux, with or without a GPU. The slow sentiment-model step
ships precomputed, so you need neither PyTorch nor a graphics card.

**What you need**
- **Python 3.10 or newer**, from [python.org](https://www.python.org/downloads/). To check, run
  `python --version` (on macOS/Linux: `python3 --version`).
- **About 2 GB of free disk space**: roughly 0.7 GB of data and 1 GB of Python packages.
- **An internet connection.** The pipeline downloads 134 MB of review data from UCSD.

**Steps.** Type these commands in a terminal: PowerShell on Windows, Terminal on macOS and Linux.

1. **Get the code.**

   ```
   git clone https://github.com/sammitbal/rating-sentiment.git
   cd rating-sentiment
   ```

   No git? On the GitHub page, choose **Code → Download ZIP**, unzip it, and open a terminal in the
   unzipped folder.

2. **Create and activate a virtual environment.** This keeps the project's packages separate from
   the rest of your system.

   | | |
   |---|---|
   | Windows (PowerShell) | `python -m venv .venv` then `.venv\Scripts\Activate.ps1` |
   | macOS / Linux | `python3 -m venv .venv` then `source .venv/bin/activate` |

   Your prompt now starts with `(.venv)`. Run the activate command again whenever you open a new
   terminal.

3. **Install the packages.**

   ```
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. **Run everything.**

   ```
   python run_pipeline.py
   ```

5. **Look at the results.** See [Looking at the results](#looking-at-the-results) below.

### What the run does

| step | what happens | writes | time* |
|---|---|---|---|
| 1. Download | fetches the reviews and product metadata, then checks their SHA-256 checksums | `data/raw/` | depends on your connection |
| 2. Clean | removes duplicates, strips HTML, blanks auto-generated "Five Stars" titles | `data/processed/reviews.parquet`, `meta.parquet` | ~1 min |
| 3. VADER + TextBlob | lexicon sentiment scores for every review | `data/processed/sentiment_lexicon.parquet` | ~2 min |
| 4. RoBERTa | installs the shipped scores, after checking they match your data | `data/processed/sentiment_roberta.parquet` | seconds |
| 5. Analysis table | joins everything and defines inconsistency | `data/processed/analysis.parquet` | ~1 min |
| 6. Notebooks | re-executes the six notebooks, which redraws every figure | `notebooks/*.ipynb`, `reports/figures/` | ~3 min |

\*Measured on the development laptop (Intel Core Ultra 7, 32 GB RAM). Expect about 10 minutes plus
the download.

You can safely run the command again: finished steps are skipped. Two options change this:
* `--force` redoes everything.
* `--skip-notebooks` stops after step 5.

## Looking at the results

| file | what it is |
|---|---|
| `reports/research_questions.md` | short answers to the research questions |
| `reports/report.md` | the full write-up |
| `notebooks/01_…` to `06_…` | the analysis, in order. Open them with `jupyter lab`, with the virtual environment active |
| `reports/figures/` | every chart, as PNG |
| `reports/validation_sample.csv` | 300 reviews with empty label columns, for hand-checking the detector (made by notebook 3) |

## Optional: recompute the RoBERTa scores on your own hardware

The shipped scores reproduce the report exactly. You only need to recompute them if you change the
cleaning code or want to verify the scores yourself.

1. Install PyTorch for your hardware (table below), then the model packages:

   ```
   python -m pip install -r requirements-roberta.txt
   ```

2. Run the pipeline with recomputation. It picks the best device automatically; add `--device <name>`
   to choose one yourself.

   ```
   python run_pipeline.py --recompute-roberta
   ```

| your hardware | install PyTorch with | `--device` | time for all reviews* |
|---|---|---|---|
| NVIDIA GPU (Windows, Linux) | the CUDA command from [pytorch.org](https://pytorch.org/get-started/locally/) | `cuda` | not timed |
| AMD GPU (Linux) | the ROCm command from [pytorch.org](https://pytorch.org/get-started/locally/) | `cuda` | not timed |
| Apple-silicon Mac (M-series) | `python -m pip install torch` | `mps` | not timed |
| Intel GPU (Arc, Core Ultra) | `python -m pip install torch openvino` | `openvino-gpu` | ~15 min |
| Intel GPU, PyTorch XPU build | the XPU command from [pytorch.org](https://pytorch.org/get-started/locally/) | `xpu` | not timed |
| any CPU | Windows/Linux: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`; macOS: `python -m pip install torch` | `cpu`, or `openvino-cpu` after `pip install openvino` | ~8 hours |

\*On the development laptop (Intel Core Ultra 7 268V with Arc 140V graphics). The CPU figure is
extrapolated from a 2,000-review run.

Some things to know before recomputing:
- **Model download.** The first run downloads the model, about 500 MB, from Hugging Face.
- **Resuming.** Progress is saved in chunks. If a run is interrupted, run the same command again and
  it continues where it stopped.
- **Try a sample first.** `python src/sentiment_roberta.py --limit 2000` shows which device is used
  and how fast it is. Its output goes to a separate file and doesn't affect the pipeline.
- **Small differences are expected.** Devices differ only in floating-point rounding: on a
  2,000-review check, GPU and CPU gave identical labels. Recomputed results can still differ very
  slightly from the report.

## Troubleshooting

| problem | what to do |
|---|---|
| `python` is not recognised, or the version is older than 3.10 | Install Python 3.10+ from python.org. On Windows, tick **"Add python.exe to PATH"** during setup, or use `py -3` instead of `python`. On Ubuntu/Debian you may also need `sudo apt install python3-venv`. |
| PowerShell says "running scripts is disabled on this system" when activating | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once. Or skip activation and use `.venv\Scripts\python` in place of `python`. |
| `pip install` fails while building a package | Run `python -m pip install --upgrade pip` and try again. Use a 64-bit Python. Very new Python releases sometimes lack ready-made packages for a few weeks; an older version (e.g. 3.12) avoids that. |
| The download fails or times out | Check your connection. Behind a proxy, set `HTTPS_PROXY`. Then run `python run_pipeline.py` again: files that finished downloading are kept. |
| "checksum mismatch" | The file on the server has changed since this project was built. To use the new file, run `python run_pipeline.py --no-verify`. This recomputes the RoBERTa scores, so it needs the optional install above. |
| "not a valid gzip file" | The downloaded file is damaged. Delete it from `data/raw/` and run the pipeline again. |
| "The shipped RoBERTa scores were computed on different cleaned data" | You changed the cleaning code or the data. Run `python run_pipeline.py --recompute-roberta`. |
| A notebook fails with `ModuleNotFoundError` | Jupyter is using a different Python than your virtual environment. Run `python -m ipykernel install --user --name rating-sentiment`, then `python run_pipeline.py --kernel rating-sentiment`. |
| `MemoryError`, or the computer becomes unresponsive | The pipeline holds all 694K reviews in memory. Close other programs and try again. It was developed on a 32 GB machine; peak memory use hasn't been measured. |
| "Recomputing the RoBERTa scores needs PyTorch and transformers" | You asked to recompute but haven't done the optional install. Follow the recompute section above. |
| `--device auto` picks the CPU although you have a GPU | You probably have the CPU-only build of PyTorch. Check with `python -c "import torch; print(torch.cuda.is_available())"`, then reinstall PyTorch using the command from pytorch.org. |
| The GPU runs out of memory | Run `python src/sentiment_roberta.py --device cuda --token-budget 8192`, then `python run_pipeline.py --recompute-roberta` to finish. |
| A Hugging Face warning about symlinks, on Windows | This is harmless. The model still downloads and works. |
| You want to start over | Run `python run_pipeline.py --force`. Deleting `data/processed/` also works. |

## Running the steps yourself

`run_pipeline.py` runs these commands for you, in this order:

```
python src/download.py
python src/prepare.py
python src/sentiment_lexicon.py
python src/precomputed.py install        # or: python src/sentiment_roberta.py --device auto
python src/build_features.py
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

**Editing notebooks.** The notebook sources are the `notebooks/*.py` files, in jupytext's "percent"
format. After editing one, regenerate its notebook with `jupytext --to ipynb notebooks/<name>.py`.

**Refreshing the shipped scores.** After recomputing, run
`python src/precomputed.py export --note "<hardware used>"` and commit `data/precomputed/`.

## Project layout

```
run_pipeline.py          runs every step (start here)
src/
  config.py              paths and constants
  download.py            downloads the raw data and verifies its SHA-256 checksums
  prepare.py             raw jsonl.gz → reviews.parquet, meta.parquet (dedup, HTML cleanup,
                         removes auto-generated "Five Stars" titles)
  sentiment_lexicon.py   VADER (title+body, title, body) and TextBlob polarity/subjectivity
  sentiment_roberta.py   cardiffnlp/twitter-roberta-base-sentiment-latest on CUDA, MPS, XPU,
                         OpenVINO or CPU; resumable
  precomputed.py         installs (or exports) the shipped RoBERTa scores, after checking they fit
  build_features.py      inconsistency definitions + leakage-free reviewer/product features
  common.py              notebook helpers: colour-blind-safe plot style, Wilson CIs, forest plots
notebooks/               *.py sources (jupytext) and *.ipynb notebooks with their outputs
  01_data_overview          replication of Kung et al.'s counts, cleaning, descriptives
  02_sentiment_measurement  model validity, agreement, thresholds, manual audit of the detector
  03_inconsistency_rates    Q1: how often (flag rates + audit-adjusted estimate), typology
  04_predictors             Q2: what predicts disagreement (logit ORs, boosting, distinctive terms)
  05_reviewers              Q3: systematic reviewers (leave-one-out concordance, permutation nulls)
  06_helpfulness            Q4: helpfulness (Poisson/logit/product FE) + leakage-free re-run of Kung et al.
data/
  precomputed/           shipped RoBERTa scores + manifest (in the repo)
  raw/, processed/       created by the pipeline (not in the repo)
reports/
  research_questions.md  short answers to the research questions
  report.md              the full write-up
  figures/               every chart in the notebooks (PNG)
  audit_labels.csv       labels for the 250 manually audited reviews (I / M / C / U), keyed by review id
requirements.txt         core packages (no PyTorch needed)
requirements-roberta.txt optional: only for recomputing the RoBERTa scores
requirements-lock.txt    the exact versions behind the published results
```

## Definitions

These definitions are implemented in `src/build_features.py`.

- **Polar rating**: 1–2★ (negative) or 4–5★ (positive). A 3★ rating has no opposite, so 3★ reviews
  are excluded.
- **Reversal (per model)**: text polarity opposite to the rating polarity. VADER and TextBlob use
  ±0.05 neutral bands. RoBERTa uses its most probable class.
- **`inconsistent`** (working definition): RoBERTa reverses **and** at least one lexicon agrees.
- **`inconsistent_confident`**: the above, plus RoBERTa P(opposite class) ≥ 0.8.
- **`inconsistent_majority`**: naive 2-of-3 vote, kept for comparison. Correlated lexicon errors
  inflate it (see the report).
- Reviewer history features are **leave-one-out** or **prior-only**, so none contains the review's own
  outcome.

## Data and citations

The review data is not stored in this repository: `src/download.py` fetches it from the McAuley Lab
at UCSD. The files in `data/precomputed/` and `reports/audit_labels.csv` contain only review ids,
scores and labels, not review text. If you use the data, please cite Hou et al. (2024).

- Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). *Bridging Language and Items for
  Retrieval and Recommendation.* arXiv:2403.03952. https://amazon-reviews-2023.github.io/
- Kung, H., Orlando, D., & Kirimlioglu, E. (2024). *Were You Helpful – Predicting Helpful Votes from
  Amazon Reviews.* arXiv:2412.02884.
- Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of
  social media text. *ICWSM.*
- Loureiro, D., Barbieri, F., Neves, L., Espinosa Anke, L., & Camacho-Collados, J. (2022). TimeLMs:
  Diachronic language models from Twitter. *ACL demo.*
- Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' words: Lexical feature selection and
  evaluation for identifying the content of political conflict. *Political Analysis.*
- Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). How to fine-tune BERT for text classification?
  *CCL.*
