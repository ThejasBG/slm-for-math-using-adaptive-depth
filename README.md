# SLM for Math using Adaptive Depth

A small language model for step-by-step math solutions, **trained entirely from scratch** (no pretrained weights, no Hugging Face model code) with **adaptive-depth inference**: easy tokens can exit early at an intermediate layer instead of running through the whole network.

The architecture follows Qwen2 (RMSNorm, RoPE, grouped-query attention, SwiGLU) and adds confidence-based early exits on top.

---

## Table of contents

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Data](#data)
- [Setup](#setup)
- [Quick start](#quick-start)
- [Model configuration](#model-configuration)
- [Results](#results)
- [Limitations](#limitations)
- [Roadmap](#roadmap)

---

## How it works

### 1. Architecture

`AdaptiveMathTransformer` (in `scratch_qwen_adaptive.py`) is a decoder-only causal LM that mirrors the Qwen2 module layout:

| Component | Implementation |
|---|---|
| Normalization | `RMSNorm` (float32 variance, no bias) |
| Positions | Rotary embeddings (`rope_theta = 1e6`) |
| Attention | Grouped-query attention, bias on Q/K/V only, PyTorch SDPA (uses Flash Attention when hardware and dtype allow) |
| MLP | SwiGLU, intermediate size ≈ 5.4 × `d_model` |
| Output | `lm_head`, optionally tied to the input embeddings |

Parameter names are intended to match `Qwen2ForCausalLM`, so the state dict layout is compatible. The only addition is the early-exit logic, which **adds no parameters**.

### 2. Adaptive depth (early exit)

- **Exit layers** (default: layers 2 and 4 of 6) reuse the model's final `norm` and `lm_head` to produce next-token logits at intermediate depth.
- **Training:** each exit gets an auxiliary cross-entropy loss, weighted by `aux_weight = 0.25` and averaged across exits, added to the main loss (`batch_loss`).
- **Inference:** `adaptive_forward` runs a batch through the layers; after each exit layer, any row whose top next-token probability is `>= confidence_threshold` stops there. The remaining rows continue deeper. Rows that never reach the threshold use the full depth.
- `generate_solution(..., use_adaptive=True, return_depths=True)` returns the layer count used for every generated token, so you can measure the speed/quality trade-off.

### 3. Training recipe

- **Tokenizer:** byte-level BPE trained from scratch on questions and solutions (8,000 tokens), with special tokens `[PAD] [UNK] [BOS] [EOS] [QUESTION] [ANSWER]`.
- **Sequence format:** `[BOS] [QUESTION] <question> [ANSWER] <solution> [EOS]`, truncated to 512 tokens (question capped at 160).
- **Loss:** training supervises the whole sequence after `[BOS]`; validation and test score **answer tokens only**.
- **Optimization:** AdamW (betas 0.9/0.95, weight decay 0.1), linear warmup then cosine decay, bf16/fp16 autocast, gradient clipping at 1.0.
- **EMA:** validation and checkpoints use an exponential moving average of the weights (decay 0.998).
- **Leak-free validation:** `grouped_split` holds out whole chapter families, then purges any validation row whose template also appears in training. This avoids the optimistic validation loss you get from a random row split.
- **Class balancing:** training uses square-root inverse-frequency sampling over chapters.
- **Token cache:** `TokenizedMathStore` can pre-encode the corpus to a memory-mapped on-disk cache so tokenization is not repeated every epoch.
- **Optional RL:** `train_grpo` implements GRPO with a verifiable, shaped math reward (`math_reward`, `extract_final_answer`, `math_answers_equal`). It is included but not used in the notebooks.

---

## Repository layout

```
.
├── scratch_qwen_adaptive.py           # Model, tokenizer, data, training, generation, GRPO
├── adaptive_depth_slm_rebuilt.ipynb   # Current end-to-end pipeline (use this one)
├── adaptive_depth_slm.ipynb           # Early encoder-decoder baseline (legacy)
├── scratch_math_tokenizer.json        # 8K byte-level BPE (used by the current pipeline)
├── math_bpe_tokenizer.json            # 4K whitespace BPE (used by the legacy notebook)
└── hendrycks_math/                    # MATH dataset, parquet, one folder per subject
    ├── algebra/
    ├── counting_and_probability/
    ├── geometry/
    ├── intermediate_algebra/
    ├── number_theory/
    ├── prealgebra/
    └── precalculus/
```

Each subject folder contains `train-00000-of-00001.parquet` and `test-00000-of-00001.parquet`.

---

## Data

**Hendrycks MATH** (included): 7 subjects, 7,500 train and 5,000 test problems. The loader reads the first column as the question and the last column as the worked solution. The notebook moves 4,000 of the 5,000 test problems into training and keeps 1,000 as the final test set.

**Big Math corpus** (not included, ~520K examples): the [Large-Scale Math Reasoning Dataset (520K)](https://www.kaggle.com/datasets/dineshkumar0705/large-scale-math-reasoning-dataset-520k) on Kaggle by `dineshkumar0705`. Download it from Kaggle and place the JSONL file at
```
big_math/datasetv2.jsonl/dataset.jsonl
```

Each line needs the fields `prompt`, `response`, `answer`, and optionally `chapter` and `problem_id`. The `problem_id` prefix (for example `NN_001`) identifies the problem template and is used for leak-free splitting. The examples seen in the notebook are templated, with solutions in a fixed "Goal / Givens / Using expression / Substitute values / Compute result" format, not free-form proofs.

You can also bring extra data with `load_extra_pairs(path, question_col, answer_col)`, which accepts `.parquet`, `.jsonl`, `.json`, or `.csv`.

---

## Setup

Python 3.10+ is required (the code uses `X | Y` type syntax).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install torch pandas numpy pyarrow tokenizers jupyter
pip install sympy                # optional: symbolic answer checking in the RL reward
pip install matplotlib nltk tqdm # only needed for the legacy notebook
```

A CUDA GPU is strongly recommended. The reference run used an NVIDIA RTX 4060 Laptop GPU. The code falls back to CPU, but training the full corpus there is impractical.

---

## Quick start

### Run the notebook

1. Place the Big Math JSONL at `big_math/datasetv2.jsonl/dataset.jsonl` (or edit the path in cell 1).
2. Open `adaptive_depth_slm_rebuilt.ipynb` and run all cells. It will:
   - load MATH and Big Math and build the grouped train/validation split,
   - train the BPE tokenizer and save `scratch_math_tokenizer.json`,
   - build the 6-layer adaptive model and train it (checkpoint saved as `big_math_adaptive_scratch_best.pt`),
   - evaluate on the held-out MATH test set and compare full-depth vs adaptive decoding.

### Use the code directly

```python
import torch
from tokenizers import Tokenizer
from scratch_qwen_adaptive import AdaptiveMathTransformer, ModelConfig, generate_solution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = Tokenizer.from_file("scratch_math_tokenizer.json")

config = ModelConfig(
    vocab_size=tokenizer.get_vocab_size(),
    d_model=192, num_heads=8, num_kv_heads=2, num_layers=6,
    max_seq_len=512, exit_layers=(2, 4),
    dropout=0.1, tie_embeddings=True,
)
model = AdaptiveMathTransformer(config)

ckpt = torch.load("big_math_adaptive_scratch_best.pt", map_location=device)
model.load_state_dict(ckpt["model"])
model.to(device).eval()

text, layers_used = generate_solution(
    model, tokenizer, "Solve: If 2x + 4 = 4, find x.", device,
    max_new_tokens=128,
    use_adaptive=True, confidence_threshold=0.95, adaptive_min_depth=2,
    return_depths=True,
)
print(text)
print("layers per token:", layers_used)
```

Set `use_adaptive=False` for standard full-depth greedy decoding. Generation applies a repetition penalty (1.15) and blocks repeated 3-grams by default.

> The trained checkpoint is not included in this repository. Run the training notebook to produce it.

---

## Model configuration

The reference configuration from the rebuilt notebook:

| Setting | Value |
|---|---|
| Parameters | 5,632,128 |
| `d_model` | 192 |
| Layers | 6 |
| Attention heads / KV heads | 8 / 2 (GQA) |
| Exit layers | 2, 4 |
| Vocabulary | 8,000 (byte-level BPE) |
| Context length | 512 |
| Dropout | 0.1 |
| Embeddings | tied |
| Batch size | 32 |
| Epochs | up to 8, early stopping with patience 3 |
| Learning rate | 3e-4 (warmup 500 steps, cosine to 10%) |

---

## Results

From the saved notebook outputs of the reference run:

| Metric | Value |
|---|---|
| Fit / validation examples | 503,919 / 27,997 |
| Validation chapters (held out) | olympiad algebra, geometry, trigonometry, combinatorics |
| Template leakage into validation | 0 / 27,997 |
| Best validation answer-token loss | **4.775** (epoch 3; early-stopped after epoch 6) |
| Training-probe answer-token loss | 0.205 (epoch 6) |
| Hendrycks MATH test answer-token loss | **2.237** |
| Exact answer, full depth (16 unseen-template prompts) | 1 / 16, 6.00 layers per token |
| Exact answer, adaptive (same prompts, threshold 0.95) | 1 / 16, **5.54** layers per token |

How to read this:

- Adaptive decoding used about **8% fewer layers per token** on that small sample with no change in exact-match count. With only 16 prompts this is an indication, not a benchmark.
- The large gap between the training-probe loss (0.21) and the held-out validation loss (4.77) shows the model fits the templates it has seen and generalizes poorly to unseen problem families.
- On a trivial prompt, the adaptive decoder frequently stopped at layer 2, which is the intended behavior.

---

## Limitations

- **Weak generalization.** The model memorizes training templates well but does not solve unseen families reliably. Final-answer accuracy on real MATH problems has not been measured, only token loss.
- **Tiny evaluation.** The exact-match check uses 16 prompts and a last-number heuristic.
- **No KV cache.** Generation re-runs the full context for every new token, and adaptive early exit is applied per step on top of that. Layers saved per token do not translate directly into wall-clock speedups here.
- **Confidence is not correctness.** The exit rule uses top-1 next-token probability. Calibrate `confidence_threshold` and `adaptive_min_depth` on validation data before trusting early exits.
- **Templated corpus.** Much of the training data uses a fixed solution format, so outputs imitate that style even for prompts that do not fit it.
- **Legacy notebook issues.** `adaptive_depth_slm.ipynb` is an early encoder-decoder experiment with a 1,000-token whitespace vocabulary. It builds its test targets from the test *questions* (`y_test_ids` is derived from `x_test_tokenized`), so its test targets are not the real solutions. Treat it as history, not as a reference.
- **GRPO is experimental.** Rewards from a randomly initialized policy are very sparse, and the training loop notes that it may learn slowly or not at all.

---

## Roadmap

Ideas that follow from the limitations above:

- Evaluate real final-answer accuracy on the MATH test split using `extract_final_answer` and `math_answers_equal`.
- Add a KV cache so early exit produces actual latency gains.
- Calibrate exit thresholds on held-out data and report an accuracy-vs-depth curve.
- Scale width, depth and data, and add GSM8K-style data via `load_extra_pairs`.
- Warm-start GRPO from the supervised checkpoint rather than from random weights.

---

## License

No license file is included yet. Add one (for example MIT or Apache-2.0) before sharing publicly. Note that the Hendrycks MATH dataset and any Big Math data carry their own licenses and terms.
