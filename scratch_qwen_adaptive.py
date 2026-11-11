"""From-scratch, Qwen-inspired adaptive-depth transformer for math solutions.

This module uses no Qwen/Hugging Face model weights. It borrows architectural
ideas (RMSNorm, RoPE, GQA, SwiGLU, causal LM training) and uses
PyTorch SDPA, which dispatches to Flash Attention when the hardware/dtype permit.
"""

from __future__ import annotations

import math
import random
import re
import statistics
import copy
import ast
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[QUESTION]", "[ANSWER]"]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_math_parquets(root: str | Path = "hendrycks_math"):
    """Load all available Hendrycks MATH subject parquet files."""
    root = Path(root)
    train_frames, test_frames = [], []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for split, collected in (("train", train_frames), ("test", test_frames)):
            files = sorted(subject_dir.glob(f"{split}-*.parquet"))
            for file in files:
                collected.append(pd.read_parquet(file))
    if not train_frames:
        raise FileNotFoundError(f"No train parquet files found under {root.resolve()}")
    train = pd.concat(train_frames, ignore_index=True)
    test = pd.concat(test_frames, ignore_index=True) if test_frames else None
    if train.shape[1] < 2:
        raise ValueError("Expected parquet rows with question and solution columns")
    return train.iloc[:, 0].fillna("").astype(str), train.iloc[:, -1].fillna("").astype(str), test


def train_tokenizer(questions: Iterable[str], answers: Iterable[str], vocab_size: int = 8000) -> Tokenizer:
    """Train a byte-level BPE on both questions and solutions; no pretrained vocab."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(iter(questions_and_answers(questions, answers)), trainer=trainer)
    return tokenizer


def questions_and_answers(questions: Iterable[str], answers: Iterable[str]):
    yield from questions
    yield from answers


class MathSolutionDataset(Dataset):
    """Causal-LM examples; train loss is applied to solution tokens, not prompt."""

    def __init__(self, questions, answers, tokenizer: Tokenizer, block_size: int = 512,
                 max_question_tokens: int = 160):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_question_tokens = max_question_tokens
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.bos_id = tokenizer.token_to_id("[BOS]")
        self.eos_id = tokenizer.token_to_id("[EOS]")
        self.question_id = tokenizer.token_to_id("[QUESTION]")
        self.answer_id = tokenizer.token_to_id("[ANSWER]")
        self.rows = list(zip(questions, answers))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        question, answer = self.rows[index]
        q_ids = self.tokenizer.encode(question).ids[:self.max_question_tokens]
        # BOS, QUESTION, ANSWER and EOS must fit in every example.
        answer_budget = max(1, self.block_size - len(q_ids) - 4)
        a_ids = self.tokenizer.encode(answer).ids[:answer_budget]
        ids = [self.bos_id, self.question_id, *q_ids, self.answer_id, *a_ids, self.eos_id]
        answer_start = 3 + len(q_ids)  # first answer token; [ANSWER] itself is prompt
        labels = [-100] * answer_start + ids[answer_start:]
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}


def collate_math(batch, pad_id: int):
    max_len = max(item["input_ids"].numel() for item in batch)
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for row, item in enumerate(batch):
        n = item["input_ids"].numel()
        ids[row, :n] = item["input_ids"]
        labels[row, :n] = item["labels"]
        mask[row, :n] = True
    return ids, labels, mask


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed.to(x.dtype) * self.weight)


def apply_rope(x, position_ids, theta: float = 1_000_000.0):
    """RoPE with half-rotation layout; x is [batch, heads, time, head_dim]."""
    head_dim = x.size(-1)
    if head_dim % 2:
        raise ValueError("RoPE requires an even attention head dimension")
    # Qwen2/Qwen2.5 default RoPE base (rope_theta) is 1,000,000.
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))
    freqs = torch.einsum("t,d->td", position_ids.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)[None, None, :, :]
    cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return x * cos + rotated * sin


class GQASelfAttention(nn.Module):
    """Qwen2-style grouped-query self-attention using SDPA/Flash when supported."""
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int,
                 rope_theta: float = 1_000_000.0):
        super().__init__()
        if d_model % num_heads or num_heads % num_kv_heads:
            raise ValueError("num_heads must divide d_model and num_kv_heads must divide num_heads")
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = d_model // num_heads
        self.rope_theta = rope_theta
        # Qwen2 uses bias on Q/K/V projections, but not on the attention output projection.
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attention_mask):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        positions = torch.arange(seq_len, device=x.device)
        q = apply_rope(q, positions, self.rope_theta)
        k = apply_rope(k, positions, self.rope_theta)
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        allowed = attention_mask[:, None, None, :] & causal[None, None, :, :]
        # SDPA automatically selects the Flash kernel when CUDA, dtype, and mask support it.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0, is_causal=False)
        return self.o_proj(y.transpose(1, 2).contiguous().view(batch, seq_len, -1))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, multiple_of: int = 64):
        super().__init__()
        hidden = int(8 * d_model / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm Qwen2 decoder block: RMSNorm, GQA, residual, RMSNorm, SwiGLU."""
    def __init__(self, d_model, num_heads, num_kv_heads, rope_theta, rms_norm_eps):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.attn = GQASelfAttention(d_model, num_heads, num_kv_heads, rope_theta)
        self.ffn_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.mlp = SwiGLU(d_model)

    def forward(self, x, attention_mask):
        x = x + self.attn(self.attn_norm(x), attention_mask)
        return x + self.mlp(self.ffn_norm(x))


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    num_heads: int = 8
    num_kv_heads: int = 2
    num_layers: int = 6
    max_seq_len: int = 512
    exit_layers: tuple[int, ...] = (2, 4)
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6


class AdaptiveMathTransformer(nn.Module):
    """Decoder-only math LM with optional confidence-based layer exits."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.d_model // config.num_heads % 2:
            raise ValueError("d_model/num_heads must be even for RoPE")
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.num_kv_heads,
                             config.rope_theta, config.rms_norm_eps)
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Qwen2/Qwen2.5 use an independent output projection by default.
        # Small, transformer-appropriate initialization is important because
        # the input embeddings are also the output vocabulary classifier.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        self.exit_layers = tuple(sorted(set(config.exit_layers) - {config.num_layers}))
        if any(layer < 1 or layer >= config.num_layers for layer in self.exit_layers):
            raise ValueError("exit_layers must be in [1, num_layers-1]")

    def _logits(self, hidden):
        return self.lm_head(self.norm(hidden))

    def forward(self, input_ids, attention_mask, return_exit_logits=False):
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(f"Input exceeds max_seq_len={self.config.max_seq_len}")
        hidden = self.embed_tokens(input_ids)
        exits = {}
        for depth, layer in enumerate(self.layers, start=1):
            hidden = layer(hidden, attention_mask)
            if return_exit_logits and depth in self.exit_layers:
                exits[depth] = self._logits(hidden)
        return self._logits(hidden), exits

    @torch.inference_mode()
    def adaptive_forward(self, input_ids, attention_mask, confidence_threshold: float = 0.90,
                         min_depth: int = 2):
        """Run only active batch rows at deeper layers; returns last-token logits and depths."""
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 1 <= min_depth <= self.config.num_layers:
            raise ValueError("min_depth must be between 1 and num_layers")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(f"Input exceeds max_seq_len={self.config.max_seq_len}")
        batch = input_ids.size(0)
        last_pos = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        active_ids = torch.arange(batch, device=input_ids.device)
        active_hidden = self.embed_tokens(input_ids)
        active_mask = attention_mask
        depths = torch.full((batch,), self.config.num_layers, device=input_ids.device, dtype=torch.long)
        chosen = [None] * batch
        for depth, layer in enumerate(self.layers, start=1):
            active_hidden = layer(active_hidden, active_mask)
            # Only exit at layers that received auxiliary answer-token loss.
            if depth in self.exit_layers and depth >= min_depth:
                logits = self._logits(active_hidden)
                local_last = last_pos.index_select(0, active_ids)
                next_logits = logits[torch.arange(active_ids.numel(), device=input_ids.device), local_last]
                confident = next_logits.float().softmax(-1).amax(-1) >= confidence_threshold
                for local in confident.nonzero(as_tuple=False).flatten().tolist():
                    original = int(active_ids[local])
                    chosen[original] = next_logits[local]
                    depths[original] = depth
                keep = ~confident
                if not keep.any():
                    break
                active_ids = active_ids[keep]
                active_hidden = active_hidden[keep]
                active_mask = active_mask[keep]
        if active_ids.numel():
            logits = self._logits(active_hidden)
            local_last = last_pos.index_select(0, active_ids)
            next_logits = logits[torch.arange(active_ids.numel(), device=input_ids.device), local_last]
            for local, original in enumerate(active_ids.tolist()):
                chosen[original] = next_logits[local]
        return torch.stack(chosen), depths


def batch_loss(model, input_ids, labels, attention_mask, aux_weight: float = 0.25):
    logits, exits = model(input_ids, attention_mask, return_exit_logits=True)
    # Next-token prediction. Prompt labels are -100; padding is also -100.
    main = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                           labels[:, 1:].reshape(-1), ignore_index=-100)
    if exits:
        aux = [F.cross_entropy(value[:, :-1].float().reshape(-1, value.size(-1)),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
               for value in exits.values()]
        main = main + aux_weight * torch.stack(aux).mean()
    return main, logits


def evaluate(model, loader, device):
    model.eval()
    total_loss, tokens = 0.0, 0
    with torch.inference_mode():
        for input_ids, labels, mask in loader:
            input_ids, labels, mask = input_ids.to(device), labels.to(device), mask.to(device)
            logits, _ = model(input_ids, mask)
            shifted_labels = labels[:, 1:]
            loss_sum = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                       shifted_labels.reshape(-1), ignore_index=-100, reduction="sum")
            count = (shifted_labels != -100).sum().item()
            total_loss += loss_sum.item()
            tokens += count
    return total_loss / max(tokens, 1)


def train_model(model, train_loader, valid_loader, device, epochs=20, learning_rate=3e-4,
                weight_decay=0.1, warmup_steps=200, patience=4, checkpoint="scratch_math_best.pt"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  betas=(0.9, 0.95), weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    def lr_scale(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    best, stale = float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        running, token_count = 0.0, 0
        for input_ids, labels, mask in train_loader:
            input_ids, labels, mask = input_ids.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, _ = batch_loss(model, input_ids, labels, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            valid_count = (labels[:, 1:] != -100).sum().item()
            running += loss.item() * valid_count
            token_count += valid_count
        train_loss = running / max(token_count, 1)
        val_loss = evaluate(model, valid_loader, device)
        print(f"epoch {epoch:03d} | train/token {train_loss:.4f} | val/token {val_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save({"model": model.state_dict(), "config": model.config.__dict__,
                        "best_val_loss": best, "epoch": epoch}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping: validation loss did not improve for {patience} epochs.")
                break
    return best


def make_loaders(train_questions, train_answers, valid_questions, valid_answers,
                 tokenizer, block_size=512, batch_size=8, num_workers=0):
    train_ds = MathSolutionDataset(train_questions, train_answers, tokenizer, block_size)
    valid_ds = MathSolutionDataset(valid_questions, valid_answers, tokenizer, block_size)
    pad_id = tokenizer.token_to_id("[PAD]")
    collate = lambda batch: collate_math(batch, pad_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate,
                              pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate,
                              pin_memory=torch.cuda.is_available())
    return train_loader, valid_loader


def generate_solution(model, tokenizer, question: str, device, max_new_tokens=256,
                       confidence_threshold=0.95, use_adaptive=False, adaptive_min_depth=2,
                       repetition_penalty=1.15, no_repeat_ngram_size=3,
                       math_only=False, return_depths=False):
    """Greedy generation with repetition controls; full depth is safer by default.

    Adaptive inference remains available with use_adaptive=True, but early exits
    are a speed/quality trade-off and should be calibrated on validation examples.
    """
    bos = tokenizer.token_to_id("[BOS]")
    q_marker = tokenizer.token_to_id("[QUESTION]")
    a_marker = tokenizer.token_to_id("[ANSWER]")
    eos = tokenizer.token_to_id("[EOS]")
    prompt = [bos, q_marker, *tokenizer.encode(question).ids, a_marker]
    prompt = prompt[-model.config.max_seq_len:]
    generated = []
    depth_history = []
    allowed = math_token_mask(tokenizer, device) if math_only else None
    model.eval()
    for _ in range(max_new_tokens):
        context = (prompt + generated)[-model.config.max_seq_len:]
        ids = torch.tensor([context], dtype=torch.long, device=device)
        mask = torch.ones_like(ids, dtype=torch.bool)
        if use_adaptive:
            logits, depths = model.adaptive_forward(
                ids, mask, confidence_threshold,
                min_depth=adaptive_min_depth,
            )
            depth_history.append(int(depths[0]))
        else:
            logits, _ = model(ids, mask)
            logits = logits[:, -1, :]
            depth_history.append(model.config.num_layers)
        scores = logits[0].float().clone()
        if allowed is not None:
            scores.masked_fill_(~allowed, -torch.inf)

        # Discourage cycling over already-emitted tokens without changing prompt logits.
        if repetition_penalty > 1.0:
            for token_id in set(generated):
                if scores[token_id] < 0:
                    scores[token_id] *= repetition_penalty
                else:
                    scores[token_id] /= repetition_penalty

        # Block a token if it would repeat an n-gram already present in the context.
        n = no_repeat_ngram_size
        sequence = prompt + generated
        if n > 1 and len(sequence) >= n - 1:
            prefix = sequence[-(n - 1):]
            banned = set()
            for start in range(len(sequence) - n + 1):
                if sequence[start:start + n - 1] == prefix:
                    banned.add(sequence[start + n - 1])
            if banned:
                scores[list(banned)] = -torch.inf

        next_id = int(scores.argmax().item())
        if next_id == eos:
            break
        generated.append(next_id)
    decoded = tokenizer.decode(generated)
    if return_depths:
        return decoded, depth_history
    return decoded


def flash_attention_status(device):
    if device.type != "cuda":
        return "CUDA unavailable: SDPA will use its CPU implementation; Flash Attention requires supported CUDA hardware."
    return (f"CUDA device: {torch.cuda.get_device_name(device)}. Attention calls use PyTorch SDPA, "
            "which dispatches to Flash Attention when the device, dtype, and mask are supported.")


def math_token_mask(tokenizer: Tokenizer, device):
    """Whitelist single math-symbol tokens so random RL rollouts stay parseable."""
    allowed_chars = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/^=().,{}[]|\\ \t")
    allowed = torch.zeros(tokenizer.get_vocab_size(), dtype=torch.bool, device=device)
    for token_id in range(tokenizer.get_vocab_size()):
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
        if len(piece) == 1 and piece in allowed_chars:
            allowed[token_id] = True
    eos_id = tokenizer.token_to_id("[EOS]")
    if eos_id is not None:
        allowed[eos_id] = True
    if not allowed.any():
        raise ValueError("The tokenizer has no usable math character tokens")
    return allowed


def extract_final_answer(text: str) -> str:
    """Extract the last balanced \\boxed{...} answer, or a conservative fallback."""
    text = str(text)
    start = text.rfind(r"\boxed")
    if start >= 0:
        brace = text.find("{", start)
        if brace >= 0:
            depth = 0
            for i in range(brace, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[brace + 1:i].strip()
    # Dataset references often say "The answer is ..." or end in a display math line.
    matches = list(re.finditer(r"(?:final answer|answer is)\s*[:=]?\s*(.+)", text, re.I))
    if matches:
        return matches[-1].group(1).strip().splitlines()[0].strip().strip("$ .")
    lines = [line.strip().strip("$ ") for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _clean_math_answer(value: str) -> str:
    value = value.strip().strip("$ ").lower()
    for token in (r"\left", r"\right", r"\displaystyle", "\\!", r"\,", r"\;", r"\quad"):
        value = value.replace(token, "")
    value = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", value)
    # Handle simple (non-nested) LaTeX fractions and roots.
    for _ in range(4):
        value = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", value)
        value = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", value)
    value = value.replace(r"\cdot", "*").replace(r"\times", "*")
    value = value.replace(r"\pi", "pi").replace("^", "**")
    value = value.replace("{", "(").replace("}", ")")
    value = re.sub(r"\\(?:,|;|!|quad)", "", value)
    return re.sub(r"\s+", "", value).strip(".")


def math_answers_equal(prediction: str, reference: str) -> bool:
    """Exact normalized or conservative symbolic equivalence (SymPy optional)."""
    pred, ref = _clean_math_answer(prediction), _clean_math_answer(reference)
    if not pred or not ref:
        return False
    if pred == ref:
        return True
    # Avoid feeding arbitrary model text to a symbolic parser.
    allowed = re.compile(r"^[a-z0-9_+*/().,=\-]+$")
    if not allowed.fullmatch(pred) or not allowed.fullmatch(ref):
        return False
    try:
        import sympy as sp
        names = set(re.findall(r"[a-z]+", pred + " " + ref))
        local_symbols = {name: sp.Symbol(name) for name in names if name not in {"sqrt", "pi"}}
        local_symbols.update({"sqrt": sp.sqrt, "pi": sp.pi})

        def parse(expr):
            if "=" in expr:
                left, right = expr.split("=", 1)
                return sp.simplify(sp.sympify(left, locals=local_symbols) - sp.sympify(right, locals=local_symbols))
            return sp.sympify(expr, locals=local_symbols)

        p, r = parse(pred), parse(ref)
        return bool(sp.simplify(p - r) == 0)
    except Exception:
        return False


def _safe_numeric_value(value: str):
    """Evaluate a plain arithmetic expression without eval/exec."""
    value = _clean_math_answer(value)
    tree = ast.parse(value, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            number = visit(node.operand)
            return number if isinstance(node.op, ast.UAdd) else -number
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            if abs(right) > 12: raise ValueError("Exponent too large")
            return left ** right
        raise ValueError("Not a plain numeric expression")

    return visit(tree)


def math_reward(completion: str, reference_solution: str) -> float:
    """Dense RL reward, with exact/symbolic correctness as the dominant signal.

    Incorrect answers receive graded lexical/structural similarity so a random
    from-scratch policy is not trapped at an identical zero-reward plateau.
    Similarity is only shaping: it can never outweigh a correct answer.
    """
    predicted = extract_final_answer(completion)
    expected = extract_final_answer(reference_solution)
    if math_answers_equal(predicted, expected):
        return 1.0
    if not predicted or not expected:
        return 0.0

    p_norm, r_norm = _clean_math_answer(predicted), _clean_math_answer(expected)
    char_similarity = SequenceMatcher(None, p_norm, r_norm, autojunk=False).ratio()
    p_terms = re.findall(r"[a-z]+|\d+(?:\.\d+)?|[^\w\s]", p_norm)
    r_terms = re.findall(r"[a-z]+|\d+(?:\.\d+)?|[^\w\s]", r_norm)
    p_counts, r_counts = {}, {}
    for term in p_terms:
        p_counts[term] = p_counts.get(term, 0) + 1
    for term in r_terms:
        r_counts[term] = r_counts.get(term, 0) + 1
    overlap = sum(min(count, r_counts.get(term, 0)) for term, count in p_counts.items())
    precision = overlap / max(sum(p_counts.values()), 1)
    recall = overlap / max(sum(r_counts.values()), 1)
    token_f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    reward = 0.20 * (0.5 * char_similarity + 0.5 * token_f1)

    # Sequence-level lexical shaping against the worked solution gives a
    # from-scratch policy signal before it can reliably emit a boxed answer.
    # This remains a reward (no token-level teacher-forced loss is used).
    completion_terms = re.findall(r"[a-z]+|\d+(?:\.\d+)?|[^\w\s]", _clean_math_answer(completion))
    solution_terms = re.findall(r"[a-z]+|\d+(?:\.\d+)?|[^\w\s]", _clean_math_answer(reference_solution))
    if completion_terms and solution_terms:
        completion_counts = {}
        for term in completion_terms:
            completion_counts[term] = completion_counts.get(term, 0) + 1
        solution_counts = {}
        for term in solution_terms:
            solution_counts[term] = solution_counts.get(term, 0) + 1
        solution_overlap = sum(min(count, solution_counts.get(term, 0))
                              for term, count in completion_counts.items())
        solution_precision = solution_overlap / max(sum(completion_counts.values()), 1)
        solution_recall = solution_overlap / max(sum(solution_counts.values()), 1)
        solution_f1 = (2 * solution_precision * solution_recall /
                       max(solution_precision + solution_recall, 1e-8))
        reward += 0.15 * solution_f1

    # Small formatting incentive; unlike the old constant bonus, this varies by completion.
    if r"\boxed" in completion:
        reward += 0.01
    try:
        p = _safe_numeric_value(predicted)
        r = _safe_numeric_value(expected)
        reward += 0.30 * math.exp(-abs(p - r) / (abs(r) + 1.0))
    except (ValueError, TypeError, OverflowError, ZeroDivisionError, SyntaxError):
        pass
    return min(reward, 0.50)


def _rollout_group(model, tokenizer, question: str, device, group_size: int,
                   max_new_tokens: int, temperature: float, allowed_tokens=None):
    """Sample a group of on-policy completions and retain their behavior log-probs."""
    bos, qmark, amark, eos = (tokenizer.token_to_id(t) for t in
                              ("[BOS]", "[QUESTION]", "[ANSWER]", "[EOS]"))
    q_ids = tokenizer.encode(str(question)).ids
    prompt = [bos, qmark, *q_ids, amark]
    room = model.config.max_seq_len - max_new_tokens
    if room < 4:
        raise ValueError("max_new_tokens leaves too little room for a prompt")
    prompt = prompt[:2] + prompt[2:-1][:room - 3] + [amark]
    prompt_len = len(prompt)
    ids = torch.tensor(prompt, dtype=torch.long, device=device).repeat(group_size, 1)
    masks = torch.ones_like(ids, dtype=torch.bool)
    generated = [[] for _ in range(group_size)]
    behavior_logps = [[] for _ in range(group_size)]
    active = torch.ones(group_size, dtype=torch.bool, device=device)
    pad_id = tokenizer.token_to_id("[PAD]")
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits, _ = model(ids, masks)
            scaled = logits[:, -1].float() / max(temperature, 1e-4)
            if allowed_tokens is not None:
                scaled = scaled.masked_fill(~allowed_tokens, -torch.inf)
            log_probs = F.log_softmax(scaled, dim=-1)
            sampled = torch.distributions.Categorical(logits=scaled).sample()
            sampled_logp = log_probs.gather(1, sampled[:, None]).squeeze(1)
            appended = torch.full((group_size,), pad_id, dtype=torch.long, device=device)
            appended_mask = active.clone()
            for row in active.nonzero(as_tuple=False).flatten().tolist():
                token = int(sampled[row])
                generated[row].append(token)
                behavior_logps[row].append(float(sampled_logp[row]))
                appended[row] = token
                if token == eos:
                    active[row] = False
            ids = torch.cat((ids, appended[:, None]), dim=1)
            masks = torch.cat((masks, appended_mask[:, None]), dim=1)
            if not active.any():
                break
    texts = []
    for completion in generated:
        if eos in completion:
            completion = completion[:completion.index(eos)]
        texts.append(tokenizer.decode(completion))
    return prompt, prompt_len, generated, behavior_logps, texts


def _pack_grpo_sequences(prompt, prompt_len, generated, behavior_logps, pad_id, device):
    inputs, targets, old_logps = [], [], []
    for response, old in zip(generated, behavior_logps):
        if not response:
            response = [pad_id]
            old = [0.0]
        # Position prompt_len-1 predicts response[0]; later positions predict later tokens.
        inputs.append(prompt + response[:-1])
        targets.append([-100] * (prompt_len - 1) + response)
        old_logps.append([0.0] * (prompt_len - 1) + old)
    width = max(map(len, inputs))
    x = torch.full((len(inputs), width), pad_id, dtype=torch.long, device=device)
    y = torch.full((len(inputs), width), -100, dtype=torch.long, device=device)
    old = torch.zeros((len(inputs), width), dtype=torch.float32, device=device)
    mask = torch.zeros((len(inputs), width), dtype=torch.bool, device=device)
    for i, (seq, target, old_row) in enumerate(zip(inputs, targets, old_logps)):
        n = len(seq)
        x[i, :n] = torch.tensor(seq, dtype=torch.long, device=device)
        y[i, :len(target)] = torch.tensor(target, dtype=torch.long, device=device)
        old[i, :len(old_row)] = torch.tensor(old_row, dtype=torch.float32, device=device)
        mask[i, :n] = True
    return x, y, old, mask


def train_grpo(model, tokenizer, questions, answers, device, num_updates=1000,
               group_size=4, max_new_tokens=96, temperature=1.0,
               learning_rate=1e-5, clip_epsilon=0.2, kl_coef=0.0,
               checkpoint="scratch_math_grpo.pt", log_every=25, seed=42,
               constrain_math_tokens=True):
    """Train a randomly initialized policy with GRPO and answer-verifiable rewards.

    This is RL from scratch: there is no supervised loss, pretrained policy, or
    SFT checkpoint. Sparse rewards mean it may learn very slowly or not at all.
    """
    if len(questions) != len(answers) or not questions:
        raise ValueError("questions and answers must be non-empty and equally sized")
    if group_size < 2:
        raise ValueError("GRPO needs group_size >= 2")
    seed_everything(seed)
    model.to(device)
    # Optional KL reference is an unchanged copy of the random initial policy.
    reference = copy.deepcopy(model).to(device).eval() if kl_coef > 0 else None
    for parameter in (reference.parameters() if reference is not None else []):
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95),
                                  weight_decay=0.01)
    pad_id = tokenizer.token_to_id("[PAD]")
    allowed_tokens = math_token_mask(tokenizer, device) if constrain_math_tokens else None
    reward_history, skipped, exact_matches = [], 0, 0
    last_grad_norm, last_update_norm = 0.0, 0.0
    for update in range(1, num_updates + 1):
        index = random.randrange(len(questions))
        prompt, prompt_len, generated, old_rows, texts = _rollout_group(
            model, tokenizer, questions[index], device, group_size, max_new_tokens, temperature,
            allowed_tokens=allowed_tokens,
        )
        rewards = torch.tensor([math_reward(text, str(answers[index])) for text in texts],
                               dtype=torch.float32, device=device)
        reward_history.extend(rewards.tolist())
        exact_matches += int((rewards >= 0.999).sum().item())
        spread = rewards.std(unbiased=False)
        if spread < 1e-6:
            skipped += 1
            if update % log_every == 0:
                recent = reward_history[-log_every * group_size:]
                print(f"GRPO update {update}/{num_updates} | mean shaped reward {statistics.mean(recent):.4f} "
                      f"| exact answers {exact_matches}/{len(reward_history)} | grad norm {last_grad_norm:.3e} "
                      f"| parameter_change {last_update_norm:.3e} | zero-advantage groups {skipped}")
            continue
        advantages = (rewards - rewards.mean()) / (spread + 1e-6)
        x, targets, old_logps, attention_mask = _pack_grpo_sequences(
            prompt, prompt_len, generated, old_rows, pad_id, device
        )
        model.train()
        logits, _ = model(x, attention_mask)
        if allowed_tokens is not None:
            logits = logits.masked_fill(~allowed_tokens[None, None, :], -torch.inf)
        new_logps_all = F.log_softmax(logits.float() / max(temperature, 1e-4), dim=-1)
        gather_targets = targets.clamp_min(0)
        new_logps = new_logps_all.gather(-1, gather_targets.unsqueeze(-1)).squeeze(-1)
        token_mask = targets.ne(-100)
        ratio = torch.exp((new_logps - old_logps).clamp(-20, 20))
        adv = advantages[:, None]
        unclipped = ratio * adv
        clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
        per_token = -torch.minimum(unclipped, clipped)
        per_sequence = (per_token * token_mask).sum(1) / token_mask.sum(1).clamp_min(1)
        loss = per_sequence.mean()
        if reference is not None and kl_coef:
            with torch.no_grad():
                ref_logits, _ = reference(x, attention_mask)
                if allowed_tokens is not None:
                    ref_logits = ref_logits.masked_fill(~allowed_tokens[None, None, :], -torch.inf)
                ref_logps = F.log_softmax(ref_logits.float() / max(temperature, 1e-4), dim=-1)
                ref_action_logps = ref_logps.gather(-1, gather_targets.unsqueeze(-1)).squeeze(-1)
            sampled_kl = ((new_logps - ref_action_logps) * token_mask).sum(1) / token_mask.sum(1).clamp_min(1)
            loss = loss + kl_coef * sampled_kl.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        tracked_parameter = model.embed_tokens.weight
        parameter_before = tracked_parameter.detach().clone()
        optimizer.step()
        last_update_norm = float((tracked_parameter.detach() - parameter_before).norm())
        model.eval()
        if update % log_every == 0:
            recent = reward_history[-log_every * group_size:]
            print(f"GRPO update {update}/{num_updates} | mean shaped reward {statistics.mean(recent):.4f} "
                  f"| exact answers {exact_matches}/{len(reward_history)} | latest group {rewards.tolist()} "
                  f"| objective {loss.item():.3e} | grad norm {last_grad_norm:.3e} "
                  f"| parameter_change {last_update_norm:.3e} | skipped groups {skipped}\n"
                  f"  sample completion: {texts[0][:160]!r}")
            torch.save({"model": model.state_dict(), "config": model.config.__dict__,
                        "update": update, "mean_reward": statistics.mean(recent)}, checkpoint)
    torch.save({"model": model.state_dict(), "config": model.config.__dict__,
                "update": num_updates,
                "mean_reward": statistics.mean(reward_history) if reward_history else 0.0}, checkpoint)
    return reward_history
