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
import json
import hashlib
import os
import time
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np
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


def template_key(problem_id) -> str:
    """Big Math ids look like 'NN_001-9a4b873a'; the prefix names the problem template."""
    return str(problem_id).split("-")[0] or "unknown"


def load_big_math_jsonl(path: str | Path = "big_math/datasetv2.jsonl/dataset.jsonl"):
    """Load only the fields needed from the 520K Big Math JSONL corpus."""
    path = Path(path)
    questions, responses, answers, chapters, problem_ids = [], [], [], [], []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            question, response, answer = row.get("prompt"), row.get("response"), row.get("answer")
            if question is None or response is None or answer is None:
                continue
            questions.append(str(question))
            responses.append(str(response))
            answers.append(str(answer))
            chapters.append(str(row.get("chapter", "unknown")))
            problem_ids.append(str(row.get("problem_id", "unknown")))
    if not questions:
        raise ValueError(f"No usable prompt/response/answer rows found in {path}")
    return questions, responses, answers, chapters, problem_ids


def grouped_split(groups, valid_groups=None, valid_fraction=0.02, seed=42,
                  max_group_fraction=0.02, purge=None):
    """Split indices so no group (template/chapter) is present on both sides.

    A random row split leaks: every validation prompt can reuse a template that
    also appears in training. Holding out whole groups measures generalisation
    to unseen problem families instead.

    `purge` takes finer-grained keys (e.g. templates) and evicts any validation
    row whose key already occurs in training, which catches catch-all groups.
    """
    groups = list(groups)
    members: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    if valid_groups is not None:
        missing = [g for g in valid_groups if g not in members]
        if missing:
            raise ValueError(f"Unknown split groups: {missing}; available: {sorted(members)[:20]}...")
        chosen = list(valid_groups)
    else:
        rng = random.Random(seed)
        keys = sorted(members)
        rng.shuffle(keys)
        total = len(groups)
        target, cap = max(1, int(valid_fraction * total)), max(1, int(max_group_fraction * total))
        # Skip groups that are too large to keep validation balanced, unless that
        # would leave nothing at all to hold out.
        pool = ([key for key in keys if len(members[key]) <= cap]
                or [min(keys, key=lambda key: len(members[key]))])
        chosen, held = [], 0
        for key in pool:
            if held >= target:
                break
            chosen.append(key)
            held += len(members[key])
    valid = {index for group in chosen for index in members[group]}
    if purge is not None:
        purge = list(purge)
        seen = {purge[index] for index in range(len(groups)) if index not in valid}
        valid = {index for index in valid if purge[index] not in seen}
    fit = [index for index in range(len(groups)) if index not in valid]
    return fit, sorted(valid), chosen


def load_extra_pairs(path, question_col, answer_col):
    """Load extra (question, solution) pairs from .parquet/.jsonl/.json/.csv, e.g. GSM8K.

    For GSM8K, convert the final '#### N' line to '\\boxed{N}' yourself if you want the
    same answer format as MATH."""
    path = Path(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix in (".jsonl", ".json"):
        frame = pd.read_json(path, lines=path.suffix == ".jsonl")
    else:
        frame = pd.read_csv(path)
    return (frame[question_col].fillna("").astype(str),
            frame[answer_col].fillna("").astype(str))


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


def dataset_text_fingerprint(questions, answers, indices=None) -> str:
    """Stable digest for deciding whether a saved tokenizer/cache matches this split."""
    digest = hashlib.sha256()
    rows = indices if indices is not None else range(len(questions))
    for index in rows:
        for text in (questions[index], answers[index]):
            encoded = str(text).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


class TokenizedMathStore:
    """Batched one-time BPE encoding backed by a compact disk cache.

    This prevents the Python tokenizer from re-encoding 520K examples on every
    epoch. Token ids are int32 on disk and converted to int64 only per batch.
    """
    CACHE_VERSION = 1

    def __init__(self, questions, answers, tokenizer, block_size=512,
                 max_question_tokens=160, cache_dir=".cache/math_tokenized",
                 encode_batch_size=2048):
        if len(questions) != len(answers):
            raise ValueError("questions and answers must have equal length")
        self.count = len(questions)
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.bos_id = tokenizer.token_to_id("[BOS]")
        self.eos_id = tokenizer.token_to_id("[EOS]")
        self.question_id = tokenizer.token_to_id("[QUESTION]")
        self.answer_id = tokenizer.token_to_id("[ANSWER]")
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_question_tokens = max_question_tokens

        data_hash = dataset_text_fingerprint(questions, answers)
        tokenizer_hash = hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest()
        signature = hashlib.sha256(
            f"{self.CACHE_VERSION}:{data_hash}:{tokenizer_hash}:{block_size}:{max_question_tokens}".encode()
        ).hexdigest()[:24]
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        self.ids_path = cache_root / f"{signature}.ids.bin"
        self.offsets_path = cache_root / f"{signature}.offsets.npy"
        self.answer_starts_path = cache_root / f"{signature}.answer_starts.npy"
        self.meta_path = cache_root / f"{signature}.json"

        valid_cache = all(path.is_file() for path in
                          (self.ids_path, self.offsets_path, self.answer_starts_path, self.meta_path))
        if not valid_cache:
            self._build(questions, answers, encode_batch_size)
        self.offsets = np.load(self.offsets_path, mmap_mode="r")
        self.answer_starts = np.load(self.answer_starts_path, mmap_mode="r")
        self.flat_ids = np.memmap(self.ids_path, dtype=np.int32, mode="r")

    def _build(self, questions, answers, batch_size):
        ids_tmp = self.ids_path.with_suffix(".ids.tmp")
        offsets_tmp = self.offsets_path.with_suffix(".offsets.tmp.npy")
        starts_tmp = self.answer_starts_path.with_suffix(".starts.tmp.npy")
        meta_tmp = self.meta_path.with_suffix(".meta.tmp")
        offsets = [0]
        answer_starts = np.empty(self.count, dtype=np.int32)
        started = time.perf_counter()
        print(f"Building token cache for {self.count:,} examples in batches of {batch_size:,}...")
        total_tokens = 0
        with ids_tmp.open("wb") as stream:
            for batch_start in range(0, self.count, batch_size):
                batch_end = min(batch_start + batch_size, self.count)
                question_batch = [str(questions[i]) for i in range(batch_start, batch_end)]
                answer_batch = [str(answers[i]) for i in range(batch_start, batch_end)]
                encoded_questions = self.tokenizer.encode_batch(question_batch)
                encoded_answers = self.tokenizer.encode_batch(answer_batch)
                for local, (q_enc, a_enc) in enumerate(zip(encoded_questions, encoded_answers)):
                    q_ids = q_enc.ids[:self.max_question_tokens]
                    answer_budget = max(1, self.block_size - len(q_ids) - 4)
                    a_ids = a_enc.ids[:answer_budget]
                    sequence = [self.bos_id, self.question_id, *q_ids,
                                self.answer_id, *a_ids, self.eos_id]
                    row = batch_start + local
                    answer_starts[row] = 3 + len(q_ids)
                    np.asarray(sequence, dtype=np.int32).tofile(stream)
                    total_tokens += len(sequence)
                    offsets.append(total_tokens)
                if (batch_start // batch_size + 1) % 25 == 0 or batch_end == self.count:
                    print(f"  tokenized {batch_end:,}/{self.count:,} examples")

        with offsets_tmp.open("wb") as stream:
            np.save(stream, np.asarray(offsets, dtype=np.int64))
        with starts_tmp.open("wb") as stream:
            np.save(stream, answer_starts)
        meta_tmp.write_text(json.dumps({"version": self.CACHE_VERSION,
                                        "examples": self.count,
                                        "tokens": total_tokens}), encoding="utf-8")
        os.replace(ids_tmp, self.ids_path)
        os.replace(offsets_tmp, self.offsets_path)
        os.replace(starts_tmp, self.answer_starts_path)
        os.replace(meta_tmp, self.meta_path)
        print(f"Token cache ready: {total_tokens:,} tokens in {time.perf_counter()-started:.1f}s")

    def get(self, index):
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        ids = torch.from_numpy(np.asarray(self.flat_ids[start:end], dtype=np.int64).copy())
        return ids, int(self.answer_starts[index])


class MathSolutionDataset(Dataset):
    """Causal-LM examples; train loss is applied to solution tokens, not prompt."""

    def __init__(self, questions, answers, tokenizer: Tokenizer, block_size: int = 512,
                 max_question_tokens: int = 160, prompt_loss: bool = False, indices=None):
        # prompt_loss=True also supervises question tokens (more signal per example).
        # Keep it False for validation/test so reported loss is answer-token loss.
        self.prompt_loss = prompt_loss
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_question_tokens = max_question_tokens
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.bos_id = tokenizer.token_to_id("[BOS]")
        self.eos_id = tokenizer.token_to_id("[EOS]")
        self.question_id = tokenizer.token_to_id("[QUESTION]")
        self.answer_id = tokenizer.token_to_id("[ANSWER]")
        self.questions = questions
        self.answers = answers
        self.indices = list(indices) if indices is not None else None

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.questions)

    def __getitem__(self, index):
        source_index = self.indices[index] if self.indices is not None else index
        question, answer = self.questions[source_index], self.answers[source_index]
        q_ids = self.tokenizer.encode(question).ids[:self.max_question_tokens]
        # BOS, QUESTION, ANSWER and EOS must fit in every example.
        answer_budget = max(1, self.block_size - len(q_ids) - 4)
        a_ids = self.tokenizer.encode(answer).ids[:answer_budget]
        ids = [self.bos_id, self.question_id, *q_ids, self.answer_id, *a_ids, self.eos_id]
        answer_start = 3 + len(q_ids)  # first answer token; [ANSWER] itself is prompt
        if self.prompt_loss:
            # Supervise everything after [BOS]; markers are cheap to predict.
            labels = [-100] + ids[1:]
        else:
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


# ---------------------------------------------------------------------------
# Qwen2 architecture, reproduced module-for-module from transformers' Qwen2:
# Qwen2RMSNorm, Qwen2RotaryEmbedding, rotate_half/apply_rotary_pos_emb,
# repeat_kv, Qwen2MLP, Qwen2Attention, Qwen2DecoderLayer and Qwen2Model.
# Parameter names match `Qwen2ForCausalLM`, so a Qwen2 state_dict loads with
# strict=True. The only addition is the set of intermediate exits that make the
# depth adaptive; they reuse Qwen2's final norm and lm_head, so they add no
# parameters and leave the Qwen2 topology untouched.
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Qwen2RMSNorm: float32 variance, no bias, multiplicative weight."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(query_states, key_states, cos, sin):
    """Qwen2's RoPE application; cos/sin are [batch, seq, head_dim]."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_embed = (query_states * cos) + (rotate_half(query_states) * sin)
    key_embed = (key_states * cos) + (rotate_half(key_states) * sin)
    return query_embed, key_embed


def repeat_kv(hidden_states, n_rep: int):
    """(batch, kv_heads, seq, head_dim) -> (batch, kv_heads*n_rep, seq, head_dim)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep,
                                                           slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RotaryEmbedding(nn.Module):
    """Qwen2RotaryEmbedding with default (unscaled) RoPE, rope_theta=1e6."""

    def __init__(self, head_dim: int, rope_theta: float = 1_000_000.0):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _activation(name: str):
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    if name == "gelu_pytorch_tanh":
        return lambda x: F.gelu(x, approximate="tanh")
    raise ValueError(f"Unsupported hidden_act: {name!r}")


class SwiGLU(nn.Module):
    """Qwen2MLP with an explicit intermediate_size and no bias."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = _activation(hidden_act)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GQASelfAttention(nn.Module):
    """Qwen2Attention with SDPA, which dispatches to Flash Attention when supported."""

    def __init__(self, config: "ModelConfig", layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        head_dim = config.head_dim or config.d_model // config.num_heads
        if config.d_model % config.num_heads or config.num_heads % config.num_kv_heads:
            raise ValueError("num_heads must divide d_model and num_kv_heads must divide num_heads")
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.head_dim = head_dim
        self.num_key_value_groups = config.num_heads // config.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        # Qwen2 puts bias on Q/K/V but not on the output projection.
        self.q_proj = nn.Linear(config.d_model, config.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.d_model, bias=False)

    def forward(self, hidden_states, position_embeddings, attention_mask):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        seq_len = query_states.size(-2)
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=query_states.device).tril()
        if attention_mask is None:
            allowed = causal[None, None, :, :]
        else:
            allowed = attention_mask[:, None, None, :] & causal[None, None, :, :]
        # SDPA selects the Flash kernel when CUDA, dtype and the bool mask support it.
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=allowed,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False, scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(*input_shape, -1)
        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    """Qwen2DecoderLayer: pre-norm attention, pre-norm SwiGLU, two residual adds."""

    def __init__(self, config: "ModelConfig", layer_idx: int):
        super().__init__()
        self.hidden_size = config.d_model
        self.self_attn = GQASelfAttention(config, layer_idx)
        self.mlp = SwiGLU(config.d_model, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        # Qwen2 has no dropout at all; the default p=0 keeps this an exact passthrough.
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, position_embeddings, attention_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + self.dropout(hidden_states)


class Qwen2Model(nn.Module):
    """Qwen2Model: embeddings, decoder layers, final norm and the RoPE module."""

    def __init__(self, config: "ModelConfig"):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx) for layer_idx in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim or config.d_model // config.num_heads,
                                          config.rope_theta)

    def forward(self, input_ids, attention_mask=None, position_ids=None, collect_depths=None):
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Input exceeds max_seq_len={self.config.max_seq_len}")
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)

        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        depths = {}
        for depth, decoder_layer in enumerate(self.layers, start=1):
            hidden_states = decoder_layer(hidden_states, position_embeddings, attention_mask)
            if collect_depths and depth in collect_depths:
                depths[depth] = hidden_states
        return self.norm(hidden_states), depths


@dataclass
class ModelConfig:
    """Qwen2 hyperparameters plus the adaptive-depth exit layers.

    Field order starts with the original arguments so older positional calls keep
    working. `intermediate_size=None` uses Qwen2's ~5.4x MLP ratio (Qwen2-0.5B
    uses 4864/896 = 5.43x, Qwen2-7B 18944/3584 = 5.29x).
    """
    vocab_size: int
    d_model: int = 256
    num_heads: int = 8
    num_kv_heads: int = 2
    num_layers: int = 6
    max_seq_len: int = 512
    exit_layers: tuple[int, ...] = (2, 4)
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = False
    intermediate_size: int | None = None
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    head_dim: int | None = None

    def __post_init__(self):
        if not self.intermediate_size:
            self.intermediate_size = 64 * max(1, round(5.4 * self.d_model / 64))
        if not self.head_dim:
            self.head_dim = self.d_model // self.num_heads


class AdaptiveMathTransformer(nn.Module):
    """Qwen2ForCausalLM plus confidence-based exits at selected depths.

    `state_dict()` keys are identical to `Qwen2ForCausalLM` (model.*, lm_head.*),
    so weights are interchangeable; only the forward pass adds early exits.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)  # Qwen2PreTrainedModel._init_weights
        if config.tie_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.exit_layers = tuple(sorted(set(config.exit_layers) - {config.num_layers}))
        if any(layer < 1 or layer >= config.num_layers for layer in self.exit_layers):
            raise ValueError("exit_layers must be in [1, num_layers-1]")

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    @property
    def embed_tokens(self):
        return self.model.embed_tokens

    @property
    def layers(self):
        return self.model.layers

    def _logits(self, hidden):
        return self.lm_head(hidden)

    def forward(self, input_ids, attention_mask=None, return_exit_logits=False, position_ids=None):
        hidden, depth_states = self.model(
            input_ids, attention_mask, position_ids,
            collect_depths=set(self.exit_layers) if return_exit_logits else None,
        )
        exits = {depth: self._logits(value) for depth, value in depth_states.items()}
        return self._logits(hidden), exits

    @torch.inference_mode()
    def adaptive_forward(self, input_ids, attention_mask, confidence_threshold: float = 0.90,
                         min_depth: int = 2):
        """Run only still-unconfident batch rows through the deeper layers."""
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 1 <= min_depth <= self.config.num_layers:
            raise ValueError("min_depth must be between 1 and num_layers")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(f"Input exceeds max_seq_len={self.config.max_seq_len}")
        batch, seq_len = input_ids.shape
        last_pos = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        position_embeddings = self.model.rotary_emb(self.model.embed_tokens(input_ids), position_ids)

        active_ids = torch.arange(batch, device=input_ids.device)
        active_hidden = self.model.embed_tokens(input_ids)
        active_mask = attention_mask
        active_positions = position_embeddings
        depths = torch.full((batch,), self.config.num_layers, device=input_ids.device, dtype=torch.long)
        chosen = [None] * batch
        for depth, layer in enumerate(self.model.layers, start=1):
            active_hidden = layer(active_hidden, active_positions, active_mask)
            # Only exit at layers that received auxiliary answer-token loss.
            if depth in self.exit_layers and depth >= min_depth:
                logits = self._logits(self.model.norm(active_hidden))
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
                active_positions = (active_positions[0][keep], active_positions[1][keep])
        if active_ids.numel():
            logits = self._logits(self.model.norm(active_hidden))
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
                weight_decay=0.1, warmup_steps=200, patience=4, checkpoint="scratch_math_best.pt",
                ema_decay=0.998, probe_loader=None):
    """Train with AdamW + cosine LR. Validation and checkpoints use an EMA of the weights,
    which usually gives a lower validation loss than the raw last-step weights.

    `probe_loader` should be an answer-only loader over a fixed slice of the training
    data. It is scored the same way as validation (EMA weights, dropout off), so the
    two losses are directly comparable; the in-epoch running loss is not.
    """
    model.to(device)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()
           if v.dtype.is_floating_point}
    ema_step = 0

    def ema_update():
        nonlocal ema_step
        ema_step += 1
        decay = min(ema_decay, (1 + ema_step) / (10 + ema_step))
        for k, v in model.state_dict().items():
            if k in ema:
                ema[k].mul_(decay).add_(v.detach().float(), alpha=1 - decay)

    def evaluate_ema(loaders):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: ema[k].to(backup[k].dtype) for k in ema}, strict=False)
        losses = [evaluate(model, loader, device) if loader is not None else None
                  for loader in loaders]
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(backup)
        return losses, state
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
            ema_update()
            valid_count = (labels[:, 1:] != -100).sum().item()
            running += loss.item() * valid_count
            token_count += valid_count
        train_loss = running / max(token_count, 1)
        (val_loss, probe_loss), ema_state = evaluate_ema((valid_loader, probe_loader))
        probe_text = " | n/a" if probe_loss is None else f" | train-probe/answer {probe_loss:.4f}"
        print(f"epoch {epoch:03d} | train/token {train_loss:.4f}{probe_text}"
              f" | val/unseen-answer {val_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save({"model": ema_state, "config": model.config.__dict__,
                        "best_val_loss": best, "epoch": epoch}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping: validation loss did not improve for {patience} epochs.")
                break
    return best


def make_loaders(train_questions, train_answers, valid_questions, valid_answers,
                 tokenizer, block_size=512, batch_size=8, num_workers=0,
                 train_indices=None, valid_indices=None, train_weights=None):
    train_ds = MathSolutionDataset(train_questions, train_answers, tokenizer, block_size,
                                   prompt_loss=True, indices=train_indices)
    valid_ds = MathSolutionDataset(valid_questions, valid_answers, tokenizer, block_size,
                                   prompt_loss=False, indices=valid_indices)
    pad_id = tokenizer.token_to_id("[PAD]")
    collate = lambda batch: collate_math(batch, pad_id)
    if train_weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(train_weights, dtype=torch.double),
            num_samples=len(train_ds), replacement=True,
        )
        shuffle = False
    else:
        sampler, shuffle = None, True
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=shuffle,
                              num_workers=num_workers, collate_fn=collate,
                              pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate,
                              pin_memory=torch.cuda.is_available())
    return train_loader, valid_loader


def make_probe_loader(questions, answers, tokenizer, indices, block_size=512,
                      batch_size=8, num_workers=0, max_examples=2048):
    """Fixed, answer-only slice of the training data, scored exactly like validation."""
    indices = list(indices)
    if max_examples and len(indices) > max_examples:
        step = len(indices) / max_examples
        indices = [indices[int(i * step)] for i in range(max_examples)]
    dataset = MathSolutionDataset(questions, answers, tokenizer, block_size,
                                  prompt_loss=False, indices=indices)
    pad_id = tokenizer.token_to_id("[PAD]")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers,
                      collate_fn=lambda rows: collate_math(rows, pad_id),
                      pin_memory=torch.cuda.is_available())


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
