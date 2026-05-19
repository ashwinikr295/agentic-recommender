"""
deep_model.py
=============
Autonomous Dynamic Personalization & Recommendation Engine
----------------------------------------------------------
Step 2: Deep Learning Sequential Recommender (SASRec-style Transformer + GRU fallback)

Architecture
~~~~~~~~~~~~
Two model options, auto-selected by hardware:

  [A] SASRec (CUDA available)
      1. Item Embedding  : item ID -> dense vector (D=128)
      2. Positional Emb  : learned positions up to MAX_SEQ_LEN=50
      3. Transformer Enc : 2 layers, 4 heads, causal self-attention
                           + FFN (d_ff=256) + LayerNorm + Dropout(0.2)
      4. Head            : Linear(D, N_ITEMS) -> logits

  [B] GRUSeqRec (CPU default — fast & practical without GPU)
      1. Item Embedding  : item ID -> dense vector (D=128)
      2. GRU             : 2 layers, hidden=256, bidirectional=False
      3. Head            : Linear(256, N_ITEMS) -> logits

Training Objective
~~~~~~~~~~~~~~~~~~
Multi-class Cross-Entropy loss on the next clicked item.

  CE(y, p) = -log(p_y)   summed over batch

  This is numerically equivalent to the binary CE in the spec
    L = -1/N * sum(yi*log(pi) + (1-yi)*log(1-pi))
  when all N_ITEMS other items are treated as negatives.

Inference
~~~~~~~~~
  get_raw_recommendations(session_item_ids, top_k=50)
    -> list[(item_id: int, score: float)]  sorted descending by score

Run:
    python deep_model.py           # train + save recommender_model.pth
    python deep_model.py --infer   # demo inference from saved weights
"""

import sys
import os
import sqlite3
import argparse
import time
from pathlib import Path
from collections import defaultdict

# Ensure Unicode output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("[ERROR] PyTorch is not installed.")
    print("        Install:  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH    = Path(__file__).parent / "recommender.db"
MODEL_PATH = Path(__file__).parent / "recommender_model.pth"

PAD_ID     = 0
N_ITEMS    = 1000
VOCAB_SIZE = N_ITEMS + 1   # slot 0 = padding

EMBED_DIM  = 128
N_HEADS    = 4
N_LAYERS   = 2
D_FF       = 256
DROPOUT    = 0.2
MAX_SEQ_LEN = 20           # CPU-practical sequence length

# Training
BATCH_SIZE  = 512
LR          = 1e-3
EPOCHS      = 20
MIN_SEQ_LEN = 2
TRAIN_SPLIT = 0.9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_TRANSFORMER = DEVICE.type == "cuda"   # use GRU on CPU, Transformer on GPU


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_user_sequences(db_path: Path) -> dict[int, list[int]]:
    """
    Return {user_id: [item_id, ...]} of chronologically sorted clicked items.
    """
    print(f"[DATA] Loading clickstream from {db_path.name} ...")
    con = sqlite3.connect(db_path)
    rows = con.execute(
        """
        SELECT user_id, item_id
        FROM   clickstream
        WHERE  clicked = 1
        ORDER  BY user_id, timestamp
        """
    ).fetchall()
    con.close()

    seqs: dict[int, list[int]] = defaultdict(list)
    for uid, iid in rows:
        seqs[uid].append(iid)

    seqs = {u: s for u, s in seqs.items() if len(s) >= MIN_SEQ_LEN}
    total = sum(len(s) for s in seqs.values())
    print(f"[DATA] {len(seqs):,} users | {total:,} click events")
    return dict(seqs)


def build_samples(user_seqs: dict[int, list[int]]) -> list[tuple[list[int], int]]:
    """
    Sliding-window: for sequence [i1,i2,...,iL] produce L-1 pairs
    (context, target) where target = next clicked item.
    """
    samples = []
    for seq in user_seqs.values():
        for t in range(1, len(seq)):
            ctx = seq[max(0, t - MAX_SEQ_LEN): t]
            samples.append((ctx, seq[t]))
    return samples


def split_samples(samples, ratio=TRAIN_SPLIT):
    cut = int(len(samples) * ratio)
    return samples[:cut], samples[cut:]


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class ClickstreamDataset(Dataset):
    def __init__(self, samples: list[tuple[list[int], int]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctx, target = self.samples[idx]
        pad_len = MAX_SEQ_LEN - len(ctx)
        padded  = [PAD_ID] * pad_len + ctx
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(target - 1, dtype=torch.long),  # 0-based for CE loss
        )


# ══════════════════════════════════════════════════════════════════════════════
# MODEL A — SASRec (Transformer, best on GPU)
# ══════════════════════════════════════════════════════════════════════════════

class CausalTransformerBlock(nn.Module):
    """Pre-norm causal Transformer block."""

    def __init__(self, d: int, h: int, d_ff: int, drop: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d_ff, d), nn.Dropout(drop),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, pad_mask):
        L = x.size(1)
        # Bool causal mask: True = "block attention to this future position"
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        r = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=causal, key_padding_mask=pad_mask,
                         is_causal=False)
        x = self.drop(x) + r
        r = x
        x = self.ff(self.norm2(x)) + r
        return x


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation.
    Input : (B, L) item IDs     Output: (B, N_ITEMS) logits
    """
    def __init__(self):
        super().__init__()
        self.item_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_ID)
        self.pos_emb  = nn.Embedding(MAX_SEQ_LEN, EMBED_DIM)
        self.drop     = nn.Dropout(DROPOUT)
        self.norm     = nn.LayerNorm(EMBED_DIM)
        self.layers   = nn.ModuleList([
            CausalTransformerBlock(EMBED_DIM, N_HEADS, D_FF, DROPOUT)
            for _ in range(N_LAYERS)
        ])
        self.head = nn.Linear(EMBED_DIM, N_ITEMS)
        self._init()

    def _init(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,  std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, seq):
        B, L    = seq.shape
        pos     = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)
        x       = self.drop(self.norm(self.item_emb(seq) + self.pos_emb(pos)))
        pad_mask = (seq == PAD_ID)
        for layer in self.layers:
            x = layer(x, pad_mask)
        # Representation at last non-pad position
        lengths  = (seq != PAD_ID).sum(1) - 1
        lengths  = lengths.clamp(min=0)
        idx      = lengths.view(-1, 1, 1).expand(-1, 1, x.size(2))
        last     = x.gather(1, idx).squeeze(1)
        return self.head(last)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL B — GRUSeqRec (fast on CPU)
# ══════════════════════════════════════════════════════════════════════════════

class GRUSeqRec(nn.Module):
    """
    GRU-based sequential recommender — CPU-friendly alternative.
    Input : (B, L) item IDs     Output: (B, N_ITEMS) logits

    Architecture:
      ItemEmb(D=128) -> GRU(hidden=256, layers=2, dropout=0.2)
      -> last hidden state -> Linear(256, N_ITEMS) -> logits
    """
    GRU_HIDDEN = 256
    GRU_LAYERS = 2

    def __init__(self):
        super().__init__()
        self.item_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_ID)
        self.gru      = nn.GRU(
            input_size=EMBED_DIM,
            hidden_size=self.GRU_HIDDEN,
            num_layers=self.GRU_LAYERS,
            batch_first=True,
            dropout=DROPOUT if self.GRU_LAYERS > 1 else 0.0,
        )
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(self.GRU_HIDDEN, N_ITEMS)
        self._init()

    def _init(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, seq):
        """
        seq: (B, L)  — left-padded with PAD_ID=0
        Use pack_padded_sequence to skip padding efficiently.
        """
        emb     = self.drop(self.item_emb(seq))       # (B, L, D)
        lengths = (seq != PAD_ID).sum(dim=1).clamp(min=1).cpu()

        packed  = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)                  # hidden: (layers, B, H)
        last_hidden = self.drop(hidden[-1])            # (B, H)
        return self.head(last_hidden)                  # (B, N_ITEMS)


def build_model() -> nn.Module:
    """Select SASRec (GPU) or GRUSeqRec (CPU) automatically."""
    if USE_TRANSFORMER:
        print("[MODEL] Using SASRec (Transformer) — CUDA detected")
        return SASRec().to(DEVICE)
    else:
        print("[MODEL] Using GRUSeqRec — CPU mode (fast & practical)")
        return GRUSeqRec().to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def acc_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    topk = logits.topk(k, dim=1).indices
    return (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[TRAIN] Device     : {DEVICE}")
    print(f"[TRAIN] Model type : {model.__class__.__name__}")
    print(f"[TRAIN] Parameters : {n_params:,}")
    print(f"[TRAIN] Batches    : {len(train_loader):,} train / {len(val_loader):,} val")
    print(f"[TRAIN] Starting {EPOCHS} epochs ...\n")

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        tr_loss, tr_acc = 0.0, 0.0

        for seq, target in train_loader:
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            logits = model(seq)
            loss   = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
            tr_acc  += acc_at_k(logits.detach(), target)

        scheduler.step()
        model.eval()
        vl_loss, vl_acc = 0.0, 0.0
        with torch.no_grad():
            for seq, target in val_loader:
                seq, target = seq.to(DEVICE), target.to(DEVICE)
                logits   = model(seq)
                vl_loss += criterion(logits, target).item()
                vl_acc  += acc_at_k(logits, target)

        tl = tr_loss / len(train_loader)
        ta = tr_acc  / len(train_loader)
        vl = vl_loss / len(val_loader)
        va = vl_acc  / len(val_loader)
        dt = time.time() - t0

        mark = " <-- best" if vl < best_val_loss else ""
        print(
            f"  Epoch {epoch:>2}/{EPOCHS}  "
            f"loss={tl:.4f}  acc@10={ta:.3f}  |  "
            f"val_loss={vl:.4f}  val_acc@10={va:.3f}  "
            f"({dt:.1f}s){mark}"
        )

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        print(f"\n[TRAIN] Restored best weights (val_loss={best_val_loss:.4f})")


def save_model(model: nn.Module, path: Path) -> None:
    ckpt = {
        "model_class": model.__class__.__name__,
        "model_state": model.state_dict(),
        "config": {
            "vocab_size":  VOCAB_SIZE,
            "embed_dim":   EMBED_DIM,
            "n_heads":     N_HEADS,
            "n_layers":    N_LAYERS,
            "d_ff":        D_FF,
            "max_seq_len": MAX_SEQ_LEN,
            "dropout":     DROPOUT,
            "n_items":     N_ITEMS,
        },
    }
    torch.save(ckpt, path)
    mb = path.stat().st_size / (1024 ** 2)
    print(f"[SAVE] Saved -> {path}  ({mb:.2f} MB)")


def load_model(path: Path) -> nn.Module:
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model at {path}. Run:  python deep_model.py"
        )
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=True)
    cls   = ckpt["model_class"]
    model = (SASRec() if cls == "SASRec" else GRUSeqRec()).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[LOAD] Loaded {cls} from {path.name}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE API
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_CACHE: nn.Module | None = None


def _get_model() -> nn.Module:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = load_model(MODEL_PATH)
    return _MODEL_CACHE


def get_raw_recommendations(
    session_item_ids: list[int],
    top_k: int = 50,
) -> list[tuple[int, float]]:
    """
    Predict the next item from a session's click history.

    Parameters
    ----------
    session_item_ids : list[int]
        Chronological item IDs the user clicked (most recent last).
        Item IDs are 1-based integers in [1, 1000].
    top_k : int
        Number of candidate recommendations (max 1000).

    Returns
    -------
    list of (item_id, score) tuples, sorted by score descending.
        item_id : int  — matches the 'items.item_id' column in recommender.db
        score   : float — softmax probability in [0, 1]

    Example
    -------
    >>> recs = get_raw_recommendations([42, 7, 315], top_k=10)
    >>> for item_id, score in recs:
    ...     print(f"item {item_id:>4}  score={score:.4f}")
    """
    if not session_item_ids:
        raise ValueError("session_item_ids cannot be empty.")
    if not all(1 <= i <= N_ITEMS for i in session_item_ids):
        raise ValueError(f"All item IDs must be in [1, {N_ITEMS}].")

    model = _get_model()
    seq   = [int(i) for i in session_item_ids[-MAX_SEQ_LEN:]]
    pad   = [PAD_ID] * (MAX_SEQ_LEN - len(seq)) + seq
    inp   = torch.tensor([pad], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits = model(inp)                             # (1, N_ITEMS)
        probs  = F.softmax(logits, dim=-1).squeeze(0)  # (N_ITEMS,)

    top_k = min(top_k, N_ITEMS)
    vals, idxs = probs.topk(top_k)
    return [(int(i) + 1, float(v)) for i, v in zip(idxs, vals)]


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_training() -> None:
    print("=" * 65)
    print("  Sequential Recommender — Training Pipeline")
    print("=" * 65)

    user_seqs = load_user_sequences(DB_PATH)
    samples   = build_samples(user_seqs)
    print(f"[DATA] Sliding-window samples : {len(samples):,}")

    train_s, val_s = split_samples(samples)
    print(f"[DATA] Train: {len(train_s):,}  |  Val: {len(val_s):,}")

    train_dl = DataLoader(ClickstreamDataset(train_s), batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
    val_dl   = DataLoader(ClickstreamDataset(val_s),   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

    model = build_model()
    train(model, train_dl, val_dl)
    save_model(model, MODEL_PATH)

    # Quick demo
    print("\n[DEMO] Inference smoke-test:")
    for demo in [[42, 7, 315], [100, 200, 300], [1]]:
        recs = get_raw_recommendations(demo, top_k=5)
        print(f"  Input {demo}  ->  top-5: {[iid for iid, _ in recs]}")

    print("\n[DONE] Done. Use get_raw_recommendations() for inference.")
    print("=" * 65)


def run_inference_demo() -> None:
    print("=" * 65)
    print("  Sequential Recommender — Inference Demo")
    print("=" * 65)

    sessions = [
        ([42, 7, 315, 88],    "Mixed sequence"),
        ([100, 200, 300, 400], "Spread IDs"),
        ([501, 502, 503],      "Close IDs"),
        ([999, 1000],          "High-end IDs"),
        ([1],                  "Single-item"),
    ]

    for seq, label in sessions:
        recs = get_raw_recommendations(seq, top_k=50)
        top5 = recs[:5]
        print(f"\n  [{label}]  session: {seq}")
        print(f"  {'Rank':<5} {'ItemID':>7}  {'Score':>9}")
        print(f"  {'─'*5} {'─'*7}  {'─'*9}")
        for rank, (iid, score) in enumerate(top5, 1):
            print(f"  {rank:<5} {iid:>7}  {score:.6f}")

    print("\n[DONE]")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential Recommender")
    parser.add_argument("--infer", action="store_true",
                        help="Run inference demo from saved recommender_model.pth")
    args = parser.parse_args()

    if args.infer:
        run_inference_demo()
    else:
        run_training()
