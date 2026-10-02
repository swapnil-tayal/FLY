"""Train the move readout on fly-brain spike counts (and a no-brain baseline).

Readout: a linear layer from standardised log(1+spike count) of responding neurons
to 4096 move logits, trained with cross-entropy against Stockfish's best move,
with illegal moves masked out. The same trainer fits a 768-dim raw board
baseline so we can say honestly how much the brain adds.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import chess, numpy as np, scipy.sparse as sp, torch

from . import connectome
from .encoder import BoardEncoder
from .features import FEAT_DIR, load_positions
from .moves import N_MOVES, board_features, encode, legal_mask

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "models"


def load_features():
    shards = sorted(FEAT_DIR.glob("shard_*.npz"))
    mats, rows = [], []
    for s in shards:
        mats.append(sp.load_npz(s)); rows.append(np.load(s.with_suffix(".rows.npy")))
    return sp.vstack(mats).tocsr(), np.concatenate(rows)


def prepare(recs, rows):
    y = np.array([encode(chess.Move.from_uci(recs[r]["best"]), chess.Board(recs[r]["fen"]).turn)
                  for r in rows])
    masks = torch.stack([legal_mask(chess.Board(recs[r]["fen"])) for r in rows])
    raw = np.stack([board_features(chess.Board(recs[r]["fen"])) for r in rows])
    return y, masks, raw


class Readout(torch.nn.Module):
    def __init__(self, d_in, hidden=0, in_dropout=0.0):
        super().__init__()
        self.in_drop = torch.nn.Dropout(in_dropout)
        if hidden:
            self.net = torch.nn.Sequential(torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
                                           torch.nn.Dropout(0.2), torch.nn.Linear(hidden, N_MOVES))
        else:
            self.net = torch.nn.Linear(d_in, N_MOVES)

    def forward(self, x, mask):
        return self.net(self.in_drop(x)).masked_fill(~mask, -1e9)


def fit(X, y, masks, Xv, yv, mv, hidden=0, epochs=30, lr=1e-3, wd=1e-4, bs=256, label="",
        in_dropout=0.0, quiet=False):
    torch.manual_seed(0)
    model = Readout(X.shape[1], hidden, in_dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X, y, Xv, yv = map(torch.as_tensor, (X, y, Xv, yv))
    best = (0, None)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            loss = torch.nn.functional.cross_entropy(model(X[b], masks[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        acc = evaluate(model, Xv, yv, mv)
        if acc[0] > best[0]:
            best = (acc[0], {k: v.clone() for k, v in model.state_dict().items()})
        if not quiet and (ep % 5 == 4 or ep == epochs - 1):
            print(f"  [{label}] epoch {ep+1:3d} loss={loss.item():.3f} val top1={acc[0]*100:.1f}% top3={acc[1]*100:.1f}%")
    model.load_state_dict(best[1])
    return model


@torch.no_grad()
def evaluate(model, X, y, masks):
    model.eval()
    logits = model(torch.as_tensor(X), masks)
    top3 = logits.topk(3, 1).indices
    y = torch.as_tensor(y)
    return float((top3[:, 0] == y).float().mean()), float((top3 == y[:, None]).any(1).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--min-active", type=int, default=5, help="keep neurons that spiked in >= this many positions")
    ap.add_argument("--wd", type=float, default=1e-4, help="weight decay for the fly readouts")
    ap.add_argument("--in-dropout", type=float, default=0.0, help="input dropout for the fly readouts")
    ap.add_argument("--sweep", action="store_true", help="grid-search wd x dropout on validation, then exit")
    a = ap.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    recs = load_positions()
    M, rows = load_features()
    print(f"positions with features: {M.shape[0]}  (dataset has {len(recs)})")
    y, masks, raw = prepare(recs, rows)

    # neuron selection: neurons that fire in enough positions. Two variants:
    #   downstream = exclude the driven visual projection neurons (they trivially
    #                re-encode the board), so only the brain's response is used
    #   all        = include them
    enc = BoardEncoder(connectome.load())
    active_per_neuron = np.asarray((M > 0).sum(0)).ravel()
    keep_all = np.flatnonzero(active_per_neuron >= a.min_active)
    keep_down = np.setdiff1d(keep_all, enc.all_input)
    print(f"responding neurons: {len(keep_all)} (all) / {len(keep_down)} (downstream only) of {M.shape[1]}")

    def featurize(keep):
        X = np.log1p(M[:, keep].toarray().astype(np.float32))
        mu, sd = X.mean(0), X.std(0) + 1e-6
        return (X - mu) / sd, mu, sd

    n = len(y); n_val = min(max(500, n // 10), n // 5)
    rng = np.random.default_rng(0); perm = rng.permutation(n)
    tr, va = perm[n_val:], perm[:n_val]

    # chance level: pick a random legal move
    chance = float(np.mean([1.0 / int(masks[i].sum()) for i in va]))
    print(f"random legal move baseline: top1={chance*100:.1f}%")

    if a.sweep:
        Xa, _, _ = featurize(keep_all)
        print(f"{'features':10s} {'wd':>7s} {'drop':>5s} {'top1':>6s} {'top3':>6s}")
        for name, Xs in (("board", raw), ("fly-all", Xa)):
            for wd in (1e-4, 1e-2, 1e-1, 3e-1):
                for drop in (0.0, 0.3, 0.6):
                    m = fit(Xs[tr], y[tr], masks[tr], Xs[va], y[va], masks[va], epochs=a.epochs,
                            wd=wd, in_dropout=drop, quiet=True)
                    acc = evaluate(m, Xs[va], y[va], masks[va])
                    print(f"{name:10s} {wd:7.0e} {drop:5.1f} {acc[0]*100:5.1f}% {acc[1]*100:5.1f}%", flush=True)
        return

    print("Training no-brain baseline (raw board one-hot -> moves)")
    base = fit(raw[tr], y[tr], masks[tr], raw[va], y[va], masks[va], hidden=a.hidden,
               epochs=a.epochs, label="board")
    b_acc = evaluate(base, raw[va], y[va], masks[va])

    fly = {}
    for name, keep in (("downstream", keep_down), ("all", keep_all)):
        print(f"Training fly-brain readout on {name} neurons ({len(keep)} features)")
        X, mu, sd = featurize(keep)
        model = fit(X[tr], y[tr], masks[tr], X[va], y[va], masks[va], hidden=a.hidden,
                    epochs=a.epochs, label=f"fly-{name}", wd=a.wd, in_dropout=a.in_dropout)
        fly[name] = (model, keep, mu, sd, evaluate(model, X[va], y[va], masks[va]))

    print(f"\nRESULTS on {n_val} held-out positions (agreement with Stockfish depth-10 best move)")
    print(f"  random legal move            : top1={chance*100:5.1f}%")
    print(f"  board one-hot (no brain)     : top1={b_acc[0]*100:5.1f}%  top3={b_acc[1]*100:5.1f}%")
    for name, (_, keep, _, _, acc) in fly.items():
        print(f"  fly brain, {name:10s} neurons: top1={acc[0]*100:5.1f}%  top3={acc[1]*100:5.1f}%  ({len(keep)} neurons)")

    for name, (model, keep, mu, sd, acc) in fly.items():
        torch.save({"state": model.state_dict(), "keep": keep, "mu": mu, "sd": sd, "hidden": a.hidden,
                    "in_dropout": a.in_dropout, "val_top1": acc[0], "variant": name},
                   MODEL_DIR / f"fly_readout_{name}.pt")
    torch.save({"state": base.state_dict(), "hidden": a.hidden, "val_top1": b_acc[0]},
               MODEL_DIR / "board_baseline.pt")
    print(f"saved -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
