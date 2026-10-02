"""Run labelled positions through the fly brain and store spike-count features.

Output: data/features/shard_XXXXX.npz with
  counts (scipy CSR, n_pos x N) spike counts of every neuron after `record_from`
  rows: the line numbers in positions.jsonl that each row corresponds to.
Resumable: existing shards are skipped. Run again after the dataset grows.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import chess, numpy as np, scipy.sparse as sp, torch

from . import connectome
from .brain import FlyBrain, LIFParams
from .encoder import BoardEncoder

ROOT = Path(__file__).resolve().parent.parent
POSITIONS = ROOT / "data" / "positions.jsonl"
FEAT_DIR = ROOT / "data" / "features"
SHARD = 512


def load_positions(path=POSITIONS):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--t-ms", type=float, default=150.0)
    ap.add_argument("--record-from", type=float, default=30.0)
    ap.add_argument("--rate", type=float, default=150.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-shards", type=int, default=10**9)
    ap.add_argument("--workers", type=int, default=1, help="total parallel feature processes")
    ap.add_argument("--worker", type=int, default=0, help="this process index (0..workers-1)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    c = connectome.load()
    enc = BoardEncoder(c, rate_hz=a.rate)
    brain = FlyBrain(c, LIFParams(dt=0.25), device=a.device, seed=0)
    recs = load_positions()
    n_shards = len(recs) // SHARD          # only full shards; rerun when more data exists
    done = 0
    for s in range(n_shards):
        out = FEAT_DIR / f"shard_{s:05d}.npz"
        if out.exists() or s % a.workers != a.worker:
            continue
        if done >= a.max_shards:
            break
        t0 = time.time()
        rows = np.arange(s * SHARD, (s + 1) * SHARD)
        blocks = []
        for i in range(0, SHARD, a.batch):
            boards = [chess.Board(recs[r]["fen"]) for r in rows[i:i + a.batch]]
            counts = brain.run(enc.drive(boards), t_ms=a.t_ms, record_from=a.record_from)
            blocks.append(sp.csr_matrix(counts.cpu().numpy().astype(np.int16)))
        m = sp.vstack(blocks).tocsr()
        np.save(out.with_suffix(".rows.npy"), rows)
        sp.save_npz(out, m, compressed=True)   # written last: presence marks the shard complete
        done += 1
        el = time.time() - t0
        print(f"shard {s}: {SHARD} positions in {el:.0f}s ({el/SHARD:.2f}s/pos), "
              f"nnz/pos={m.nnz/SHARD:.0f}", flush=True)
    print(f"features complete for {n_shards} shards ({n_shards*SHARD} positions)")


if __name__ == "__main__":
    main()
