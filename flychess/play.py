"""Let the fly play chess.

Modes:
  python -m flychess.play random  --games 20     fly vs random mover
  python -m flychess.play stockfish --games 10   fly vs Stockfish (Skill Level 0, depth 1)
  python -m flychess.play human                   you (white or black) vs the fly in the terminal

Each fly move: encode the position onto its visual projection neurons, run the
whole-brain LIF simulation, read descending/central spike counts through the
trained linear readout, and play the highest-scoring legal move.
"""
from __future__ import annotations

import argparse, random, shutil, sys, time
from pathlib import Path

import chess, chess.engine, numpy as np, torch

from . import connectome
from .brain import FlyBrain, LIFParams
from .encoder import BoardEncoder
from .moves import decode, legal_mask
from .train import Readout, MODEL_DIR

STOCKFISH = shutil.which("stockfish") or "/opt/homebrew/bin/stockfish"


class FlyPlayer:
    def __init__(self, model_path=MODEL_DIR / "fly_readout_downstream.pt", device="cpu", t_ms=150.0,
                 record_from=30.0, rate=150.0, temperature=0.0, seed=0):
        ck = torch.load(model_path, weights_only=False)
        self.c = connectome.load()
        self.enc = BoardEncoder(self.c, rate_hz=rate)
        self.brain = FlyBrain(self.c, LIFParams(dt=0.25), device=device, seed=seed)
        self.keep = torch.as_tensor(ck["keep"])
        self.mu, self.sd = torch.as_tensor(ck["mu"]), torch.as_tensor(ck["sd"])
        self.model = Readout(len(self.keep), ck["hidden"], ck.get("in_dropout", 0.0)); self.model.load_state_dict(ck["state"]); self.model.eval()
        self.t_ms, self.record_from, self.temperature = t_ms, record_from, temperature
        self.last = {}

    @torch.no_grad()
    def think(self, board: chess.Board, avoid_repetition: bool = False):
        t0 = time.time()
        counts = self.brain.run(self.enc.drive([board]), t_ms=self.t_ms, record_from=self.record_from)[0]
        x = (torch.log1p(counts[self.keep].float()) - self.mu) / self.sd
        mask = legal_mask(board)
        logits = self.model(x[None], mask[None])[0]
        if self.temperature > 0:
            idx = torch.multinomial(torch.softmax(logits / self.temperature, 0), 1).item()
            move = decode(idx, board)
        else:
            order = logits.argsort(descending=True)
            move = None
            for idx in order[: int(mask.sum())].tolist():
                cand = decode(idx, board)
                if not avoid_repetition:
                    move = cand; break
                # skip moves that repeat a position already seen twice (rule outside the brain)
                board.push(cand); rep = board.is_repetition(2); board.pop()
                if not rep:
                    move = cand; break
            if move is None:
                move = decode(int(order[0]), board)
        self.last = dict(spikes=int(counts.sum()), neurons=int((counts > 0).sum()),
                         dn_spikes=int(counts[self.c.select(super_class="descending")].sum()),
                         secs=time.time() - t0)
        return move


def play_game(fly, opponent_move, fly_color, verbose=False, max_plies=300, avoid_repetition=False):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        mv = fly.think(board, avoid_repetition) if board.turn == fly_color else opponent_move(board)
        if verbose and board.turn == fly_color:
            print(f"  fly plays {board.san(mv):7s} ({fly.last['neurons']} neurons, {fly.last['spikes']} spikes, "
                  f"{fly.last['dn_spikes']} descending spikes, {fly.last['secs']:.1f}s)")
        board.push(mv)
    res = board.result(claim_draw=True)
    if res == "*":
        res = "1/2-1/2"
    return res, board


def score_for(res, color):
    if res == "1/2-1/2":
        return 0.5
    return 1.0 if (res == "1-0") == (color == chess.WHITE) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["random", "stockfish", "human"])
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--skill", type=int, default=0)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--color", default="white", help="human mode: your colour")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--avoid-repetition", action="store_true",
                    help="skip moves that repeat a position twice (a rule outside the brain)")
    ap.add_argument("--variant", default="downstream", choices=["downstream", "all"],
                    help="readout trained on downstream-only neurons or all responding neurons")
    a = ap.parse_args()
    torch.set_num_threads(8)
    fly = FlyPlayer(model_path=MODEL_DIR / f"fly_readout_{a.variant}.pt", temperature=a.temperature)
    rng = random.Random(0)

    if a.mode == "human":
        human = chess.WHITE if a.color == "white" else chess.BLACK
        board = chess.Board()
        print(board, "\n")
        while not board.is_game_over(claim_draw=True):
            if board.turn == human:
                s = input("your move (SAN or UCI, q to quit): ").strip()
                if s == "q":
                    return
                try:
                    mv = board.parse_san(s)
                except ValueError:
                    try:
                        mv = chess.Move.from_uci(s)
                        assert mv in board.legal_moves
                    except Exception:
                        print("illegal move"); continue
            else:
                mv = fly.think(board, a.avoid_repetition)
                print(f"fly plays {board.san(mv)}  [{fly.last['neurons']} neurons fired, "
                      f"{fly.last['dn_spikes']} descending spikes, {fly.last['secs']:.1f}s]")
            board.push(mv)
            print(board, "\n")
        print("result:", board.result(claim_draw=True))
        return

    if a.mode == "random":
        opp = lambda b: rng.choice(list(b.legal_moves))
        label = "random mover"
    else:
        eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
        eng.configure({"Skill Level": a.skill, "UCI_LimitStrength": True, "UCI_Elo": 1320})
        opp = lambda b: eng.play(b, chess.engine.Limit(depth=a.depth)).move
        label = f"Stockfish skill={a.skill} depth={a.depth} (Elo cap 1320)"

    pts, results = 0.0, []
    for g in range(a.games):
        color = chess.WHITE if g % 2 == 0 else chess.BLACK
        res, board = play_game(fly, opp, color, verbose=a.verbose, avoid_repetition=a.avoid_repetition)
        s = score_for(res, color); pts += s; results.append(res)
        print(f"game {g+1:2d}: fly as {'white' if color else 'black'} -> {res}  "
              f"({board.ply()} plies, {'checkmate' if board.is_checkmate() else board.outcome(claim_draw=True).termination.name.lower() if board.outcome(claim_draw=True) else 'cutoff'})", flush=True)
    print(f"\nfly vs {label}: {pts}/{a.games} points  "
          f"(W {sum(score_for(r, chess.WHITE if i%2==0 else chess.BLACK)==1 for i,r in enumerate(results))} "
          f"D {results.count('1/2-1/2')} "
          f"L {sum(score_for(r, chess.WHITE if i%2==0 else chess.BLACK)==0 for i,r in enumerate(results))})")
    if a.mode == "stockfish":
        eng.quit()


if __name__ == "__main__":
    main()
