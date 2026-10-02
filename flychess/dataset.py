"""Generate Stockfish-labelled chess positions for training the readout.

Positions come from diverse self-play games where moves are sampled from
Stockfish's top candidates with some randomness (and occasional random moves),
so the readout sees realistic but varied positions. Each record stores the FEN,
Stockfish's best move at a fixed depth, and the engine's candidate moves with
centipawn scores so that training can use a soft target.
"""
from __future__ import annotations

import argparse, json, random, shutil, time
from pathlib import Path

import chess, chess.engine

STOCKFISH = shutil.which("stockfish") or "/opt/homebrew/bin/stockfish"
OUT = Path(__file__).resolve().parent.parent / "data" / "positions.jsonl"


def score_cp(info, pov: chess.Color) -> int:
    s = info["score"].pov(pov)
    return s.score(mate_score=10000)


def gen_game(engine, depth: int, multipv: int, temp: float, p_random: float, rng: random.Random,
             max_plies: int = 160):
    board = chess.Board()
    records = []
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        cands = [(i["pv"][0], score_cp(i, board.turn)) for i in infos if "pv" in i and i["pv"]]
        if not cands:
            break
        best, best_cp = cands[0]
        records.append({
            "fen": board.fen(),
            "best": best.uci(),
            "cands": [[m.uci(), cp] for m, cp in cands],
        })
        if rng.random() < p_random:
            move = rng.choice(list(board.legal_moves))
        else:
            # softmax over candidate scores (in pawns) with temperature
            ws = [pow(2.718, (cp - best_cp) / 100.0 / max(temp, 1e-3)) for _, cp in cands]
            move = rng.choices([m for m, _ in cands], weights=ws, k=1)[0]
        board.push(move)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-positions", type=int, default=20000)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--multipv", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--p-random", type=float, default=0.08)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    engine.configure({"Threads": a.threads, "Hash": 256})
    n, games, t0 = 0, 0, time.time()
    with open(a.out, "a") as f:
        while n < a.n_positions:
            recs = gen_game(engine, a.depth, a.multipv, a.temp, a.p_random, rng)
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n += len(recs); games += 1
            if games % 10 == 0:
                print(f"games={games} positions={n} elapsed={time.time()-t0:.0f}s", flush=True)
    engine.quit()
    print(f"done: {n} positions from {games} games -> {a.out}")


if __name__ == "__main__":
    main()
