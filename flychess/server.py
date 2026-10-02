"""Web UI for flychess: play the fly and watch its brain think.

    .venv/bin/python -m flychess.server      # http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse, asyncio, json, threading, time
from pathlib import Path

import chess, numpy as np, torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import connectome
from .brain import FlyBrain, LIFParams
from .encoder import BoardEncoder
from .moves import decode, legal_mask
from .train import Readout, MODEL_DIR

STATIC = Path(__file__).resolve().parent / "static"
CLASS_CODES = {"sensory": 0, "optic": 1, "visual_projection": 2, "central": 3,
               "descending": 4, "ascending": 5, "motor": 6, "visual_centrifugal": 7,
               "sensory_ascending": 8, "endocrine": 9, "unknown": 10}

app = FastAPI(title="flychess")
torch.set_num_threads(8)
C = connectome.load()
ENC = BoardEncoder(C)
BRAIN = FlyBrain(C, LIFParams(dt=0.25), device="cpu", seed=0)
NEU = C.neurons
POS = NEU[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float32)
CLS = NEU["super_class"].map(CLASS_CODES).fillna(10).to_numpy(dtype=np.int8)
DN = set(C.select(super_class="descending").tolist())
INPUT = set(ENC.all_input.tolist())


class Model:
    def __init__(self, variant: str):
        ck = torch.load(MODEL_DIR / f"fly_readout_{variant}.pt", weights_only=False)
        self.keep = torch.as_tensor(ck["keep"])
        self.mu, self.sd = torch.as_tensor(ck["mu"]), torch.as_tensor(ck["sd"])
        self.net = Readout(len(self.keep), ck["hidden"], ck.get("in_dropout", 0.0))
        self.net.load_state_dict(ck["state"]); self.net.eval()
        self.val_top1 = ck.get("val_top1")


MODELS = {v: Model(v) for v in ("downstream", "all") if (MODEL_DIR / f"fly_readout_{v}.pt").exists()}
T_MS, RECORD_FROM = 150.0, 30.0


class MoveRequest(BaseModel):
    fen: str
    variant: str = "downstream"
    avoid_repetition: bool = True
    history: list[str] = []          # previous FENs (for repetition avoidance)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    return {
        "neurons": C.n_neurons, "synapses": C.n_synapses,
        "input_neurons": len(INPUT), "variants": {k: m.val_top1 for k, m in MODELS.items()},
        "t_ms": T_MS, "record_from": RECORD_FROM,
        "encoder": ENC.describe().split("\n"),
    }


@app.get("/api/positions")
def positions():
    """Binary 3D positions of every neuron for the point cloud.

    Body: float32 xyz for N neurons (z scaled x10 so voxel units are isotropic),
    then uint8 class codes for N neurons. Neurons without a position get NaN."""
    xyz = POS.copy(); xyz[:, 2] *= 10.0
    return Response(content=xyz.astype(np.float32).tobytes() + CLS.astype(np.uint8).tobytes(),
                    media_type="application/octet-stream",
                    headers={"X-Neurons": str(C.n_neurons)})


def choose_move(board, model, counts, avoid_repetition, history):
    x = (torch.log1p(counts[model.keep].float()) - model.mu) / model.sd
    mask = legal_mask(board)
    logits = model.net(x[None], mask[None])[0]
    probs = torch.softmax(logits, 0)
    order = logits.argsort(descending=True)[: int(mask.sum())]
    seen = {}
    for f in history:
        k = " ".join(f.split()[:4]); seen[k] = seen.get(k, 0) + 1
    cands, chosen = [], None
    for idx in order.tolist():
        mv = decode(idx, board)
        board.push(mv); key = " ".join(board.fen().split()[:4]); board.pop()
        rep = seen.get(key, 0) >= 1
        cands.append({"uci": mv.uci(), "san": board.san(mv), "p": float(probs[idx]), "repeats": rep})
        if chosen is None and (not avoid_repetition or not rep):
            chosen = mv
    if chosen is None:
        chosen = decode(int(order[0]), board)
    return chosen, cands


def stats_for(board, counts):
    resp = counts.clone(); resp[list(INPUT)] = 0
    classes = {}
    for name, code in CLASS_CODES.items():
        sel = np.flatnonzero(CLS == code)
        if len(sel):
            classes[name] = int(resp[sel].sum())
    return {"driven": int((ENC.drive([board])[0] > 0).sum()),
            "spikes": int(resp.sum()), "neurons": int((resp > 0).sum()),
            "dn_spikes": int(counts[list(DN)].sum()), "dn_neurons": int((counts[list(DN)] > 0).sum()),
            "by_class": classes}


@app.websocket("/ws/move")
async def ws_move(ws: WebSocket):
    """Live mode: run the brain and stream spikes while it computes.

    Client sends {fen, variant, avoid_repetition, history, slowdown}. Server streams
    {"t": brain_ms, "n": [neuron indices]} frames, then {"done": true, move, san,
    candidates, stats}. `slowdown` paces the simulation so 1 ms of brain time takes
    `slowdown` ms of wall time (default 20x)."""
    await ws.accept()
    try:
        req = json.loads(await ws.receive_text())
    except WebSocketDisconnect:
        return
    try:
        board = chess.Board(req["fen"])
    except (KeyError, ValueError):
        await ws.send_text(json.dumps({"error": "bad FEN"})); await ws.close(); return
    model = MODELS.get(req.get("variant", "downstream")) or next(iter(MODELS.values()))
    slow = float(req.get("slowdown", 20.0))
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    dt = BRAIN.p.dt
    frame_steps = max(1, int(round(1.0 / dt)))          # one frame per brain-ms
    pending, t_wall0 = [], time.time()

    def on_step(step, nz):
        if nz.numel():
            pending.extend(nz[:, 1].tolist())
        if (step + 1) % frame_steps == 0:
            t_ms = (step + 1) * dt
            loop.call_soon_threadsafe(queue.put_nowait, {"t": t_ms, "n": pending.copy()})
            pending.clear()
            # pace the simulation to the requested slow-motion factor
            target = t_wall0 + t_ms * slow / 1000.0
            now = time.time()
            if target > now:
                time.sleep(target - now)

    def simulate():
        with torch.no_grad():
            counts = BRAIN.run(ENC.drive([board]), t_ms=T_MS, record_from=RECORD_FROM, on_step=on_step)[0]
            chosen, cands = choose_move(board, model, counts, req.get("avoid_repetition", True), req.get("history", []))
            result = {"done": True, "move": chosen.uci(), "san": board.san(chosen), "candidates": cands[:8],
                      "stats": {**stats_for(board, counts), "sim_seconds": round(time.time() - t_wall0, 2)}}
        loop.call_soon_threadsafe(queue.put_nowait, result)

    threading.Thread(target=simulate, daemon=True).start()
    try:
        while True:
            msg = await queue.get()
            await ws.send_text(json.dumps(msg))
            if msg.get("done"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/api/layout")
def layout(n_background: int = 14000):
    """A frontal 2D silhouette of the brain: every input + readout neuron plus a
    random background sample. Returns [x, y, class] triples in FlyWire voxel units."""
    rng = np.random.default_rng(0)
    keep = set(INPUT)
    for m in MODELS.values():
        keep |= set(m.keep.tolist())
    valid = np.flatnonzero(~np.isnan(POS[:, 0]))
    bg = rng.choice(valid, min(n_background, len(valid)), replace=False)
    idx = np.unique(np.concatenate([np.fromiter(keep, dtype=np.int64), bg]))
    idx = idx[~np.isnan(POS[idx, 0])]
    pts = np.column_stack([POS[idx, 0], POS[idx, 1], CLS[idx]]).round(0).astype(int).tolist()
    return {"points": pts, "bounds": [float(np.nanmin(POS[:, 0])), float(np.nanmin(POS[:, 1])),
                                      float(np.nanmax(POS[:, 0])), float(np.nanmax(POS[:, 1]))]}


@app.post("/api/move")
@torch.no_grad()
def move(req: MoveRequest):
    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(400, "bad FEN")
    if board.is_game_over(claim_draw=True):
        raise HTTPException(400, "game is over")
    model = MODELS.get(req.variant) or next(iter(MODELS.values()))

    t0 = time.time()
    counts, spikes = BRAIN.run(ENC.drive([board]), t_ms=T_MS, record_from=RECORD_FROM, return_spikes=True)
    sim_s = time.time() - t0
    counts = counts[0]
    x = (torch.log1p(counts[model.keep].float()) - model.mu) / model.sd
    mask = legal_mask(board)
    logits = model.net(x[None], mask[None])[0]
    probs = torch.softmax(logits, 0)
    order = logits.argsort(descending=True)[: int(mask.sum())]

    # repetition avoidance (a rule outside the brain): skip moves repeating a seen position
    seen = {}
    for f in req.history:
        k = " ".join(f.split()[:4]); seen[k] = seen.get(k, 0) + 1
    cands, chosen = [], None
    for idx in order.tolist():
        mv = decode(idx, board)
        board.push(mv); key = " ".join(board.fen().split()[:4]); board.pop()
        rep = seen.get(key, 0) >= 1
        cands.append({"uci": mv.uci(), "san": board.san(mv), "p": float(probs[idx]), "repeats": rep})
        if chosen is None and (not req.avoid_repetition or not rep):
            chosen = mv
    if chosen is None:
        chosen = decode(int(order[0]), board)

    # spike stream for the animation: [x, y, t_ms, class, kind]  kind: 0 input, 1 brain, 2 descending
    n_idx = spikes[:, 1].numpy(); t = (spikes[:, 2].numpy() * BRAIN.p.dt)
    ok = ~np.isnan(POS[n_idx, 0])
    kind = np.where(np.isin(n_idx, list(DN)), 2, np.where(np.isin(n_idx, list(INPUT)), 0, 1))
    stream = np.column_stack([POS[n_idx, 0], POS[n_idx, 1], t, CLS[n_idx], kind])[ok]
    stream = stream[np.argsort(stream[:, 2])].round(1).tolist()

    resp = counts.clone(); resp[list(INPUT)] = 0
    classes = {}
    for name, code in CLASS_CODES.items():
        sel = np.flatnonzero(CLS == code)
        if len(sel):
            classes[name] = int(resp[sel].sum())
    return {
        "move": chosen.uci(), "san": board.san(chosen),
        "candidates": cands[:8],
        "stats": {"sim_seconds": round(sim_s, 2), "driven": int((ENC.drive([board])[0] > 0).sum()),
                  "spikes": int(resp.sum()), "neurons": int((resp > 0).sum()),
                  "dn_spikes": int(counts[list(DN)].sum()), "dn_neurons": int((counts[list(DN)] > 0).sum()),
                  "by_class": classes},
        "spikes": stream,
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
