"""Move encoding shared by training and play.

Moves are encoded in the canonical (side-to-move = white) orientation as
from_square * 64 + to_square (4096 classes). Promotions default to queen.
"""
import chess
import numpy as np
import torch

N_MOVES = 64 * 64


def canonical_move(move: chess.Move, turn: chess.Color) -> chess.Move:
    """Mirror a move if black is to move so it matches the mirrored board."""
    if turn == chess.WHITE:
        return move
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square),
                      promotion=move.promotion)


def encode(move: chess.Move, turn: chess.Color) -> int:
    m = canonical_move(move, turn)
    return m.from_square * 64 + m.to_square


def decode(idx: int, board: chess.Board) -> chess.Move:
    """Turn a class index back into a legal move on the real (unmirrored) board."""
    frm, to = divmod(int(idx), 64)
    if board.turn == chess.BLACK:
        frm, to = chess.square_mirror(frm), chess.square_mirror(to)
    for promo in (None, chess.QUEEN):
        mv = chess.Move(frm, to, promotion=promo)
        if mv in board.legal_moves:
            return mv
    raise ValueError("decoded move is not legal")


def legal_mask(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(N_MOVES, dtype=torch.bool)
    for mv in board.legal_moves:
        mask[encode(mv, board.turn)] = True
    return mask


def board_features(board: chess.Board) -> np.ndarray:
    """768-dim one-hot board features in canonical orientation (the 'no brain' baseline)."""
    b = board if board.turn == chess.WHITE else board.mirror()
    x = np.zeros(768, dtype=np.float32)
    for sq, p in b.piece_map().items():
        x[(p.piece_type - 1 + (0 if p.color == chess.WHITE else 6)) * 64 + sq] = 1.0
    return x
