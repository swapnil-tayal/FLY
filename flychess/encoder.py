"""Project a chess position onto the fly's visual projection neurons (VPNs).

Biology: lobula columnar (LC) and related VPN types each tile the visual field
retinotopically and carry one visual feature (looming, small objects, bars...) into
a dedicated optic glomerulus in the central brain. We use them as the fly's
"view" of the board:

  * piece type  -> a set of VPN cell types (pawn, knight, bishop, rook, queen, king
                   each get their own feature channels)
  * piece colour -> eye: the side to move always sees its own pieces with the
                   right eye and the opponent's with the left eye. The board is
                   mirrored when black is to move, so the fly always "plays white".
  * square      -> a retinotopic bin: neurons of each type are sorted by position
                   and split into 64 spatial groups, one per square.

Every piece on the board therefore drives a small, consistent group of VPNs at a
fixed Poisson rate. Castling rights and en-passant are ignored for now.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np
import torch

from .connectome import Connectome

# Piece type -> VPN cell types. Chosen so each channel has a few hundred neurons
# per side and is known to project into the central brain.
PIECE_CHANNELS = {
    chess.PAWN:   ["LC12", "LC9"],
    chess.KNIGHT: ["LC17", "LC13"],
    chess.BISHOP: ["LC10a", "LC10d"],
    chess.ROOK:   ["LPLC2", "LPLC1"],
    chess.QUEEN:  ["LC11", "LC21", "LC15"],
    chess.KING:   ["LC4", "LC6", "LC16"],
}
OWN_SIDE, OPP_SIDE = "right", "left"


def _bin_by_position(df, n_bins: int = 64) -> np.ndarray:
    """Assign each neuron a bin 0..63 arranged as an 8x8 grid over its 2D position.

    Positions are projected onto the two principal axes of the population and
    quantised into 8 rank-ordered rows and 8 columns within each row.
    """
    xyz = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)
    xyz = np.nan_to_num(xyz, nan=np.nanmean(xyz))
    xyz = xyz - xyz.mean(0)
    _, _, vt = np.linalg.svd(xyz, full_matrices=False)
    uv = xyz @ vt[:2].T
    n = len(df)
    row = np.empty(n, dtype=int)
    order = np.argsort(uv[:, 0], kind="stable")
    row[order] = np.minimum((np.arange(n) * 8) // n, 7)
    col = np.empty(n, dtype=int)
    for r in range(8):
        sel = np.flatnonzero(row == r)
        o = sel[np.argsort(uv[sel, 1], kind="stable")]
        col[o] = np.minimum((np.arange(len(o)) * 8) // max(len(o), 1), 7)
    return row * 8 + col


@dataclass
class BoardEncoder:
    c: Connectome
    rate_hz: float = 150.0

    def __post_init__(self):
        n = self.c.neurons
        # groups[(piece_type, own:bool)] -> list of 64 index arrays
        self.groups: dict[tuple[int, bool], list[np.ndarray]] = {}
        for pt, types in PIECE_CHANNELS.items():
            for own, side in ((True, OWN_SIDE), (False, OPP_SIDE)):
                sel = n[(n.cell_type.isin(types)) & (n.side == side)]
                bins = _bin_by_position(sel)
                idx = sel.index.to_numpy()
                self.groups[(pt, own)] = [idx[bins == b] for b in range(64)]
        self.all_input = np.unique(np.concatenate(
            [g for gl in self.groups.values() for g in gl]))

    def describe(self) -> str:
        lines = []
        for (pt, own), gl in self.groups.items():
            sizes = [len(g) for g in gl]
            lines.append(f"{chess.piece_name(pt):6s} {'own' if own else 'opp'}: "
                         f"{sum(sizes):4d} neurons, {min(sizes)}-{max(sizes)} per square")
        return "\n".join(lines)

    @staticmethod
    def canonical(board: chess.Board) -> chess.Board:
        """Return the board from the side-to-move's perspective (white to move)."""
        return board if board.turn == chess.WHITE else board.mirror()

    def driven_indices(self, board: chess.Board) -> np.ndarray:
        b = self.canonical(board)
        parts = []
        for sq, piece in b.piece_map().items():
            parts.append(self.groups[(piece.piece_type, piece.color == chess.WHITE)][sq])
        return np.concatenate(parts) if parts else np.zeros(0, dtype=int)

    def drive(self, boards: list[chess.Board]) -> torch.Tensor:
        N = self.c.n_neurons
        d = torch.zeros((len(boards), N), dtype=torch.float32)
        for i, bd in enumerate(boards):
            idx = self.driven_indices(bd)
            if len(idx):
                d[i, torch.as_tensor(idx)] = self.rate_hz
        return d
