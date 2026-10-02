"""Load the FlyWire v783 connectome (Shiu et al. 2024 release) into PyTorch tensors.

Data sources (both already on disk):
  external/Drosophila_brain_model/Completeness_783.csv   - neuron list (index = root_id)
  external/Drosophila_brain_model/Connectivity_783.parquet - pre, post, signed synapse count
  data/flywire_annotations.tsv - Schlegel et al. 2024 cell-type annotations
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "external" / "Drosophila_brain_model"
DATA_DIR = ROOT / "data"
CACHE = DATA_DIR / "cache" / "connectome_783.pt"

COMPLETENESS = MODEL_DIR / "Completeness_783.csv"
CONNECTIVITY = MODEL_DIR / "Connectivity_783.parquet"
ANNOTATIONS = DATA_DIR / "flywire_annotations.tsv"

ANNOTATION_COLS = [
    "root_id", "pos_x", "pos_y", "pos_z", "super_class", "cell_class",
    "cell_sub_class", "cell_type", "hemibrain_type", "top_nt", "side",
]


@dataclass
class Connectome:
    """CSR-by-presynaptic representation of the signed connectome.

    indptr[i]:indptr[i+1] slices `indices`/`weights` for the outgoing synapses of
    neuron i. weights are signed synapse counts (Excitatory x Connectivity).
    """
    root_ids: np.ndarray          # (N,) int64 FlyWire root ids, position = model index
    indptr: torch.Tensor          # (N+1,) int64
    indices: torch.Tensor         # (E,) int32 postsynaptic index
    weights: torch.Tensor         # (E,) float32 signed synapse count
    neurons: pd.DataFrame         # (N rows) annotations aligned to model index

    @property
    def n_neurons(self) -> int:
        return len(self.root_ids)

    @property
    def n_synapses(self) -> int:
        return int(self.indices.numel())

    def index_of(self, root_ids) -> np.ndarray:
        lut = pd.Series(np.arange(self.n_neurons), index=self.root_ids)
        return lut.loc[np.asarray(root_ids)].to_numpy()

    def select(self, **filters) -> np.ndarray:
        """Model indices of neurons matching all column==value filters.

        A value may be a list/tuple/set to match any of several values.
        """
        mask = np.ones(self.n_neurons, dtype=bool)
        for col, val in filters.items():
            s = self.neurons[col]
            if isinstance(val, (list, tuple, set, np.ndarray)):
                mask &= s.isin(list(val)).to_numpy()
            else:
                mask &= (s == val).to_numpy()
        return np.flatnonzero(mask)

    def to(self, device) -> "Connectome":
        return Connectome(
            self.root_ids,
            self.indptr.to(device),
            self.indices.to(device),
            self.weights.to(device),
            self.neurons,
        )


def _build() -> Connectome:
    comp = pd.read_csv(COMPLETENESS, index_col=0)
    root_ids = comp.index.to_numpy(dtype=np.int64)
    n = len(root_ids)

    con = pd.read_parquet(
        CONNECTIVITY,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    pre = con["Presynaptic_Index"].to_numpy(dtype=np.int64)
    post = con["Postsynaptic_Index"].to_numpy(dtype=np.int64)
    w = con["Excitatory x Connectivity"].to_numpy(dtype=np.float32)
    order = np.argsort(pre, kind="stable")
    pre, post, w = pre[order], post[order], w[order]
    counts = np.bincount(pre, minlength=n)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    ann = pd.read_csv(ANNOTATIONS, sep="\t", usecols=ANNOTATION_COLS, low_memory=False)
    ann = ann.drop_duplicates("root_id").set_index("root_id")
    neurons = ann.reindex(root_ids).reset_index().rename(columns={"index": "root_id"})
    neurons["root_id"] = root_ids
    neurons["super_class"] = neurons["super_class"].fillna("unknown")

    return Connectome(
        root_ids=root_ids,
        indptr=torch.from_numpy(indptr),
        indices=torch.from_numpy(post.astype(np.int32)),
        weights=torch.from_numpy(w),
        neurons=neurons,
    )


def load(use_cache: bool = True) -> Connectome:
    """Load the connectome, building and caching tensors on first use."""
    if use_cache and CACHE.exists():
        blob = torch.load(CACHE, weights_only=False)
        return Connectome(**blob)
    c = _build()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(root_ids=c.root_ids, indptr=c.indptr, indices=c.indices,
             weights=c.weights, neurons=c.neurons),
        CACHE,
    )
    return c


if __name__ == "__main__":
    c = load(use_cache=False)
    print(f"neurons={c.n_neurons:,} synaptic connections={c.n_synapses:,}")
    print(c.neurons.super_class.value_counts())
