"""Batched leaky integrate-and-fire simulation of the whole fly brain.

Model follows Shiu et al. 2024 (Nature), "A leaky integrate-and-fire computational
model based on the connectome of the entire adult Drosophila brain":

    dv/dt = (v_0 - v + g) / t_mbr        (unless refractory)
    dg/dt = -g / tau                      (unless refractory)
    spike when v > v_th; then v = v_rst, g = 0; refractory for t_rfc
    presynaptic spike -> after t_dly, g_post += w_syn * signed_synapse_count

Driven ("Poisson input") neurons fire as a Poisson process at a given rate; in the
original model each Poisson event forces a spike, which is what we do directly.

Propagation is event driven: cost scales with spikes x out-degree, not with the
15M synapses, so it runs comfortably on a laptop CPU for batches of positions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .connectome import Connectome


@dataclass
class LIFParams:
    dt: float = 0.25          # ms, integration step
    v_0: float = -52.0        # mV, rest
    v_rst: float = -52.0      # mV, reset
    v_th: float = -45.0       # mV, threshold
    t_mbr: float = 20.0       # ms, membrane time constant
    tau: float = 5.0          # ms, synaptic time constant
    t_rfc: float = 2.2        # ms, refractory period
    t_dly: float = 1.8        # ms, synaptic delay
    w_syn: float = 0.275      # mV per synapse

    @property
    def delay_steps(self) -> int:
        return max(1, int(round(self.t_dly / self.dt)))

    @property
    def rfc_steps(self) -> int:
        return max(1, int(round(self.t_rfc / self.dt)))


class FlyBrain:
    """Whole-brain LIF network over a batch of independent trials.

    Usage:
        brain = FlyBrain(connectome, device="cpu")
        counts = brain.run(drive, t_ms=100, record_from=t_record)
    where `drive` is a (B, N) tensor of Poisson rates in Hz (0 = not driven).
    Returns (B, N) spike counts recorded after `record_from` ms.
    """

    def __init__(self, c: Connectome, params: LIFParams | None = None,
                 device: str | torch.device = "cpu", seed: int | None = None):
        self.p = params or LIFParams()
        self.device = torch.device(device)
        self.c = c.to(self.device)
        self.N = c.n_neurons
        self.gen = torch.Generator(device="cpu")
        if seed is not None:
            self.gen.manual_seed(seed)
        p = self.p
        self.decay_v = math.exp(-p.dt / p.t_mbr)
        self.decay_g = math.exp(-p.dt / p.tau)

    # ------------------------------------------------------------------ core
    def run(self, drive: torch.Tensor, t_ms: float = 100.0, record_from: float = 0.0,
            return_trace: bool = False, return_spikes: bool = False, on_step=None):
        """Simulate. `on_step(step, nz)` is called every step with the (batch, neuron)
        index tensor of neurons that spiked, for live streaming."""
        p, N, dev = self.p, self.N, self.device
        drive = drive.to(dev, torch.float32)
        B = drive.shape[0]
        driven = drive > 0
        drv_flat = driven.view(-1).nonzero().flatten()          # driven flat indices
        p_fire = (drive.view(-1)[drv_flat] * (p.dt / 1000.0)).clamp(0, 1).cpu()

        v = torch.full((B, N), p.v_0, device=dev)
        g = torch.zeros((B, N), device=dev)
        rfc = torch.zeros((B, N), dtype=torch.int32, device=dev)   # steps left
        D = p.delay_steps
        buf = torch.zeros((D, B * N), device=dev)                   # delayed g increments
        counts = torch.zeros((B, N), dtype=torch.int32, device=dev)
        trace = [] if return_trace else None
        spikes = [] if return_spikes else None

        n_steps = int(round(t_ms / p.dt))
        rec_step = int(round(record_from / p.dt))
        indptr, indices, weights = self.c.indptr, self.c.indices, self.c.weights
        w_syn = p.w_syn
        batch_offsets = (torch.arange(B, device=dev) * N)

        for step in range(n_steps):
            slot = step % D
            # 1. deliver delayed synaptic input, then decay
            g = g * self.decay_g + buf[slot].view(B, N)
            buf[slot].zero_()
            # 2. membrane update (exponential Euler, g held over the step)
            active = rfc <= 0
            v_new = p.v_0 + g + (v - p.v_0 - g) * self.decay_v
            v = torch.where(active, v_new, v)
            # 3. spikes: threshold crossing or forced Poisson events on driven neurons
            spk = (v > p.v_th) & active
            if drv_flat.numel():
                forced = drv_flat[(torch.rand(drv_flat.numel(), generator=self.gen) < p_fire).to(dev)]
                if forced.numel():
                    spk.view(-1)[forced] = True
            # 4. reset + refractory (driven neurons have no refractory period)
            v = torch.where(spk, torch.full_like(v, p.v_rst), v)
            g = torch.where(spk, torch.zeros_like(g), g)
            rfc = torch.where(spk & ~driven, torch.full_like(rfc, p.rfc_steps), rfc - 1)
            if step >= rec_step:
                counts += spk
            if trace is not None:
                trace.append(spk.sum(1))
            # 5. propagate spikes to delayed buffer (event driven gather)
            nz = spk.nonzero(as_tuple=False)
            if on_step is not None:
                on_step(step, nz)
            if spikes is not None and nz.numel():
                spikes.append(torch.cat([nz, torch.full((nz.shape[0], 1), step, device=dev)], 1))
            if nz.numel():
                b_idx, pre = nz[:, 0], nz[:, 1].to(torch.int64)
                starts, ends = indptr[pre], indptr[pre + 1]
                lens = ends - starts
                total = int(lens.sum())
                if total:
                    rep_start = torch.repeat_interleave(starts, lens)
                    rep_b = torch.repeat_interleave(b_idx, lens)
                    offs = torch.arange(total, device=dev) - torch.repeat_interleave(
                        torch.cumsum(lens, 0) - lens, lens)
                    flat = rep_start + offs
                    post = indices[flat].to(torch.int64)
                    w = weights[flat] * w_syn
                    target = buf[(step + D) % D]
                    target.index_add_(0, batch_offsets[rep_b] + post, w)

        out = [counts]
        if return_trace:
            out.append(torch.stack(trace, 1))
        if return_spikes:
            sp = torch.cat(spikes, 0) if spikes else torch.zeros((0, 3), dtype=torch.int64)
            out.append(sp)                      # columns: batch, neuron, step (t = step * dt)
        return out[0] if len(out) == 1 else tuple(out)

    # ------------------------------------------- active-set variant (experimental)
    def run_active(self, drive: torch.Tensor, t_ms: float = 100.0, record_from: float = 0.0,
            return_trace: bool = False, refresh_every: int = 16, eps: float = 0.05):
        """Active-set LIF simulation. Identical model to `run`, but state
        updates are restricted to neurons that are driven, refractory, or have
        non-negligible membrane/synaptic deviation from rest, plus any neuron
        that has a spike in flight toward it. Everything else sits exactly at
        rest, so skipping it is lossless up to a 1e-4 mV tolerance."""
        p, N, dev = self.p, self.N, self.device
        drive = drive.to(dev, torch.float32)
        B = drive.shape[0]
        BN = B * N
        drv_flat = (drive > 0).view(-1).nonzero().flatten()           # driven flat indices
        p_fire = (drive.view(-1)[drv_flat] * (p.dt / 1000.0)).clamp(0, 1)
        driven_mask = torch.zeros(BN, dtype=torch.bool, device=dev)
        driven_mask[drv_flat] = True

        v = torch.full((BN,), p.v_0, device=dev)
        g = torch.zeros(BN, device=dev)
        rfc = torch.zeros(BN, dtype=torch.int32, device=dev)
        D = p.delay_steps
        buf = torch.zeros((D, BN), device=dev)
        counts = torch.zeros(BN, dtype=torch.int32, device=dev)
        active = driven_mask.clone()
        idx = drv_flat.clone()
        dirty = False
        trace = [] if return_trace else None

        n_steps = int(round(t_ms / p.dt))
        rec_step = int(round(record_from / p.dt))
        indptr, indices, weights = self.c.indptr, self.c.indices, self.c.weights
        w_syn = p.w_syn
        self.active_sizes = []
        v_th, v_0, v_rst = p.v_th, p.v_0, p.v_rst

        for step in range(n_steps):
            if step % refresh_every == 0 and step:
                # drop neurons that have relaxed back to rest (|dv|,|g| below eps mV)
                active = driven_mask | (rfc > 0) | (g.abs() > eps) | ((v - v_0).abs() > eps)
                for d in range(D):
                    active |= buf[d] != 0
                v = torch.where(active, v, torch.full_like(v, v_0))
                g = torch.where(active, g, torch.zeros_like(g))
                idx = active.nonzero().flatten()
                self.active_sizes.append(int(idx.numel()) / B)
            slot = step % D
            inc = buf[slot][idx]
            buf[slot].zero_()
            g_a, v_a, rfc_a = g[idx], v[idx], rfc[idx]
            refr = rfc_a > 0
            g_a = torch.where(refr, g_a + inc, g_a * self.decay_g + inc)
            v_new = v_0 + g_a + (v_a - v_0 - g_a) * self.decay_v
            v_a = torch.where(refr, v_a, v_new)
            spk = (v_a > v_th) & ~refr
            if drv_flat.numel():
                forced = torch.rand(drv_flat.numel(), generator=self.gen) < p_fire.cpu()
                if forced.any():
                    f = torch.zeros(BN, dtype=torch.bool, device=dev)
                    f[drv_flat[forced.to(dev)]] = True
                    spk |= f[idx]
            v_a = torch.where(spk, torch.full_like(v_a, v_rst), v_a)
            g_a = torch.where(spk, torch.zeros_like(g_a), g_a)
            drv_a = driven_mask[idx]
            rfc_a = torch.where(spk & ~drv_a, torch.full_like(rfc_a, p.rfc_steps), rfc_a - 1)
            v[idx], g[idx], rfc[idx] = v_a, g_a, rfc_a
            spk_idx = idx[spk]
            if step >= rec_step and spk_idx.numel():
                counts[spk_idx] += 1
            if trace is not None:
                trace.append(torch.bincount(spk_idx // N, minlength=B))
            if spk_idx.numel():
                b_idx, pre = spk_idx // N, spk_idx % N
                starts, ends = indptr[pre], indptr[pre + 1]
                lens = ends - starts
                total = int(lens.sum())
                if total:
                    rep_start = torch.repeat_interleave(starts, lens)
                    rep_b = torch.repeat_interleave(b_idx, lens)
                    offs = torch.arange(total, device=dev) - torch.repeat_interleave(
                        torch.cumsum(lens, 0) - lens, lens)
                    flat = rep_start + offs
                    tgt = rep_b * N + indices[flat].to(torch.int64)
                    buf[(step + D) % D].index_add_(0, tgt, weights[flat] * w_syn)
                    # targets become active (a spike is in flight toward them)
                    newly = ~active[tgt]
                    if bool(newly.any()):
                        new_idx = torch.unique(tgt[newly])
                        active[new_idx] = True
                        idx = torch.cat([idx, new_idx])

        counts = counts.view(B, N)
        if return_trace:
            return counts, torch.stack(trace, 1)
        return counts

    # --------------------------------------------------------------- helpers
    def make_drive(self, B: int, idx_per_batch, rate_hz: float = 150.0) -> torch.Tensor:
        """Build a (B, N) drive tensor from per-batch lists of neuron indices."""
        drive = torch.zeros((B, self.N), dtype=torch.float32)
        for b, idx in enumerate(idx_per_batch):
            if len(idx):
                drive[b, torch.as_tensor(idx, dtype=torch.int64)] = rate_hz
        return drive
