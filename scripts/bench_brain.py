"""Sanity + speed check of the whole-brain LIF simulation."""
import sys, time
import numpy as np, torch
from flychess import connectome
from flychess.brain import FlyBrain, LIFParams

torch.set_num_threads(8)
c = connectome.load()
photo = c.select(super_class="sensory", cell_class="visual")
rng = np.random.default_rng(0)
patch = rng.choice(photo, 400, replace=False)      # ~ one 'square' worth of retina

def summarize(counts, t_rec_ms):
    rate = counts.float() / (t_rec_ms / 1000)
    for sc in ["sensory", "optic", "visual_projection", "central", "descending", "motor"]:
        idx = c.select(super_class=sc)
        r = rate[:, idx]
        print(f"  {sc:18s} n={len(idx):6d}  active={float((r>0).float().mean()*100):5.1f}%  mean Hz={float(r.mean()):6.2f}  max Hz={float(r.max()):6.1f}")

for dev in sys.argv[1:] or ["cpu"]:
    brain = FlyBrain(c, LIFParams(dt=0.25), device=dev, seed=1)
    for B in [1, 8, 32]:
        drive = brain.make_drive(B, [patch] * B, rate_hz=150.0)
        t0 = time.time()
        counts, trace = brain.run(drive, t_ms=100.0, record_from=20.0, return_trace=True)
        dt = time.time() - t0
        print(f"[{dev}] B={B:2d} 100ms sim: {dt:6.2f}s  ({dt/B:5.2f}s/position)  spikes/step first/last: "
              f"{int(trace[0,:40].float().mean())}/{int(trace[0,-40:].float().mean())}")
    summarize(counts, 80.0)
    # with no drive the network must stay silent
    quiet = brain.run(torch.zeros(1, c.n_neurons), t_ms=50.0)
    print(f"[{dev}] undriven network total spikes: {int(quiet.sum())}")
