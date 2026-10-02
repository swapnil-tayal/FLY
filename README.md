# flychess: a fruit fly brain that plays chess

The complete adult *Drosophila* brain from the FlyWire connectome (v783: 138,639
neurons, 15.1 million synaptic connections) is simulated as a leaky
integrate-and-fire network, exactly following Shiu et al. 2024 (*Nature*). A chess
position is projected onto the fly's visual projection neurons, the brain runs
for 150 ms, and a trained linear readout on the resulting spike counts picks a
legal move.

## How it works

1. **Connectome** (`flychess/connectome.py`): loads the Shiu et al. release of
   FlyWire v783 (signed synapse counts, neurotransmitter-based sign) plus the
   Schlegel et al. 2024 cell-type annotations. Cached as PyTorch tensors.
2. **Brain** (`flychess/brain.py`): batched whole-brain LIF simulation. Same
   parameters as the paper (rest -52 mV, threshold -45 mV, 20 ms membrane, 5 ms
   synapse, 2.2 ms refractory, 1.8 ms delay, 0.275 mV per synapse). Event-driven
   propagation, so cost scales with spikes rather than synapses: about 0.2 s per
   position on an Apple M5 CPU in batches of 32.
3. **Encoder** (`flychess/encoder.py`): the fly "sees" the board through its
   lobula columnar visual projection neurons. Each piece type uses two or three LC
   cell types, own pieces go to the right eye and the opponent's to the left, and
   each square maps to a retinotopic bin of that population. The board is mirrored
   when black is to move so the fly always plays as white.
4. **Readout** (`flychess/train.py`): logistic regression from standardised
   log spike counts of every responding neuron to 4096 from-to move classes,
   trained against Stockfish depth-10 best moves with illegal moves masked. A
   no-brain baseline (same readout on raw board one-hot features) is trained
   alongside so the brain's contribution is measured, not assumed.
5. **Play** (`flychess/play.py`): fly vs random mover, fly vs weak Stockfish, or
   you vs the fly in the terminal.
6. **Web UI** (`flychess/server.py` + `flychess/static/`): FastAPI server with a
   WebSocket that streams spikes live from the running simulation, and a
   single-page interface: 2D board, Three.js brain point cloud with a corner
   fly avatar, readout panel.

## Validation

Driving the 20 sugar-sensing gustatory neurons from the paper's example at 150 Hz
reproduces its headline result: the proboscis motor neurons (CB0700, MNx01, MN10)
and the same central interneurons (CB0248, CB0192, CB0118, CB0445) fire at
100-180 Hz, matching the paper's own published spike data in identity, ranking
and approximate rate.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
brew install stockfish
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model external/Drosophila_brain_model
curl -L -o data/flywire_annotations.tsv https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv
```

## Web UI

```bash
.venv/bin/python -m flychess.server      # then open http://127.0.0.1:8765
```

Play against the fly on a drag-and-drop 2D board while its brain fills the
centre panel in 3D: all 138,639 neurons at their real FlyWire positions,
lighting up spike by spike (visual input in blue, brain in amber, descending
neurons in red) *while the simulation is running*. Spikes stream over a
WebSocket as they are computed, paced to slow motion (1 ms of brain time per
18 ms), so the fly visibly thinks for about 2.7 s per move. A small 3D fly sits
in the corner as the avatar, with a miniature of the same glowing brain in its
head; its wings buzz while it thinks. Drag to orbit, scroll to zoom,
double-click to reset the view. The readout panel shows the candidate moves,
their confidence, and per-class spike counts. Choose your colour, the readout
variant, and whether the fly may repeat positions.

## Pipeline

```bash
.venv/bin/python -m flychess.dataset --n-positions 20000   # Stockfish-labelled positions
.venv/bin/python -m flychess.features                      # run them through the brain
.venv/bin/python -m flychess.train                         # fit readout + baseline
.venv/bin/python -m flychess.play random --games 20        # evaluate
.venv/bin/python -m flychess.play human --color white      # play it yourself
```

## Honest framing

The connectome does not "know" chess. It is a fixed, biologically wired
nonlinear dynamical system that we use as a reservoir: the only thing that learns
is the linear readout. Whether that readout does better than the same readout on
raw board features is the empirical question this project measures. See the
results section below.

## Results (2 Oct 2026)

Dataset: 20,061 positions from 186 diverse Stockfish self-play games, each labelled
with Stockfish's depth-10 best move. 19,968 positions were run through the brain
(150 ms each, spike counts recorded from 30 ms). Readouts are linear with input
dropout 0.3; evaluation on 1,996 held-out positions.

| Move chooser                                   | Agrees with Stockfish (top-1) | top-3 |
|------------------------------------------------|------------------------------:|------:|
| Random legal move                               |   8.9% |      |
| Linear readout on raw board one-hot (no brain)  |  25.8% | 47.7% |
| Linear readout on all 4,930 responding neurons  |  24.5% | 44.7% |
| Linear readout on 2,611 downstream neurons only |  21.2% | 39.4% |

Games (fly moves chosen greedily; `downstream` readout unless noted):

| Match                                           | Games | W | D | L |
|-------------------------------------------------|------:|--:|--:|--:|
| vs random mover                                  | 6 | 0 | 6 | 0 |
| vs random mover, repetition avoidance on         | 6 | 1 | 5 | 0 |
| vs random mover, `all` readout                   | 6 | 1 | 4 | 1 |
| vs Stockfish skill 0, depth 1                    | 10 | 0 | 0 | 10 |

What this says:

* The brain's spike counts carry real information about the position: a linear
  readout on downstream neurons alone, which never see the board directly, picks
  Stockfish's move 2.4x more often than chance.
* It does not beat a linear model on the raw board. The connectome acts as a
  fixed, noisy nonlinear expansion; with single-trial Poisson input and ~20k
  training positions, the expansion costs about as much in noise as it gains in
  nonlinearity.
* Against a random mover the fly almost never loses but rarely converts: the
  readout imitates middlegame move choice and has no notion of mating. Most games
  end in repetition or the 300-ply cutoff.
* Against an engine it loses every game, as any one-ply imitation policy would.

## Ideas to try next

* Average spike counts over several trials per position to cut Poisson noise.
* A small hidden layer on both readouts, so the comparison stays fair.
* Encode castling rights and en passant (currently ignored).
* Close the loop: let the readout's chosen move feed back as visual input, so
  the fly "watches" its own move before the opponent replies.
* Drive the actual photoreceptors once a release with denser optic-lobe
  proofreading is available.
