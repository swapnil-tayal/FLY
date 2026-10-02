I taught a fruit fly brain to play chess this week. Sort of.

Google and HHMI Janelia open-sourced the complete wiring diagram of a fruit fly brain, and people have been plugging it into Doom, Minecraft and even a Rubik's cube. I wanted to see what it would do with chess.

What I built, in one evening with Claude Code:

🧠 The whole FlyWire connectome (138,639 neurons, 15 million connections) simulated as a spiking network, following the Shiu et al. 2024 Nature model. First thing I did was check it reproduces the paper: stimulate the sugar-sensing neurons and the same proboscis motor neurons fire. They do.

♟️ The chess position gets projected onto the fly's visual projection neurons. Its own pieces go to one eye, the opponent's to the other, and each square is a patch of the visual field.

📈 The brain runs for 150 ms. A simple linear readout, trained on 20,000 Stockfish-labelled positions, turns the spike counts into a move. Nothing inside the brain was changed. Only the readout learned.

🖥️ A web UI with a 2D board and the brain in 3D, every neuron at its real position, glowing live as the spikes stream out of the simulation. You can watch it think.

The honest results:

• The readout agrees with Stockfish's best move 25% of the time, versus 9% for a random legal move.
• It does NOT beat a plain linear model on the raw board. The brain is a fixed, noisy nonlinear transform, and that costs about as much as it gains.
• Against a random mover it almost never loses but rarely wins. Against Stockfish on its weakest setting it lost every game.

So the fly brain carries real information about the position, but a connectome is a wiring diagram, not a chess player. Which is exactly the kind of thing you only learn by building it.

Stack: PyTorch, python-chess, Stockfish, FastAPI, Three.js. Runs on a MacBook.

#neuroscience #connectomics #FlyWire #chess #AI #openscience
