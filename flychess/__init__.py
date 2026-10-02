"""flychess: a fruit fly connectome (FlyWire v783) that plays chess.

The whole adult fly brain (138,639 neurons, ~15M connections) is simulated as a
leaky integrate-and-fire network following Shiu et al. 2024. A chess position is
projected onto the fly's photoreceptors, the brain is run for a short time, and a
trained linear readout on the resulting spike counts picks a move.
"""
__version__ = "0.1.0"
