"""Shared label vocabulary; order is part of the checkpoint contract."""

ALPHABET = "0123456789ABCEHKMOPTXY"
LETTERS = "ABCEHKMOPTXY"
TRANSLATION = str.maketrans("АВСЕНКМОРТХУ", "ABCEHKMOPTXY")
WEIGHT_FILES = (
    "detector.pt",
    "sequence.pt",
    "sequence_aux.pt",
    "characters.pt",
    "characters_aux.pt",
)
