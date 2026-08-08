"""
utils.py
--------
Shared constants, path helpers, and small utility functions used across
the project (data loading paths, interaction weights, logging helper).
"""

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (single source of truth so every module agrees on where files live)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Interaction weights
# ---------------------------------------------------------------------------
# WHY THESE WEIGHTS?
# Not all interactions signal the same strength of preference. A "purchase"
# is a strong, costly signal of genuine interest (the user spent money).
# "Wishlist" and "add_to_cart" show clear intent but no commitment yet.
# A "view" is the weakest and noisiest signal -- a user might view a product
# by accident, out of curiosity, or because it was simply shown to them.
# These weights follow the intuitive and widely-used e-commerce ordering:
#       purchase > wishlist > add_to_cart > view
# The exact numbers are chosen so that a single purchase outweighs many
# views (reflecting real conversion-funnel drop-off, where views vastly
# outnumber purchases but matter far less individually), while still letting
# accumulated weaker signals (e.g. many views) contribute meaningfully to a
# user's implicit preference score.
INTERACTION_WEIGHTS = {
    "purchase": 5.0,
    "wishlist": 3.0,
    "add_to_cart": 2.0,
    "view": 1.0,
}

RANDOM_SEED = 42


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger so every module prints consistent output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(name)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
