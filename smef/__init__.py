"""
SMEF — Semantic Manifold Encoded Format
==================================================================

  Pass 1: Structural dictionary factorization (LZ77-inspired)   — pass1_dict.py
  Pass 2: Cross-layer delta encoding (Git/VCDIFF-inspired)      — pass2_delta.py
  Pass 3: Entropy-aware byte-stream separation (ZipNN/JPEG-inspired) — pass3_streams.py
  Pass 4: Context-adaptive entropy coding (LZMA + rANS)         — pass4_entropy.py / pass4_rans.py

Quick start:
    import numpy as np
    from smef import SmefEncoder, SmefDecoder

    weights = {"layer0.weight": np.random.randn(768, 768).astype(np.float32)}

    SmefEncoder(num_atoms=64, atom_rank=128).encode(weights, "model.smef")

    state = SmefDecoder("model.smef").decode_all()
    assert np.allclose(state["layer0.weight"], weights["layer0.weight"])

See the accompanying README.md for how each file maps.

"""

from .encoder import SmefEncoder
from .decoder import SmefDecoder
from .format import (
    SmefHeader, FLAGS_ALL,
    FLAG_PASS1_DICT, FLAG_PASS2_DELTA, FLAG_PASS3_SPLIT, FLAG_PASS4_ANS,
)

__version__ = "0.1.0"
__all__ = [
    "SmefEncoder", "SmefDecoder",
    "SmefHeader", "FLAGS_ALL",
    "FLAG_PASS1_DICT", "FLAG_PASS2_DELTA", "FLAG_PASS3_SPLIT", "FLAG_PASS4_ANS",
]
