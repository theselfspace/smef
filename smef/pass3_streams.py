"""
smef/pass3_streams.py
=====================
Pass 3 — Entropy-Aware Byte-Stream Separation  (inspired by ZipNN, JPEG DCT split)

Float32 bit layout:
  [1-bit sign] [8-bit exponent] [23-bit mantissa]

BF16 (stored as uint16) bit layout:
  [1-bit sign] [8-bit exponent] [7-bit mantissa]

FP16 bit layout:
  [1-bit sign] [5-bit exponent] [10-bit mantissa]

Key insight: exponent bytes have highly non-uniform distribution
(weights cluster near zero → exponent concentrates near 127 for fp32/bf16).
Mantissa bytes are more uniform but have context-dependent structure.
Sign bits are near i.i.d.

We separate the three streams so each can be compressed optimally:
  stream_S : packed sign bits   → raw bit-packing (incompressible)
  stream_E : exponent bytes     → Huffman coding (very compressible)
  stream_M : mantissa bytes     → LZMA (medium entropy)

This module handles split/merge. The actual entropy coding is in pass4_entropy.py.
"""

import numpy as np
from typing import Tuple, Dict


# ── Float layout constants ────────────────────────────────────────────────────

class FloatLayout:
    # (total_bits, sign_bits, exp_bits, mant_bits, exp_bias)
    FP32 = (32,  1, 8, 23, 127)
    BF16 = (16,  1, 8,  7, 127)   # stored as uint16
    FP16 = (16,  1, 5, 10,  15)
    FP8  = ( 8,  1, 4,  3,   7)   # E4M3 format (NF4 approximation)


def _dtype_layout(arr: np.ndarray) -> Tuple:
    if arr.dtype == np.float32:  return FloatLayout.FP32
    if arr.dtype == np.uint16:   return FloatLayout.BF16
    if arr.dtype == np.float16:  return FloatLayout.FP16
    if arr.dtype == np.uint8:    return FloatLayout.FP8
    # Fallback: cast to float32
    return FloatLayout.FP32


# ── Split: float array → (signs, exponents, mantissas) ───────────────────────

def split_streams(arr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Split a float array into three separate byte/bit streams.

    Returns dict with keys: 'sign', 'exp', 'mant', 'layout', 'shape', 'dtype'
    """
    flat = arr.ravel()
    layout = _dtype_layout(flat)
    total_bits, sign_bits, exp_bits, mant_bits, _ = layout

    if total_bits == 32:
        u = flat.view(np.uint32)
        signs = ((u >> 31) & 0x1).astype(np.uint8)
        exps  = ((u >> 23) & 0xFF).astype(np.uint8)
        mants = (u & 0x7FFFFF).astype(np.uint32)   # 23-bit mantissa → uint32

        return {
            'sign':   _pack_bits(signs),
            'exp':    exps.tobytes(),
            'mant':   mants.tobytes(),   # 4 bytes per element
            'layout': 'fp32',
            'shape':  arr.shape,
            'dtype':  'fp32',
        }

    elif total_bits == 16:
        if arr.dtype == np.float16:
            u = flat.view(np.uint16)
        else:
            u = flat  # bf16 already stored as uint16

        signs = ((u >> 15) & 0x1).astype(np.uint8)

        if layout == FloatLayout.BF16:
            exps  = ((u >> 7) & 0xFF).astype(np.uint8)
            mants = (u & 0x7F).astype(np.uint8)
        else:  # FP16
            exps  = ((u >> 10) & 0x1F).astype(np.uint8)
            mants = (u & 0x3FF).view(np.uint8).tobytes()
            return {
                'sign':   _pack_bits(signs),
                'exp':    exps.tobytes(),
                'mant':   mants,
                'layout': 'fp16',
                'shape':  arr.shape,
                'dtype':  'fp16',
            }

        return {
            'sign':   _pack_bits(signs),
            'exp':    exps.tobytes(),
            'mant':   mants.tobytes(),
            'layout': 'bf16',
            'shape':  arr.shape,
            'dtype':  'bf16',
        }

    else:  # FP8 / uint8
        u = flat.astype(np.uint8)
        signs = ((u >> 7) & 0x1).astype(np.uint8)
        exps  = ((u >> 3) & 0x0F).astype(np.uint8)
        mants = (u & 0x07).astype(np.uint8)
        return {
            'sign':   _pack_bits(signs),
            'exp':    exps.tobytes(),
            'mant':   mants.tobytes(),
            'layout': 'fp8',
            'shape':  arr.shape,
            'dtype':  'fp8',
        }


# ── Merge: (signs, exponents, mantissas) → float array ───────────────────────

def merge_streams(streams: Dict) -> np.ndarray:
    """Reconstruct the original float array from split streams."""
    layout = streams['layout']
    shape  = streams['shape']
    n      = int(np.prod(shape))

    sign_bits = _unpack_bits(streams['sign'], n)

    if layout == 'fp32':
        exps  = np.frombuffer(streams['exp'],  dtype=np.uint8)[:n]
        mants = np.frombuffer(streams['mant'], dtype=np.uint32)[:n]
        u = ((sign_bits.astype(np.uint32) << 31) |
             (exps.astype(np.uint32) << 23) |
             (mants & 0x7FFFFF))
        return u.view(np.float32).reshape(shape)

    elif layout == 'bf16':
        exps  = np.frombuffer(streams['exp'],  dtype=np.uint8)[:n]
        mants = np.frombuffer(streams['mant'], dtype=np.uint8)[:n]
        u = ((sign_bits.astype(np.uint16) << 15) |
             (exps.astype(np.uint16) << 7) |
             mants.astype(np.uint16))
        return u.astype(np.uint16).reshape(shape)

    elif layout == 'fp16':
        exps  = np.frombuffer(streams['exp'],  dtype=np.uint8)[:n]
        mants = np.frombuffer(streams['mant'], dtype=np.uint8)
        mant_u16 = mants.view(np.uint16)[:n]
        u = ((sign_bits.astype(np.uint16) << 15) |
             (exps.astype(np.uint16) << 10) |
             mant_u16.astype(np.uint16))
        return u.view(np.float16).reshape(shape)

    else:  # fp8
        exps  = np.frombuffer(streams['exp'],  dtype=np.uint8)[:n]
        mants = np.frombuffer(streams['mant'], dtype=np.uint8)[:n]
        u = ((sign_bits.astype(np.uint8) << 7) |
             (exps.astype(np.uint8) << 3) |
             mants.astype(np.uint8))
        return u.reshape(shape)


# ── Bit packing helpers ───────────────────────────────────────────────────────

def _pack_bits(bits: np.ndarray) -> bytes:
    """Pack a uint8 array of 0/1 values into a compact bytes object."""
    n = len(bits)
    padded = np.zeros(((n + 7) // 8) * 8, dtype=np.uint8)
    padded[:n] = bits
    # Pack 8 bits → 1 byte
    packed = np.packbits(padded)
    # Prepend length so we know how many bits to unpack
    import struct
    return struct.pack("<I", n) + packed.tobytes()


def _unpack_bits(data: bytes, n: int) -> np.ndarray:
    """Unpack bytes back into uint8 array of 0/1 values."""
    import struct
    stored_n = struct.unpack("<I", data[:4])[0]
    packed = np.frombuffer(data[4:], dtype=np.uint8)
    bits = np.unpackbits(packed)
    return bits[:stored_n].astype(np.uint8)


# ── Entropy analysis (used for compression reporting) ────────────────────────

def stream_entropy(stream_bytes: bytes) -> float:
    """Compute Shannon entropy (bits per byte) of a byte sequence."""
    arr = np.frombuffer(stream_bytes, dtype=np.uint8)
    if len(arr) == 0:
        return 0.0
    counts = np.bincount(arr, minlength=256).astype(np.float64)
    probs = counts[counts > 0] / len(arr)
    return float(-np.sum(probs * np.log2(probs)))


def analyse_streams(streams: Dict) -> Dict:
    """Report entropy of each stream (diagnostic utility)."""
    return {
        'sign_entropy_bpb':  stream_entropy(streams['sign']),
        'exp_entropy_bpb':   stream_entropy(streams['exp']),
        'mant_entropy_bpb':  stream_entropy(streams['mant']),
        'sign_size_bytes':   len(streams['sign']),
        'exp_size_bytes':    len(streams['exp']),
        'mant_size_bytes':   len(streams['mant']),
    }
