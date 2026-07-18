"""
smef/pass2_delta.py
===================
Pass 2 — Cross-Layer Delta Encoding  (inspired by Git delta, VCDIFF, video P-frames)

After Pass 1 produces per-layer coefficient matrices C_l and residuals R_l,
this pass:
  1. Groups layers by semantic similarity (via KL-divergence proxy on C distributions).
  2. Stores the first layer of each group as an I-frame (full copy).
  3. Stores subsequent layers in each group as P-frames (delta from previous layer):
       delta_C_l = C_l XOR_or_sub C_{l-1}
       delta_R_l = R_l - R_{l-1}

Deltas are sparse and small-magnitude → much more compressible in Passes 3+4.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


# ── Group segmentation ────────────────────────────────────────────────────────

def _distribution_distance(C1: np.ndarray, C2: np.ndarray) -> float:
    """
    Proxy for KL divergence: compare value histograms of two coefficient matrices.
    Fast histogram overlap measure (1 - intersection).
    """
    flat1 = C1.ravel().astype(np.float32)
    flat2 = C2.ravel().astype(np.float32)
    # Use 32 bins over joint range
    lo = min(flat1.min(), flat2.min())
    hi = max(flat1.max(), flat2.max()) + 1e-8
    bins = 32
    h1, _ = np.histogram(flat1, bins=bins, range=(lo, hi), density=True)
    h2, _ = np.histogram(flat2, bins=bins, range=(lo, hi), density=True)
    # Intersection distance
    intersection = np.minimum(h1, h2).sum() / (bins * (hi - lo) / bins + 1e-8)
    return float(1.0 - min(intersection, 1.0))


def build_layer_groups(coeff_list: List[np.ndarray],
                       drift_threshold: float = 0.35,
                       min_group_size: int = 2,
                       max_group_size: int = 8) -> List[List[int]]:
    """
    Segment layers into groups of consecutive similar layers.
    Returns list of groups; each group is a list of layer indices.

    Parameters
    ----------
    coeff_list       : C matrices from Pass 1, one per layer
    drift_threshold  : if distance(C_l, C_{l-1}) > threshold, start new group
    min_group_size   : minimum layers per group
    max_group_size   : maximum layers per group (forces new group)
    """
    if not coeff_list:
        return []

    groups: List[List[int]] = [[0]]

    for i in range(1, len(coeff_list)):
        # Compare to last layer in current group
        prev_idx = groups[-1][-1]
        dist = _distribution_distance(coeff_list[prev_idx], coeff_list[i])
        current_size = len(groups[-1])

        if dist > drift_threshold or current_size >= max_group_size:
            groups.append([i])
        else:
            groups[-1].append(i)

    # Merge tiny tail group into previous if possible
    if len(groups) > 1 and len(groups[-1]) < min_group_size:
        groups[-2].extend(groups[-1])
        groups.pop()

    return groups


# ── Delta encoder ─────────────────────────────────────────────────────────────

class DeltaEncoder:
    """
    Encode/decode a sequence of (C, R) pairs using delta compression.

    I-frames store full data. P-frames store only the difference from the
    previous layer (within the same group).
    """

    def __init__(self, drift_threshold: float = 0.35):
        self.drift_threshold = drift_threshold
        self.groups: List[List[int]] = []

    # ── encode ────────────────────────────────────────────────────────────────

    def encode(self,
               coeff_list:  List[np.ndarray],
               resid_list:  List[np.ndarray],
               ) -> Tuple[List[Dict], List[List[int]]]:
        """
        Encode sequence of (C_l, R_l) pairs.

        Returns
        -------
        frames : list of dicts with keys:
                 'is_iframe', 'C' (full or delta), 'R' (full or delta),
                 'shape_C', 'shape_R'
        groups : group structure [[layer_idx, ...], ...]
        """
        n = len(coeff_list)
        assert n == len(resid_list)

        self.groups = build_layer_groups(coeff_list,
                                         drift_threshold=self.drift_threshold)
        frames = [None] * n

        for group in self.groups:
            iframe_idx = group[0]
            # I-frame: store full, preserving whatever dtype Pass 1 chose
            # (e.g. float16 when convert_fp16 is on) — forcing float32 here
            # would silently discard that storage-size reduction.
            frames[iframe_idx] = {
                'is_iframe': True,
                'C': coeff_list[iframe_idx],
                'R': resid_list[iframe_idx],
                'shape_C': coeff_list[iframe_idx].shape,
                'shape_R': resid_list[iframe_idx].shape,
            }
            # P-frames: store delta from previous layer
            for j in range(1, len(group)):
                cur_idx  = group[j]
                prev_idx = group[j - 1]

                C_cur  = coeff_list[cur_idx]
                C_prev = coeff_list[prev_idx]
                R_cur  = resid_list[cur_idx]
                R_prev = resid_list[prev_idx]

                # Only store as P-frame if shapes match exactly
                # (mismatched shapes can't be delta-encoded losslessly)
                if C_cur.shape == C_prev.shape and R_cur.shape == R_prev.shape:
                    # Subtract in float32 for numerical safety, then cast the
                    # delta back down to the original storage dtype.
                    dC = (C_cur.astype(np.float32) - C_prev.astype(np.float32)).astype(C_cur.dtype)
                    dR = (R_cur.astype(np.float32) - R_prev.astype(np.float32)).astype(R_cur.dtype)
                    frames[cur_idx] = {
                        'is_iframe': False,
                        'C': dC,
                        'R': dR,
                        'shape_C': C_cur.shape,
                        'shape_R': R_cur.shape,
                    }
                else:
                    # Shape mismatch — store as I-frame
                    frames[cur_idx] = {
                        'is_iframe': True,
                        'C': C_cur,
                        'R': R_cur,
                        'shape_C': C_cur.shape,
                        'shape_R': R_cur.shape,
                    }

        return frames, self.groups

    # ── decode ────────────────────────────────────────────────────────────────

    def decode(self,
               frames: List[Dict],
               groups: List[List[int]],
               ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Reconstruct full (C_l, R_l) from frames.
        """
        n = len(frames)
        coeff_out = [None] * n
        resid_out = [None] * n

        for group in groups:
            iframe_idx = group[0]
            f = frames[iframe_idx]
            coeff_out[iframe_idx] = f['C'].copy()
            resid_out[iframe_idx] = f['R'].copy()

            for j in range(1, len(group)):
                cur_idx  = group[j]
                prev_idx = group[j - 1]
                f = frames[cur_idx]

                coeff_out[cur_idx] = _safe_add(coeff_out[prev_idx], f['C'],
                                               f['shape_C'])
                resid_out[cur_idx] = _safe_add(resid_out[prev_idx], f['R'],
                                               f['shape_R'])

        return coeff_out, resid_out


# ── Shape-safe arithmetic helpers ────────────────────────────────────────────

def _safe_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Subtract b from a, handling shape mismatches by cropping to min shape."""
    rows = min(a.shape[0], b.shape[0])
    cols = min(a.shape[1], b.shape[1])
    return (a[:rows, :cols] - b[:rows, :cols]).astype(np.float32)


def _safe_add(prev: np.ndarray, delta: np.ndarray,
              target_shape: tuple) -> np.ndarray:
    """Add delta to prev, then restore to target_shape (pad with zeros)."""
    rows = min(prev.shape[0], delta.shape[0])
    cols = min(prev.shape[1], delta.shape[1])
    result = np.zeros(target_shape, dtype=prev.dtype)
    r_rows = min(rows, target_shape[0])
    r_cols = min(cols, target_shape[1])
    summed = (prev[:r_rows, :r_cols].astype(np.float32) +
              delta[:r_rows, :r_cols].astype(np.float32))
    result[:r_rows, :r_cols] = summed.astype(prev.dtype)
    return result


# ── Compression stats helper ──────────────────────────────────────────────────

def delta_compression_stats(coeff_list: List[np.ndarray],
                             frames: List[Dict]) -> Dict:
    """Report the ratio of delta magnitudes to original magnitudes."""
    orig_total = sum(np.abs(c).sum() for c in coeff_list)
    delta_total = sum(
        np.abs(f['C']).sum() for f in frames if not f['is_iframe']
    )
    n_pframes = sum(1 for f in frames if not f['is_iframe'])
    n_iframes = sum(1 for f in frames if f['is_iframe'])
    return {
        'n_iframes':     n_iframes,
        'n_pframes':     n_pframes,
        'delta_ratio':   float(delta_total / (orig_total + 1e-8)),
        'expected_gain': float(1.0 / max(delta_total / (orig_total + 1e-8), 0.01)),
    }
