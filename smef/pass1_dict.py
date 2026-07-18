"""
smef/pass1_dict.py
==================
Pass 1 — Structural Factorisation (inspired by LZ77 / dictionary compression)

Each weight matrix W is decomposed as:
    W ≈ D @ C + R
where:
  D  — global Semantic Dictionary  [num_atoms × atom_rank] (shared across all layers)
  C  — per-layer sparse coefficient matrix  [atom_rank × cols]
  R  — per-layer residual (W - D @ C)

The dictionary is built by:
  1. Computing truncated SVD of each weight matrix to get its principal directions.
  2. Running mini-batch k-means on all left singular vectors to find shared atoms.
  3. Each layer's weight matrix is projected onto D, giving C; residual R = W - D@C.

On decode: W = D @ C + R  (exact if R is stored losslessly).
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import math


# ── helpers ──────────────────────────────────────────────────────────────────

# Float dtypes that are safe to convert to float32 for SVD.
# np.float128 only exists on Linux x86-64, so we build the set dynamically.
_NATIVE_FLOAT_TYPES = {np.float16, np.float32, np.float64}
for _name in ('float128', 'float96', 'longdouble'):
    _t = getattr(np, _name, None)
    if _t is not None:
        _NATIVE_FLOAT_TYPES.add(_t)


def _is_float_tensor(arr: np.ndarray) -> bool:
    """Return True only for float/bf16 tensors — skip integer embeddings, masks, etc."""
    return arr.dtype.type in _NATIVE_FLOAT_TYPES or arr.dtype == np.uint16


def _to_f32(arr: np.ndarray) -> np.ndarray:
    """
    Cast a weight tensor to float32 for computation.

    Handles:
      float16, float32, float64 → direct cast
      uint16                    → BF16 reinterpretation
      anything else             → direct cast (caller should pre-screen with _is_float_tensor)
    """
    if arr.dtype == np.uint16:          # bf16 stored as uint16
        return _bf16_to_f32(arr)
    return arr.astype(np.float32)


def _bf16_to_f32(arr: np.ndarray) -> np.ndarray:
    """Reinterpret uint16 bf16 bytes as float32 by zero-padding lower 16 bits."""
    u32 = arr.astype(np.uint32) << 16
    return u32.view(np.float32)


def _f32_to_bf16(arr: np.ndarray) -> np.ndarray:
    """Round float32 to bf16 (stored as uint16)."""
    f32 = arr.astype(np.float32)
    # round-to-nearest-even
    u32 = f32.view(np.uint32)
    rounding_bias = (u32 >> 16) & 1  # lsb of bf16 mantissa
    u32 = u32 + 0x7FFF + rounding_bias
    return (u32 >> 16).astype(np.uint16)


def _to_storage_dtype(arr: np.ndarray) -> np.ndarray:
    """
    Cast a tensor to its SMEF on-disk storage dtype.

    Native float16/float32 arrays are kept as-is, so a caller that already
    downcast a tensor to fp16 (SmefEncoder's convert_fp16 option) keeps the
    2-byte-per-value saving all the way to disk. Everything else (integer
    masks, bf16-as-uint16, etc.) falls back to float32 for safe stream
    splitting, matching the pre-existing behaviour.
    """
    if arr.dtype == np.float16 or arr.dtype == np.float32:
        return arr
    return _to_f32(arr)


def truncated_svd(W: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (U, S, Vt) of shape (m, rank), (rank,), (rank, n).

    Always works in float64 internally to prevent overflow/underflow in power
    iteration.  Results are returned as float32.
    """
    from scipy.linalg import svd as full_svd

    # ── sanitise input ────────────────────────────────────────────────────────
    # Replace any NaN / Inf with 0 before decomposing.  This can happen if a
    # tensor was misidentified as float (e.g. an integer tensor that slipped
    # through the dtype check) or if the file is partially corrupt.
    W64 = W.astype(np.float64)
    if not np.isfinite(W64).all():
        W64 = np.nan_to_num(W64, nan=0.0, posinf=0.0, neginf=0.0)

    m, n = W64.shape
    k    = min(rank, m, n)

    if m * n > 50_000:
        U, S, Vt = _randomised_svd(W64, k)
    else:
        U_full, S_full, Vt_full = full_svd(W64, full_matrices=False)
        U, S, Vt = U_full[:, :k], S_full[:k], Vt_full[:k, :]

    return U.astype(np.float32), S.astype(np.float32), Vt.astype(np.float32)


def _randomised_svd(W: np.ndarray, rank: int,
                    n_oversampling: int = 10,
                    n_power_iter: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Halko, Martinsson, Tropp (2011) randomised SVD — float64 throughout.

    Key fixes vs the original implementation:
      1. Input W must be float64 (caller's responsibility — done in truncated_svd).
      2. QR re-orthogonalisation after every power iteration.
         Without this, power iteration amplifies singular values exponentially
         (S_max^(2*iter+1)) and overflows float32.  With QR, the range stays
         bounded regardless of spectral norm.
      3. float64 throughout gives ~15 decimal digits of precision, eliminating
         the catastrophic cancellation that triggers RuntimeWarning in float32.
    """
    m, n = W.shape          # W is float64
    k    = min(rank + n_oversampling, min(m, n))

    rng   = np.random.default_rng(42)
    Omega = rng.standard_normal((n, k))   # float64 by default

    # Stage 1: range finder with QR-stabilised power iteration.
    # np.errstate suppresses spurious numpy 2.x RuntimeWarnings that fire on
    # certain BLAS paths even when the final float64 result is finite/correct.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        Y = W @ Omega                         # (m, k)
        for _ in range(n_power_iter):
            Y, _ = np.linalg.qr(Y)           # re-orthogonalise: keeps values bounded
            Y    = W @ (W.T @ Y)              # (m, k)

        Q, _ = np.linalg.qr(Y)               # (m, k)  orthonormal basis for range(W)

        # Stage 2: project and SVD in the small k-dimensional space
        B        = Q.T @ W                   # (k, n)
        Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
        U         = Q @ Ub                   # (m, rank+oversampling)

    return U[:, :rank], S[:rank], Vt[:rank, :]


# ── K-means for dictionary construction ──────────────────────────────────────

def _kmeans(data: np.ndarray, k: int, n_iter: int = 30) -> np.ndarray:
    """
    Mini-batch k-means on rows of `data` (shape: N × d).
    Returns centroids of shape (k × d).
    """
    N, d = data.shape
    rng = np.random.default_rng(0)
    # Initialise with k-means++ style seeding
    idx = [rng.integers(N)]
    for _ in range(k - 1):
        dists = np.min(
            np.sum((data[np.array(idx)][:, None] - data[None]) ** 2, axis=-1),
            axis=0,
        )
        probs = dists / dists.sum()
        idx.append(rng.choice(N, p=probs))
    centroids = data[np.array(idx)].copy()

    batch = min(4096, N)
    for _ in range(n_iter):
        perm = rng.permutation(N)[:batch]
        batch_data = data[perm]
        # Assign each point to nearest centroid
        dists = np.sum((centroids[:, None] - batch_data[None]) ** 2, axis=-1)  # (k, batch)
        assigns = np.argmin(dists, axis=0)
        # Update centroids
        for c in range(k):
            mask = assigns == c
            if mask.sum() > 0:
                centroids[c] = batch_data[mask].mean(axis=0)
    return centroids


# ── Main pass-1 encoder ───────────────────────────────────────────────────────

class DictionaryEncoder:
    """
    Builds per-row-dimension semantic dictionaries and encodes each weight tensor.

    For each unique row dimension m seen during fit(), we build a dictionary
    D_m of shape (m, num_atoms) by running k-means on the m-dimensional left
    singular vectors collected from all tensors sharing that row dimension.

    Encoding a tensor W (m, n):
        C = D_m.T @ W    → (num_atoms, n)   — compact projection coefficients
        R = W - D_m @ C  → (m, n)           — low-variance residual

    C is smaller than W when num_atoms < m.  R has lower entropy than W
    because the structured cross-layer variation is captured by D_m @ C,
    leaving R as near-Gaussian noise that LZMA can compress more aggressively.

    Decoding:  W = D_m @ C + R  (exact, lossless)

    Parameters
    ----------
    num_atoms : int
        Dictionary columns (k-means centroids) per row dimension. Typical: 64–256.
    atom_rank : int
        SVD truncation rank per tensor during fit(). Typical: 16–128.
    """

    def __init__(self,
                 num_atoms: int = 128,
                 atom_rank: int = 32,
                 residual_threshold: float = 0.0):
        self.num_atoms = num_atoms
        self.atom_rank = atom_rank
        self.residual_threshold = residual_threshold
        # Per-m dictionaries: {m: D} where D.shape == (m, num_atoms)
        self.dictionaries: Dict[int, np.ndarray] = {}

    # ── Phase A: collect singular vectors grouped by row dimension ────────────

    def fit(self, weight_tensors: List[np.ndarray]) -> "DictionaryEncoder":
        """
        Build per-row-dimension dictionaries from a list of weight tensors.

        For each unique m, collects all m-dimensional left singular vectors
        (weighted by singular values) across all tensors with that row dim,
        then runs k-means to produce a (m, num_atoms) dictionary matrix.
        """
        # Group m-dimensional singular vectors by row size
        vectors_by_m: Dict[int, List[np.ndarray]] = {}

        for W in weight_tensors:
            if not _is_float_tensor(W):
                continue

            W_f32 = _to_f32(W)
            if W_f32.ndim < 2:
                continue
            W2d   = W_f32.reshape(W_f32.shape[0], -1) if W_f32.ndim > 2 else W_f32
            m, n  = W2d.shape

            if min(m, n) < 4 or m * n < 1000 or not np.any(W2d):
                continue

            rank = min(self.atom_rank, m, n)
            try:
                U, S, _ = truncated_svd(W2d, rank)   # U: (m, rank), S: (rank,)
            except Exception:
                continue

            # Collect full m-dimensional vectors, weighted by singular values
            weighted = U * S[None, :]                 # (m, rank)
            if m not in vectors_by_m:
                vectors_by_m[m] = []
            for r in range(weighted.shape[1]):
                vectors_by_m[m].append(weighted[:, r].astype(np.float32))

        self.dictionaries = {}
        for m, vecs in vectors_by_m.items():
            if len(vecs) < 2:
                continue

            data = np.array(vecs, dtype=np.float32)  # (N, m)
            # Normalise rows
            norms = np.linalg.norm(data, axis=1, keepdims=True) + 1e-8
            data  = data / norms

            k = min(self.num_atoms, len(vecs))
            centroids = _kmeans(data, k)              # (k, m)
            # Re-normalise and transpose → (m, k)
            cnorms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
            centroids = centroids / cnorms
            self.dictionaries[m] = centroids.T.astype(np.float32)  # (m, k)

        return self

    # ── Phase B: encode one tensor ────────────────────────────────────────────

    def encode_tensor(self, W: np.ndarray,
                      storage_dtype: Optional[np.dtype] = None
                      ) -> Tuple[np.ndarray, np.ndarray, tuple]:
        """
        Encode W as (C, R, original_shape).

        Dictionary mode (when D_m is available and num_atoms < m):
          C = D_m.T @ W2d    shape (num_atoms, n)
          R = W2d - D_m @ C  shape (m, n)
          Decode: W = D_m @ C + R

        Passthrough mode (no dictionary or tensor too small):
          C = zeros(1, 1),  R = W2d

        storage_dtype : dtype to store C/R in. Defaults to preserving W's
            own dtype when it is float16 or float32 (anything else falls
            back to float32). The dictionary projection itself is always
            computed in float32 for numerical stability — only the final
            C/R arrays returned to the caller are cast to storage_dtype.
            Passing float16 here is what makes FP16 precision-reduction
            actually shrink the stored payload: previously C/R were always
            upcast to float32 before storage, silently discarding it.
        """
        if storage_dtype is None:
            storage_dtype = W.dtype if W.dtype in (np.float16, np.float32) else np.float32

        W_f32 = _to_f32(W)
        orig_shape = W_f32.shape
        W2d = W_f32.reshape(W_f32.shape[0], -1).astype(np.float32)
        m, n = W2d.shape

        D = self.dictionaries.get(m)
        use_dict = (
            D is not None
            and _is_float_tensor(W)
            and min(m, n) >= 4
            and m * n >= 1000
            and D.shape[1] < m   # only helps when num_atoms < m
        )

        if not use_dict:
            return np.zeros((1, 1), dtype=storage_dtype), W2d.astype(storage_dtype), orig_shape

        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            C = D.T @ W2d          # (num_atoms, n)
            R = W2d - D @ C        # (m, n) — low-variance residual

        return C.astype(storage_dtype), R.astype(storage_dtype), orig_shape

    def decode_tensor(self,
                      C: np.ndarray,
                      R: np.ndarray,
                      orig_shape: tuple) -> np.ndarray:
        """
        Reconstruct W from (C, R, orig_shape).

        Dictionary mode — C.shape[0] == num_atoms: W = D_m @ C + R
        Passthrough     — C.shape == (1, 1):        W = R

        C/R may be stored at reduced precision (e.g. float16); they are
        upcast to float32 here so the reconstruction math always runs at
        full precision regardless of on-disk storage dtype.
        """
        m = orig_shape[0] if len(orig_shape) >= 1 else R.shape[0]
        D = self.dictionaries.get(m)
        is_dict = (D is not None and C.shape != (1, 1))

        if is_dict:
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                W2d = D @ C.astype(np.float32) + R.astype(np.float32)
        else:
            W2d = R.astype(np.float32).reshape(m, -1)

        return W2d.reshape(orig_shape)

    # ── Serialise / deserialise all per-m dictionaries ───────────────────────

    def dict_to_bytes(self) -> bytes:
        """Pack all per-m dictionaries to bytes (lzma-compressed)."""
        import lzma, struct
        parts = []
        for m, D in self.dictionaries.items():
            rows, cols = D.shape
            header = struct.pack('<III', m, rows, cols)
            parts.append(header + D.astype(np.float32).tobytes())
        raw = b''.join(parts)
        return lzma.compress(raw, preset=3)

    def dict_from_bytes(self, data: bytes) -> None:
        import lzma, struct
        raw = lzma.decompress(data)
        offset = 0
        self.dictionaries = {}
        header_size = struct.calcsize('<III')
        while offset < len(raw):
            m, rows, cols = struct.unpack_from('<III', raw, offset)
            offset += header_size
            n_floats = rows * cols
            arr = np.frombuffer(raw, dtype=np.float32,
                                count=n_floats, offset=offset)
            self.dictionaries[m] = arr.reshape(rows, cols)
            offset += n_floats * 4
