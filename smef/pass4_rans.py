"""
smef/pass4_rans.py
===================
rANS (range Asymmetric Numeral System) entropy coder — an alternative
backend to the LZMA/zlib coder in pass4_entropy.py.

Static-frequency-table, byte-alphabet rANS, following the classic
byte-wise renormalizing scheme (Fabian Giesen's "rans_byte.h" reference,
https://github.com/rygorous/ryg_rans — public-domain description):

    encode:  x = ((x // freq[s]) << scale_bits) + (x % freq[s]) + start[s]
    decode:  x = freq[s] * (x >> scale_bits) + slot - start[s]

with byte-wise renormalization (RANS_BYTE_L lower bound, 8-bit output digits).

Vectorization: the inherently-sequential state recurrence can't be
parallelized within one rANS stream, so this module splits the input
round-robin across NUM_LANES independent interleaved states. The Python-level
loop then only runs ceil(N / NUM_LANES) times, each iteration updating all
lanes at once via numpy array ops — instead of N sequential per-byte steps.

Encode processes columns in reverse (rANS is a stack: last symbol first);
decode processes forward. Both sides derive the exact same renormalization
decisions from their own state (that's the rANS invariant), so only the
compact stream of *actually emitted* bytes needs to be stored — no separate
mask/bookkeeping data.
"""

import struct
import numpy as np

RANS_BYTE_L = 1 << 23      # lower bound of the renormalization interval
SCALE_BITS  = 15           # M = 2**15 = 32768 total frequency budget (fits uint16)
M           = 1 << SCALE_BITS
NUM_LANES   = 4096         # interleaved independent rANS states
MAX_ITERS   = 2            # renorm iterations per symbol; see proof below

_X_MAX_UNIT = (RANS_BYTE_L >> SCALE_BITS) << 8

# Proof that MAX_ITERS=2 always suffices (RANS_BYTE_L=2**23, byte=2**8,
# scale_bits=15, min quantized freq=1):
#   Encode: pre-step x < RANS_BYTE_L*256 = 2**31. Each renorm shift divides
#   by 256; after 2 shifts x < 2**15 <= x_max (x_max >= RANS_BYTE_L*256/M
#   = 2**16 for freq=1), so the loop always terminates within 2 iterations.
#   Decode: post-update x can be as low as ~freq*(RANS_BYTE_L>>scale_bits)
#   >= 2**8 for freq=1; each renorm read multiplies by 256, and
#   256**2 * 2**8 = 2**24 >= RANS_BYTE_L, so 2 reads always suffice.


def _build_freq_table(arr: np.ndarray) -> np.ndarray:
    """Static per-stream byte histogram, quantized to sum exactly to M."""
    counts = np.bincount(arr, minlength=256).astype(np.float64)
    present = counts > 0
    if not present.any():
        freq = np.zeros(256, dtype=np.int64)
        freq[0] = M   # arbitrary placeholder; never referenced (stream is empty)
        return freq.astype(np.uint32)

    scaled = counts / counts.sum() * M
    freq = np.where(present, np.maximum(1, np.floor(scaled)), 0).astype(np.int64)

    remainder = M - int(freq.sum())
    if remainder > 0:
        # Give extra budget to the highest-count symbols first — cheapest
        # place to spend it in terms of coding-length impact.
        order = np.argsort(-counts)
        freq[order[:remainder]] += 1
    elif remainder < 0:
        order = np.argsort(-freq)
        i = 0
        while remainder < 0:
            idx = order[i % len(order)]
            if freq[idx] > 1:
                freq[idx] -= 1
                remainder += 1
            i += 1
    return freq.astype(np.uint32)


def _cum_freq(freq: np.ndarray) -> np.ndarray:
    start = np.zeros(256, dtype=np.uint32)
    start[1:] = np.cumsum(freq.astype(np.uint64))[:-1]
    return start


def _slot_to_symbol(freq: np.ndarray, start: np.ndarray) -> np.ndarray:
    slot_sym = np.zeros(M, dtype=np.uint8)
    for s in range(256):
        f = int(freq[s])
        if f:
            st = int(start[s])
            slot_sym[st:st + f] = s
    return slot_sym


def rans_encode(data: bytes) -> bytes:
    """Encode `data` with a static-table byte-alphabet rANS coder."""
    arr = np.frombuffer(data, dtype=np.uint8)
    n = arr.size
    freq = _build_freq_table(arr)
    freq16 = freq.astype('<u2')  # safe: max single freq == M == 32768 < 65536
    header = struct.pack('<I', n) + freq16.tobytes()

    if n == 0:
        return header + struct.pack('<III', 0, 0, 0)

    start = _cum_freq(freq)
    lanes  = min(NUM_LANES, n)
    n_cols = (n + lanes - 1) // lanes
    padded = np.zeros(n_cols * lanes, dtype=np.uint8)
    padded[:n] = arr
    cols   = padded.reshape(n_cols, lanes)
    f_cols = freq[cols].astype(np.uint64)
    c_cols = start[cols].astype(np.uint64)
    idx    = np.arange(n_cols * lanes, dtype=np.int64).reshape(n_cols, lanes)
    valid  = idx < n

    x = np.full(lanes, RANS_BYTE_L, dtype=np.uint64)
    chunks = []   # built in reverse column order, prepended -> ends up ascending

    for c in range(n_cols - 1, -1, -1):
        valid_col = valid[c]
        f, cst = f_cols[c], c_cols[c]
        x_max = _X_MAX_UNIT * f

        # Different lanes may need 0, 1, or 2 renorm bytes at this column
        # (MAX_ITERS=2). Decode re-derives its own need-mask fresh at each
        # of its 2 iterations from its own state, so it does NOT group
        # bytes by "which encode iteration produced them" — it groups by
        # "how many bytes from the end, per lane". Concretely: decode's
        # first consumption pass wants every emitting lane's *last*
        # emission (whether that lane emitted once or twice); its second
        # pass wants only the *first* emission of lanes that emitted twice.
        # (need1 is always a subset of need0: a lane can't need a second
        # shift without needing the first, since x is unchanged between
        # checks for lanes where need0 was False.)
        need0  = (x >= x_max) & valid_col
        byte0  = (x & 0xFF).astype(np.uint8)
        x1     = np.where(need0, x >> 8, x).astype(np.uint64)

        need1  = (x1 >= x_max) & valid_col
        byte1  = (x1 & 0xFF).astype(np.uint8)
        x2     = np.where(need1, x1 >> 8, x1).astype(np.uint64)

        last_byte = np.where(need1, byte1, byte0)   # each emitting lane's final byte
        step_chunks = []
        if need0.any():
            step_chunks.append(last_byte[need0])    # decode's 1st pass for this column
        if need1.any():
            step_chunks.append(byte0[need1])        # decode's 2nd pass for this column
        chunks = step_chunks + chunks

        x = x2
        f_safe = np.where(valid_col, f, np.uint64(1))  # avoid /0 on padding lanes
        new_x = ((x // f_safe) << SCALE_BITS) + (x % f_safe) + cst
        x = np.where(valid_col, new_x, x).astype(np.uint64)

    compact = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint8)
    body = (struct.pack('<III', lanes, n_cols, compact.size)
            + x.astype('<u4').tobytes()
            + compact.tobytes())
    return header + body


def rans_decode(blob: bytes) -> bytes:
    """Decode a blob produced by rans_encode back to the original bytes."""
    n = struct.unpack_from('<I', blob, 0)[0]
    off = 4
    freq = np.frombuffer(blob, dtype='<u2', count=256, offset=off).astype(np.uint32)
    off += 512

    lanes, n_cols, compact_len = struct.unpack_from('<III', blob, off)
    off += 12

    if n == 0:
        return b""

    start    = _cum_freq(freq)
    slot_sym = _slot_to_symbol(freq, start)

    x = np.frombuffer(blob, dtype='<u4', count=lanes, offset=off).astype(np.uint64).copy()
    off += lanes * 4
    compact = np.frombuffer(blob, dtype=np.uint8, count=compact_len, offset=off)

    idx   = np.arange(n_cols * lanes, dtype=np.int64).reshape(n_cols, lanes)
    valid = idx < n
    out   = np.zeros(n_cols * lanes, dtype=np.uint8)

    pos = 0
    for c in range(n_cols):
        valid_col = valid[c]
        slot = (x & (M - 1)).astype(np.int64)
        s    = slot_sym[slot]
        out[c * lanes:(c + 1) * lanes] = np.where(valid_col, s, 0)

        f, cst = freq[s].astype(np.uint64), start[s].astype(np.uint64)
        new_x  = f * (x >> SCALE_BITS) + (x & (M - 1)).astype(np.uint64) - cst
        x = np.where(valid_col, new_x, x).astype(np.uint64)

        for _ in range(MAX_ITERS):
            need = (x < RANS_BYTE_L) & valid_col
            k = int(need.sum())
            if k:
                nb = np.zeros(lanes, dtype=np.uint64)
                nb[need] = compact[pos:pos + k].astype(np.uint64)
                pos += k
                x = np.where(need, (x << 8) | nb, x).astype(np.uint64)

    return out[:n].tobytes()
