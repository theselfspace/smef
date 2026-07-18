"""
smef/pass4_entropy.py
=====================
Pass 4 — Context-Adaptive Entropy Coding  (inspired by PAQ, Zstandard, LZMA)

Production strategy per stream:
  stream_S (sign bits)     → zlib level 1  (near-random, minimal overhead)
  stream_E (exponent bytes)→ LZMA level 6  (non-uniform → very compressible)
  stream_M (mantissa bytes)→ LZMA level 3  (local tensor correlations)

HuffmanCodec is retained as a reference implementation per the paper.
The StreamCompressor uses robust LZMA throughout.
"""

import heapq, struct, lzma, zlib, collections, numpy as np
from typing import Tuple, Dict, Optional, List

from .pass4_rans import rans_encode, rans_decode

ENTROPY_BACKENDS = ('lzma', 'rans')


class HuffmanCodec:
    """Reference Huffman implementation (see paper Section 3.4)."""

    def __init__(self):
        self.encode_table: Dict[int, Tuple[int, int]] = {}
        self.decode_root: Optional[dict] = None

    def build_from_bytes(self, data: bytes) -> "HuffmanCodec":
        counts = collections.Counter(data)
        for b in range(256):
            counts.setdefault(b, 1)
        node_id = 256
        nodes = {b: (f, b, None, None) for b, f in counts.items()}
        pq = list(nodes.values())
        heapq.heapify(pq)
        while len(pq) > 1:
            f1, n1, *_ = heapq.heappop(pq)
            f2, n2, *_ = heapq.heappop(pq)
            nodes[node_id] = (f1+f2, node_id, n1, n2)
            heapq.heappush(pq, nodes[node_id])
            node_id += 1
        self.encode_table = {}
        self._codes(nodes, pq[0][1], 0, 0)
        self.decode_root = {}
        for val, (code, ln) in self.encode_table.items():
            node = self.decode_root
            for i in range(ln-1, -1, -1):
                b = (code >> i) & 1
                node = node.setdefault(b, {})
            node['val'] = val
        return self

    def _codes(self, nodes, nid, code, ln):
        _, ident, left, right = nodes[nid]
        if left is None:
            self.encode_table[ident] = (code, max(ln, 1))
        else:
            self._codes(nodes, left,  (code<<1)|0, ln+1)
            self._codes(nodes, right, (code<<1)|1, ln+1)

    def table_to_bytes(self) -> bytes:
        return b"".join(struct.pack("<IB", *self.encode_table.get(b, (0,0))) for b in range(256))

    @classmethod
    def from_table_bytes(cls, data: bytes) -> "HuffmanCodec":
        c = cls()
        c.encode_table = {}
        for b in range(256):
            code, ln = struct.unpack("<IB", data[b*5:b*5+5])
            if ln: c.encode_table[b] = (code, ln)
        c.decode_root = {}
        for val, (code, ln) in c.encode_table.items():
            node = c.decode_root
            for i in range(ln-1,-1,-1):
                node = node.setdefault((code>>i)&1, {})
            node['val'] = val
        return c


class StreamCompressor:
    """
    Stream compressor for Pass 4 — sign always uses zlib (packed sign bits
    are near-random; entropy coding them isn't worth the per-stream header
    cost either backend would add). exp/mant use a selectable `backend`:

    backend='lzma' (default): exp → LZMA-6, mant → LZMA-3. Slower, but has
        no fixed per-stream header overhead, so it wins on small tensors and
        on data with structure a static histogram can't capture (LZMA's LZ77
        stage + adaptive range coder).
    backend='rans': exp/mant → static-table rANS (pass4_rans.py). ~15-20x
        faster than LZMA at compressing/decompressing exponent streams (measured on
        real GPT-2 data — see DEV_NOTES Session 8) and slightly smaller on
        skewed data like exponents, at the cost of a ~528-byte fixed header
        per stream — a real loss on very small tensors.

    huffman_table parameter is a 1280-byte format-compatibility placeholder
    (retained from an earlier design; not used by either backend today).
    """

    def __init__(self, backend: str = 'lzma'):
        if backend not in ENTROPY_BACKENDS:
            raise ValueError(f"Unknown entropy backend {backend!r}; expected one of {ENTROPY_BACKENDS}")
        self.backend = backend

    def compress_streams(self, streams: Dict, is_iframe: bool,
                         iframe_huffman: Optional[bytes] = None,
                         ) -> Tuple[bytes, bytes, bytes, bytes]:
        c_sign = zlib.compress(streams['sign'], level=1)
        if self.backend == 'rans':
            c_exp  = rans_encode(streams['exp'])
            c_mant = rans_encode(streams['mant'])
        else:
            c_exp  = lzma.compress(streams['exp'],  preset=6)
            c_mant = lzma.compress(streams['mant'], preset=3)
        placeholder = bytes(1280) if is_iframe else b""
        return c_sign, c_exp, c_mant, placeholder

    def decompress_streams(self, c_sign: bytes, c_exp: bytes, c_mant: bytes,
                           layout: str, shape: tuple, huffman_table: bytes) -> Dict:
        if self.backend == 'rans':
            exp_bytes  = rans_decode(c_exp)
            mant_bytes = rans_decode(c_mant)
        else:
            exp_bytes  = lzma.decompress(c_exp)
            mant_bytes = lzma.decompress(c_mant)
        return {
            'sign':   zlib.decompress(c_sign),
            'exp':    exp_bytes,
            'mant':   mant_bytes,
            'layout': layout, 'shape': shape, 'dtype': layout,
        }


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    return original_bytes / max(compressed_bytes, 1)

def format_ratio(ratio: float) -> str:
    return f"{ratio:.2f}×"
