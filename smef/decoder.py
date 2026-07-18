"""
smef/decoder.py
===============
SMEF Decoder — reads a .smef file and reconstructs the original weight tensors.

Supports two loading modes:
  1. Full decode  — decompress all layers to memory (fastest inference)
  2. Streaming    — decompress one layer at a time (lowest memory footprint)

Usage:
    from smef.decoder import SmefDecoder

    dec = SmefDecoder("model.smef")
    state = dec.decode_all()          # full decode → {name: np.array}

    # OR for streaming (one layer at a time):
    for name, tensor in dec.stream_layers():
        ...
"""

import io
import os
import json
import struct
import lzma
import hashlib
import time
from typing import Dict, Any, List, Tuple, Optional, Iterator
import numpy as np

from .format import (
    SmefHeader, GroupEntry, FrameHeader,
    HEADER_PADDED, GROUP_ENTRY_SIZE,
    PREC_FP32, PREC_BF16, PREC_FP16,
    TensorRegistry,
)
from .pass1_dict   import DictionaryEncoder
from .pass2_delta  import DeltaEncoder
from .pass3_streams import merge_streams
from .pass4_entropy import StreamCompressor, HuffmanCodec
from .encoder       import _unpack_layer_block


class SmefDecoder:
    """
    Decode a SMEF file back to a model state dict.

    Parameters
    ----------
    path    : str     path to .smef file
    verbose : bool    print progress
    """

    def __init__(self, path: str, verbose: bool = True):
        self.path    = path
        self.verbose = verbose
        self._fh: Optional[io.BufferedReader] = None
        self._header: Optional[SmefHeader] = None
        self._meta:   Optional[Dict]        = None
        self._loaded = False

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [SMEF] {msg}")

    # ── File open / header read ───────────────────────────────────────────────

    def open(self) -> "SmefDecoder":
        self._fh = open(self.path, 'rb')
        hdr_bytes = self._fh.read(HEADER_PADDED)
        self._header = SmefHeader.from_bytes(hdr_bytes)
        self._header.validate()
        self._load_metadata()
        self._loaded = True
        return self

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self): return self.open()
    def __exit__(self, *_): self.close()

    def _load_metadata(self):
        self._fh.seek(self._header.meta_offset)
        meta_len = struct.unpack("<I", self._fh.read(4))[0]
        meta_raw = lzma.decompress(self._fh.read(meta_len))
        self._meta = json.loads(meta_raw.decode())

    # ── Read raw layer block bytes from file ──────────────────────────────────

    def _read_layer_block(self, layer_idx: int) -> bytes:
        """Read one layer's compressed block from file using seek table."""
        n = self._header.num_layers
        # Seek table is at the very end
        file_size = os.path.getsize(self.path)
        seek_table_start = file_size - n * 8
        self._fh.seek(seek_table_start + layer_idx * 8)
        layer_offset_in_data = struct.unpack("<Q", self._fh.read(8))[0]

        # Absolute offset: past header + meta + dict + group_index
        abs_offset = self._header.layers_offset + layer_offset_in_data

        # Determine block size: distance to next layer (or end of layers section)
        if layer_idx < n - 1:
            self._fh.seek(seek_table_start + (layer_idx + 1) * 8)
            next_offset = struct.unpack("<Q", self._fh.read(8))[0]
            block_size = next_offset - layer_offset_in_data
        else:
            block_size = seek_table_start - abs_offset

        self._fh.seek(abs_offset)
        return self._fh.read(block_size)

    # ── Load dictionary from file ─────────────────────────────────────────────

    def _load_dictionary(self) -> Optional[DictionaryEncoder]:
        meta = self._meta
        if meta.get('num_atoms', 0) == 0:
            return None

        self._fh.seek(self._header.dict_offset)
        dict_len = struct.unpack("<I", self._fh.read(4))[0]
        dict_raw = self._fh.read(dict_len)

        dict_enc = DictionaryEncoder(
            num_atoms=meta['num_atoms'],
            atom_rank=meta['atom_rank'],
        )
        if dict_raw:
            dict_enc.dict_from_bytes(dict_raw)
        return dict_enc

    # ── Full decode ───────────────────────────────────────────────────────────

    def decode_all(self) -> Dict[str, np.ndarray]:
        """
        Decompress all layers and return {tensor_name: numpy_array}.
        """
        if not self._loaded:
            self.open()

        t0 = time.time()
        meta     = self._meta
        n        = self._header.num_layers
        names    = meta['tensor_names']
        shapes   = [tuple(s) for s in meta['tensor_shapes']]
        dtypes   = meta['tensor_dtypes']
        groups   = meta['groups']
        flags    = meta.get('flags', 0)

        self._log(f"Decoding {n} tensors …")

        # Load dictionary (Pass 1)
        dict_enc = self._load_dictionary()

        # Read all compressed layer blocks
        raw_blocks = [self._read_layer_block(i) for i in range(n)]

        # Per-group Huffman tables (I-frames store them, P-frames reuse)
        group_huffman_R: Dict[int, bytes] = {}
        group_huffman_C: Dict[int, bytes] = {}
        layer_to_group: Dict[int, int] = {}
        for g_idx, group in enumerate(groups):
            for l_idx in group:
                layer_to_group[l_idx] = g_idx

        # Decode all layers in group order (I-frame before P-frames)
        # Build pass-2 frames first
        decoded_frames = [None] * n
        sc = StreamCompressor(backend=meta.get('entropy_backend', 'lzma'))

        for g_idx, group in enumerate(groups):
            prev_C: Optional[np.ndarray] = None
            prev_R: Optional[np.ndarray] = None

            for pos_in_group, layer_idx in enumerate(group):
                raw = raw_blocks[layer_idx]
                frame, _ = _unpack_layer_block(raw)
                is_iframe = frame['is_iframe']

                # Retrieve or set Huffman tables
                if is_iframe:
                    if frame['huf_R'] and len(frame['huf_R']) == 1280:
                        group_huffman_R[g_idx] = frame['huf_R']
                    if frame['huf_C'] and len(frame['huf_C']) == 1280:
                        group_huffman_C[g_idx] = frame['huf_C']

                huf_R = group_huffman_R.get(g_idx, b'\x00' * 1280)
                huf_C = group_huffman_C.get(g_idx, b'\x00' * 1280)
                if len(huf_R) < 1280:
                    huf_R = huf_R + b'\x00' * (1280 - len(huf_R))
                if len(huf_C) < 1280:
                    huf_C = huf_C + b'\x00' * (1280 - len(huf_C))

                # Pass 4 → 3: Decompress streams
                streams_R = sc.decompress_streams(
                    frame['cs_R'], frame['ce_R'], frame['cm_R'],
                    layout=frame['layout'],
                    shape=frame['shape_R'],
                    huffman_table=huf_R,
                )
                streams_C = sc.decompress_streams(
                    frame['cs_C'], frame['ce_C'], frame['cm_C'],
                    layout=frame['layout'],
                    shape=frame['shape_C'],
                    huffman_table=huf_C,
                )

                # Pass 3: merge streams → arrays
                R = merge_streams(streams_R).astype(np.float32)
                C = merge_streams(streams_C).astype(np.float32)

                # Pass 2: reconstruct from delta
                if not is_iframe and prev_C is not None:
                    # P-frame: add delta to previous only if shapes match
                    if (C.shape == prev_C.shape and R.shape == prev_R.shape):
                        C = prev_C + C
                        R = prev_R + R
                    # else: treat as I-frame (data is already absolute)

                decoded_frames[layer_idx] = {'C': C, 'R': R,
                                             'orig_shape': frame['orig_shape']}
                prev_C = C.copy()
                prev_R = R.copy()

        # Pass 1: reconstruct tensors from D @ C + R
        state_dict: Dict[str, np.ndarray] = {}

        for layer_idx, name in enumerate(names):
            fr = decoded_frames[layer_idx]
            C  = fr['C']
            R  = fr['R']
            orig_shape = fr['orig_shape'] or shapes[layer_idx]

            if dict_enc is not None and dict_enc.dictionaries:
                W = dict_enc.decode_tensor(C, R, orig_shape)
            else:
                W = R.reshape(orig_shape)

            # Cast back to original dtype
            target_dtype = np.dtype(dtypes[layer_idx])
            if target_dtype == np.dtype('uint16'):   # bf16
                state_dict[name] = W.astype(np.float32)  # keep as f32
            elif target_dtype == np.dtype('float16'):
                state_dict[name] = W.astype(np.float16)
            else:
                state_dict[name] = W.astype(np.float32)

        elapsed = time.time() - t0
        self._log(f"Decoded {n} tensors in {elapsed:.2f}s")
        return state_dict

    # ── Streaming decode ──────────────────────────────────────────────────────

    def stream_layers(self) -> Iterator[Tuple[str, np.ndarray]]:
        """
        Yield (name, tensor) one at a time in layer order.
        Keeps only the current and previous layer in memory — suitable for
        very low-memory inference environments.

        Note: streaming decode requires sequential access; random-order access
        would require re-reading I-frames. For now, yields in natural order.
        """
        if not self._loaded:
            self.open()

        meta     = self._meta
        n        = self._header.num_layers
        names    = meta['tensor_names']
        shapes   = [tuple(s) for s in meta['tensor_shapes']]
        dtypes   = meta['tensor_dtypes']
        groups   = meta['groups']

        dict_enc = self._load_dictionary()

        group_huffman: Dict[int, bytes] = {}
        layer_to_group: Dict[int, int] = {}
        for g_idx, group in enumerate(groups):
            for l_idx in group:
                layer_to_group[l_idx] = g_idx

        # We process in natural layer order.
        # For each layer, we need the previous layer's C,R in the same group.
        prev_by_group: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        sc = StreamCompressor(backend=meta.get('entropy_backend', 'lzma'))

        for layer_idx in range(n):
            g_idx = layer_to_group[layer_idx]
            raw   = self._read_layer_block(layer_idx)
            frame, _ = _unpack_layer_block(raw)
            is_iframe = frame['is_iframe']

            if is_iframe and frame['huf_R']:
                group_huffman[g_idx] = frame['huf_R']
            huf_table = group_huffman.get(g_idx, b'\x00' * 1280)
            if len(huf_table) < 1280:
                huf_table = huf_table + b'\x00' * (1280 - len(huf_table))

            streams_R = sc.decompress_streams(
                frame['cs_R'], frame['ce_R'], frame['cm_R'],
                frame['layout'], frame['shape_R'], huf_table)
            streams_C = sc.decompress_streams(
                frame['cs_C'], frame['ce_C'], frame['cm_C'],
                frame['layout'], frame['shape_C'], huf_table)

            R = merge_streams(streams_R).astype(np.float32)
            C = merge_streams(streams_C).astype(np.float32)

            if not is_iframe and g_idx in prev_by_group:
                prev_C, prev_R = prev_by_group[g_idx]
                if C.shape == prev_C.shape and R.shape == prev_R.shape:
                    C = prev_C + C
                    R = prev_R + R

            prev_by_group[g_idx] = (C.copy(), R.copy())

            orig_shape = frame['orig_shape'] or shapes[layer_idx]
            if dict_enc is not None and dict_enc.dictionaries:
                W = dict_enc.decode_tensor(C, R, orig_shape)
            else:
                W = R.reshape(orig_shape)

            target_dtype = np.dtype(dtypes[layer_idx])
            if target_dtype == np.dtype('float16'):
                W = W.astype(np.float16)
            else:
                W = W.astype(np.float32)

            yield names[layer_idx], W

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def metadata(self) -> Dict:
        if not self._loaded: self.open()
        return self._meta

    @property
    def tensor_names(self) -> List[str]:
        return self.metadata['tensor_names']

    @property
    def tensor_shapes(self) -> List[Tuple]:
        return [tuple(s) for s in self.metadata['tensor_shapes']]
