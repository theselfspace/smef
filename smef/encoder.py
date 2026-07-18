"""
smef/encoder.py
===============
SMEF Encoder — orchestrates all 4 passes to produce a .smef file.

Usage:
    from smef.encoder import SmefEncoder
    from smef.model import SmefModel   # see model.py

    model = SmefModel(...)
    enc   = SmefEncoder(num_atoms=128, atom_rank=32)
    enc.encode(model, "model.smef")
"""

import io
import os
import json
import struct
import lzma
import hashlib
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from .format import (
    SmefHeader, GroupEntry, FrameHeader,
    SMEF_MAGIC, FLAGS_ALL, FLAG_PASS1_DICT, FLAG_PASS2_DELTA,
    FLAG_PASS3_SPLIT, FLAG_PASS4_ANS,
    HEADER_PADDED, GROUP_ENTRY_SIZE, FRAME_HDR_SIZE,
    PREC_FP32, PREC_BF16, PREC_FP16,
    TensorRegistry,
)
from .pass1_dict  import DictionaryEncoder, _to_f32, _to_storage_dtype
from .pass2_delta import DeltaEncoder, delta_compression_stats
from .pass3_streams import split_streams, analyse_streams
from .pass4_entropy import StreamCompressor, compression_ratio


def _quick_compressed_size(arr: np.ndarray) -> int:
    """
    Fast proxy for 'how many bytes will this take after Pass 3+4'.

    Splits into sign/exp/mantissa streams and LZMA-compresses each with a
    fast preset. Used only to decide, per-tensor, whether Pass 1's
    dictionary projection is worth its raw storage overhead — see the
    adaptive fallback in SmefEncoder.encode. Measuring directly (instead of
    assuming the projection always helps) matters because on real trained
    weights it frequently doesn't: the extra coefficient matrix can cost
    more raw bytes than the residual's lower entropy recovers.
    """
    if arr.size == 0:
        return 0
    streams = split_streams(arr)
    return (len(lzma.compress(streams['sign'], preset=1)) +
            len(lzma.compress(streams['exp'],  preset=1)) +
            len(lzma.compress(streams['mant'], preset=1)))


# ─── Encoder ─────────────────────────────────────────────────────────────────

class SmefEncoder:
    """
    Encode a model (dict of tensor name → numpy array) to SMEF format.

    Parameters
    ----------
    num_atoms : int
        Dictionary size for Pass 1.
    atom_rank : int
        SVD truncation rank per tensor for Pass 1.
    drift_threshold : float
        KL-divergence threshold for grouping layers (Pass 2).
    flags : int
        Bitmask selecting which passes to apply.
    verbose : bool
        Print progress information.
    """

    def __init__(self,
                 num_atoms: int = 64,
                 atom_rank: int = 128,
                 drift_threshold: float = 0.35,
                 flags: int = FLAGS_ALL,
                 verbose: bool = True,
                 convert_fp16: bool = False,
                 entropy_backend: str = 'lzma'):
        self.num_atoms       = num_atoms
        self.atom_rank       = atom_rank
        self.drift_threshold = drift_threshold
        self.flags           = flags
        self.verbose         = verbose
        self.convert_fp16    = convert_fp16
        # 'lzma' (default, no fixed per-stream overhead, best on small
        # tensors) or 'rans' (much faster, slightly smaller on skewed
        # streams like exponents — see DEV_NOTES Session 8). Recorded in
        # the file's metadata so the decoder never has to be told again.
        self.entropy_backend = entropy_backend

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [SMEF] {msg}")

    def encode(self, model_state: Dict[str, np.ndarray],
               output_path: str,
               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Encode model_state to a SMEF file.

        Parameters
        ----------
        model_state : dict  {tensor_name: numpy_array}
        output_path : str   path to write .smef file
        metadata    : dict  arbitrary JSON-serialisable metadata (config, tokenizer, etc.)

        Returns dict with compression statistics.
        """
        t0 = time.time()
        stats = {}

        # ── Prepare tensor list (sort by name for determinism) ───────────────
        names   = sorted(model_state.keys())
        tensors = [model_state[n] for n in names]
        n_layers = len(tensors)
        self._log(f"Encoding {n_layers} tensors …")

        # Optional FP16 conversion (demonstrates lower-precision compression gain)
        if self.convert_fp16:
            self._log("Converting FP32 → FP16 (--fp16 mode) …")
            tensors = [
                t.astype(np.float16) if t.dtype == np.float32 else t
                for t in tensors
            ]
            # Update model_state so dtype is stored correctly in metadata
            model_state = {n: t for n, t in zip(names, tensors)}

        orig_bytes = sum(t.nbytes for t in tensors)
        stats['original_bytes'] = orig_bytes

        registry = TensorRegistry()
        for name in names:
            registry.register(name)

        # ── Pass 1: Dictionary factorisation ─────────────────────────────────
        if self.flags & FLAG_PASS1_DICT:
            self._log("Pass 1 — building semantic dictionary …")
            # Only fit on 2-D float weight tensors.
            # Explicitly exclude integer tensors (position IDs, token type IDs,
            # attention masks) — their large integer values would overflow SVD.
            from .pass1_dict import _is_float_tensor
            fit_tensors = [
                _to_f32(t) for t in tensors
                if t.ndim >= 2
                and min(t.shape[:2]) >= 2
                and _is_float_tensor(t)
            ]
            dict_enc = DictionaryEncoder(
                num_atoms=self.num_atoms,
                atom_rank=self.atom_rank,
            )
            dict_enc.fit(fit_tensors)
            self._log(f"  Dictionary: {dict_enc.num_atoms} atoms × rank {dict_enc.atom_rank}")

            coeff_list = []
            resid_list = []
            n_dict_used = 0
            n_eligible  = 0
            for t in tensors:
                # Preserve fp16 storage precision when convert_fp16 downcast
                # this tensor; everything else stores at float32 as before.
                target_dtype = t.dtype if t.dtype in (np.float16, np.float32) else np.float32
                C_dict, R_dict, _ = dict_enc.encode_tensor(t, storage_dtype=target_dtype)

                if C_dict.shape == (1, 1):
                    # No dictionary available for this tensor's shape —
                    # passthrough is the only option.
                    coeff_list.append(C_dict)
                    resid_list.append(R_dict)
                    continue

                n_eligible += 1
                # Adaptive fallback: only keep the dictionary projection if
                # it actually measures smaller once entropy-coded. Otherwise
                # fall back to passthrough for this tensor.
                Wf = _to_f32(t)
                R_pass = Wf.reshape(Wf.shape[0], -1).astype(target_dtype)
                C_pass = np.zeros((1, 1), dtype=target_dtype)

                size_dict = _quick_compressed_size(C_dict) + _quick_compressed_size(R_dict)
                size_pass = _quick_compressed_size(R_pass)

                if size_dict < size_pass:
                    coeff_list.append(C_dict)
                    resid_list.append(R_dict)
                    n_dict_used += 1
                else:
                    coeff_list.append(C_pass)
                    resid_list.append(R_pass)

            self._log(f"  Adaptive Pass1: dictionary kept on {n_dict_used}/{n_eligible} "
                      f"eligible tensors (passthrough measured smaller on the rest)")

            dict_bytes = dict_enc.dict_to_bytes()
            p1_bytes   = sum(c.nbytes + r.nbytes for c, r in zip(coeff_list, resid_list))
            stats['pass1_dict_bytes'] = len(dict_bytes)
            stats['pass1_coeff_resid_bytes'] = p1_bytes
            stats['pass1_dict_used'] = n_dict_used
            stats['pass1_eligible']  = n_eligible
            self._log(f"  P1 coefficient+residual size: {p1_bytes/1e6:.1f} MB")
        else:
            # Skip Pass 1: treat raw float arrays as "residuals", preserving
            # fp16 storage precision when convert_fp16 downcast a tensor.
            dict_enc  = None
            dict_bytes = b""
            coeff_list = [np.zeros((1, 1), dtype=(t.dtype if t.dtype in (np.float16, np.float32) else np.float32))
                          for t in tensors]
            resid_list = [_to_storage_dtype(t) for t in tensors]

        # ── Pass 2: Cross-layer delta encoding ───────────────────────────────
        if self.flags & FLAG_PASS2_DELTA:
            self._log("Pass 2 — cross-layer delta encoding …")
            delta_enc = DeltaEncoder(drift_threshold=self.drift_threshold)
            frames, groups = delta_enc.encode(coeff_list, resid_list)
            p2_stats = delta_compression_stats(coeff_list, frames)
            self._log(f"  Groups: {len(groups)}  "
                      f"I-frames: {p2_stats['n_iframes']}  "
                      f"P-frames: {p2_stats['n_pframes']}  "
                      f"delta ratio: {p2_stats['delta_ratio']:.3f}")
            stats['pass2'] = p2_stats
        else:
            # No delta: every layer is an I-frame
            frames = [{'is_iframe': True,
                       'C': coeff_list[i],
                       'R': resid_list[i],
                       'shape_C': coeff_list[i].shape,
                       'shape_R': resid_list[i].shape}
                      for i in range(n_layers)]
            groups = [[i] for i in range(n_layers)]

        # ── Pass 3 + 4: Byte-stream split and entropy coding ─────────────────
        self._log("Pass 3+4 — stream splitting and entropy coding …")
        compressed_layers: List[bytes] = []
        seek_offsets: List[int] = []
        group_huffman_tables: Dict[int, bytes] = {}  # group_idx → huffman table

        # Map layer → group index
        layer_to_group: Dict[int, int] = {}
        for g_idx, group in enumerate(groups):
            for l_idx in group:
                layer_to_group[l_idx] = g_idx

        sc = StreamCompressor(backend=self.entropy_backend)
        current_offset = 0

        for layer_idx, frame in enumerate(frames):
            g_idx      = layer_to_group[layer_idx]
            is_iframe  = frame['is_iframe']
            iframe_huf = group_huffman_tables.get(g_idx)

            # Combine C and R into one array for stream encoding
            # For memory efficiency, encode R (larger) via streams, C separately
            # NOTE: no forced float32 cast here — R/C already carry whatever
            # storage dtype Pass 1/2 decided (float16 when convert_fp16 is
            # on), and split_streams natively supports fp16/bf16/fp32/fp8.
            R = frame['R']
            C = frame['C']

            # Pass 3: Split into sign/exp/mant streams
            if self.flags & FLAG_PASS3_SPLIT:
                streams_R = split_streams(R)
                streams_C = split_streams(C)
            else:
                streams_R = {'sign': R.tobytes(), 'exp': R.tobytes(), 'mant': b'',
                             'layout': 'fp32', 'shape': R.shape, 'dtype': 'fp32'}
                streams_C = {'sign': C.tobytes(), 'exp': C.tobytes(), 'mant': b'',
                             'layout': 'fp32', 'shape': C.shape, 'dtype': 'fp32'}

            # Pass 4: Entropy code each stream
            if self.flags & FLAG_PASS4_ANS:
                cs_R, ce_R, cm_R, ht_R = sc.compress_streams(
                    streams_R, is_iframe, iframe_huf)
                cs_C, ce_C, cm_C, ht_C = sc.compress_streams(
                    streams_C, is_iframe, iframe_huf)
            else:
                import zlib
                cs_R = zlib.compress(streams_R['sign'], 1)
                ce_R = streams_R['exp']
                cm_R = streams_R['mant']
                ht_R = b""
                cs_C = zlib.compress(streams_C['sign'], 1)
                ce_C = streams_C['exp']
                cm_C = streams_C['mant']
                ht_C = b""

            if is_iframe and ht_R:
                group_huffman_tables[g_idx] = ht_R

            # Serialise this layer's block
            layer_block = _pack_layer_block(
                is_iframe=is_iframe,
                # R streams
                cs_R=cs_R, ce_R=ce_R, cm_R=cm_R,
                layout_R=streams_R['layout'], shape_R=R.shape,
                # C streams
                cs_C=cs_C, ce_C=ce_C, cm_C=cm_C,
                layout_C=streams_C['layout'], shape_C=C.shape,
                # Huffman table (stored per group I-frame)
                huffman_R=ht_R if is_iframe else b"",
                huffman_C=ht_C if is_iframe else b"",
                # Shape info for residual original shape
                orig_shape=tensors[layer_idx].shape,
            )
            seek_offsets.append(current_offset)
            compressed_layers.append(layer_block)
            current_offset += len(layer_block)

        p34_bytes = sum(len(b) for b in compressed_layers)
        stats['pass34_bytes'] = p34_bytes
        stats['compression_ratio'] = orig_bytes / max(p34_bytes, 1)
        self._log(f"  Compressed: {p34_bytes/1e6:.1f} MB  "
                  f"(ratio: {stats['compression_ratio']:.2f}×)")

        # ── Assemble SMEF file ────────────────────────────────────────────────
        self._log("Assembling SMEF file …")

        # Metadata block
        meta_obj = {
            'tensor_names': names,
            'tensor_shapes': [list(t.shape) for t in tensors],
            'tensor_dtypes': [str(t.dtype) for t in tensors],
            'registry': registry.to_dict(),
            'groups': groups,
            'entropy_backend': self.entropy_backend,
            'group_huffman_present': list(group_huffman_tables.keys()),
            'num_atoms': dict_enc.num_atoms if dict_enc else 0,
            'atom_rank': dict_enc.atom_rank if dict_enc else 0,
            'flags': self.flags,
            **(metadata or {}),
        }
        meta_bytes = lzma.compress(json.dumps(meta_obj).encode(), preset=3)

        # Group index
        group_index_bytes = b""
        group_entry_offset = 0
        for g_idx, group in enumerate(groups):
            iframe_idx = group[0]
            ge = GroupEntry(
                iframe_layer_idx=iframe_idx,
                p_frame_count=len(group) - 1,
                iframe_offset=seek_offsets[iframe_idx],
            )
            group_index_bytes += ge.to_bytes()

        # Seek table
        seek_table_bytes = struct.pack(f"<{n_layers}Q", *seek_offsets)

        # Layer data concatenated
        all_layers_bytes = b"".join(compressed_layers)

        # Compute offsets
        meta_offset   = HEADER_PADDED
        dict_offset   = meta_offset + 4 + len(meta_bytes)   # 4-byte length prefix
        layers_offset = dict_offset + 4 + len(dict_bytes)
        group_idx_off = layers_offset + 4 + len(group_index_bytes)
        seek_offset   = group_idx_off + len(all_layers_bytes)

        # Checksum over all content
        checksum_data = meta_bytes + dict_bytes + group_index_bytes + all_layers_bytes
        chk = int.from_bytes(hashlib.sha256(checksum_data).digest()[:8], 'little')
        # Mask to signed int64 range (struct 'q' format)
        if chk >= 2**63:
            chk -= 2**64

        header = SmefHeader(
            num_layers    = n_layers,
            num_atoms     = dict_enc.num_atoms if dict_enc else 0,
            atom_rank     = dict_enc.atom_rank if dict_enc else 0,
            flags         = self.flags,
            group_count   = len(groups),
            meta_offset   = meta_offset,
            dict_offset   = dict_offset,
            layers_offset = group_idx_off,   # point past group index
            seek_offset   = seek_offset,
            checksum      = chk,
        )

        # Write file
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(header.to_bytes())
            f.write(struct.pack("<I", len(meta_bytes)))
            f.write(meta_bytes)
            f.write(struct.pack("<I", len(dict_bytes)))
            f.write(dict_bytes)
            f.write(struct.pack("<I", len(group_index_bytes)))
            f.write(group_index_bytes)
            f.write(all_layers_bytes)
            f.write(seek_table_bytes)

        file_size = os.path.getsize(output_path)
        stats['smef_file_bytes'] = file_size
        stats['final_ratio']     = orig_bytes / max(file_size, 1)
        stats['encode_time_s']   = time.time() - t0
        self._log(f"Done — {output_path}  "
                  f"({file_size/1e6:.1f} MB, "
                  f"{stats['final_ratio']:.2f}× compression, "
                  f"{stats['encode_time_s']:.1f}s)")
        return stats


# ── Layer block serialisation ─────────────────────────────────────────────────

def _pack_layer_block(*,
                      is_iframe: bool,
                      cs_R: bytes, ce_R: bytes, cm_R: bytes,
                      layout_R: str, shape_R: tuple,
                      cs_C: bytes, ce_C: bytes, cm_C: bytes,
                      layout_C: str, shape_C: tuple,
                      huffman_R: bytes, huffman_C: bytes,
                      orig_shape: tuple) -> bytes:
    """Pack all streams for one layer into a single bytes blob."""
    # Layout tag → 1 byte
    layout_tag = {'fp32': 0, 'bf16': 1, 'fp16': 2, 'fp8': 3}.get(layout_R, 0)

    def _pack_array(data: bytes) -> bytes:
        return struct.pack("<I", len(data)) + data

    def _pack_shape(s: tuple) -> bytes:
        # up to 4 dims
        dims = list(s)[:4]
        dims += [0] * (4 - len(dims))
        return struct.pack("<4I", *dims) + struct.pack("<B", len(s))

    parts = [
        struct.pack("<B", int(is_iframe)),
        struct.pack("<B", layout_tag),
        _pack_shape(shape_R),
        _pack_shape(shape_C),
        _pack_shape(orig_shape),
        _pack_array(cs_R),
        _pack_array(ce_R),
        _pack_array(cm_R),
        _pack_array(cs_C),
        _pack_array(ce_C),
        _pack_array(cm_C),
        _pack_array(huffman_R),
        _pack_array(huffman_C),
    ]
    return b"".join(parts)


def _unpack_layer_block(data: bytes, offset: int = 0) -> Tuple[Dict, int]:
    """Unpack a layer block; returns (frame_dict, bytes_consumed)."""
    pos = offset

    def _read(n: int) -> bytes:
        nonlocal pos
        chunk = data[pos:pos + n]
        pos += n
        return chunk

    def _read_array() -> bytes:
        sz = struct.unpack("<I", _read(4))[0]
        return _read(sz)

    def _read_shape() -> tuple:
        dims = struct.unpack("<4I", _read(16))
        ndim = struct.unpack("<B", _read(1))[0]
        return tuple(dims[:ndim])

    is_iframe   = bool(struct.unpack("<B", _read(1))[0])
    layout_tag  = struct.unpack("<B", _read(1))[0]
    layout      = ['fp32', 'bf16', 'fp16', 'fp8'][layout_tag]

    shape_R     = _read_shape()
    shape_C     = _read_shape()
    orig_shape  = _read_shape()

    cs_R = _read_array()
    ce_R = _read_array()
    cm_R = _read_array()
    cs_C = _read_array()
    ce_C = _read_array()
    cm_C = _read_array()
    huf_R = _read_array()
    huf_C = _read_array()

    return {
        'is_iframe':  is_iframe,
        'layout':     layout,
        'shape_R':    shape_R,
        'shape_C':    shape_C,
        'orig_shape': orig_shape,
        'cs_R': cs_R, 'ce_R': ce_R, 'cm_R': cm_R,
        'cs_C': cs_C, 'ce_C': ce_C, 'cm_C': cm_C,
        'huf_R': huf_R, 'huf_C': huf_C,
    }, pos - offset
