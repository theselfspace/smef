"""
smef/format.py
==============
Binary format constants, header layout, and shared data structures for SMEF.

Binary layout (single file):
  [SMEF_HEADER   128 bytes]
  [METADATA_BLOCK  variable, lzma-compressed JSON]
  [DICTIONARY      variable, shared weight atoms]
  [LAYER_GROUP_INDEX  variable]
  [LAYER_DATA      variable, per-layer streams]
  [LAYER_SEEK_TABLE  8 bytes × num_layers]
"""

import struct
import dataclasses
from typing import Optional, List, Dict, Any
import numpy as np

# ── Magic & version ──────────────────────────────────────────────────────────
SMEF_MAGIC   = 0x534D4546   # b"SMEF"
SMEF_VERSION = 0x0001

# ── Precision enum ───────────────────────────────────────────────────────────
PREC_FP32 = 0
PREC_BF16 = 1
PREC_FP16 = 2
PREC_FP8  = 3   # stored as uint8 with E4M3 layout

PREC_NAMES = {PREC_FP32: "fp32", PREC_BF16: "bf16",
              PREC_FP16: "fp16", PREC_FP8:  "fp8"}

PREC_NUMPY = {
    PREC_FP32: np.float32,
    PREC_BF16: np.uint16,    # numpy has no native bf16; we store as uint16
    PREC_FP16: np.float16,
    PREC_FP8:  np.uint8,
}

# ── Compression flags ────────────────────────────────────────────────────────
FLAG_PASS1_DICT   = 0x01   # structural dictionary factorisation
FLAG_PASS2_DELTA  = 0x02   # cross-layer delta encoding
FLAG_PASS3_SPLIT  = 0x04   # byte-stream exponent/mantissa separation
FLAG_PASS4_ANS    = 0x08   # context-adaptive entropy coding (lzma proxy)
FLAG_NATIVE_TRAIN = 0x10   # model was SMEF-native trained

FLAGS_ALL = FLAG_PASS1_DICT | FLAG_PASS2_DELTA | FLAG_PASS3_SPLIT | FLAG_PASS4_ANS

# ── Header struct  (128 bytes, little-endian) ────────────────────────────────
# Format string: magic(I) version(H) arch(H) num_layers(I) num_atoms(I)
#                atom_rank(I) orig_prec(H) smef_prec(H) flags(I)
#                group_count(I) meta_offset(Q) dict_offset(Q)
#                layers_offset(Q) seek_offset(Q) checksum(Q)
#                reserved(18s)
HEADER_FMT    = "<IHHIIIHHIIQQQQq18s"
HEADER_SIZE   = struct.calcsize(HEADER_FMT)   # should be 88; pad to 128
assert HEADER_SIZE <= 128, f"header too large: {HEADER_SIZE}"
HEADER_PADDED = 128


@dataclasses.dataclass
class SmefHeader:
    magic:          int = SMEF_MAGIC
    version:        int = SMEF_VERSION
    arch_type:      int = 0           # 0 = transformer, 1 = cnn
    num_layers:     int = 0
    num_atoms:      int = 0
    atom_rank:      int = 0
    orig_prec:      int = PREC_BF16
    smef_prec:      int = PREC_BF16
    flags:          int = FLAGS_ALL
    group_count:    int = 0
    meta_offset:    int = 0
    dict_offset:    int = 0
    layers_offset:  int = 0
    seek_offset:    int = 0
    checksum:       int = 0           # xxhash-like; we use a 64-bit CRC

    def to_bytes(self) -> bytes:
        core = struct.pack(
            HEADER_FMT,
            self.magic, self.version, self.arch_type,
            self.num_layers, self.num_atoms, self.atom_rank,
            self.orig_prec, self.smef_prec, self.flags,
            self.group_count,
            self.meta_offset, self.dict_offset,
            self.layers_offset, self.seek_offset,
            self.checksum,
            b'\x00' * 18,
        )
        return core + b'\x00' * (HEADER_PADDED - len(core))

    @classmethod
    def from_bytes(cls, data: bytes) -> "SmefHeader":
        assert len(data) >= HEADER_SIZE
        fields = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
        return cls(
            magic=fields[0], version=fields[1], arch_type=fields[2],
            num_layers=fields[3], num_atoms=fields[4], atom_rank=fields[5],
            orig_prec=fields[6], smef_prec=fields[7], flags=fields[8],
            group_count=fields[9],
            meta_offset=fields[10], dict_offset=fields[11],
            layers_offset=fields[12], seek_offset=fields[13],
            checksum=fields[14],
        )

    def validate(self):
        if self.magic != SMEF_MAGIC:
            raise ValueError(f"Bad magic: 0x{self.magic:08X}")
        if self.version != SMEF_VERSION:
            raise ValueError(f"Unsupported version: {self.version}")


# ── Layer group index entry ──────────────────────────────────────────────────
# Each group has an I-frame layer; remaining layers in the group are P-frames.
GROUP_ENTRY_FMT  = "<IIQ"   # iframe_layer_idx(I) p_frame_count(I) iframe_offset(Q)
GROUP_ENTRY_SIZE = struct.calcsize(GROUP_ENTRY_FMT)

@dataclasses.dataclass
class GroupEntry:
    iframe_layer_idx: int
    p_frame_count:    int
    iframe_offset:    int   # byte offset of I-frame layer data

    def to_bytes(self) -> bytes:
        return struct.pack(GROUP_ENTRY_FMT,
                           self.iframe_layer_idx,
                           self.p_frame_count,
                           self.iframe_offset)

    @classmethod
    def from_bytes(cls, data: bytes) -> "GroupEntry":
        f = struct.unpack(GROUP_ENTRY_FMT, data[:GROUP_ENTRY_SIZE])
        return cls(*f)


# ── Per-layer frame header ────────────────────────────────────────────────────
# Stored before the three byte-streams (sign / exponent / mantissa).
FRAME_HDR_FMT  = "<BIQQQ"   # is_iframe(B) reserved(I) sz_sign(Q) sz_exp(Q) sz_mant(Q)
FRAME_HDR_SIZE = struct.calcsize(FRAME_HDR_FMT)

@dataclasses.dataclass
class FrameHeader:
    is_iframe:  bool
    sz_sign:    int
    sz_exp:     int
    sz_mant:    int

    def to_bytes(self) -> bytes:
        return struct.pack(FRAME_HDR_FMT,
                           int(self.is_iframe), 0,
                           self.sz_sign, self.sz_exp, self.sz_mant)

    @classmethod
    def from_bytes(cls, data: bytes) -> "FrameHeader":
        f = struct.unpack(FRAME_HDR_FMT, data[:FRAME_HDR_SIZE])
        return cls(is_iframe=bool(f[0]),
                   sz_sign=f[2], sz_exp=f[3], sz_mant=f[4])


# ── Tensor name registry  ─────────────────────────────────────────────────────
# Maps human-readable tensor names to integer IDs for compact encoding.
class TensorRegistry:
    def __init__(self):
        self._name_to_id: Dict[str, int] = {}
        self._id_to_name: Dict[int, str] = {}

    def register(self, name: str) -> int:
        if name not in self._name_to_id:
            idx = len(self._name_to_id)
            self._name_to_id[name] = idx
            self._id_to_name[idx] = name
        return self._name_to_id[name]

    def name(self, idx: int) -> str:
        return self._id_to_name[idx]

    def to_dict(self) -> Dict[str, int]:
        return dict(self._name_to_id)

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "TensorRegistry":
        r = cls()
        r._name_to_id = d
        r._id_to_name = {v: k for k, v in d.items()}
        return r
