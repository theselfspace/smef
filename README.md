# SMEF — blog companion code

This codebase has the
eight files that the two-part blog series ("SMEF: Chasing Pocket-Sized Language Models Through the Guts of GGUF" [Link: https://medium.com/@theself.space/smef-chasing-pocket-sized-language-models-through-the-guts-of-gguf-a7954be9d094 /
"SMEF: Building a Four-Pass Weight Compressor - and Why "Lossless" Was the Wrong Lever" [Link: https://medium.com/@theself.space/smef-building-a-four-pass-weight-compressor-and-why-lossless-was-the-wrong-lever-90f5c759f562] ) actually talks about. It's
meant to be read alongside the posts, not as the full research project.

It is a real, working package — pure Python + numpy/scipy, no PyTorch, no
CUDA — and it round-trips losslessly. Two dependencies, listed in
`requirements.txt` (numpy + scipy) — install those and run the example below
right now.

## What's here, and which part of the blog it maps to

| File | Blog reference | What it does |
|---|---|---|
| `smef/format.py` | Part 2, point 4 ("the container") | The `.smef` binary layout: header, magic number, flags, seek table. |
| `smef/pass1_dict.py` | Part 1, point 3 ("the manifold hypothesis") · Part 2's randomized-SVD bug story | **Pass 1** — builds a shared dictionary of weight "atoms" via SVD + k-means, projects each layer onto it (`W = D·C + R`). |
| `smef/pass2_delta.py` | Part 1, point 2 ("cross-layer delta encoding") | **Pass 2** — groups similar layers and stores later ones as diffs from the first (I-frame/P-frame style). |
| `smef/pass3_streams.py` | Part 1, point 3 ("the float bit-field insight") | **Pass 3** — splits each tensor into separate sign / exponent / mantissa byte streams. |
| `smef/pass4_entropy.py` | Part 2, point 4 | **Pass 4** — entropy-codes each stream; selects between the LZMA and rANS backends. |
| `smef/pass4_rans.py` | Part 2's rANS ordering-bug story | The vectorized rANS (range Asymmetric Numeral Systems) coder — the fast alternative to LZMA. |
| `smef/encoder.py` | Part 2, point 4–5 | `SmefEncoder` — orchestrates all four passes, including the adaptive per-tensor Pass-1 fallback that the benchmarks depend on. |
| `smef/decoder.py` | Part 2, point 4 | `SmefDecoder` — full and streaming decode; this is what the lossless / inference-equivalence claims are checked against. |

**Deliberately left out** (not discussed in the blog, and would only add
noise here): the SMEF-native training regularizers (DPT/LSR/EAP), the
PyTorch/GPT integration, the experimental INT8 quantized pipeline, and the
MiniTransformer toy model.

## Try it yourself

```bash
pip install -r requirements.txt
```

```python
import numpy as np
from smef import SmefEncoder, SmefDecoder

# Any dict of {name: numpy array} works — this stands in for a model's weights.
weights = {
    "layer0.attn.weight": np.random.randn(768, 2304).astype(np.float32) * 0.02,
    "layer0.mlp.weight":  np.random.randn(768, 3072).astype(np.float32) * 0.02,
}

SmefEncoder(num_atoms=64, atom_rank=128).encode(weights, "model.smef")

state = SmefDecoder("model.smef").decode_all()
assert all(np.allclose(state[k], weights[k], atol=1e-5) for k in weights)
print("round-trip OK — bit-exact within floating-point tolerance")
```

Running this prints the same kind of encode log the blog describes —
including the adaptive Pass 1 check deciding, tensor by tensor, whether the
dictionary is worth keeping. On random (untrained) weights like this example,
the compression ratio will be modest by design — the blog's real numbers
(1.185× lossless, 2.38× with FP16) come from running this same pipeline
against actual trained GPT-2 weights, not random ones.

## Reading order

If you're going through the blog and want to follow along in code:

1. Start with `pass3_streams.py` — it's the shortest, and it's the one
   technique the blog says "genuinely works." `split_streams()` /
   `merge_streams()` are the whole idea.
2. Then `pass1_dict.py` — `DictionaryEncoder.fit()` builds the dictionary,
   `encode_tensor()` / `decode_tensor()` do the `W = D·C + R` projection.
3. `pass2_delta.py` is the smallest conceptual leap from there.
4. `pass4_entropy.py` and `pass4_rans.py` together are the final squeeze —
   read `pass4_rans.py` last, it's the most intricate (and the one with the
   ordering bug the blog tells the story of).
5. `encoder.py` and `decoder.py` tie all four passes together; `encoder.py`'s
   `_quick_compressed_size()` is the adaptive Pass-1 fallback mentioned in
   the benchmarks section.
