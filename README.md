# ggufverify

**Is this GGUF file complete and internally consistent?**

A 20 GB download that died 40 MB short. A copy interrupted by a full
disk. A file mangled by a sync tool. `llama-server` answers with one
cryptic error line; `ggufverify` answers with the actual problem in
under a second — it reads only the header and the tensor table, never
the model data.

```
ggufpick (pick) → ggufcheck (fit) → ggufverify (integrity) → ggufdiff (compare)
```

## Install

Single file, Python 3.9+, zero dependencies:

```bash
curl -O https://raw.githubusercontent.com/Kiwi172/ggufverify/main/ggufverify.py
python3 ggufverify.py --help
```

## Example

```
$ ggufverify ~/models/*.gguf "$TMPDIR/trunc.gguf"
OK           13.8 GB  Huihui-gpt-oss-20b-mxfp4-abliterated.gguf
                      Huihui Gpt Oss 20b Mxfp4 Abliterated v2 · gpt-oss · 459 tensors
OK           21.2 GB  Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M.gguf
                      Huihui Qwen3.6 35B A3B Abliterated · qwen35moe · 733 tensors
TRUNCATED     1.1 MB  trunc.gguf
                      - file is 100,000 bytes short (last tensor blk.4.ffn_norm.weight
                        needs 256 bytes, ends at 1,185,376, file is 1,085,376)
```

Exit code is 0 only when every file is OK — scriptable in a download
pipeline:

```bash
ggufpick unsloth/some-model-GGUF --vram 12 --get best --out ~/models \
  && ggufverify ~/models/some-best.gguf
```

## What it checks

- magic and GGUF version
- declared tensor count vs. what is actually parseable (truncated table)
- duplicate tensor names, zero/missing dimensions
- overlapping or misaligned tensor regions
- **per-tensor size cross-check**: the gap to the next tensor must be at
  least what the quant format requires (a 256-element super-block table
  for k-quants/i-quants, verified against real files)
- **truncation**: the last tensor's data must fit inside the file — it
  reports exactly how many bytes are missing
- tensors *larger* than the format minimum are noted, not failed —
  custom re-quantizers (imatrix variants, "UD"/"i1" style files) add
  per-block data, and that is not corruption

Unknown quant types skip the size cross-check but still get
overlap/truncation checks via offset gaps.

## Flags

| Flag | Meaning |
|---|---|
| `--json` | machine-readable output |
| multiple files | check a whole directory at once |

## How it works

GGUF stores a tensor table: every tensor's name, dimensions, quant
format, and byte offset into the data section. That table is a set of
mutual promises — each tensor's region must fit between its offset and
the next one's, and the last one must fit inside the file. Checking
those promises needs only the first few hundred KB of the file, so a
20 GB model verifies in milliseconds.

## Privacy

No network. The file is opened, the header is read, the verdict is
printed. Nothing else.

## Tests

```bash
python3 tests/test_ggufverify.py      # standalone
python3 -m pytest tests/              # or pytest
```

Synthetic GGUFs are built byte-by-byte (metadata + tensor table +
data), so every failure mode — truncation, overlap, duplicate names,
undersized tensors, unknown types — is proven without real models.

## License

MIT
