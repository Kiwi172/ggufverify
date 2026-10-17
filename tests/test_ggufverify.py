"""Tests for ggufverify. Run with: python3 -m pytest tests/ (or python3 tests/test_ggufverify.py)

Synthetic GGUF files are built byte-by-byte (metadata + tensor table +
data), so every failure mode -- truncation, overlap, duplicate names,
undersized tensors -- is proven without real models.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = str(Path(__file__).resolve().parent.parent / "ggufverify.py")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ggufverify as gv  # noqa: E402


def _w_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _w_kv_str(k, v):
    return _w_str(k) + struct.pack("<I", 8) + _w_str(v)


def _w_kv_u32(k, v):
    return _w_str(k) + struct.pack("<I", 4) + struct.pack("<I", v)


# q4_k: 144 bytes per 256 elements; f16: 2 bytes/element
def tsize(ty, dims):
    if ty not in gv.TYPES:
        return 0
    _, bs, blk = gv.TYPES[ty]
    elems = 1
    for d in dims:
        elems *= d
    return elems // blk * bs


def build(tensors, align=32, n_tensors=None, extra_tail=0):
    """tensors: list of (name, dims, type). Returns file bytes."""
    metas = [
        _w_kv_str("general.architecture", "testarch"),
        _w_kv_str("general.name", "Test Model"),
    ]
    if align != 32:
        metas.append(_w_kv_u32("general.alignment", align))
    out = b"GGUF" + struct.pack("<I", 3)
    out += struct.pack("<Q", n_tensors if n_tensors is not None
                       else len(tensors))
    out += struct.pack("<Q", len(metas))
    out += b"".join(metas)
    off = 0
    data = b""
    for name, dims, ty in tensors:
        out += _w_str(name) + struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", ty) + struct.pack("<Q", off)
        size = tsize(ty, dims)
        data += b"\x00" * size
        off += size
    out += b"\x00" * (-len(out) % align)   # pad to data start
    return out + data + b"\x00" * extra_tail


GOOD = [("token_embd.weight", [256, 4], 1),        # f16
        ("blk.0.ffn_down.weight", [512, 8], 12)]   # q4_k


def write(tmp: Path, name: str, blob: bytes) -> Path:
    p = tmp / name
    p.write_bytes(blob)
    return p


def run_cli(*paths, extra=()):
    return subprocess.run([sys.executable, TOOL, *paths, *extra],
                          capture_output=True, text=True)


# ---------------------------------------------------------------------------

def test_good_file_ok(tmp_path):
    p = write(tmp_path, "good.gguf", build(GOOD))
    verdict, issues, summary = gv.verify(p)
    assert verdict == "OK", issues
    assert summary["tensors"] == 2


def test_truncated_tail(tmp_path):
    blob = build(GOOD)[:-1000]
    p = write(tmp_path, "trunc.gguf", blob)
    verdict, issues, _ = gv.verify(p)
    assert verdict == "TRUNCATED"
    assert "1,000 bytes short" in issues[-1]


def test_undersized_middle_tensor(tmp_path):
    # move the second tensor 10 bytes too early -> gap < required
    blob = bytearray(build(GOOD))
    # find the second tensor's offset field: patch by rebuilding instead
    tensors = [GOOD[0]]
    out = build(tensors)  # not used; simpler: custom build with bad offsets
    # hand-build with overlapping offsets
    metas = [_w_kv_str("general.architecture", "a"),
             _w_kv_str("general.name", "T")]
    hdr = b"GGUF" + struct.pack("<I", 3)
    hdr += struct.pack("<Q", 2) + struct.pack("<Q", len(metas))
    hdr += b"".join(metas)
    for name, dims, ty, off in [("a.weight", [512, 8], 12, 0),
                                ("b.weight", [256, 4], 1, 100)]:
        hdr += _w_str(name) + struct.pack("<I", len(dims))
        for d in dims:
            hdr += struct.pack("<Q", d)
        hdr += struct.pack("<I", ty) + struct.pack("<Q", off)
    hdr += b"\x00" * (-len(hdr) % 32)
    p = write(tmp_path, "overlap.gguf", bytes(hdr) + b"\x00" * 4096)
    verdict, issues, _ = gv.verify(p)
    assert verdict == "CORRUPT"
    assert any("needs" in i and "only" in i for i in issues)


def test_duplicate_names(tmp_path):
    dup = [GOOD[0], GOOD[0]]
    p = write(tmp_path, "dup.gguf", build(dup))
    verdict, issues, _ = gv.verify(p)
    assert verdict == "CORRUPT"
    assert any("duplicate tensor name" in i for i in issues)


def test_extra_tail_bytes_note(tmp_path):
    p = write(tmp_path, "extra.gguf", build(GOOD, extra_tail=64))
    verdict, issues, _ = gv.verify(p)
    assert verdict == "OK"
    assert any(i.startswith("note:") and "extra bytes" in i for i in issues)


def test_unknown_type_tail(tmp_path):
    t = [GOOD[0], ("weird.weight", [256, 4], 999)]
    blob = bytearray(build(t))
    p = write(tmp_path, "unknown.gguf", bytes(blob))
    verdict, issues, _ = gv.verify(p)
    assert verdict == "OK"
    assert any("unknown to this checker" in i for i in issues)


def test_not_gguf(tmp_path):
    p = write(tmp_path, "bad.gguf", b"XXXX" + b"\x00" * 100)
    verdict, issues, _ = gv.verify(p)
    assert verdict == "CORRUPT" and "not a GGUF" in issues[0]


def test_truncated_table(tmp_path):
    # declares 5 tensors but the file ends right after the 2nd entry
    metas = [
        _w_kv_str("general.architecture", "testarch"),
        _w_kv_str("general.name", "Test Model"),
    ]
    hdr_len = 8 + 8 + 8 + sum(len(m) for m in metas)
    for name, dims, ty in GOOD:
        hdr_len += 8 + len(name.encode()) + 4 + 8 * len(dims) + 4 + 8
    blob = build(GOOD, n_tensors=5)[:hdr_len]
    p = write(tmp_path, "badtab.gguf", blob)
    verdict, issues, _ = gv.verify(p)
    assert verdict == "TRUNCATED"
    assert "tensor table truncated" in issues[0]


def test_cli_json(tmp_path):
    good = write(tmp_path, "good.gguf", build(GOOD))
    bad = write(tmp_path, "trunc.gguf", build(GOOD)[:-1000])
    r = run_cli(str(good), str(bad), extra=("--json",))
    assert r.returncode == 1
    d = json.loads(r.stdout)
    vs = {x["file"]: x["verdict"] for x in d["results"]}
    assert vs[str(good)] == "OK" and vs[str(bad)] == "TRUNCATED"


if __name__ == "__main__":
    import inspect
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                with tempfile.TemporaryDirectory() as d:
                    if inspect.signature(fn).parameters:
                        fn(Path(d))
                    else:
                        fn()
                print(f"  ok  {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
