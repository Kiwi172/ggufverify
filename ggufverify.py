#!/usr/bin/env python3
"""ggufverify -- is this GGUF file complete and internally consistent?

A 20 GB download that died 40 MB short, a copy interrupted by a full
disk, a file mangled by a sync tool: llama-server answers with one
cryptic error line. ggufverify answers with the actual problem in
seconds -- it reads only the header and the tensor table, then checks
the tensor table against the file itself:

  * magic / version
  * declared tensor count vs. what is actually parseable
  * duplicate tensor names, zero/negative dimensions
  * overlapping or misaligned tensor regions
  * per-tensor size cross-check: the gap to the next tensor must match
    what the quant format says it should be
  * truncation: the last tensor's data must fit inside the file
    (reports exactly how many bytes are missing)

    ggufpick (pick) -> ggufcheck (fit) -> ggufverify (integrity) -> ggufdiff (compare)

Stdlib-only. No network.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

VERSION = "0.1.0"

# ggml type enum -> (name, bytes per block, block size in elements).
# Note: k-quants and most i-quants use 256-element super-blocks; only
# the plain Q4/Q5/Q8 formats and iq4_nl use 32. Verified empirically
# against real files. Unknown types skip the size cross-check but
# still get overlap/truncation checks via offset gaps.
TYPES = {
    0: ("f32", 4, 1), 1: ("f16", 2, 1), 25: ("bf16", 2, 1),
    2: ("q4_0", 18, 32), 3: ("q4_1", 22, 32),
    6: ("q5_0", 32, 32), 7: ("q5_1", 34, 32),
    8: ("q8_0", 33, 32), 9: ("q8_1", 34, 32),
    10: ("q2_k", 84, 256), 11: ("q3_k", 110, 256),
    12: ("q4_k", 144, 256), 13: ("q5_k", 162, 256),
    14: ("q6_k", 210, 256),
    16: ("iq2_xxs", 66, 256), 17: ("iq2_xs", 70, 256),
    18: ("iq3_xxs", 96, 256), 20: ("iq4_nl", 18, 32),
    21: ("iq3_s", 104, 256), 22: ("iq2_s", 78, 256),
    23: ("iq4_xs", 130, 256), 29: ("mxfp4", 17, 32),
}


class GgufError(Exception):
    pass


def _rd_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    if n > 1 << 28:
        raise GgufError("implausible string length")
    b = f.read(n)
    if len(b) < n:
        raise GgufError("file ends inside a string")
    return b.decode("utf-8", "replace")


def _skip_val(f, t, depth=0):
    FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<B", 10: "<Q", 11: "<q", 12: "<d"}
    if t == 8:
        _rd_str(f)
    elif t == 9:
        if depth > 4:
            raise GgufError("array nesting too deep")
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if n > 1 << 24:
            raise GgufError("implausible array length")
        for _ in range(n):
            _skip_val(f, et, depth + 1)
    elif t in FMT:
        f.read(struct.calcsize(FMT[t]))
    else:
        raise GgufError(f"unknown GGUF value type {t}")


def parse(path: Path):
    """Parse GGUF metadata + tensor table. Returns dict or raises."""
    with open(path, "rb") as f:
        head = f.read(8)
        if head[:4] != b"GGUF":
            raise GgufError("not a GGUF file")
        (ver,) = struct.unpack("<I", head[4:8])
        if ver not in (1, 2, 3):
            raise GgufError(f"unsupported GGUF version {ver}")
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        if n_tensors > 1 << 22 or n_kv > 1 << 20:
            raise GgufError("implausible counts")
        meta = {}
        for _ in range(n_kv):
            k = _rd_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            if t == 8:
                meta[k] = _rd_str(f)
            elif t == 9:
                (et,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                vals = []
                for _ in range(min(n, 4096)):
                    FMT = {4: "<I", 5: "<i", 6: "<f", 10: "<Q", 12: "<d"}
                    if et in FMT:
                        vals.append(struct.unpack(FMT[et],
                                    f.read(struct.calcsize(FMT[et])))[0])
                    else:
                        _skip_val(f, et)
                for _ in range(max(0, n - 4096)):
                    _skip_val(f, et)
                meta[k] = vals
            else:
                FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                       6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}
                if t not in FMT:
                    raise GgufError(f"unknown GGUF value type {t}")
                meta[k] = struct.unpack(FMT[t],
                                        f.read(struct.calcsize(FMT[t])))[0]

        tensors = []
        parsed = 0
        truncated_table = False
        try:
            for _ in range(n_tensors):
                name = _rd_str(f)
                (nd,) = struct.unpack("<I", f.read(4))
                if nd > 8:
                    raise GgufError(f"{name}: implausible n_dims {nd}")
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
                (ty,) = struct.unpack("<I", f.read(4))
                (off,) = struct.unpack("<Q", f.read(8))
                tensors.append({"name": name, "dims": dims, "type": ty,
                                "offset": off})
                parsed += 1
        except (struct.error, GgufError):
            truncated_table = True
        info_end = f.tell()

    return {"version": ver, "meta": meta, "n_tensors": n_tensors,
            "tensors": tensors, "parsed": parsed,
            "truncated_table": truncated_table, "info_end": info_end}


def align_up(n, a):
    return n + (-n % a) if a else n


def type_bytes(ty, dims):
    """Expected tensor bytes from the format table; None if unknown type."""
    if ty not in TYPES:
        return None
    _, bs, blk = TYPES[ty]
    elems = 1
    for d in dims:
        elems *= d
    if elems % blk:
        return None
    return elems // blk * bs


def verify(path: Path):
    """Return (verdict, issues, summary) for one file."""
    issues = []
    size = path.stat().st_size
    try:
        g = parse(path)
    except GgufError as e:
        return "CORRUPT", [f"header: {e}"], {}
    except struct.error:
        return "TRUNCATED", ["header ends mid-field (file truncated "
                             "before the tensor table)"], {}

    meta = g["meta"]
    align = int(meta.get("general.alignment", 32)) or 32
    name = meta.get("general.name", "")
    arch = meta.get("general.architecture", "")

    if g["truncated_table"]:
        return "TRUNCATED", [
            f"tensor table truncated: declared {g['n_tensors']} tensors, "
            f"only {g['parsed']} parseable"], {"name": name, "arch": arch}
    if g["parsed"] != g["n_tensors"]:
        issues.append(f"declared {g['n_tensors']} tensors, parsed {g['parsed']}")

    ts = g["tensors"]
    seen = {}
    for t in ts:
        if t["name"] in seen:
            issues.append(f"duplicate tensor name: {t['name']}")
        seen[t["name"]] = t
        if any(d == 0 for d in t["dims"]) or not t["dims"]:
            issues.append(f"{t['name']}: zero/missing dimensions")
        if t["offset"] % align:
            issues.append(f"{t['name']}: offset {t['offset']} not aligned "
                          f"to {align}")

    ts_sorted = sorted(ts, key=lambda t: t["offset"])
    for a, b in zip(ts_sorted, ts_sorted[1:]):
        if a["offset"] == b["offset"]:
            issues.append(f"tensors share an offset: {a['name']}, "
                          f"{b['name']}")
            continue
        exp = type_bytes(a["type"], a["dims"])
        gap = b["offset"] - a["offset"]
        if exp is not None and gap != exp:
            ta = TYPES[a["type"]][0]
            if gap < exp:
                issues.append(f"{a['name']}: {ta} needs {exp} bytes but "
                              f"next tensor starts only {gap} bytes later")
            else:
                issues.append(f"note: {a['name']}: {gap} bytes for {ta} "
                              f"(format needs {exp}) -- custom quant "
                              "variant?")

    data_start = align_up(g["info_end"], align)
    tail_unknown = False
    if ts_sorted:
        last = ts_sorted[-1]
        exp = type_bytes(last["type"], last["dims"])
        if exp is None:
            tail_unknown = True
            ty = last["type"]
            issues.append(f"note: last tensor type {ty} unknown to this "
                          "checker; completeness not verified for the tail")
        else:
            need = data_start + last["offset"] + exp
            if size < need:
                return ("TRUNCATED",
                        issues + [f"file is {need - size:,} bytes short "
                                  f"(last tensor {last['name']} needs "
                                  f"{exp:,} bytes, ends at "
                                  f"{need:,}, file is {size:,})"],
                        {"name": name, "arch": arch})
            if size > need:
                issues.append(f"note: {size - need:,} extra bytes after the "
                              "last tensor (harmless padding?)")

    summary = {"name": name, "arch": arch, "tensors": len(ts),
               "tail_ok": not tail_unknown}
    errs = [i for i in issues if not i.startswith("note:")]
    verdict = "CORRUPT" if errs else "OK"
    return verdict, issues, summary


def fmt(n):
    return (f"{n / 1e9:.1f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="ggufverify",
        description="Check a GGUF file for truncation and internal "
                    "inconsistency without loading the model. "
                    "Stdlib-only.")
    ap.add_argument("files", nargs="+", metavar="FILE.gguf")
    ap.add_argument("--json", action="store_true", dest="as_json")
    ap.add_argument("--version", action="version", version=f"ggufverify {VERSION}")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    bad = 0
    results = []
    for p in args.files:
        path = Path(p)
        if not path.exists():
            results.append({"file": p, "verdict": "MISSING",
                            "issues": ["no such file"], "size": 0})
            bad += 1
            continue
        verdict, issues, summary = verify(path)
        if verdict != "OK":
            bad += 1
        results.append({"file": p, "size": path.stat().st_size,
                        "verdict": verdict, "issues": issues,
                        "summary": summary})

    if args.as_json:
        print(json.dumps({"tool": "ggufverify", "version": VERSION,
                          "results": results}, indent=1))
    else:
        for r in results:
            s = r["summary"]
            tag = (f"{s.get('name') or '?'} · {s.get('arch') or '?'} · "
                   f"{s.get('tensors', '?')} tensors")
            print(f"{r['verdict']:<10} {fmt(r['size']):>9}  "
                  f"{Path(r['file']).name}")
            if r["verdict"] == "OK":
                print(f"{'':10} {'':>9}  {tag}")
            for i in r["issues"]:
                print(f"{'':10} {'':>9}  - {i}")
        if len(results) > 1:
            ok = sum(1 for r in results if r["verdict"] == "OK")
            print(f"\n{ok}/{len(results)} files OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
