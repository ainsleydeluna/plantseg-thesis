#!/usr/bin/env python3
"""Smoke for scripts/am7_clip_value.py (AM-7a item 5; lane L-KD-HARDEN).

Synthetic E1 telemetry only (train_e1's row schema: one run_meta row, train rows with grad_norm, val
rows), under a temp dir; nothing is written in the repository (no bytecode either) and no torch is
needed.

  S   the 1-2-5 ceiling on a table of values, exact at the series points
  V   seeds 43 and 44 -> the per-seed maxima and their first iterations, M = the larger, 1.5 x M and
      its series value; inputs recorded with sha256 and seed; the RESULT line
  X   exactness: M = 3.3333333333333335 (the logged repr of 10/3) gives 1.5 x M = 5.00000000000000025
      and the value 10, where float arithmetic would give 5.0 and 5; M = 3.333333333333333 gives 5;
      the record keeps the target as exact text only; a 0.1 value prints as 0.1, not 0.10
  R   refusals, exit 2 with the named code, nothing written, never a traceback: a file without
      grad_norm rows, a train row without grad_norm, a NaN, Infinity or -Infinity token, a grad_norm
      that overflows a float, one or three files, the same file twice, seeds 42 and 43 or 43 twice, a
      dry run, a second run_meta row, a bool seed, resumed_from set, a run resumed into a fresh
      directory (no resumed_from, first iteration 2), gapped, decreasing or float iterations, a clipped
      run, an unfinished run, a torn last line, a malformed or non-object middle line, a non-UTF-8 file,
      files with different max_iters, all-zero gradient norms, a missing file, and an existing output
      (left unchanged)
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True                   # nothing is written in the checkout, not even a .pyc

import scripts.am7_clip_value as acv  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def telemetry(path: Path, seed: int, norms, *, mode="real", max_iters=None, meta_rows=1, drop_at=None,
              raw_norm=None, torn=False, iters=None, meta_extra=None) -> Path:
    """An e1_telemetry.jsonl: run_meta (train_e1's resumed_from and grad_clip_norm null unless
    `meta_extra` says otherwise), then one train row per grad_norm (iterations 1..n), a val row at the
    end. `raw_norm` = (iteration, text) writes that grad_norm token verbatim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    iters = list(iters) if iters is not None else list(range(1, len(norms) + 1))
    meta = {"event": "run_meta", "wall_clock": 1.0, "mode": mode, "seed": seed, "num_workers": 12,
            "batch_size": 16, "max_iters": iters[-1] if max_iters is None else max_iters,
            "resumed_from": None, "grad_clip_norm": None, **(meta_extra or {})}
    lines = [json.dumps(meta)] * meta_rows
    for it, g in zip(iters, norms):
        row = {"event": "train", "iter": it, "loss": 1.0, "ce": 0.5, "dice": 0.5, "lr": 0.01,
               "grad_norm": g, "wall_clock": 1.0 + it, "iter_seconds": 0.1, "samples_per_sec": 160.0}
        if drop_at == it:
            del row["grad_norm"]
        text = json.dumps(row)
        if raw_norm is not None and raw_norm[0] == it:
            text = text.replace(f'"grad_norm": {json.dumps(g)}', f'"grad_norm": {raw_norm[1]}')
        lines.append(text)
    lines.append(json.dumps({"event": "val", "iter": iters[-1], "all_class_miou": 0.3, "val_seconds": 1.0,
                             "wall_clock": 99.0}))
    text = "\n".join(lines) + "\n"
    if torn:
        text += '{"event": "train", "it'
    path.write_text(text, encoding="utf-8")
    return path


def run_cli(argv) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = acv.main(argv)
    return rc, out.getvalue() + err.getvalue()


def result_line(log: str) -> str:
    return next((ln for ln in log.splitlines() if ln.startswith("RESULT:")), "")


def test_series() -> None:
    table = {"0.3": "0.5", "1": "1", "1.5": "2", "2": "2", "2.0000001": "5", "4.8": "5", "5": "5",
             "5.000001": "10", "7.5": "10", "10": "10", "0.0151": "0.02", "1234": "2000", "0.5": "0.5",
             "99.9": "100", "100.1": "200"}
    bad = {x: format(acv.series_ceiling(Decimal(x)), "f") for x in table
           if acv.series_ceiling(Decimal(x)) != Decimal(table[x])}
    check("S_series_ceiling_table", not bad, str(bad))
    try:
        acv.series_ceiling(Decimal(0))
        check("S_series_ceiling_needs_positive", False, "no error for 0")
    except ValueError:
        check("S_series_ceiling_needs_positive", True)


def test_value(tmp: Path) -> None:
    a = telemetry(tmp / "v" / "s43" / "e1_telemetry.jsonl", 43, [1.2, 2.1, 0.9, 2.1, 1.0])
    b = telemetry(tmp / "v" / "s44" / "e1_telemetry.jsonl", 44, [0.7, 1.1, 3.2, 0.4, 0.5])
    out = tmp / "v" / "am7_clip_value.json"
    rc, log = run_cli(["--telemetry", str(b), str(a), "--out", str(out)])
    doc = json.loads(out.read_text()) if out.exists() else {}
    check("V_exit0_and_written", rc == 0 and bool(doc), log.strip()[-200:])
    check("V_per_seed_max_and_first_iteration",
          doc.get("per_seed", {}).get("43", {}).get("max_grad_norm") == 2.1
          and doc["per_seed"]["43"]["iter"] == 2 and doc["per_seed"]["44"]["max_grad_norm"] == 3.2
          and doc["per_seed"]["44"]["iter"] == 3 and doc["per_seed"]["43"]["n_train_rows"] == 5,
          json.dumps(doc.get("per_seed")))
    check("V_max_factor_target_series",
          doc.get("max_grad_norm") == 3.2 and doc.get("max_seed") == 44 and doc.get("factor") == 1.5
          and doc.get("target_text") == "4.80" and doc.get("series_value") == 5.0
          and doc.get("series_value_text") == "5", json.dumps({k: doc.get(k) for k in
                                                              ("max_grad_norm", "target_text",
                                                               "series_value_text")}))
    check("V_inputs_with_sha256_and_seed",
          [i.get("seed") for i in doc.get("inputs", [])] == [43, 44]
          and [i.get("sha256") for i in doc.get("inputs", [])] == [acv.sha256_file(a), acv.sha256_file(b)])
    check("V_result_line", result_line(log).startswith("RESULT: AM7_CLIP_VALUE = 5 (1.5 x 3.2 [seed 44] = "
                                                       "4.80; 1-2-5 series)"), result_line(log))


def test_exactness(tmp: Path) -> None:
    for label, norm, want in (("above", 3.3333333333333335, "10"), ("below", 3.333333333333333, "5")):
        a = telemetry(tmp / f"x_{label}" / "s43.jsonl", 43, [1.0, norm])
        b = telemetry(tmp / f"x_{label}" / "s44.jsonl", 44, [1.0, 2.0])
        out = tmp / f"x_{label}" / "out.json"
        rc, log = run_cli(["--telemetry", str(a), str(b), "--out", str(out)])
        doc = json.loads(out.read_text()) if out.exists() else {}
        check(f"X_exact_decimal_{label}", rc == 0 and doc.get("series_value_text") == want
              and Decimal(doc.get("target_text", "0")) == Decimal("1.5") * Decimal(repr(norm))
              and "target" not in doc, f"float 1.5*M = {1.5 * norm!r}; target {doc.get('target_text')} "
                                       f"value {doc.get('series_value_text')}")
    # the 10 x 10^e candidate at a negative exponent prints without a trailing zero (0.1, not 0.10)
    a = telemetry(tmp / "x_small" / "s43.jsonl", 43, [0.01, 0.06])
    b = telemetry(tmp / "x_small" / "s44.jsonl", 44, [0.02, 0.03])
    out = tmp / "x_small" / "out.json"
    rc, log = run_cli(["--telemetry", str(a), str(b), "--out", str(out)])
    doc = json.loads(out.read_text()) if out.exists() else {}
    check("X_series_text_normalized", rc == 0 and doc.get("series_value_text") == "0.1"
          and doc.get("series_value") == 0.1, f"rc={rc} {doc.get('series_value_text')!r}")


def test_refusals(tmp: Path) -> None:
    good43 = telemetry(tmp / "r" / "good43.jsonl", 43, [1.0, 2.0, 3.0])
    good44 = telemetry(tmp / "r" / "good44.jsonl", 44, [1.0, 2.0, 3.0])
    r = tmp / "r"
    nog = r / "a.jsonl"                  # train rows that never carry grad_norm
    nog.write_text("\n".join(json.dumps(x) for x in (
        {"event": "run_meta", "mode": "real", "seed": 43, "max_iters": 2},
        {"event": "train", "iter": 1, "loss": 1.0}, {"event": "train", "iter": 2, "loss": 1.0})) + "\n",
        encoding="utf-8")
    cases = {
        "no_grad_norm_rows": ([nog], "no_grad_norm"),
        "train_row_without_grad_norm": ([telemetry(r / "b.jsonl", 43, [1.0, 2.0, 3.0], drop_at=2)],
                                        "bad_grad_norm"),
        "nan_token": ([telemetry(r / "c.jsonl", 43, [1.0, 2.0], raw_norm=(2, "NaN"))], "nonfinite"),
        "infinity_token": ([telemetry(r / "d.jsonl", 43, [1.0, 2.0], raw_norm=(1, "Infinity"))], "nonfinite"),
        "negative_grad_norm": ([telemetry(r / "e.jsonl", 43, [1.0, -2.0])], "bad_grad_norm"),
        "dry_run": ([telemetry(r / "f.jsonl", 43, [1.0, 2.0], mode="dry")], "not_real"),
        "two_run_meta_rows": ([telemetry(r / "g.jsonl", 43, [1.0, 2.0], meta_rows=2)], "run_meta_rows"),
        "unfinished": ([telemetry(r / "h.jsonl", 43, [1.0, 2.0], max_iters=80000)], "run_unfinished"),
        "torn_last_line": ([telemetry(r / "i.jsonl", 43, [1.0, 2.0], torn=True)], "run_unfinished"),
        "iterations_not_increasing": ([telemetry(r / "j.jsonl", 43, [1.0, 2.0, 3.0], iters=[1, 3, 2],
                                                 max_iters=2)], "iterations"),
        "missing_file": ([r / "absent.jsonl"], "input_missing"),
        "resumed_from_set": ([telemetry(r / "k.jsonl", 43, [1.0, 2.0, 3.0],
                                        meta_extra={"resumed_from": "/w/e1/last.pt"})], "resumed"),
        "resumed_into_fresh_dir": ([telemetry(r / "l.jsonl", 43, [2.0, 3.0], iters=[2, 3])], "iterations"),
        "gapped_iterations": ([telemetry(r / "m.jsonl", 43, [1.0, 3.0], iters=[1, 3])], "iterations"),
        "clipped_run": ([telemetry(r / "n.jsonl", 43, [1.0, 2.0, 3.0], meta_extra={"grad_clip_norm": 1.0})],
                        "clipped"),
        "minus_infinity_token": ([telemetry(r / "o.jsonl", 43, [1.0, 2.0], raw_norm=(1, "-Infinity"))],
                                 "nonfinite"),
        "grad_norm_overflows_float": ([telemetry(r / "p.jsonl", 43, [1.0, 2.0], raw_norm=(2, "1e400"))],
                                      "bad_grad_norm"),
        "bool_seed": ([telemetry(r / "q.jsonl", True, [1.0, 2.0])], "run_meta_rows"),
        "float_iteration": ([telemetry(r / "t.jsonl", 43, [1.0, 2.0], iters=[1, 2.0], max_iters=2)],
                            "iterations"),
    }
    bad = r / "u.jsonl"                  # a malformed line that is not the last one
    good_lines = (r / "good43.jsonl").read_text(encoding="utf-8").splitlines()
    bad.write_text("\n".join([good_lines[0], '{"event": "train", "iter": 1, "grad', *good_lines[1:]]) + "\n",
                   encoding="utf-8")
    nonobj = r / "v.jsonl"               # a JSON value that is not an object
    nonobj.write_text("\n".join([good_lines[0], "[1, 2]", *good_lines[1:]]) + "\n", encoding="utf-8")
    binary = r / "w.jsonl"               # not UTF-8
    binary.write_bytes(b'{"event": "run_meta"}\n\xff\xfe\n')
    cases.update({"malformed_middle_line": ([bad], "unreadable_input"),
                  "non_object_line": ([nonobj], "unreadable_input"),
                  "not_utf8": ([binary], "unreadable_input")})
    for label, (files, code) in cases.items():
        out = tmp / f"r_{label}.json"
        rc, log = run_cli(["--telemetry", str(files[0]), str(good44), "--out", str(out)])
        check(f"R_{label}_refused", rc == 2 and f"[{code}]" in log and not out.exists()
              and "Traceback" not in log, f"rc={rc} {log.strip()[-200:]}")
    for label, argv, code in (
            ("one_file", [str(good43)], "input_count"),
            ("three_files", [str(good43), str(good44), str(good44)], "input_count"),
            ("same_file_twice", [str(good43), str(good43)], "duplicate_input"),
            ("seeds_42_43", [str(telemetry(r / "s42.jsonl", 42, [1.0, 2.0])), str(good43)], "seeds"),
            ("seeds_43_43", [str(good43), str(telemetry(r / "s43b.jsonl", 43, [1.0, 2.0]))], "seeds"),
            ("horizon_mismatch", [str(good43), str(telemetry(r / "s44short.jsonl", 44, [1.0, 2.0]))],
             "horizon_mismatch"),
            ("all_zero_grad_norms", [str(telemetry(r / "z43.jsonl", 43, [0.0, 0.0, 0.0])),
                                     str(telemetry(r / "z44.jsonl", 44, [0.0, 0.0, 0.0]))], "bad_grad_norm")):
        out = tmp / f"r_{label}.json"
        rc, log = run_cli(["--telemetry", *argv, "--out", str(out)])
        check(f"R_{label}_refused", rc == 2 and f"[{code}]" in log and not out.exists()
              and "Traceback" not in log, f"rc={rc} {log.strip()[-200:]}")
    out = tmp / "r_exists.json"
    out.write_text('{"keep": true}\n', encoding="utf-8")
    rc, log = run_cli(["--telemetry", str(good43), str(good44), "--out", str(out)])
    check("R_existing_output_never_overwritten", rc == 2 and "[output_exists]" in log
          and out.read_text() == '{"keep": true}\n', f"rc={rc} {log.strip()[-160:]}")


def main() -> int:
    print("=" * 78)
    print("AM-7a CLIP VALUE SMOKE — synthetic E1 telemetry only; nothing written in the repository")
    print("=" * 78)
    tmp = Path(tempfile.mkdtemp(prefix="kdh_clip_"))
    test_series()
    for fn in (test_value, test_exactness, test_refusals):
        fn(tmp)
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:44}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail and not ok else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
