# Lane-spec errata (CP-007d, 2026-09-28)

These entries override docs/lane_specs/part1.md and part2.md where they conflict. Lane sessions read this file together with the specs.

- E-1 PTQ calibration: the 128-image calibration list is fixed by AM-10 and is not selected on VAL. Part 2 lane 6's phrase "one calibration configuration selected on VAL" is wrong; AM-4a item 3 governs the Chapter 4 disclosure.
- E-2 MDE guard: AM-17 item 3(c) now includes the three-point Monte Carlo guard (CP-007d D5); the spec and the amendment agree.
- E-3 Teacher–student gap: 2.26 pp is the gap against E1's GPU comparator (AM-17 item 1(b)); 2.27 pp is the gap on the CPU pairing used by L-AM17B-GAP. Both are correct in their context.
- E-4 QNNPACK Sigmoid: support was measured under DL-18 (d44cce9), and the cloud smoke passed smoke_qnnpack_head and smoke_qnnpack_full_student. Lane 8's operator census runs as a regression check only; it does not reopen the question.
- E-5 Hook-safe commands: the protected-file hook refuses git hash-object and whole-tree git archive. The frozen-blob check computes the git blob id in Python: sha1 over the ASCII bytes "blob ", the file size in decimal, one NUL byte, then the file's bytes. The result is compared with cbd5fa86…. Cross-commit exports use git archive on a commit with explicit paths only (src, configs, scripts and any other path the test needs, never docs/reference/), plus the hook's required exclude suffix.
- E-6 Governed set: AGENTS rule 8 is the definition. The B66 G0 check additionally includes Dockerfile (stricter), and lanes treat Dockerfile as governed too.
- E-7 Tests follow the repository's smoke-script convention (the spec header), so the spec's tests/… paths become scripts/smoke_*.py.
- E-8 Cloud clones contain the protected PDF; see DL-37.
