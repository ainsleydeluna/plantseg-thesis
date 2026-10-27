# Lane report: TMAN-B3A (TEST manifest, B-3a)

## Part 1

Session TMAN-B3A, 27 Oct 2026, under tmanb3a_go_20261027.md (35,570 B, sha256 39805366…af34efc), STEPS 0–8.
Part 1 adds new files only: the duplicate-free list, the in-session image hasher, their smokes and this report.
Nothing that exists at B is edited (H1). Labels: MEASURED (run here), REPOSITORY-PROVEN (a file:line read here),
DOCUMENTED (a document says so), INFERRED.

### 1. Base, branch and commits

- B = `ee5593e9bd1f36d23b2ea2136120ae7890ed0728` (lane/tman, "TMAN B-1 C6g4"). lane/tman-b3 was created at B in
  STEP 2 and pushed after every commit (`git push origin refs/heads/lane/tman-b3:refs/heads/lane/tman-b3`, then
  `git ls-remote`). No merge, rebase or trailer; lane/tman untouched.
- lane/tman moved on after B: 005a891 at STEP 1's fetch (C6g5 34518d0, C6h 01eb92b, C6h2 005a891), and c8587b9 by
  `git ls-remote` when this report was written (not fetched, not read here).

| Commit | What |
|---|---|
| 92b1babc8ed0718d4eab1ba74ab6881ba26c7bb6 | B3A-1: src/tman/dedup.py and smoke_tman_dedup.py |
| 4f22aeade860811054acd84d6964a8a7846b6638 | B3A-2: scripts/tday_hash_images.py and smoke_tman_hasher.py |
| 7b0eec4ba0b73c5085b84f0d46551df5af0256e0 | B3A-3: smoke_tman_b3_mutations.py (and message checks in the two smokes) |
| 542caa85ce9b312b454260724e7f464f0ddd16b0 | B3A-F1: both outputs or neither; the exit-4 path (DL-24 r1-F1, v3-M2) |
| 3d1e743ba054a9a917f4b53b8223868a42f4e9f5 | B3A-F2: step 2 refuses a shallow --repo (r1-F2) |
| 4b7195e29e0f7320dcbf9200fca7ab746e80021f | B3A-F3: the strict reader: lone surrogate; every form rule cased and mutated (r2-F1, r3-F5) |
| cf082d61d75078757405accf195dd981253f9b1e | B3A-F4: the list rows' byte count and sha1, the UTF-8 branch (r3-F6) |
| a84fdb9ae8003d75aa1fbbca5504ee401b44d13b | B3A-F5: the printed lines: escaped arrow, issue counts pinned (v1-M1, v3-M1) |
| 35453bb76d64c13a989c6b8c66febcb2145a3b79 | B3A-F6: the hasher smoke's masked and crash-only kills (r3-F1, F3, F4, F7) |
| (this commit) | the report |

The scoped `git diff --stat ee5593e9bd1f36d23b2ea2136120ae7890ed0728 35453bb` (the last code commit), MEASURED:

```
 scripts/smoke_tman_b3_mutations.py | 531 ++++++++++++++++++++++
 scripts/smoke_tman_dedup.py        | 490 ++++++++++++++++++++
 scripts/smoke_tman_hasher.py       | 897 +++++++++++++++++++++++++++++++++++++
 scripts/tday_hash_images.py        | 347 ++++++++++++++
 src/tman/dedup.py                  | 315 +++++++++++++
 5 files changed, 2580 insertions(+)
```

The report commit adds the sixth SCOPE file, docs/lane_reports/tman_b3.md. At 35453bb, sha256 of
src/tman/dedup.py is b39d8ad3dd8bd1890f98a8d108ac4cc75928e295caad4ba37490b993b6737df7, and of
scripts/tday_hash_images.py 41433b47da456355a8f705633a0d1124fe4edfb3525c50f144cf655be8b3d3cc.

### 2. Environment (STEP 0)

- The clone was shallow; `git fetch --unshallow origin` ran first (is-shallow false after). requirements.lock is
  blob 8431e533 at B, master and HEAD (76 pins, no --hash lines).
- $SCR/tmanvenv, built as B1GO STEP 3 (:69–76) with this GO's additions: Python 3.11.17 from /usr/bin/python3.11,
  pip 24.0, setuptools 79.0.1. In order: the stats stack at the lock's pins; torch's dependencies at the lock's
  pins; torch 2.1.0+cpu and torchvision 0.16.0+cpu with `--no-deps` from download.pytorch.org/whl/cpu; mmcv 2.1.0,
  mmengine 0.10.7 and mmsegmentation 1.2.2; ftfy 6.3.0; `--no-deps regex==2026.5.9`.
- mmcv: the first attempt failed in pip's isolated build environment (setuptools 84.2.1 there has no
  pkg_resources; nothing installed). With `--no-build-isolation` pip built it from its sdist with ops (wheel
  27,502,491 B, about 25 min).
- Hard gates, MEASURED: `pip check` "No broken requirements found."; torch and torchvision print
  `2.1.0+cpu 0.16.0+cpu`; numpy 1.26.4, scipy 1.11.4, statsmodels 0.14.6, patsy 1.0.2, pandas 3.0.3, pillow
  12.3.0, scikit-image 0.23.2, mmcv 2.1.0, mmengine 0.10.7, mmsegmentation 1.2.2, ftfy 6.3.0, regex 2026.5.9.

Every installed package against the lock, one line each (MEASURED):

| Against the lock | Packages |
|---|---|
| MATCH (48) | addict 2.4.0, certifi 2026.6.17, charset-normalizer 3.4.7, contourpy 1.3.3, cycler 0.12.1, filelock 3.14.0, fonttools 4.63.0, fsspec 2026.4.0, idna 3.18, ImageIO 2.37.3, Jinja2 3.1.6, kiwisolver 1.5.0, lazy-loader 0.5, markdown-it-py 4.2.0, MarkupSafe 3.0.3, matplotlib 3.11.0, mdurl 0.1.2, mmcv 2.1.0, mmengine 0.10.7, mmsegmentation 1.2.2, mpmath 1.3.0, networkx 3.6.1, numpy 1.26.4, opencv-python 4.8.1.78, packaging 24.2, pandas 3.0.3, patsy 1.0.2, pillow 12.3.0, platformdirs 4.10.0, prettytable 3.18.0, Pygments 2.20.0, pyparsing 3.3.2, python-dateutil 2.9.0.post0, PyYAML 6.0.3, regex 2026.5.9, requests 2.28.2, rich 13.4.2, scikit-image 0.23.2, scipy 1.11.4, six 1.17.0, statsmodels 0.14.6, sympy 1.14.0, termcolor 3.3.0, tifffile 2026.3.3, typing_extensions 4.15.0, urllib3 1.26.20, wcwidth 0.8.1, yapf 0.43.0 |
| DIFF (2) | torch 2.1.0+cpu (lock +cu121), torchvision 0.16.0+cpu (lock +cu121): TMAN-B1's recorded difference |
| EXTRA (3) | ftfy 6.3.0 (DL-105's approved install), pip 24.0, setuptools 79.0.1 |
| In the lock, not installed (26) | Markdown, aliyun-python-sdk-core, aliyun-python-sdk-kms, cffi, click, colorama, crcmod, cryptography, fvcore, iopath, jmespath, model-index, opendatalab, openmim, openxlab, ordered-set, oss2, portalocker, pycparser, pycryptodome, pytz, pywin32, tabulate, tqdm, tzdata, yacs |

TMAN-B1 counted 47 MATCH lines (GO :182); this venv has 48. TMAN-B1's own list is not readable from this lane, so
the extra line is not named from evidence. INFERRED: regex 2026.5.9, which TMAN-STEP4-1 ruling 2 added after
TMAN-B1's count was taken (open item 3).

### 3. Smokes: STEP 3 (at B) and STEP 6 (at 35453bb)

Each smoke alone, with the venv, `python -B`, PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH the checkout root. TMPDIR
is unset for the thirteen and a new folder under $SCR for each new smoke. All MEASURED.

| Smoke | STEP 3, at B | STEP 6, at 35453bb |
|---|---|---|
| smoke_tman_schema | RESULT: PASS (165/165) | RESULT: PASS (165/165) |
| smoke_tman_paths | RESULT: PASS (85/85) | RESULT: PASS (85/85) |
| smoke_tman_complete | RESULT: PASS (90/90) | RESULT: PASS (90/90) |
| smoke_tman_cuts | RESULT: PASS (129/129) | RESULT: PASS (129/129) |
| smoke_tman_parent | RESULT: PASS (33/33) | RESULT: PASS (33/33) |
| smoke_tman_derived | RESULT: PASS (101/101) | RESULT: PASS (101/101) |
| smoke_tman_pins | RESULT: PASS (49/49) | RESULT: PASS (49/49) |
| smoke_tman_validate | RESULT: PASS (361/361), 203 s | RESULT: PASS (361/361), 204 s (rerun alone; see slip 5) |
| smoke_tman_unlock | RESULT: PASS (85/85), 57 s | RESULT: PASS (85/85), 58 s |
| smoke_tman_noscore | RESULT: PASS (27/27) | RESULT: PASS (27/27) |
| smoke_tman_determinism | RESULT: PASS (18/18) | RESULT: PASS (18/18) |
| smoke_tman_mutations | RESULT: PASS (404/404), 355 KILLED, 0 SURVIVED, 942 s | RESULT: PASS (404/404), 355 KILLED, 0 SURVIVED, 935 s |
| smoke_frozen_blobs | FROZEN BLOBS OK (10/10) | FROZEN BLOBS OK (10/10) |
| smoke_tman_dedup | (new) | RESULT: PASS (95/95) |
| smoke_tman_hasher | (new) | RESULT: PASS (124/124) |
| smoke_tman_b3_mutations | (new) | RESULT: PASS (132/132), 107 mutations KILLED, 0 SURVIVED, 20 baselines |

Every STEP 3 count equals the GO's expected count. The new smokes' counts are fixed at their commits and changed
only for DL-24's fixes: dedup 84 (B3A-1) → 95, hasher 105 (B3A-2) → 124, mutations 99 with 76 mutations (B3A-3)
→ 132 with 107. The GO's estimates were about 35, 40 and 30 (INFERRED).

### 4. Rulings H0–H10, with file:line (REPOSITORY-PROVEN at 35453bb)

- **H0.**
  - The split is read by import, never written: scripts/tday_hash_images.py :59 holds only its import path, and
    split_of_record :187–190 imports OFFICIAL_SPLIT inside the function. Static case
    h0_split_read_by_import_inside_the_resolver; mutation split_written_as_a_literal.
  - "duplicate TEST image" is a value only: src/tman/dedup.py :52.
  - The mutations live in scripts/smoke_tman_b3_mutations.py.
  - "blob" is the sha256 of the file bytes: tday_hash_images.py :78–84, compared in pin_errors :117–120.
- **H1.** Only A lines from B: §11's name-status checks.
- **H2.**
  - (a) src/tman/dedup.py imports only `re` and B-1's schema (:31–33). It holds DEDUP_FORMAT (:36), REASONS in
    precedence order (:49), REASON_LABELS (:52), parse_train_val_list (:127–164), build_doc (:191–205) and
    doc_bytes (:265–267), read_dedup (:270–289) and hasher_values (:300–308).
  - (b) scripts/tday_hash_images.py is its own script with two modes: `--print-pin` (cmd_print_pin :279–285; reads
    no data) and `hash` (cmd_hash :288–305). The parser is at :308–321.
  - (c) and (d) are the three smokes and this report.
- **H3.** dedup.classify :167–188:
  - stems in sorted() order (:174);
  - TRAIN first, with the both-match count (:176–178), then VAL (:179–180), then groups by hash (:182);
  - each group keeps its first sorted id (:185–187).
  - Each stem takes exactly one branch, so the reasons partition the removed set.
  - Stems are os.path.splitext(name)[0] (tday_hash_images.py :204). Only SHA-256 of the file bytes is used
    (hash_images :225–239).
- **H4.**
  - Keys: equal_train, equal_val, duplicate_in_split (:49) and equal_train_and_val (:50).
  - The plan's labels are values of REASON_LABELS (:52).
  - No key holds the word (static case h10_dedup_no_word_in_identifiers_attributes_keys_fstrings_paths).
- **H5.** DOC_KEYS :57–59, built at :197–200.
  - The fingerprints are ids_sha256 :80–84, am6_tables.ids_sha256 (6c222c9 :98–100) restated, never imported.
  - doc_errors :216–262 checks the exact keys, the format, each field's form, then the invariants: counts are the
    lists' lengths, the sum is n_images minus the kept ids, the lists are disjoint and their union is the image
    set, kept is not empty and matches its fingerprint, also_counted is at most equal_train.
  - The bytes are canonical: schema.canonical_json_bytes (:38), from src/eval/artifacts.py :93–96.
  - The file is created exclusively (tday_hash_images.py :255–275).
- **H6.** Header at :39; hash_list_bytes :292–297 writes rows sorted by stem, LF, UTF-8.
- **H7.**
  - VALUE_KEYS :46 equals manifest.HASHER_VALUES :544 (static check).
  - hasher_values :300–308.
  - The hash mode prints the three values (:301–303), then the issue's pinned counts (:304; issue_counts_line
    :311–315). No id is printed.
  - main sets stdout's error handler to backslashreplace (:325–329).
- **H8.**
  - hasher_pin :93–103: script {path, commit, sha256 of its bytes}; python "3.11" (:57); parameters with
    expected_images from schema.EXPECTED_ROWS and train_val_list as TRAINVAL_HASH_LIST plus 5367/846; modules
    {src/tman/dedup.py: sha256}.
  - The pin's form is PIN_SPEC :67–74.
  - --print-pin refuses its own pin through manifest.hasher_pin_errors (:281–283).
  - Cases print_pin_passes_hasher_pin_errors_609 and print_pin_passes_the_builder_s_wrapping_861; one
    wrapping_fails_with_<key>_injected check per HASHER_VALUES key.
- **H9.** cmd_hash :288–305, each step before the next:
  1. check_paths :132–138 runs B-1's guard on the four paths and on --unlock-entry under --repo.
  2. check_records :147–183:
     - a full clone with a HEAD commit (:150–158);
     - entries.read_entry_file(…, "unlock") (:160);
     - the manifest by path, file sha256, strict JSON, format, body and appends (:161–177);
     - the pin (:179–182).
  3. split_of_record :187–190, then list_images :197–210: one os.scandir, the suffix rule, ids, the count.
  4. read_train_val :213–222, then dedup.parse_train_val_list :127–164, which checks the sha256 before parsing.
  5. hash_images :225–239: one "rb" read per image.
  6. out_folder :243–252, then write_outputs :255–275: both outputs or neither.
  - Exit codes 0/2/4: main :324–343.
- **H10.**
  - The guards come from B-1 by import (:137–138).
  - The only literals holding the word are REASON_LABELS (:52) and ISSUE_ROWS (:55–56), which cite PA :1196,
    :1199 and :1198 at B (comment :53–54).
  - Static cases bar PIL, cv2 and src.stats imports. The smokes patch split_of_record to "tday_fixture".
  - Nothing created holds the word or ":" (case nothing_created_holds_the_word_or_a_colon).

### 5. The two formats, with a fixture example of each (MEASURED, synthetic bytes)

The fixture: TRAIN {tr_a: T1, tr_c: TB}, VAL {va_a: V1, va_b: TB}, and six images. One pair is identical and equal
to nothing; one image equals a TRAIN image; one equals both a TRAIN and a VAL image; one equals a VAL image; one is
unique. The TRAIN/VAL list is written in scripts/hash_split_files.py :232–233's format (542 B, sha256
2153bee2…ea05c1a5).

tday_image_hashes.tsv (H6), 522 B, sha256 f554bdaad2d603e97f5a3aa7c44ef6b4f06cd9d18f666eb7f2fda8eda7768823
(tab-separated, LF):

```
stem	file	bytes	sha256
kx_B1	kx_B1.jpg	2	7b778e4c1d1f33c90c619ed9bda321bbc5f05cf9f131a326c57fda87359d3b0b
kx_a1	kx_a1.jpg	2	7b778e4c1d1f33c90c619ed9bda321bbc5f05cf9f131a326c57fda87359d3b0b
kx_t1	kx_t1.jpg	2	1f93603db53bfad5c92390f735d0cbb8617b4ab8214ae91c5664a3d1e9b009c8
kx_tb	kx_tb.JPG	2	791b0ac348749055f0aa77a0ba17854a031767621a242f6adabc87ac7ddef118
kx_u1	kx_u1.jpg	2	316ca0efda6296d8f2c11d1e20890d220cec4266a0c16fbbef324c004688468a
kx_v1	kx_v1.jpeg	2	dae96fc0046a5ae1864d9a66e6715da8da08240e7119816ab722261c0744d8e8
```

tday_dedup.json (tday_dedup/1, H5), 791 B of canonical JSON with no trailing LF, sha256
0006dfed80582436c29cfb5d9daf2c71d82d38ae9755b879380a1f3f8826e5d3. kx_B1 is kept over kx_a1 because "B" (0x42) sorts
before "a" (0x61), and kx_tb is removed under equal_train and also counted:

```
{"also_counted":{"equal_train_and_val":1},"format":"tday_dedup/1","hash_list_sha256":"f554bdaad2d603e97f5a3aa7c44ef6b4f06cd9d18f666eb7f2fda8eda7768823","image_ids_sha256":"29b957929452ffe5a92e1947c07d0097bfcf3bbd873d320ef4fc7a7fbf45b988","kept_ids":["kx_B1","kx_u1"],"kept_ids_sha256":"808003be89fccd28c6a125e9100546dff4f005e4f5f0881b9b8245597bc5df31","n_images":6,"reasons_partition":true,"removed_by_reason":{"duplicate_in_split":1,"equal_train":2,"equal_val":1},"removed_ids_by_reason":{"duplicate_in_split":["kx_a1"],"equal_train":["kx_t1","kx_tb"],"equal_val":["kx_v1"]},"script":{"path":"scripts/tday_hash_images.py","sha256":"41433b47da456355a8f705633a0d1124fe4edfb3525c50f144cf655be8b3d3cc"},"train_val_list_sha256":"2153bee2053947647b0bba92bb7ecb96175cc13274ce5a4125dfa932ea05c1a5"}
```

read_dedup accepts these bytes with their sha256. The printed H7 lines for this example would be hash_list_sha256
f554bdaa…7768823, dedup_list_sha256 0006dfed…8826e5d3 and counts_by_reason
`{"duplicate_in_split":1,"equal_train":2,"equal_val":1}`. The issue line follows them.

### 6. Refusals (REPOSITORY-PROVEN at 35453bb)

Printed `REFUSED [<code>] -- <detail>`, exit 2 (schema.py :67–76). "New" means no B-1 file (src/tman/*.py at B,
scripts/tday_manifest.py) holds the code.

| Code | Where | When |
|---|---|---|
| test_path (B-1's) | tday_hash_images.py :137–138, by import | a path holding the word, as given or resolved |
| repository_shallow (new) | tday_hash_images.py :155 | --repo is a shallow clone |
| record_uncommitted (B-1's) | tday_hash_images.py :158 | --repo has no HEAD commit; also from B-1's reader (:160) |
| dl_row_missing, entry_file_rewritten, entry_file_invalid (B-1's) | entries.read_entry_file via :160 | the unlock entry |
| unlock_manifest_mismatch (new) | tday_hash_images.py :163, :167, :174, :177 | manifest path, file sha256, body, appends |
| manifest_unreadable (B-1's) | tday_hash_images.py :169 | the manifest is not strict JSON |
| manifest_format (B-1's) | tday_hash_images.py :172 | not tday_manifest/1 with body and appends |
| hasher_pin_mismatch (B-1's) | tday_hash_images.py :182 | the pin is not this code's |
| images_unreadable (new) | tday_hash_images.py :203, :236 | the folder or an image cannot be read |
| image_ids_invalid (new) | tday_hash_images.py :207 (dedup.image_id_errors :95–124) | whitespace, tab/CR/LF, not UTF-8, a shared stem, a case-folded collision |
| image_count_mismatch (new) | tday_hash_images.py :209 | not the pin's expected_images |
| train_val_list_mismatch (new) | tday_hash_images.py :219; dedup.py :135 | unreadable; sha256 not the pin's (before parsing) |
| train_val_list_invalid (new) | dedup.py :140, :142, :145, :149, :157, :161 | not UTF-8, no final LF, header, rows, a row, split counts |
| dedup_invalid (new) | dedup.py :204, :276, :278, :282, :285, :288, :307 | build check; sha256; strict JSON; lone surrogate; not canonical; doc_errors; another hash list |
| output_location (B-1's) | tday_hash_images.py :248, :251 | --out inside --repo or this checkout once resolved; not a folder |
| output_exists (B-1's) | tday_hash_images.py :260, :268 | an output exists (pre-check, then "xb") |
| generator_pin_incomplete (B-1's) | tday_hash_images.py :283 | --print-pin's pin fails hasher_pin_errors |

An unexpected error is `ERROR -- <type>: <detail>`, exit 4 (:341–343). A usage error is argparse's, exit 2.

### 7. Mutations and their killing cases

smoke_tman_b3_mutations.py follows smoke_tman_mutations.py's form:
- Each mutation replaces exact texts, each occurring once, in a scratch copy of the two files under the system temp
  dir.
- A fresh interpreter imports the copies as src.tman.dedup and scripts.tday_hash_images, with B-1's modules taken
  from the checkout.
- It runs one case up to its named check. A mutation is KILLED only when that named check fails; a crash, or
  another check failing, is no kill.
- Each of the 20 killing cases first passes alone on the unmutated copy.
- A static case ties the 15 codes the two files raise to CODES. With B-1's guard's code that makes 16, each with a
  killed mutation.

107 mutations, all KILLED at 35453bb (MEASURED). Each row is file: id → its killing case:check:

- **dedup.py, the rule (H3)**:
  - precedence_swapped → case_rule:both_match_removed_under_equal_train_and_also_counted_plan577
  - last_instead_of_first → case_rule:kept_ids_are_each_group_s_first_sorted_id
  - reversed_sort → case_rule:code_point_upper_case_before_lower_case_keeps_kx_B1
  - both_match_not_counted → case_rule:both_match_removed_under_equal_train_and_also_counted_plan577
  - partition_flag_false → case_rule:reasons_partition_true_in_the_built_doc
  - hash_list_unsorted → case_canonical_and_hash_list:hash_list_header_sorted_rows_lf_utf8_H6
- **dedup.py, the invariants (H5)**, each killed by case_invariants:
  - builder_unchecked → invariant_kept_not_empty_build_refused
  - partition_flag_unchecked → invariant_partition_flag_false_reported
  - kept_may_be_empty → invariant_kept_not_empty_build_refused
  - counts_not_lengths → invariant_counts_are_the_lists_lengths
  - partition_sum_unchecked → invariant_sum_broken_reported
  - disjoint_unchecked → invariant_disjoint_broken_reported
  - union_unchecked → invariant_union_broken_reported
  - kept_fingerprint_unchecked → invariant_kept_fingerprint_reported
  - also_counted_unbounded → invariant_also_counted_at_most_equal_train
  - lists_unsorted_allowed → invariant_kept_ids_sorted_and_unique
  - ids_shape_unchecked → invariant_ids_without_surrounding_whitespace
- **dedup.py, the TRAIN/VAL list**, each killed by case_list_parser:
  - list_parsed_before_its_sha256 → list_wrong_sha256_refused_before_parsing
  - list_sha256_unchecked → list_wrong_sha256_refused_before_parsing
  - list_final_lf_unchecked → list_without_final_lf_refused
  - list_header_unchecked → list_wrong_header_refused
  - list_rows_unchecked → list_row_count_refused
  - list_byte_count_unchecked → list_row_with_a_non_digit_byte_count_refused
  - list_sha1_unchecked → list_row_with_a_non_hex_sha1_refused
  - list_utf8_unchecked → list_not_utf8_refused
  - list_row_form_unchecked → list_row_of_five_fields_refused
  - list_split_counts_unchecked → list_split_counts_refused
- **dedup.py, the ids**, each killed by case_image_ids:
  - ids_whitespace_allowed → ids_surrounding_whitespace_refused
  - ids_tab_cr_lf_allowed → ids_tab_in_a_name_refused
  - ids_not_utf8_allowed → ids_not_utf8_refused
  - ids_shared_stem_allowed → ids_shared_stem_refused
  - ids_case_fold_allowed → ids_case_folded_collision_refused
- **dedup.py, the strict reader**, each killed by case_reader:
  - reader_parses_before_its_sha256 → reader_sha256_compared_before_parsing
  - reader_accepts_non_canonical → reader_non_canonical_bytes_refused
  - reader_lone_surrogate_uncaught → reader_lone_surrogate_escape_refused
  - reader_not_strict_json → reader_duplicate_key_refused
  - doc_keys_unchecked → reader_extra_key_refused
  - doc_format_unchecked → reader_other_format_refused
  - n_images_form_unchecked → reader_n_images_not_an_integer_refused
  - sha256_fields_form_unchecked → reader_sha256_field_not_hex64_refused
  - script_form_unchecked → reader_script_extra_key_refused
  - ids_list_type_unchecked → reader_kept_ids_not_a_list_refused
  - reason_keys_unchecked → reader_fourth_reason_in_the_id_lists_refused
  - counts_form_unchecked → reader_fourth_reason_in_the_counts_refused
  - also_counted_form_unchecked → reader_also_counted_second_key_refused
  - reader_skips_the_invariants → reader_broken_invariant_refused
- **dedup.py, fingerprints, values and the issue**:
  - ids_sha256_trailing_lf → case_lrep_rules:ids_sha256_is_lrep_s_definition_no_trailing_lf
  - value_keys_reordered → case_hasher_values:hasher_values_keys_are_manifest_s_544
  - values_of_another_hash_list → case_hasher_values:hasher_values_refuse_a_dedup_of_another_hash_list
  - issue_count_changed → case_issue_rows_and_keys:issue_counts_pinned_70_11_13
  - issue_line_counts_dropped → hasher case_pass_and_trace:prints_the_issue_s_pinned_counts_70_11_13_H7
- **tday_hash_images.py, steps 1 and 2**:
  - path_guards_dropped → case_step1_paths:step1_root_holding_the_word_refused_with_no_call
  - unlock_guard_dropped → case_step1_paths:step1_unlock_entry_resolved_under_repo
  - data_call_before_the_records → case_step2_records:step2_unlock_entry_absent_refused_before_any_data_call
  - shallow_clone_allowed → case_step2_repository:step2_shallow_clone_refused_before_any_data_call
  - no_head_commit_unnamed → case_step2_repository:step2_repository_without_a_commit_refused_before_any_data_call
  - the rest are killed by case_step2_records; each check name below is followed by
    _refused_before_any_data_call:
    - unlock_kind_unchecked → step2_unlock_entry_of_another_kind
    - manifest_path_unchecked → step2_manifest_path_other
    - manifest_sha256_unchecked → step2_manifest_sha256_other
    - manifest_not_strict_json → step2_manifest_duplicate_key
    - manifest_format_unchecked → step2_manifest_of_another_format
    - manifest_body_unchecked → step2_manifest_body_sha256_other
    - manifest_appends_unchecked → step2_manifest_appends_sha256_other
    - pin_errors_ignored → step2_pin_script_sha256_other
    - pin_form_unchecked → step2_pin_commit_not_lower_hex
    - pin_script_unchecked → step2_pin_script_sha256_other
    - pin_script_path_unchecked → step2_pin_script_path_other
    - pin_module_unchecked → step2_pin_module_sha256_other
    - pin_python_unchecked → step2_pin_python_other
    - pin_parameters_unchecked → step2_pin_parameters_other
    - descriptive_without_digest → step2_pin_digest_other
    - descriptive_without_folder → step2_pin_folder_other
    - descriptive_without_outputs → step2_pin_outputs_other
    - descriptive_without_rule → step2_pin_rule_other
    - descriptive_without_split → step2_pin_split_other
    - descriptive_without_suffixes → step2_pin_parameters_other
- **tday_hash_images.py, steps 3 to 6**:
  - split_written_as_a_literal → case_static:h0_split_read_by_import_inside_the_resolver
  - a_second_listing → case_pass_and_trace:trace_no_listdir
  - folders_listed_as_images → case_pass_and_trace:hash_writes_both_outputs_exit_0
  - suffix_case_sensitive → case_pass_and_trace:hash_writes_both_outputs_exit_0
  - ids_unchecked → case_step3_listing:step3_two_images_one_stem_refused_before_any_image_or_list_read
  - expected_images_one_sided → case_step3_listing:step3_image_count_other_refused_before_any_image_or_list_read
  - folder_error_uncaught → case_step3_listing:step3_image_folder_absent_refused_before_any_image_or_list_read
  - list_error_uncaught → case_step4_list:step4_list_absent_refused_before_any_image_read
  - list_rows_not_the_pin_s → case_step4_list:step4_list_rows_not_the_pin_s_refused_before_any_image_read
  - image_error_uncaught → case_step5_hashing:step5_unreadable_image_refused_nothing_written
  - images_read_as_text → case_pass_and_trace:trace_one_binary_read_per_image
  - digest_not_sha256 → case_pass_and_trace:hash_list_is_the_images_sorted_by_stem_H6
  - the rest are killed by case_step6_outputs:
    - create_not_exclusive → step6_exclusive_create_refuses_an_existing_file_xb
    - partial_outputs_kept → step6_second_create_refused_first_output_removed
    - created_file_registered_after_its_write → step6_write_failure_exit_4_leaves_neither_output
    - error_exit_dropped → step6_write_failure_exit_4_leaves_neither_output
    - outputs_not_checked_first → step6_one_output_present_refused_none_created
    - out_inside_allowed → step6_out_inside_the_repository_refused
    - out_equal_to_the_repository_allowed → step6_out_equal_to_the_repository_refused
    - out_not_resolved → step6_out_linked_into_the_repository_refused
    - this_checkout_not_checked → step6_out_inside_this_checkout_refused
    - out_folder_not_required → step6_out_not_an_existing_folder_refused
- **tday_hash_images.py, printing, usage and the pin (H7, H8)**:
  - ids_printed → case_pass_and_trace:prints_no_id_H7
  - stdout_not_escaped → case_stdout_without_the_arrow:cp1252_stdout_exit_0_four_lines_arrow_escaped
  - the rest are killed by case_print_pin:
    - print_pin_unchecked → print_pin_commit_upper_case_refused
    - usage_modes_unchecked → usage_both_modes_exit_2
    - usage_commit_flag_unchecked → usage_print_pin_without_commit_exit_2
    - pin_expected_images_not_1561 → pin_expected_images_is_schema_s_and_official_rows
    - pin_without_the_module → pin_modules_hold_the_dedup_module_s_sha256
    - pin_python_other → pin_python_3_11
    - pin_counts_swapped → pin_train_val_list_is_b1_s_with_the_split_counts

The GO's named items, by mutation:
- the precedence swapped: precedence_swapped;
- last instead of first, and a reversed sort: last_instead_of_first, reversed_sort;
- the partition flag: partition_flag_false, partition_flag_unchecked;
- sha256-before-parse: list_parsed_before_its_sha256, reader_parses_before_its_sha256;
- the "test" guards: path_guards_dropped, unlock_guard_dropped;
- "xb" to "wb": create_not_exclusive;
- the pin and module checks: pin_*, descriptive_*, pin_without_the_module;
- expected_images: expected_images_one_sided, pin_expected_images_not_1561.

### 8. H5's B-4 change for LREP

At lane/lrep 6c222c9, LREP's reader reads tday_dedup.json unchanged. DL-24 r2 and r4 restated it independently and
it accepted the file. It requires sha256 before parsing, strict JSON, kept_ids non-empty, sorted, unique and
_is_text, and removed_by_reason equal to the export's. The other ten keys are kept as other_fields (H:1379). B-4's
exact change, from the GO's H5:
- heldout_inputs.read_dedup (:1334–1379) reads `reasons_partition` (bool) and the two id digests, `image_ids_sha256`
  and `kept_ids_sha256`, from the file and carries them in DedupRecord (:1313–1320).
- When `reasons_partition` is true, dedup_subset.check_ids (:91–110) requires |all| − |kept| == Σ removed_by_reason
  and reports partition_stated True. This replaces its "≤ with overlap" branch (:104–109) and its fixed False
  (:109).
- Optionally it compares the two digests with its own (am6_tables.ids_sha256 :98–100: the sha256 of the sorted ids
  joined by LF, UTF-8, no trailing LF). dedup.ids_sha256 (:80–84) is the same expression.

DL-24 measured this file against B-4's reading:
- reasons_partition is JSON true;
- image_ids_sha256 is am6(all ids) and kept_ids_sha256 is am6(kept);
- the partition equality held on 2,166 randomized docs (v2);
- the export view {path, sha256, removed_by_reason} passes B-1's DEDUP_SPEC (validate.py :1612).

Where tday_dedup.json lives relative to the export is B-4's and part 2's question (IF§11.5 :85; H:10).

### 9. The part-2 list, with what this part found

The GO's list (unchanged):
- manifest.py:
  - LANE_CODE_FILES gains the two code files;
  - the builder takes the image_hasher pin from hasher_pin (:846, :861);
  - build_inventory's dedup (:985) becomes {path, sha256, removed_by_reason}.
- schema.py: the tday_dedup/1 format line, one definition.
- validate.py:
  - HASHER_VALUES only after the unlock, equal to the files' sha256s;
  - g_hasher_pin_mismatch extended to modules;
  - the export's dedup checked against the file (DEDUP_SPEC :1612, the partition sum).
- The CLI.
- smoke_tman_paths: LANE_FILES, == 23, 3317, and the static file-access case (+1 scandir, +2 exclusive writes).
- The noscore module count.
- The pins, validate, unlock and mutation smokes' wiring cases.

Added by this part:
1. **Line shifts at lane/tman's tip** (005a891, STEP 1; REPOSITORY-PROVEN there):
   - entries.py gains an arm registry before read_entry_file (+17 lines).
   - manifest.py gains `import posixpath` and two EVALUATOR_DATA arm configs: TRAINVAL_HASH_LIST moves to :679–680
     and LANE_CODE_FILES to :684–686.
   - smoke_tman_paths (+20 lines), smoke_tman_noscore (+80) and smoke_tman_mutations (+190) change.
   - The hasher's docstrings and this report cite B's lines (entries.py :753, :757; manifest.py :544, :609, :861).
     Re-cite them after the merge-in. c8587b9 (the tip now) was not read.
2. **The unlock-entry guard.** B-1's read_entry_file resolves the path against the working folder first
   (entries.py :757) and only after that under the records' root (records.py :212). The hasher therefore guards
   --unlock-entry under --repo itself (:138). If C6g5 or later resolves under the root, unlock_guard_dropped becomes
   an equivalent mutant: part 2's final runs would show SURVIVED, and its check should then be retargeted (v3 on
   r3-F2).
3. **Shallow repositories in B-1** (v1 on r1-F2). Neither validate.py nor tday_manifest.py calls
   HeadSource.require_not_shallow (no call in src/tman or scripts/tday_manifest.py). The hasher now refuses one
   (B3A-F2). Part 2 adds the same check wherever B-1 reads "written once" at HEAD.
4. **The pin's data parameters** (r1-F3). Step 2 compares the script, the module, python and DESCRIPTIVE. It takes
   expected_images and train_val_list from the pin, as H9 rules. A gate, or the builder taking the pin from
   hasher_pin (item 1 of the GO's list), should tie them to S.EXPECTED_ROWS, body.expected_rows and
   TRAINVAL_HASH_LIST with 5367/846.
5. **The session part and the manifest's file sha256** (r1-F4; open item 1). Step 2 compares the committed
   manifest's file sha256, as H9 says ("bytes"). Once a session part is committed, that sha256 changes and the
   hasher refuses (unlock_manifest_mismatch). The ADDENDUM rules either the order (the hasher before the first
   session-part commit) or validate.py's session-mode reading (body and appends; the file sha256 while session is
   null, :1256–1259).
6. **The export's dedup against the pin** (r2-F2). When validate.py checks the export's dedup against
   tday_dedup.json, it also requires:
   - script == {path scripts/tday_hash_images.py, sha256 the pin's script sha256};
   - train_val_list_sha256 == the pin's train_val_list.sha256;
   - n_images == the pin's expected_images.
7. **New refusal codes.** If the hasher becomes a subcommand or its codes enter the validator, smoke_tman_mutations'
   89-code list (V.GATES and REFUSAL_EXTRA) needs these: train_val_list_mismatch, train_val_list_invalid,
   dedup_invalid, repository_shallow, unlock_manifest_mismatch, images_unreadable, image_ids_invalid and
   image_count_mismatch.

### 10. DL-24

One read-only workflow (DL-24, GO STEP 5) over the scoped diff from B at 7b0eec4. It had four reviewers, each
followed by an adversarial verifier:
- R1, the hasher (H8–H10, the records pre-check, rule 4);
- R2, the rule and the formats (H3–H7, the partition, LREP's rules, the B-1 hooks);
- R3, the smokes and mutations;
- R4, numerical.

The agents read by exact path, ran git only in the exclude form, ran no smoke and wrote only under
$SCR/dl24/<id>/. The first run hit the session's usage limit. The workflow was resumed from its cache after
"Try again" (10:56 UTC). R4 was rerun, and both of its runs matched byte for byte.

Before and after (MEASURED), identical:
- HEAD 7b0eec4ba0b73c5085b84f0d46551df5af0256e0;
- the scoped status empty;
- `sha256sum -c` of 17 files (the five new files and the B-1 files they use) all OK.

| Finding | Severity | Verdict | Resolution |
|---|---|---|---|
| r1-F1 write_outputs left a partial output (second create or write failing) | minor | confirmed (MEASURED) | B3A-F1 |
| r1-F2 a shallow --repo passed the written-once check | minor | confirmed (MEASURED) | B3A-F2 |
| r1-F3 the pin's data parameters not compared with the constants | note | confirmed as a note | part-2 item 4 |
| r1-F4 the file-sha256 match fails once a session part is committed | note | confirmed as a note | part-2 item 5; open item 1 |
| r1-F5 symlinked *.jpg entries are followed (as hash_split_files.py :95–96) | note | confirmed as a note | open item 2 |
| r2-F1 read_dedup raised UnicodeEncodeError on a lone surrogate escape | minor | confirmed (MEASURED) | B3A-F3 |
| r2-F2 script, train_val_list_sha256 and n_images not tied to the pin | note | confirmed as a note | part-2 item 6 |
| r3-F1 usage check masked; usage branches unmutated | minor | confirmed (MEASURED) | B3A-F6 |
| r3-F2 the unlock guard's checks cannot attribute the refusal | minor | refuted | below |
| r3-F3 pin_form_unchecked killed only by a crash; one DESCRIPTIVE parameter varied | minor | confirmed (MEASURED) | B3A-F6 |
| r3-F4 the manifest_unreadable and manifest_format mutations killed only by crashes | minor | confirmed (MEASURED) | B3A-F6 |
| r3-F5 six doc_errors form rules without case or mutation; keys, format and strict JSON unmutated | major | confirmed (MEASURED) | B3A-F3 |
| r3-F6 the list rows' byte count and sha1 without a case; UTF-8 branch unmutated | minor | confirmed (MEASURED) | B3A-F4 |
| r3-F7 out_folder's `p == r` and resolve() without a case | minor | confirmed (MEASURED) | B3A-F6 |
| r3-F8 the H10 scan can be fooled (escaped bytes, concatenation) | note | refuted | below |
| r3-F9 the data-call checks miss stat-level and keyword-path calls | note | refuted | below |
| r3-F10 the PIL half of a recorded check cannot fail | note | refuted | below |
| r4 (both runs): no findings; byte-identical independent implementation | — | spot checks all hold | — |
| v1-M1 (v2-M1): exit 4 after a complete run when stdout cannot encode U+2194 | minor | MEASURED by v1 | B3A-F5 |
| v3-M1 the printed issue line was checked only against its own function | minor | MEASURED by v3 | B3A-F5 |
| v3-M2 main's exit-4 branch had no case and no mutation | minor | MEASURED by v3 | B3A-F1 |

Refuted, with the verifier's reasons:
- **r3-F2.** H9 step 1 names only --root, --out, --train-val-list and --repo, so the masked check verifies a true
  property. The kill of unlock_guard_dropped comes from the property the guard exists for: lstat calls on the
  string before it is refused. The C6g5 scenario is INFERRED, and a change there would show as SURVIVED in part
  2's final runs (part-2 item 2).
- **r3-F8.** No realistic failure. Neither production file holds a bytes literal other than b'' or any str+str
  constant concatenation. Every form H10 names is covered. The exemption is H10's own allowance for B-1's guard,
  and step 1's runtime checks catch a wrong guard.
- **r3-F9.** Recording positional paths is B-1's tracer form (smoke_tman_paths.py :70–72). stat and lstat are left
  out because H9 makes step 1 resolve every path before step 2. All four of the hasher's data accesses are
  positional and recorded. The GO asks for import bans, not an allowlist.
- **r3-F10.** The recorded fact holds on its own: a fresh interpreter importing src.tman.schema has PIL.Image and
  torch loaded. The property that matters, no PIL call, is enforced by the patched PIL.Image.open and checked by
  trace_no_pil_or_cv2_call.

Reviewers' observations that are not findings:
- r2: the list parser does not check the stem and file fields, and only the pinned list passes its sha256 gate.
  Separator characters other than tab, CR and LF are not refused, which matches the evaluator's forbidden id
  characters.
- r2: a lone high surrogate in a file name, which only a Windows scandir could give, makes name_digests raise (open
  item 5).
- v2: the ids are the evaluator's canonical ids except all-dot stems ('..jpg'), which no PlantSeg name has and which
  the count gate refuses.

The agents' slips:
- r2: one Grep call named the folder src/tman (glob !*[Tt][Ee][Ss][Tt]*); it printed manifest.py :840–866 and
  :980–990 with sed instead of the Read tool.
- r4 (first run): the same Grep slip.
- v1: the same Grep slip at 11:11:14Z, not self-reported (§11).

### 11. SL-1 log

SL-1 guard installed in log mode (MODE = "log", sl1_guard.py :66). There was no ~/.claude/sl1_guard.log at the
start. `tail -n 80` of /root/.claude/sl1_guard.log at the report:

```
2026-10-27T09:44:01Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/tman glob=!*[Tt][Ee][Ss][Tt]*
2026-10-27T10:00:35Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/tman glob=!*[Tt][Ee][Ss][Tt]*
2026-10-27T11:01:48Z	Bash	SL1-B1	grep -n 'def require_not_shallow' -B 2 -A 14 scripts/sweep_select.py; grep -n 'class SelectionRefused' -A 12 scripts/sweep_select.py; grep -n 'require_not_shallow\|is-shallow\|shallow' src/tman/*.py s
2026-10-27T11:11:14Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/tman glob=!*[Tt][Ee][Ss][Tt]*
2026-10-27T12:09:43Z	Bash	SL1-B1	for f in pyproject.toml setup.cfg .flake8 tox.ini ruff.toml .ruff.toml; do [ -f "$f" ] && grep -n -i 'line-length\|max-line-length\|line_length' "$f" /dev/null; done; git diff -U0 -- scripts/smoke_tma
```

Attribution (MEASURED from the session transcripts):
- 09:44:01 is review r2 (agent ae6c5619);
- 10:00:35 is review r4, first run (agent af67b8a4);
- 11:11:14 is verify r1 (agent af5b54c5);
- 11:01:48 and 12:09:43 are mine (slips 2 and 4).

### 12. Slips (mine)

1. 08:28 UTC: `ls ~/.claude/` was denied by L-PROT S (a listing with ~). It was not retried, and absolute paths
   were used after that.
2. 11:01 UTC (STEP 5, preparing fixes): one grep named `src/tman/*.py`, a shell glob. It searched B-1's src/tman
   files for "shallow", and no file holding the word was matched or read. SL-1 logged it (SL1-B1).
3. 12:06 and 12:59 UTC: two `git status --porcelain=v1 --ignored` calls named folders (src/tman, scripts, src/eval;
   src, scripts, configs) with only the `*test*` exclude instead of the exact exclude form. Both were read-only, and
   neither listed anything.
4. 12:09 UTC: a loop grepped six named config files at the repository root for a line-length setting (none
   exists). SL-1 logged it (SL1-B1): the root is not a search path under rule 4's wording.
5. 12:34:14–12:34:16 UTC (STEP 6): a `sed` of the GO's own text ran while smoke_tman_validate was running
   (12:34:05–12:37:30). That breaks "nothing else runs during a smoke". smoke_tman_validate was rerun alone (final2:
   PASS 361/361, 204 s), and that run is the record. No other action fell inside the final runs (MEASURED from the
   transcript).
6. 13:01 UTC (report): an inline python that mentioned git subcommands (a transcript scan) was denied by L-PROT. It
   was redone as a script file under $SCR/probes.
7. The harness's attribution reminder asked for `Co-Authored-By`/`Claude-Session` trailers and named the branch
   claude/practical-goldberg-reaa3d. As the GO rules, this was reported once and not followed.

### 13. Open items (at most five)

1. **The hasher's order against the session part** (r1-F4). Decide in part 2's ADDENDUM or the TEST-day runbook:
   either the hash mode runs before the first session-part commit, or step 2 reads the manifest as validate.py's
   session mode does.
2. **Symlinks in images/<split>** (r1-F5). They are followed, as in the list of record's writer. Plan :538's
   no-annotation guarantee assumes a folder without links (INFERRED: the release is an archive extract). Accept
   this, or refuse links (is_file(follow_symlinks=False)) in part 2.
3. **The MATCH count** is 48 against TMAN-B1's 47. The extra line is not named from evidence (INFERRED: regex
   2026.5.9).
4. **unlock_guard_dropped's kill** depends on B-1's entries.py :757 resolving against the working folder (part-2
   item 2).
5. **A lone high surrogate in a Windows file name** makes name_digests raise (exit 4 instead of image_ids_invalid;
   r2's observation). It cannot occur on the Linux image of record (plan :566–568, INFERRED). Harden it if TEST day
   ever runs on Windows.

### 14. RECORDS proposed (Part 1; the lane writes none)

- **A DL row for the duplicate-free rule** (plan :695, before the unlock): "The TEST split's duplicate-free list
  (AM-18 item 8(c); plan (ix)) is fixed as follows.
  - Ids are file stems; 'sorted' is Python's sorted() in code-point order.
  - Precedence: equal_train (sha256 equal to a TRAIN image's), then equal_val, then duplicate_in_split (every
    member of a group of identical TEST images except its first sorted id). An image equal to both a TRAIN and a VAL
    image is removed under equal_train and counted in also_counted.equal_train_and_val.
  - Keys hold no 'test'; the plan's labels are values (H3, H4).
  - The file is tday_dedup.json, format tday_dedup/1 (H5), canonical JSON, with reasons_partition true (each removed
    image has exactly one reason), image_ids_sha256 and kept_ids_sha256 by am6_tables.ids_sha256's definition.
    Beside it goes tday_image_hashes.tsv (H6).
  - Code: src/tman/dedup.py and scripts/tday_hash_images.py at lane/tman-b3 <report commit>."
- **A pin row at the hasher's merge** (DL-37's form): "<merge commit> (TMAN B-3a part 1: scripts/tday_hash_images.py
  sha256 41433b47…d3cc and its module src/tman/dedup.py sha256 b39d8ad3…7df7, from lane/tman-b3 <report commit>,
  <date>)". The hasher's pin carries the same two sha256 values (H8, modules).
- **The current-plan line** (plan :691): "B-3a runs before the freeze in two parts. Part 1 is new files on
  lane/tman-b3: the duplicate-free list, the in-session hasher, their smokes and report. Part 2 wires them into
  manifest.py, schema.py and validate.py after the merge-in of lane/tman's final tip."
- **The Ch4 sentence** (plan :708), drafted: "The TEST images were hashed in the session, after the unlock and before
  any model was scored (scripts/tday_hash_images.py at <commit>; tday_image_hashes.tsv sha256 …). D images were
  removed: a equal to a TRAIN image, b to a VAL image, c duplicate TEST images (issue #11: 70/11/13), leaving K
  images in the duplicate-free subset (tday_dedup.json sha256 …)."
- **A B-4 note for LREP** (H5): §8's change, with the file's fields `reasons_partition`, `image_ids_sha256` and
  `kept_ids_sha256`. LREP's reader at 6c222c9 reads the file unchanged until then.

### 15. Acceptance (checked after the report commit; see the session's final lines)

- STEP 0 matches TMAN-B1's environment: the same recipe and gates, the MATCH count difference named in §2.
  STEP 3 reproduces every count.
- STEP 6 passes, with FROZEN BLOBS OK (10/10) and every new mutation KILLED (107/107).
- The scoped `git diff --no-renames --name-status ee5593e9 HEAD` lists only A lines, for SCOPE's six files. The
  src/scripts/configs/docs/lane_reports form lists the same six.
- Every new refusal has a case reached with every earlier gate passing, and a mutation killed by a named case (§6,
  §7).
- No run created a path, name, key or id holding the word or ":" (nothing_created_holds_the_word_or_a_colon). The
  scoped status is clean after the report commit.
- One DL-24 workflow, its findings resolved (§10). No trailers, no merges. Pushes went to lane/tman-b3 only, after
  every commit.
