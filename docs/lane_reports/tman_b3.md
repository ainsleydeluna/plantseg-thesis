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

## Part 2

Session TMAN-B3A part 2, 29 Oct 2026, under tman_b3a_part2_addendum_20261029.md (the ADDENDUM: 44,258 B, sha256
05b435b9…a180afd0), STEPS P0–P7, and ORCH-4's ruling P2R2-AMEND-1 on the STOP held at P2-1. Part 2 wires part 1
into the TMAN files: the hasher's pin, the duplicate-free list's format line, the inventory export's dedup block,
the session's hasher values, and the validator's and records' checks. Labels as in Part 1: MEASURED,
REPOSITORY-PROVEN, DOCUMENTED, INFERRED. File:line is at f36ce8f (the last code commit) unless another commit is
named.

### 1. MB3, branch and commits

- MB3 = `815d138fa7008326739ff447903dd313a2ff5874`, Ice's merge of lane/tman (39a6798) into lane/tman-b3
  (parents 79d5b2b and 39a6798; tree 7e279152; MEASURED in P0). 39a6798..MB3 lists six A lines, part 1's files.
  lane/tman-b3 was moved to MB3 in P0 (fetched 79d5b2b..815d138; lane/tman 005a891..39a6798) and pushed after every
  commit (`git push origin refs/heads/lane/tman-b3:refs/heads/lane/tman-b3`, then `git ls-remote`). No merge,
  rebase or trailer; lane/tman untouched.

| Commit | What |
|---|---|
| aba5bf1a76c6811761aa367f3867afa1267e69bd | P2-0: the hasher smoke's freeze fixture takes B-2's freeze fields (P2R10) |
| baa1f35575dfa7e3e8665bd40cd7ebd1e5fca481 | P2-1: the wiring -- one pin, DEDUP_FORMAT, LANE_CODE_FILES, `inventory --dedup` (P2R1–P2R5, P2R11) |
| e1a5dcd749bf51d580f50be4bf8c14a878e2d754 | P2-2: the validator and the records (P2R6, P2R7) |
| b64a868570bdc6159fd7c606210c37cd4bf07189 | P2-3: the hasher -- step 2 by validate.py's rule, links refused, B-era citations (P2R7, P2R8) |
| ceac84805d6713b08879629690b82395bd7367ed | P2-4: mutations (P2R13) |
| 67d4073d9f24894c9f43d9e6a91bad065fbc2077 | P2-F1: the command line refuses a shallow clone first (DL-24 R2-F1) |
| 7fc834de03cc99f1a08abd4c5db50950ab815674 | P2-F2: export_dedup: a link loop, a name not UTF-8 (R1-F2) |
| 5737d33c70369d32cd5f88c17e80118e3fe51801 | P2-F3: the dedup file's two rule-4 guards cased (R3-F4) |
| e02901dc5a84becdf5429a1b746ee1c7f3b28583 | P2-F4: hasher values and pin data typed; fourth key, list (R2-F2, R4-1, R4-2, R3-F3) |
| 5bada14793f8659b085938b75ca51bba100d79f4 | P2-F5: dedup.path's form into schema.py; the H10 scan folds `+` (R1-N1) |
| d470a887f1ec816b67ac5b18f0a1895219e537ac | P2-F6: the hasher: running script, links, step 2's manifest shape (R1-F1, R3-F1, R3-F2) |
| f402ba3b17d62878f8e509f252593c2354b74a1b | P2-F7: citations and comments (R1-N4, R3-F5, R3-N4, R3-N5, k3) |
| ca38c20d447a6e4f2b5d1f0dab7f120ef90364fe | P2-F8: B-1's one_unlock_row killed again (found by P5) |
| f36ce8f83c0752ef35e57a4925c5e42e216d1a30 | P2-F9: two citations P2-F4 moved |
| (this commit) | the report |

The scoped `git diff --no-renames --stat 815d138fa7008326739ff447903dd313a2ff5874 f36ce8f` (the last code commit),
MEASURED:

```
 scripts/smoke_tman_b3_mutations.py | 110 +++++++++------
 scripts/smoke_tman_dedup.py        |  22 ++-
 scripts/smoke_tman_hasher.py       | 154 ++++++++++++++++----
 scripts/smoke_tman_mutations.py    | 246 ++++++++++++++++++++++++++++++--
 scripts/smoke_tman_noscore.py      |  17 ++-
 scripts/smoke_tman_paths.py        |  59 +++++---
 scripts/smoke_tman_pins.py         |  75 +++++++++-
 scripts/smoke_tman_unlock.py       | 158 ++++++++++++++++++++-
 scripts/smoke_tman_validate.py     | 282 ++++++++++++++++++++++++++++++++++++-
 scripts/tday_hash_images.py        | 165 +++++++++-------------
 scripts/tday_manifest.py           |  40 ++++--
 src/tman/dedup.py                  |  96 ++++++++++++-
 src/tman/manifest.py               |  62 ++++++--
 src/tman/records.py                |  17 +++
 src/tman/schema.py                 |  19 ++-
 src/tman/validate.py               | 139 ++++++++++++++++--
 16 files changed, 1398 insertions(+), 263 deletions(-)
```

All 16 are SCOPE files, all M; the report commit adds docs/lane_reports/tman_b3.md. At f36ce8f, sha256 of
scripts/tday_hash_images.py is 5f65eb4893c49e6b41e363d924cf0b5697dc0d0a9b736a983c015b91f549096a (17,340 B), and of
src/tman/dedup.py 76fe0c6b815e98c8311a5a4959f650f16ba2b7a65db89451944accae8c78f079 (24,061 B).

### 2. Environment (P0)

The five attachments' bytes and sha256 matched. The is-shallow check gave false. Part 1's $SCR/tmanvenv was kept,
MEASURED:
- Python 3.11.17 from /usr/bin/python3.11; `pip check` clean;
- torch and torchvision print `2.1.0+cpu 0.16.0+cpu`, and the twelve pinned versions are exact;
- requirements.lock is blob 8431e533 at B and at MB3;
- the package-against-lock table is identical to Part 1's §2: MATCH 48, DIFF 2, EXTRA 3, ABSENT 26.

TMAN2-RULE-3 item 3 rules the MATCH count (48 against TMAN-B1's 47) a note, not a gate.

### 3. Smokes: P1 (at MB3) and P5 (at f36ce8f)

Each smoke ran alone, with the venv, `python -B`, PYTHONDONTWRITEBYTECODE=1, PYTHONPATH the checkout root and its
own new TMPDIR under $SCR; nothing else ran meanwhile. All MEASURED. P1 ran 06:13–07:05 UTC; the P5 of record ran
16:20–17:17 UTC on f36ce8f.

| Smoke | P1, at MB3 | P5, at f36ce8f |
|---|---|---|
| smoke_tman_schema | RESULT: PASS (167/167) | RESULT: PASS (167/167) |
| smoke_tman_paths | RESULT: PASS (89/89) | RESULT: PASS (94/94) |
| smoke_tman_complete | RESULT: PASS (93/93) | RESULT: PASS (93/93) |
| smoke_tman_cuts | RESULT: PASS (204/204) | RESULT: PASS (204/204) |
| smoke_tman_parent | RESULT: PASS (33/33) | RESULT: PASS (33/33) |
| smoke_tman_derived | RESULT: PASS (101/101) | RESULT: PASS (101/101) |
| smoke_tman_pins | RESULT: PASS (53/53) | RESULT: PASS (61/61) |
| smoke_tman_validate | RESULT: PASS (369/369), 299 s | RESULT: PASS (417/417), 309 s |
| smoke_tman_unlock | RESULT: PASS (86/86), 88 s | RESULT: PASS (109/109), 129 s |
| smoke_tman_noscore | RESULT: PASS (36/36) | RESULT: PASS (37/37) |
| smoke_tman_determinism | RESULT: PASS (18/18) | RESULT: PASS (18/18) |
| smoke_tman_mutations | RESULT: PASS (647/647), 563 KILLED, 2201 s | RESULT: PASS (714/714), 623 KILLED, 0 SURVIVED, 2426 s |
| smoke_tman_am22 | RESULT: PASS (191/191), 347 s | RESULT: PASS (191/191), 344 s |
| smoke_frozen_blobs | FROZEN BLOBS OK (10/10) | FROZEN BLOBS OK (10/10) |
| smoke_tman_dedup | RESULT: PASS (95/95) | RESULT: PASS (96/96) |
| smoke_tman_hasher | RESULT: FAIL (37/47); P2-0: PASS (124/124) | RESULT: PASS (136/136) |
| smoke_tman_b3_mutations | RESULT: FAIL (73/132); P2-0: PASS (132/132), 107 KILLED | RESULT: PASS (141/141), 114 KILLED, 0 SURVIVED |

P1 reproduced FACTS' counts for the fifteen others. The hasher smoke and the b3 mutation smoke failed for one
cause, a lane/tman change since B (P2R10's exception). B-2's freeze fields d_date and conditions (B2R5,
src/tman/entries.py :65–69 at 39a6798) made B-1's reader refuse the hasher smoke's freeze fixture ("entry freeze:
missing ['conditions', 'd_date']"). P2-0 gave the World's freeze B-2's item-1(f) fields (FREEZE_1F,
smoke_tman_hasher.py :67–68, smoke_tman_validate.py :46's form) and changed no format. Its runs are the hasher's
and the b3 smoke's baseline.

Two P5 runs before the record do not count:
- On f402ba3, smoke_tman_mutations gave RESULT: FAIL (713/714): B-1's one_unlock_row SURVIVED (slip 3). Fixed in
  P2-F8.
- On ca38c20, every count was the record's. P2-F9 then re-cited two citations, and P5 ran once more.

Every changed count, with its cases (MEASURED from the check lists of the P1 or P2-0 log and the P5 log):
- **smoke_tman_paths 89 → 94.**
  - Added: cli_validate_dedup_holding_test and cli_inventory_dedup_holding_test (PATH_ARGS gains `dedup`);
    validator_dedup_holding_test_refused_before_any_call_P2R6a (g_test_path covers --dedup);
    inventory_dedup_given_resolved_where_read_R3F4 and export_dedup_path_resolved_where_read_R3F4 (P2-F3).
  - Two checks renamed for P2R11: listings_are_the_teacher_log_the_runs_folder_and_the_image_folder_only (12
    modules; the hasher's scandir), and writes_only_through_three_exclusive_opens_and_the_hasher_s_cleanup_P2R11.
  - LANE_FILES 25 → 31 and the names total 3325 keep their check names.
- **smoke_tman_pins 53 → 61.** case_one_pin (P2R2 and P2R2-AMEND-1), eight checks:
  - one_definition_of_the_hasher_pin_P2R2, builder_and_print_pin_call_the_one_pin_P2R2,
    no_other_module_builds_the_pin_s_shape_P2R2;
  - the_pin_passes_hasher_pin_errors_and_test_value_errors_H8;
  - builder_takes_the_function_s_pin_P2R2, builder_takes_the_function_s_pin_whatever_the_sources_hold;
  - cli_build_sources_pin_null_writes_the_function_s_pin, cli_build_sources_pin_of_another_python_writes_the_function_s.
- **smoke_tman_validate 369 → 417.**
  - case_record_gates +6: P2R6(d)'s three variants and draft mode leaving the data parameters, plus the typed pair
    hasher_pin_expected_images_1561_0_R4_1 and hasher_pin_train_val_rows_6213_0_R4_1 (P2-F4).
  - case_inventory_dedup +18: the builder's dedup, P2R4 and P2R5. Each refusal: outside, "..", backslash, space,
    absent, a folder, not canonical, the three pin equalities, and P2-F2's link loop and non-UTF-8 name. The
    command line's `inventory --dedup`.
  - case_inventory_dedup_reader +16: read_inventory's export dedup, P2R6(c)(i). The working folder never read; a
    subfolder; nine path forms; absent; the sha256 before parsing; strict reading; removed_by_reason; DEDUP_SPEC's
    shape staying inventory_invalid.
  - case_shallow_records +8: P2R6(e) in the validator, the inventory and the command line. P2-F1's two command-line
    checks. Each reader first, once per Records, and no HEAD commit giving record_uncommitted.
- **smoke_tman_unlock 86 → 109.**
  - case_unlock_gates +2: dedup_module_not_its_pin_P2R6b, unlock_entry_naming_another_manifest_path_P2R7.
  - case_hasher_pin_modules +1 (P2R6(b)).
  - case_session_hasher_values +17: P2R6(a) and (c)(ii), plus P2-F4's fourth key, floats, true, a list and the
    spec-keys check.
  - case_cli_session_dedup +3.
- **smoke_tman_noscore 36 → 37.** + dedup_format_literal_once_in_the_lane_s_code_schema_s_P2R3; the module count
  check renamed for 12 modules (P2R11).
- **smoke_tman_dedup 95 → 96.**
  - + the_pin_s_data_imported_inside_data_parameters_only_P2R2.
  - The module-level import check renamed, ..._at_module_level_P2R2.
  - Two re-cited names: export_passes_validate_s_dedup_spec and hasher_values_keys_are_manifest_s, which lost their
    B-era line numbers.
- **smoke_tman_hasher 124 → 136.**
  - P2R7: three case_step2_session_part checks.
  - P2R8: two link variants.
  - P2-F6: a link to itself, three manifest-shape variants, case_running_script's three checks.
  - Three case_print_pin checks renamed, losing B-era line numbers.
- **smoke_tman_b3_mutations 132 → 141.**
  - Eleven new mutations and two new killing-case baselines (case_running_script, case_step2_session_part).
  - Four mutations retired by P2-3 (§6).
- **smoke_tman_mutations 647 → 714.** Sixty new mutations and seven new killing-case baselines (§6).

### 4. Rulings P2R1–P2R13 and TMAN2-RULE-3, with file:line (REPOSITORY-PROVEN at f36ce8f)

- **P2R1, the CLI.** The hasher stays scripts/tday_hash_images.py with its two modes: cmd_print_pin :247 and
  cmd_hash :259.
  - tday_manifest.py has no hash command.
  - Its only new options are the two `--dedup` flags:
    - `validate --dedup` (:648) is read in session mode only. Any other mode refuses hasher_values_mismatch before
      any read (:161–163).
    - `inventory --dedup` (:696) is passed as (file, --out) (:583).
  - PATH_ARGS holds "dedup" (:721).
- **P2R2, one pin, as P2R2-AMEND-1 rules.**
  - dedup.hasher_pin (src/tman/dedup.py :376–382) is the one definition. Its parts:
    - the script's and the module's bytes from this checkout (code_bytes :349, CODE_ROOT :329);
    - descriptive_parameters :360;
    - data_parameters :366, which imports src.eval.teacher_diag and manifest inside the call;
    - the form PIN_SPEC :339.
  - Its users: the builder (manifest.py :879, `DD.hasher_pin(e)`), --print-pin (tday_hash_images.py :251) and step
    2 (pin_errors :74–95).
  - smoke_tman_pins.py :195–204 finds exactly one definition and no other builder of the pin's shape.
  - **As P2R2-AMEND-1 item 4 asks: since P2-1 (baa1f35) the builder ignores sources.generator_pins.image_hasher.**
    The cases that show it are in smoke_tman_pins.py case_one_pin:
    - builder_takes_the_function_s_pin_whatever_the_sources_hold (:217): the sources' pin null or of another
      python, the built pin is dedup.hasher_pin's;
    - cli_build_sources_pin_of_another_python_writes_the_function_s (:226), on the command line.
  - The withdrawn refusal and its withdrawn mutation are not made. smoke_tman_am22.py is untouched (191/191).
  - Since P2-F6, running_script_error (tday_hash_images.py :66–71) refuses at step 2 and at --print-pin when the
    running file is not the bytes the pin hashes (DL-24 R1-F1).
- **P2R3.** DEDUP_FORMAT is schema.py :50, its one definition. dedup.py :44 aliases it. The static case is
  smoke_tman_noscore.py :259.
- **P2R4.**
  - LANE_CODE_FILES (manifest.py :686–689) gains src/tman/dedup.py and scripts/tday_hash_images.py.
  - manifest.py imports dedup at module level (:36). The builder's pin is :879.
  - build_inventory (:1014) takes `dedup=(file, export folder)` and calls export_dedup (:983–1011), which returns
    {path, sha256, removed_by_reason}. The file is read strictly (dedup.read_dedup) and held to the pin (:1007–1010).
  - cmd_inventory (tday_manifest.py :573–583) resolves --out before building, and a refusal writes nothing.
- **P2R5.**
  - dedup.path's form is schema.relative_path_error (:256–268), LREP's path_problem restated, beside B-1's guards
    since P2-F5.
  - The builder: the resolved file must lie in the resolved export folder (manifest.py :990–995), with its relative
    name in that form (:995). A name that is not UTF-8 is refused (:996–1000; P2-F2).
  - The reader resolves dedup.path against the export file's folder (validate.py :1895), never the working folder.
- **P2R6.**
  - (a) g_hasher_values_mismatch (validate.py :1656–1687) is the last session gate (:1742).
    - values.image_hasher and --dedup go together (:1664–1668).
    - The values hold exactly the three keys, typed by HASHER_VALUES_SPEC (:1652–1653, :1672–1675; P2-F4).
    - The file is read strictly (:1676–1680) and held to the pin (:1681–1683).
    - The values are its bytes' sha256, its hash_list_sha256 and its removed_by_reason (:1684–1687).
    - Ctx gains `dedup` (:91, :96), passed by the command line (tday_manifest.py :184).
  - (b) g_hasher_pin_mismatch (validate.py :1474): modules is exactly [src/tman/dedup.py] (:1486), with its sha256
    at S (:1490).
  - (c)(i) read_inventory (:1906) calls read_export_dedup (:1924 → :1886–1903). The steps:
    - DEDUP_SPEC's shape (:1812) stays inventory_invalid;
    - then the path form (:1892–1894), the resolution (:1895) and B-1's guard (:1896);
    - the read (:1897–1900), the sha256 before parsing (:1901) and removed_by_reason (:1902–1903).
  - (c)(ii) The builder (manifest.py :1007–1010) and the session gate (validate.py :1681–1683) both call
    dedup.dedup_pin_errors (dedup.py :386–399).
  - (d) g_hasher_pin_data_mismatch (validate.py :722–731), a full-validation gate right after
    generator_cells_mismatch (:1707–1708).
    - The parameters are typed by PIN_SPEC's own specs (:729; P2-F4).
    - They then equal dedup.data_parameters() (:730–731).
  - (e) records.py full_history (:241–252) checks once per Records.
    - It maps repository_shallow to itself and anything else to record_uncommitted (:250–251).
    - adding_commits, written_once_error and deleted_record_error call it first (:254–264).
    - Since P2-F1 the command line also calls it where it opens the records (tday_manifest.py :92–97, :179). §7
      gives the call sites and the gate order.
- **P2R7** (TMAN2-RULE-3 item 1 (b)). validate.unlock_manifest_errors (:1229–1245) is the session-mode rule,
  factored once.
  - It returns {part: reason} for path, body, file and appends; file_binds is the manifest file's sha256.
  - Its users: g_append_after_unlock (:1248), g_run_added_after_freeze (:1304), and the hasher's check_records
    (tday_hash_images.py :137, binding the file while the session part is null).
  - A run after a session-part commit verifies; a session commit changing the body or the appends is refused
    (smoke_tman_hasher case_step2_session_part).
- **P2R8** (TMAN2-RULE-3 item 2). list_images (tday_hash_images.py :158–178) makes one os.scandir.
  - Each entry's is_symlink() and is_file(follow_symlinks=False) are taken (:164; no link is followed since P2-F6).
  - Any link, of any name, is refused image_link_refused (:167–170), before any image or the TRAIN/VAL list is read.
- **P2R9** (TMAN2-RULE-3 item 4). unlock_guard_dropped was KILLED at P2-0 and at P5. Its killing check is
  case_step1_paths:step1_unlock_entry_resolved_under_repo (smoke_tman_b3_mutations.py :200;
  smoke_tman_hasher.py :457), as at 35453bb. This is the expected outcome: B-1's reader is unchanged (entries.py
  :1007).
- **P2R10.** §3: the fifteen at FACTS' counts and FROZEN BLOBS OK (10/10); the hasher's and the b3 smoke's failure
  had one cause, fixed in P2-0.
- **P2R11.**
  - smoke_tman_paths: LANE_FILES (:48–54) holds 31 (:131) and the names total is 3325 (:129).
  - Its static case: 12 modules (:311), the hasher's scandir among the listings (:312), opens rb, rb, xb plus B-1's
    two xb (:327–328), and write_outputs' cleanup exempted by name (:320–322).
  - smoke_tman_noscore: 12 modules (:255).
- **P2R12** (TMAN2-RULE-3 items 1 and 5). No real data here: synthetic images and a fixture split only. The real
  hash runs in the TEST session inside the Linux image of record (a runbook line in §14b).
- **P2R13.** New validator, builder, records and command-line codes are in smoke_tman_mutations; codes 94 → 99:
  - the gates hasher_pin_data_mismatch and hasher_values_mismatch;
  - REFUSAL_EXTRA's dedup_invalid, dedup_pin_mismatch and repository_shallow.

  The hasher's new code is in smoke_tman_b3_mutations: CODES gains image_link_refused (16 → 17).
- **TMAN2-RULE-3 item 3** is §2's note; **item 5** is §14b's runbook line.

### 5. The export's dedup, the session's values and the pin, with Part 1's fixture (MEASURED)

Part 1 §5's fixture is reused: six images, TRAIN {tr_a: T1, tr_c: TB}, VAL {va_a: V1, va_b: TB}. The repository's
functions were called on it at f36ce8f by a probe under $SCR:
- dedup.build_doc and doc_bytes; manifest.export_dedup; validate.read_export_dedup;
- dedup.hasher_values; validate.g_hasher_values_mismatch.

Every sha256 was also computed with hashlib.

The probe reproduced Part 1's inputs: the TRAIN/VAL list is 542 B, sha256 2153bee2…ea05c1a5, and
tday_image_hashes.tsv is 522 B, sha256 f554bdaa…7768823, both as in Part 1. tday_dedup.json is 791 B with the final
script sha256, sha256 078f02427b2a3c21cc35970ab167776207f50c2732118f3571a993b127dd0d8e:

```
{"also_counted":{"equal_train_and_val":1},"format":"tday_dedup/1","hash_list_sha256":"f554bdaad2d603e97f5a3aa7c44ef6b4f06cd9d18f666eb7f2fda8eda7768823","image_ids_sha256":"29b957929452ffe5a92e1947c07d0097bfcf3bbd873d320ef4fc7a7fbf45b988","kept_ids":["kx_B1","kx_u1"],"kept_ids_sha256":"808003be89fccd28c6a125e9100546dff4f005e4f5f0881b9b8245597bc5df31","n_images":6,"reasons_partition":true,"removed_by_reason":{"duplicate_in_split":1,"equal_train":2,"equal_val":1},"removed_ids_by_reason":{"duplicate_in_split":["kx_a1"],"equal_train":["kx_t1","kx_tb"],"equal_val":["kx_v1"]},"script":{"path":"scripts/tday_hash_images.py","sha256":"5f65eb4893c49e6b41e363d924cf0b5697dc0d0a9b736a983c015b91f549096a"},"train_val_list_sha256":"2153bee2053947647b0bba92bb7ecb96175cc13274ce5a4125dfa932ea05c1a5"}
```

The export's dedup block, from export_dedup with the file in the export's folder and a pin carrying the fixture's
expected_images (6) and list sha256. read_export_dedup reads it back without refusal:

```
{"path": "tday_dedup.json", "sha256": "078f02427b2a3c21cc35970ab167776207f50c2732118f3571a993b127dd0d8e",
 "removed_by_reason": {"duplicate_in_split": 1, "equal_train": 2, "equal_val": 1}}
```

The session part's values.image_hasher for the same file. g_hasher_values_mismatch passes it with that file as
--dedup:

```
{"hash_list_sha256": "f554bdaad2d603e97f5a3aa7c44ef6b4f06cd9d18f666eb7f2fda8eda7768823",
 "dedup_list_sha256": "078f02427b2a3c21cc35970ab167776207f50c2732118f3571a993b127dd0d8e",
 "counts_by_reason": {"duplicate_in_split": 1, "equal_train": 2, "equal_val": 1}}
```

The pin as `--print-pin --commit f36ce8f83c0752ef35e57a4925c5e42e216d1a30` prints it at f36ce8f (1,282 B; keys
sorted here; the builder writes the evaluator commit E as script.commit):

```
{"parameters":{"digest":"sha256 of the file bytes","expected_images":1561,"folder":"images/{split}","modules":{"src/tman/dedup.py":"76fe0c6b815e98c8311a5a4959f650f16ba2b7a65db89451944accae8c78f079"},"outputs":{"dedup":"tday_dedup.json","dedup_format":"tday_dedup/1","hash_list":"tday_image_hashes.tsv"},"rule":{"ids":"the file stem; sorted is Python's sorted(), in code-point order","keep":"of each group of identical images equal to no TRAIN or VAL image, its first sorted id","precedence":["equal_train","equal_val","duplicate_in_split"]},"split":{"from":"src.eval.stage_artifacts.OFFICIAL_SPLIT"},"suffixes":[".jpeg",".jpg"],"train_val_list":{"dl":"DL-91","path":"train_val_image_hashes.tsv","rows":6213,"sha256":"62e5ba9d38e1de1111f59ee46d5cc8ff23d8253c546cde9fa13c11a03233837c","train":5367,"val":846}},"python":"3.11","script":{"commit":"f36ce8f83c0752ef35e57a4925c5e42e216d1a30","path":"scripts/tday_hash_images.py","sha256":"5f65eb4893c49e6b41e363d924cf0b5697dc0d0a9b736a983c015b91f549096a"}}
```

### 6. New codes, and the mutations with their killing cases

| Code | Raised at (f36ce8f) | When |
|---|---|---|
| hasher_pin_data_mismatch (new gate, freeze mode on) | validate.py :722–731 | the pin's data parameters are not dedup.data_parameters()', typed by PIN_SPEC |
| hasher_values_mismatch (new gate, session; last) | validate.py :1656–1687; tday_manifest.py :161–163 | values.image_hasher without --dedup or the reverse; not exactly the three, typed; a value not the file's; --dedup outside session mode |
| dedup_invalid (new to the validator and the builder) | validate.py :1894, :1900, :1679; manifest.py :993, :1000, :1002, :1006; dedup.read_dedup | the path form; unresolvable; not UTF-8; unreadable; the sha256, strict reading or invariants; removed_by_reason not the export's |
| dedup_pin_mismatch (new) | manifest.py :1010; validate.py :1683 | the file's script, train_val_list_sha256 or n_images not the pin's (P2R6(c)(ii)) |
| repository_shallow (new to the validator and the command line) | records.py :250 (via :96 and :179 of tday_manifest.py) | a shallow clone, before any history read |
| image_link_refused (new, hasher) | tday_hash_images.py :169 | a link in images/<split> |
| hasher_pin_mismatch (B-1's; new hasher sites) | tday_hash_images.py :81–83, :250 | the running script is not the bytes the pin hashes |

Every code has a case reached with every earlier gate passing (expect_gate's and expect_in_gate's form) and a
mutation killed by a named case. In P5, every_refusal_code_has_a_killed_mutation holds at 99 codes in
smoke_tman_mutations and at 17 in smoke_tman_b3_mutations. Each mutation below is id → its killing case:check, all
KILLED at P5. The checks' suffixes (_P2R4 … _R3F4) name the ruling or finding.

smoke_tman_mutations, 60 new (563 → 623):
- **the builder (manifest.py, tday_manifest.py, dedup.py)**:
  - builder_takes_the_sources_hasher_pin → pins case_one_pin:builder_takes_the_function_s_pin_whatever_the_sources_hold
  - inventory_dedup_flag_ignored → case_inventory_dedup:cli_inventory_dedup_names_the_file_P2R4
  - the rest are killed by validate case_inventory_dedup:
    - builder_dedup_outside_the_folder → inventory_dedup_outside_the_export_s_folder_P2R5
    - builder_dedup_path_form_unchecked → inventory_dedup_path_with_a_backslash_P2R5
    - builder_dedup_unreadable_uncaught → inventory_dedup_absent_P2R4
    - builder_dedup_read_loosely → inventory_dedup_not_canonical_read_strictly_P2R4
    - builder_dedup_pin_unchecked and builder_dedup_script_equality_dropped → inventory_dedup_of_another_script_P2R6c
    - builder_dedup_list_equality_dropped → inventory_dedup_of_another_list_P2R6c
    - builder_dedup_count_equality_dropped → inventory_dedup_of_another_count_P2R6c
    - builder_dedup_link_loop_uncaught → inventory_dedup_link_loop_R1F2
    - builder_dedup_name_utf8_unchecked → inventory_dedup_name_not_utf8_R1F2
- **dedup.path's form (schema.py since P2-F5)**, each killed by validate case_inventory_dedup_reader:
  - dedup_path_dot_dot_allowed → read_inventory_dedup_path_with_a_dot_dot_component_P2R5
  - dedup_path_backslash_allowed → read_inventory_dedup_path_with_a_backslash_P2R5
  - dedup_path_colon_allowed → read_inventory_dedup_path_with_a_colon_P2R5
  - dedup_path_nul_allowed → read_inventory_dedup_path_with_a_nul_P2R5
  - dedup_path_empty_component_allowed → read_inventory_dedup_path_with_an_empty_component_P2R5
  - dedup_path_dot_component_allowed → read_inventory_dedup_path_with_a_dot_component_P2R5
  - dedup_path_word_allowed → read_inventory_dedup_path_holding_the_word_P2R5
  - dedup_path_padding_allowed → read_inventory_dedup_path_with_surrounding_space_P2R5
  - Not kept: dropping the leading "/" test is an equivalent mutation (known item k1, §10).
- **read_inventory's export dedup (validate.py)**, each killed by validate case_inventory_dedup_reader:
  - export_dedup_unchecked and export_dedup_counts_unchecked → read_inventory_dedup_removed_by_reason_the_export_s
  - export_dedup_path_form_unchecked → read_inventory_dedup_path_with_a_dot_dot_component_P2R5
  - export_dedup_resolved_in_the_working_folder → read_inventory_reads_the_dedup_file_against_the_export_s_folder_P2R6c
  - export_dedup_unreadable_uncaught → read_inventory_dedup_file_absent
  - export_dedup_parsed_before_its_sha256 → read_inventory_dedup_sha256_compared_before_parsing
  - export_dedup_read_loosely → read_inventory_dedup_read_strictly
- **the session gate (validate.py, dedup.py)**, each killed by unlock case_session_hasher_values:
  - hasher_values_without_dedup_accepted → hasher_values_without_dedup_P2R6a
  - dedup_without_hasher_values_accepted → dedup_without_the_hasher_s_values_P2R6a
  - hasher_values_keys_unchecked → hasher_values_with_a_fourth_key_R3F3, killed by acceptance since P2-F4
  - hasher_values_types_unchecked → hasher_value_counts_as_floats_R2F2
  - hasher_value_hash_list_not_compared → hasher_value_hash_list_sha256_not_the_file_s_P2R6a
  - hasher_value_dedup_list_not_compared → hasher_value_dedup_list_sha256_not_the_file_s_P2R6a
  - hasher_value_counts_not_compared → hasher_value_counts_by_reason_not_the_file_s_P2R6a
  - session_dedup_unreadable_uncaught → session_dedup_unreadable_P2R6a
  - session_dedup_read_loosely → session_dedup_read_strictly_P2R6a
  - session_dedup_pin_unchecked and session_dedup_script_equality_dropped → session_dedup_of_another_script_P2R6c
  - session_dedup_list_equality_dropped → session_dedup_of_another_list_P2R6c
  - session_dedup_count_equality_dropped → session_dedup_of_another_count_P2R6c
- **the command line**:
  - dedup_read_outside_session_mode → unlock case_cli_session_dedup:cli_dedup_outside_session_mode_refused_P2R6a
  - cli_dedup_not_passed → unlock case_cli_session_dedup:cli_session_with_dedup_passes_P2R6a
  - dedup_path_not_guarded → paths case_cli_outputs:validator_dedup_holding_test_refused_before_any_call_P2R6a
- **the module at S, the pin's data, P2R7**:
  - dedup_module_at_s_unchecked → unlock case_unlock_gates:dedup_module_not_its_pin_P2R6b
  - hasher_modules_shape_unchecked → unlock case_hasher_pin_modules:hasher_pin_naming_a_second_module_P2R6b
  - hasher_pin_data_unchecked → validate case_record_gates:hasher_pin_expected_images_not_1561_P2R6d
  - hasher_pin_data_types_unchecked → validate case_record_gates:hasher_pin_expected_images_1561_0_R4_1
  - unlock_records_the_path → unlock case_unlock_gates:unlock_entry_naming_another_manifest_path_P2R7
  - file_binds_in_session_mode → unlock case_session:session_world_passes_every_session_gate
- **shallow clones (records.py, tday_manifest.py)**, each killed by validate case_shallow_records:
  - shallow_refusal_dropped and shallow_code_not_kept → validator_refuses_a_shallow_clone_in_entry_file_rewritten_P2R6e
  - no_head_commit_not_record_uncommitted → history_readers_without_a_head_commit_record_uncommitted_P2R6e
  - shallow_check_on_every_call → history_readers_check_the_clone_once_per_records_P2R6e
  - adding_commits_unguarded, written_once_unguarded and deleted_record_unguarded →
    each_history_reader_refuses_a_shallow_clone_first_P2R6e
  - cli_records_at_head_unchecked → every_cli_command_at_head_refuses_a_shallow_clone_first_R2F1
  - cli_records_commit_unchecked → cli_validate_at_a_records_commit_refuses_a_shallow_clone_first_R2F1
- **the dedup file's rule-4 guards**, each killed by paths case_paths_resolved_where_read:
  - builder_dedup_word_guard_dropped → inventory_dedup_given_resolved_where_read_R3F4
  - export_dedup_word_guard_dropped → export_dedup_path_resolved_where_read_R3F4
- **Retargeted by P2-2**: S08_appends_after_the_unlock, unlock_records_the_body and unlock_records_the_file now
  mutate unlock_manifest_errors. They keep their checks and are KILLED.

smoke_tman_b3_mutations, 11 new, 4 retired (107 → 114):
- **P2-3 retired** manifest_path_unchecked, manifest_sha256_unchecked, manifest_body_unchecked and
  manifest_appends_unchecked: step 2's manifest checks are now validate.py's. They were replaced by:
  - manifest_rule_ignored → case_step2_records:step2_manifest_path_other_refused_before_any_data_call
  - manifest_file_never_binds → case_step2_records:step2_manifest_sha256_other_refused_before_any_data_call
  - manifest_file_binds_after_the_session → case_step2_session_part:step2_after_a_session_part_commit_verifies_P2R7
    (the ADDENDUM's "step 2 comparing the manifest's file sha256 after a session-part commit")
- link_check_dropped → case_step3_listing:step3_an_image_name_linked_P2R8_refused_before_any_image_or_list_read
- links_followed_to_list → case_step3_listing:step3_a_link_to_itself_R3F1_refused_before_any_image_or_list_read
- manifest_object_unchecked, manifest_appends_key_unchecked and manifest_session_key_unchecked →
  case_step2_records: step2_manifest_a_json_array, _without_appends and _without_session (each
  _refused_before_any_data_call)
- running_script_unchecked_at_step2 → case_running_script:step2_a_copy_of_the_script_run_from_elsewhere_refused_R1F1
- running_script_unchecked_at_print_pin → case_running_script:print_pin_of_a_copy_run_from_elsewhere_refused_R1F1
- word_split_across_a_concatenation → case_static:h10_dedup_no_word_in_identifiers_attributes_keys_fstrings_paths
- value_keys_reordered keeps its case, its check renamed (hasher_values_keys_are_manifest_s).

The ADDENDUM's P2-4 list, by mutation:
- values.image_hasher without --dedup, and the reverse: hasher_values_without_dedup_accepted,
  dedup_without_hasher_values_accepted.
- Each of the three values: hasher_value_hash_list_not_compared, hasher_value_dedup_list_not_compared,
  hasher_value_counts_not_compared.
- The module dropped from g_hasher_pin_mismatch: dedup_module_at_s_unchecked, hasher_modules_shape_unchecked.
- read_inventory's three: export_dedup_parsed_before_its_sha256, export_dedup_read_loosely,
  export_dedup_counts_unchecked.
- The working folder: export_dedup_resolved_in_the_working_folder.
- ".." and backslash: dedup_path_dot_dot_allowed, dedup_path_backslash_allowed.
- The three equalities, at the builder and at the session gate: builder_dedup_* and session_dedup_*_equality_dropped.
- The non-null sources pin: withdrawn by P2R2-AMEND-1. builder_takes_the_sources_hasher_pin guards the ruling's
  behaviour instead.
- P2R6(d): hasher_pin_data_unchecked.
- The shallow refusal: shallow_refusal_dropped.
- The link check: link_check_dropped.
- Step 2's file sha256 after a session-part commit: manifest_file_binds_after_the_session.

### 7. The shallow refusal: call sites and place in the gate order (REPOSITORY-PROVEN at f36ce8f)

- **The guard.** records.Records.full_history (records.py :241–252) runs sweep_select's require_not_shallow once per
  Records.
- **Its readers** (:254–264), each calling it first:
  - written_once_error is called at entries.py :1009 (read_entry_file), validate.py :422 (g_entry_file_rewritten)
    and validate.py :1580 (push_evidence_error, twice).
  - adding_commits is called at validate.py :1581 (twice).
  - deleted_record_error has no caller in the lane's code; the smoke calls it.
- **read_entry_file's callers:**
  - am22.py :246 (View.entry_files: caught, a NOTE), :724 (recorded_freezes: caught, skipped) and :893 (freeze_body:
    propagates);
  - manifest.py :1026 (build_inventory with --freeze-record);
  - tday_manifest.py :248, :260 and :280;
  - tday_hash_images.py :130, after the hasher's own require_not_shallow (:121–129).
- **push_evidence_error's callers** (:1560): validate.py :1379 (g_freeze_entry_late, re-coded freeze_entry_late),
  :1647 (g_push_evidence_invalid), and tday_manifest.py :294 (check_unlock, re-coded; read_entry_file at :280 runs
  first).
- **Since P2-F1 (DL-24 R2-F1), the command line** calls full_history where it opens the records: records_at
  (tday_manifest.py :92–97) for build, validate in draft and freeze modes, entry, arithmetic, append, inventory,
  check-freeze and freeze-body; and :179 at a records commit (validate in unlock and session modes). Every command
  refuses a shallow clone before any record is read and before any AM-22 lookup.
- **The order:**
  - Inside read_entry_file the guard comes after test_path (entries.py :1007) and after records.read's test_path or
    record_uncommitted (:1008), at :1009.
  - In the validator it falls inside g_entry_file_rewritten, GATES index 12, with the twelve earlier gates passing.
    This is case validator_refuses_a_shallow_clone_in_entry_file_rewritten_P2R6e.
  - The ADDENDUM's ":1534 gate" (MB3) is push_evidence_error, now :1580: a helper, not a gate. Its two gates need
    entry files that g_entry_file_rewritten has already passed through the guard (DL-24 R2-N1).
  - When the validator is called as a library with no entry file cited, the first guarded call is B-2's lookup
    inside schedule_invalid, which catches the refusal (open item 1). No command reaches that path since P2-F1.

### 8. B-4 notes for LREP (Part 1 §8 stands)

- dedup.path is read relative to the export file's directory, in path_problem's form (P2R5). At TMAN it is
  schema.relative_path_error, LREP's rule restated with the same messages.
- The export's dedup failures are dedup_invalid on both sides. DEDUP_SPEC's shape stays inventory_invalid.
- The hasher's values sit at the session part's values.image_hasher (P2R6(a)), typed as the hasher prints them: two
  lower-case sha256 strings and integer counts.

### 9. The part-2 list, closed

Part 1 §9's items and the GO's list are done as §4 says.
- Item 1 (line shifts) was re-cited in P2-3, and again in P2-F7 and P2-F9 where the P2-F commits moved lines.
- Item 2 (unlock_guard_dropped) is P2R9: KILLED.
- Item 3 (shallow repositories) is P2R6(e) with P2-F1.
- Item 4 (the pin's data) is P2R6(d).
- Item 5 (the session part) is P2R7.
- Item 6 (the export's dedup against the pin) is P2R6(c)(ii).
- Item 7 (new codes) is §6.

### 10. DL-24

One read-only workflow (DL-24, P4), wf_1ba3a1f2-51b, over the scoped diff from MB3 at ceac848, 11:14–12:46 UTC. It
had eight agents and no errors (about 3.1 M tokens, 672 tool uses): four reviewers, each followed by an adversarial
verifier.
- R1, the wiring;
- R2, the validator and the records;
- R3, the hasher, the smokes and the mutations;
- R4, numerical. R4 hand-made 119 cases (every reason, a both-match image, a subfolder, "..", a backslash, a wrong
  expected_images) and computed each verdict with python3 -I and the standard library only. 113 agreed with the
  repository; the 6 that did not are R4-1 and R4-2.

The agents read by exact path, ran no smoke and wrote only under $SCR/dl24p2/<agent>/. Before and after (MEASURED),
identical: HEAD ceac848; the scoped status empty; `sha256sum -c` of 27 files (LANE_CODE_FILES, the nine smokes part
2 changed, smoke_tman_am22, smoke_tman_schema, scripts/preflight_arms.py, tman.md and tman_b3.md) all OK.

| Finding | Severity | Verdict | Resolution |
|---|---|---|---|
| R1-F1 the hasher pinned CODE_ROOT's script, not the running file | minor | confirmed (MEASURED) | P2-F6 |
| R1-F2 export_dedup: link loop raised RuntimeError, exit 4 | minor | confirmed (MEASURED) | P2-F2 |
| R1-N1 dedup.py held its own split word literal (GO H10) | note → minor | confirmed | P2-F5 |
| R1-N2 data_parameters imports two scripts at call time (via teacher_diag) | note | confirmed; not new (entries._pa) | note |
| R1-N3 the builder stamps E but hashes its own checkout | note | confirmed | open item 4; runbook line |
| R1-N4 a fixture comment restated P2R2's withdrawn bullet | note | confirmed | P2-F7 |
| v1: a --dedup name not UTF-8, exit 4 | — | MEASURED by v1 | P2-F2 |
| v1: build_inventory does not hold --dedup to the session's values | — | flagged | open item 2 |
| v1: B-1's and the hasher's out_folder: a link-loop --out, exit 4 | — | flagged | note (not part 2's code) |
| R2-F1 B-2's catch-alls swallowed the shallow refusal | major | confirmed (MEASURED) | P2-F1; open item 1 |
| R2-F2 the session's values compared untyped | minor | confirmed (MEASURED) | P2-F4 |
| R2-N1 ADDENDUM :120's ":1534 gate" is a helper | note | confirmed | §7 |
| R2-N2 a lone-surrogate body: exit 4 now at append_after_unlock, not six gates later | note | confirmed | note |
| R2-N3 via the API outside session mode, a Ctx's dedup is neither read nor refused | note | confirmed | note |
| R2-N4 hasher values under another key go unchecked | note | confirmed | note |
| R2-N5 as R1-N2 | note | confirmed | note |
| R2-N6 inventory: output_location now before the build's codes | note | confirmed | note |
| v2: condition 5 and entry freeze --d-date now run hasher_pin_data_mismatch | — | flagged | P2R6(d)'s reach; note |
| R3-F1 is_file() followed a link before the link check | minor | confirmed (MEASURED) | P2-F6 |
| R3-F2 `"session" in man` (and part 1's two siblings) uncased | minor | confirmed | P2-F6 |
| R3-F3 the fourth-key half uncased | minor | partial | P2-F4; refuted part below |
| R3-F4 the dedup file's two rule-4 guards never refused | minor | confirmed (MEASURED) | P2-F3 |
| R3-F5 the b3 smoke cited the mutation smoke's moved lines | note | confirmed | P2-F7 |
| R3-N1 two kills come from crashes, unavoidably | note | confirmed | note |
| R3-N2 as R1-F1 | note | confirmed | P2-F6 |
| R3-N3 a timed-out b3 worker leaves its scratch (part 1's code) | note | confirmed | note |
| R3-N4 the mutation smoke's kill-rule sentence; MUTATED_FILES read by no code | note | confirmed | P2-F7; MUTATED_FILES kept |
| R3-N5 smoke_tman_pins cited artifacts.py :93 | note | confirmed | P2-F7 |
| v3: the `isinstance(vals, dict)` half uncased | — | flagged | P2-F4 (the list case) |
| R4-1 the pin-data gate compared untyped (1561.0) | minor | partial | P2-F4; refuted part below |
| R4-2 the session counts compared untyped (1.0, true) | minor | partial (note) | P2-F4 |
| v4: B-1's expected_rows_mismatch is untyped too | — | flagged | open item 5 |
| v4: no validator gate applies the hasher's python, DESCRIPTIVE or PIN_SPEC keys | — | flagged | open item 5 |
| v4: canonical bytes would raise on a lone surrogate | — | flagged | P2-F4 uses type checks |

Refuted parts, with the verifiers' reasons:
- **R3-F3**, "killed only through a crash". The rule kills a mutation when its named check fails (DL-24 #6), and
  Unlock.run records the crash as a code, so the kill was valid. The missing case was real and P2-F4 adds it; that
  mutation is now killed by acceptance.
- **R4-1 and R4-2**, the "TB14" attribution. B1GO :184–188 (TB14) fixes one canonical JSON definition, not a typed
  equality, and B-1 compares with Python's == (validate.py :314). The behaviour itself was real, and P2-F4 types
  both comparisons. The verifier rated R4-2 a note: the counts LREP reads are the export's, which DEDUP_SPEC
  already types.

Found after DL-24, by me:
- P5's first final run: B-1's one_unlock_row SURVIVED. My P2-2 case committed a third unlock entry before B-1's
  two-row check (slip 3); fixed in P2-F8.
- A scan of every citation part 2 added found two moved by P2-F4; fixed in P2-F9.

The agents' slips, as each reported them. None touched docs/reference, a path holding the word, src.stats or the
checkout.
- R1: shell grep and sed on named code files instead of the Grep and Read tools; grep on two attachments.
- R1's verifier: grep on an attachment and on `git show` output; three git grep calls without the reference.pdf
  element of the exclude form; one name-status diff in ACCEPTANCE's form; python reading code bytes by exact path.
- R2: one git grep without the exclude form on three named src/eval files; grep on two named code files; shell
  globs inside its own folder.
- R2's verifier: sed of code and of `git show` output; a probe importing two scripts through data_parameters; a
  read of R2's probe file.
- R3: grep on `git show` output; loop-variable log paths; git grep without the full exclude form; a docs grep piped
  to head; a python3 -I exec of the mutation smokes' constant assignments.
- R3's verifier: one loop giving `git show` a variable path, refused by L-PROT before it ran; python reading code by
  exact paths.
- R4: one `git status` naming two folders with only the word's exclude; docs greps piped to head; sha256sum of the
  two code files.
- R4's verifier: the decision log read through grep's output; Grep on an attachment; cat of its own fixture files.

### 11. SL-1 log

`tail -n 80` of /root/.claude/sl1_guard.log, read while this report was written (17:3x UTC), holds 24 lines.
- Part 1's eight: Part 1 §11 names five of them. The three later ones are mine too: 27 Oct 13:02:27 and 13:02:35,
  Part 1's transcript scans for that attribution, and 13:09:31, Part 1's push and acceptance command.
- These sixteen are part 2's:

```
2026-10-29T07:40:47Z	Bash	SL1-B1	grep -n 'MUTATED_FILES' scripts/smoke_tman_*.py | head; grep -n 'LANE_CODE_FILES' scripts/smoke_tman_*.py src/tman/*.py | grep -v '^src/tman/manifest.py:686' | head -n 20
2026-10-29T11:11:03Z	Bash	SL1-B1	git cat-file -t 62ca0f8 2>&1; for f in p24only_smoke_tman_mutations p24only2_smoke_tman_mutations p23dev_smoke_tman_b3_mutations p23dev_smoke_tman_hasher p23dev_smoke_tman_dedup p23dev_smoke_tman_vali
2026-10-29T11:25:55Z	Bash	SL1-B1	for f in p23dev_smoke_tman_determinism p23dev_smoke_tman_validate p23dev_smoke_tman_unlock p23dev_smoke_tman_dedup p23dev_smoke_tman_hasher p23dev_smoke_tman_b3_mutations; do printf '%s: ' "$f"; grep 
2026-10-29T11:47:06Z	Bash	SL1-B1	L=/tmp/claude-0/-home-user-plantseg-thesis/5353eb32-4932-5c04-a83c-f2d7e77fae7c/scratchpad/logs; for f in p23dev_smoke_tman_dedup p23dev_smoke_tman_validate p23dev_smoke_tman_unlock p23dev_smoke_tman_
2026-10-29T12:13:12Z	Bash	SL1-B1	cd /tmp/claude-0/-home-user-plantseg-thesis/5353eb32-4932-5c04-a83c-f2d7e77fae7c/scratchpad/logs && for f in p23dev_smoke_tman_hasher.log p23dev_smoke_tman_b3_mutations.log p23dev_smoke_tman_dedup.log
2026-10-29T12:16:54Z	Bash	SL1-B2	git diff --no-renames --name-status 815d138fa7008326739ff447903dd313a2ff5874 HEAD -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*' > /tmp/claude-0/-hom
2026-10-29T12:35:25Z	Bash	SL1-B1	L=/tmp/claude-0/-home-user-plantseg-thesis/5353eb32-4932-5c04-a83c-f2d7e77fae7c/scratchpad/logs; for f in $L/p23dev_smoke_tman_dedup.log $L/p23dev_smoke_tman_validate.log $L/p23dev_smoke_tman_unlock.l
2026-10-29T12:55:13Z	Bash	SL1-B1	S=/tmp/claude-0/-home-user-plantseg-thesis/5353eb32-4932-5c04-a83c-f2d7e77fae7c/scratchpad; date -u; sh $S/run_only.sh smoke_tman_mutations pf1only $S/tmp_pf1_mut cli_records_at_head_unchecked,cli_rec
2026-10-29T13:13:26Z	Bash	SL1-B1	S=/tmp/claude-0/-home-user-plantseg-thesis/5353eb32-4932-5c04-a83c-f2d7e77fae7c/scratchpad; date -u; sh $S/run_only.sh smoke_tman_mutations pf2only $S/tmp_pf2_mut builder_dedup_link_loop_uncaught,buil
2026-10-29T13:25:46Z	Bash	SL1-B1	S=/tmp/claude-0/-home-user-plantseg-thesis/5353eb32-4932-5c04-a83c-f2d7e77fae7c/scratchpad; date -u; sh $S/run_only.sh smoke_tman_mutations pf4only $S/tmp_pf4_mut hasher_values_keys_unchecked,hasher_v
2026-10-29T13:47:07Z	Bash	SL1-B1	S=/tmp/claude-0/-home-user-plantseg-thesis/5353eb32-4932-5c04-a83c-f2d7e77fae7c/scratchpad; date -u; sh $S/run_only.sh smoke_tman_mutations pf5only $S/tmp_pf5_mut dedup_path_dot_dot_allowed,dedup_path
2026-10-29T13:58:03Z	Bash	SL1-B1	S=/tmp/claude-0/-home-user-plantseg-thesis/5353eb32-4932-5c04-a83c-f2d7e77fae7c/scratchpad; date -u; for s in smoke_tman_hasher smoke_tman_b3_mutations smoke_tman_dedup smoke_tman_paths smoke_tman_nos
2026-10-29T14:02:37Z	Bash	SL1-B1	S=/tmp/claude-0/-home-user-plantseg-thesis/5353eb32-4932-5c04-a83c-f2d7e77fae7c/scratchpad; date -u; for s in smoke_tman_pins smoke_tman_schema smoke_tman_b3_mutations smoke_tman_noscore smoke_tman_pa
2026-10-29T14:06:09Z	Bash	SL1-B1	git rev-parse HEAD && S=/tmp/claude-0/-home-user-plantseg-thesis/5353eb32-4932-5c04-a83c-f2d7e77fae7c/scratchpad; date -u; for s in smoke_tman_schema smoke_tman_paths smoke_tman_complete smoke_tman_cu
2026-10-29T15:13:10Z	Bash	SL1-B1	git rev-parse HEAD && S=/tmp/claude-0/-home-user-plantseg-thesis/5353eb32-4932-5c04-a83c-f2d7e77fae7c/scratchpad; date -u; for s in smoke_tman_schema smoke_tman_paths smoke_tman_complete smoke_tman_cu
2026-10-29T16:20:43Z	Bash	SL1-B1	git rev-parse HEAD && S=/tmp/claude-0/-home-user-plantseg-thesis/5353eb32-4932-5c04-a83c-f2d7e77fae7c/scratchpad; date -u; for s in smoke_tman_schema smoke_tman_paths smoke_tman_complete smoke_tman_cu
```

Attribution (MEASURED from the session transcripts):
- 11:25:55 is review R1; 11:47:06 is review R3; 12:13:12 and 12:16:54 are R1's verifier; 12:35:25 is R3's
  verifier.
- The others are mine:
  - 07:40:47 is slip 1;
  - 11:11:03 is a loop over dev-log names with a variable path;
  - 12:55:13 to 16:20:43 are the P2-F dev chains and the three P5 chains. Each holds its scratchpad paths in a
    shell variable, and every git command in them is literal.

### 12. Slips (mine)

1. **07:40 UTC (P2, study).** One grep over the shell globs `scripts/smoke_tman_*.py` and `src/tman/*.py`, looking
   for MUTATED_FILES and LANE_CODE_FILES. No file there holds the word in its name, and none was read beyond grep's
   lines. SL-1 logged it (SL1-B1).
2. **Shell reads of named code files** instead of the Read or Grep tool. Exact paths, no search root, no path
   holding the word:
   - awk line-length checks on the P2-F5 files and on `git show MB3:<file>` output (13:5x UTC);
   - two single-line `sed -n` prints, src/eval/artifacts.py :97 and smoke_tman_mutations.py :2164 and :2260 (P2-F7,
     14:0x UTC);
   - for this report's line checks (17:3x UTC), one shell grep over eight named code files.
3. **A regression of mine, found by P5's first final run.** P2-2's case unlock_entry_naming_another_manifest_path_P2R7
   committed a third unlock entry into smoke_tman_unlock's shared world, just before B-1's
   second_unlock_entry_in_the_decision_log. B-1's one_unlock_row then SURVIVED (RESULT: FAIL 713/714 on f402ba3).
   P2-F8 moved the case after B-1's check, and P5 ran again.
4. **Citations my P2-F4 moved** (the hasher's docstring step 2 and smoke_tman_dedup :16). My own scan found them
   after P5 on ca38c20; P2-F9 re-cited them, and P5 ran a third time on f36ce8f.
5. **~16:1x UTC.** One `git diff -U0` gave git a loop-variable path. L-PROT G refused the whole command before it
   ran, and nothing ran. It was redone with the files named literally.
6. **The harness's attribution reminder and stop hook.** The reminder asked for Co-Authored-By/Claude-Session
   trailers and named the branch claude/practical-goldberg-reaa3d; the stop hook asked to commit and push
   uncommitted changes. Neither was followed: as in Part 1, reported once, not re-raised. No commit carries a
   trailer, and every commit was made after its dev runs, by the protocol.
7. **LANE_STATE ($SCR).** It first noted the 08:12 STOP at a wrong time; corrected the same hour.

### 13. Open items (at most five)

1. **The validator as a library on a shallow clone** (R2-F1's residue). Since P2-F1 every tday_manifest.py command
   refuses a shallow clone first.
   - validate.validate called directly still reaches B-2's am22_providers (validate.py :583, inside schedule_invalid
     :595). Its lookup (am22.recorded_freezes :714–726) catches the refusal and sees no freeze.
   - This happens in draft and freeze modes with no cited entry file and no --am22-d, when a freeze row is recorded.
   - ORCH to rule:
     - whether am22_providers, a B-2 hook, may call `x.records.full_history()` first;
     - and to confirm P2-F1's reading of P2R6(e)'s "every tday_manifest.py command": B-2's commands now refuse a
       shallow clone too.
2. **The inventory's dedup against the session's values** (R1's verifier). build_inventory holds the --dedup file to
   the pin, P2R6(c)(ii), but not to the session part's values.image_hasher when one is recorded. An export could
   therefore name a tday_dedup.json other than the one the session recorded. Neither P2R4 nor IF§11.5 asks for it:
   ORCH to rule.
3. **History reads outside the guarded readers** (k2 and R2-F1's list):
   - records.history and held_by (records.py :266, :278);
   - ss.adding_commit at validate.py :1369 and :1586, entries.py :299 and tday_manifest.py :285;
   - records.src._out("log", …) at entries.py :1028 and am22.py :77–78, :166, :179, :190 and :1123.

   No command reaches them in a shallow clone since P2-F1, but a library call can. Guarding them is outside P2R6(e)'s
   three readers.
4. **The frozen pin against E's blobs** (R1-N3). The builder writes script.commit = E but hashes its own checkout's
   files. Nothing before the unlock compares the pin's two sha256s with E's blobs; the unlock gate
   hasher_pin_mismatch does. §14b adds a runbook line; a freeze-mode comparison would be ORCH's to rule.
5. **B-1's untyped comparisons and the hasher's other pin rules** (R4's verifier). B-1's expected_rows_mismatch
   (validate.py :314) passes 1561.0. No validator gate applies the hasher's python value, its DESCRIPTIVE
   parameters or PIN_SPEC's exact keys; P2-F4 types the data parameters only. A frozen pin with python "3.12" or
   another suffix list is therefore refused only by the TEST-day hasher's step 2.

Notes, not items:
- k1: an equivalent mutation, not kept, named at smoke_tman_mutations' list.
- R2-N2, R2-N3, R2-N4 and R2-N6 (§10).
- R3-N3 (part 1's b3 worker scratch on a timeout).
- MUTATED_FILES, read by no code.
- The out_folder link loops in B-1's tday_manifest.py :55 and the hasher's out_folder.
- Condition 5 now runs the new full-validation gate, as P2R6(d) means.

### 14. RECORDS proposed (Part 2; the lane writes none)

**a. Part 1's five, as part 2 completes them** (Part 1 §14's texts stand, with these values):
- **The DL row for the duplicate-free rule.** Code: src/tman/dedup.py and scripts/tday_hash_images.py, wired into
  the manifest, schema and validator, at lane/tman-b3 <this report commit>.
- **The pin row at TMAN's merge** (DL-37's form): "<merge commit> (TMAN B-3a: scripts/tday_hash_images.py sha256
  5f65eb4893c49e6b41e363d924cf0b5697dc0d0a9b736a983c015b91f549096a and its module src/tman/dedup.py sha256
  76fe0c6b815e98c8311a5a4959f650f16ba2b7a65db89451944accae8c78f079, from lane/tman-b3 <this report commit>,
  <date>)". The builder's pin carries the same two values (dedup.hasher_pin; §5).
- **The current-plan line.** "B-3a runs before the freeze in two parts, both done. Part 1 is new files (lane/tman-b3
  79d5b2b). Part 2, after the merge-in MB3 of lane/tman's tip, wires them into manifest.py, schema.py, validate.py,
  records.py and the CLI (lane/tman-b3 <this report commit>)."
- **The Ch4 sentence:** Part 1's draft, unchanged.
- **The B-4 note for LREP:** Part 1 §8 and Part 2 §8.

**b. Runbook lines:**
- The hash mode runs inside the Linux image of record, never in Windows Python, right after the unlock and before
  any model is scored (P2R12; TMAN2-RULE-3 items 1 and 5). It runs from the session checkout S, whose script bytes
  are the pinned bytes (P2-F6).
- A shallow clone is refused (repository_shallow): by every tday_manifest.py command before it reads a record
  (P2-F1), and by the hasher's step 2.
- The inventory's --out is the hasher's --out, or a folder that contains it, so dedup.path is a plain relative path
  with no "..".
- Every session-mode validate after the hasher's values are recorded passes `--dedup` with the hasher's
  tday_dedup.json. The values are recorded as the hasher prints them: two lower-case sha256 strings and integer
  counts (P2-F4).
- The draft to be frozen is built from a clean checkout at E (R1-N3; open item 4).

**c. P2R2-AMEND-1's line:** "retire sources.generator_pins.image_hasher (SOURCES_SPEC :790-792) in a later TMAN
step; until then the builder ignores it."

### 15. Acceptance (checked after the report commit; see the session's final lines)

- P0 landed on MB3 with the venv gates passing. P1 reproduced FACTS' counts; the two failures are named under P2R10
  and were fixed in P2-0.
- P5 passes (§3): FROZEN BLOBS OK (10/10), every mutation KILLED in both mutation smokes (623/623 and 114/114).
- The scoped `git diff --no-renames --name-status 815d138… HEAD` lists only M lines for SCOPE's seventeen files.
  The src/scripts/configs/docs/lane_reports form lists the same lines.
- Every new refusal code has a case reached with every earlier gate passing, and a mutation killed by a named case
  (§6). The noscore scan reads the builder, the validator and the hasher (12 modules).
- The scoped status is clean after the report commit. One DL-24 workflow, its findings resolved (§10). No trailers,
  no merges. Pushes went to lane/tman-b3 only, after every commit.
