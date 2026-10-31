"""The TEST-day manifest (lane TMAN; AM-17 item 9, AM-18 item 8(c), AM-19 Code, AM-22 Code, AM-23 item 4).

Sub-phase B-1 builds the manifest's formats and catalogue, the score-blind readers of the run records, completion
at an instant, the freeze resolver, the entry printers, the derived-artifact entries and appends, the generator
pins, the builder, the validator and the CLI (scripts/tday_manifest.py). B-3a adds the in-session TEST image hasher
and the duplicate-free list, B-3b the binding, B-2 AM-22's freeze check and its entries.

Modules: schema (formats, vocabularies, names, catalogue), records (readers), completion, schedule, entries,
manifest, validate. Importing the package has no side effects: no git call, no filesystem access, no data read.
"""
