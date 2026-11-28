# Contributing

## Documentation citations

Docs cite source with a line range (`Sources/<File>.swift:12-40`) or a Swift symbol (`Sources/<File>.swift#Type.member`). Symbol citations stay valid when the declaration moves inside the same file. They fail only if the symbol is gone or renamed.

Before tagging a release, run:

```bash
python3 scripts/verify_docs.py
```

If line-range citations fail because code moved, retarget them with:

```bash
python3 scripts/verify_docs.py --fix
```

`--fix` only rewrites a citation when the previously hashed text still exists uniquely in the same file, or when the range still clearly names one Swift symbol. It reports anything it cannot resolve instead of guessing.
