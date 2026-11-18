---
name: ablation
description: Measure what one feature (or a group) contributes, on validation only, by refitting ridge and hgb on identical rows with and without it. Use when deciding whether to keep or drop a feature, or when a before/after comparison also changed which rows survive.
allowed-tools: Read, Edit, Bash(make features*), Bash(python -m src.ablation*)
---

# Current feature set

!`python -c "import pandas as pd; from src.features import feature_columns; print(', '.join(feature_columns(pd.read_parquet('data/processed/features.parquet'))))"`

# Ablating: $ARGUMENTS

1. **Refresh the table.** Run `make features` so the parquet matches `src/`.
2. **Run** `python -m src.ablation $ARGUMENTS`. If it says a name is not a
   feature, stop and report the list it printed; do not guess a spelling.
3. **Sanity-check.** The ridge and hgb "with" rows must equal the `valid`
   entries in `reports/metrics.json`. If they differ, `metrics.json` is
   stale: run `make train` and say so; do not paper over it.
4. **Interpret.** A ridge delta under 0.01 R^2 is inside the noise of a
   182-day validation set: say "no measurable effect", not "helps a bit".
   A negative delta is a finding, report it as one.
5. **Record.** Append one line to `docs/DECISIONS.md`, starting with
   today's date: the feature(s), ridge and hgb valid R^2 with -> without,
   and a *recommendation*. Keeping or dropping is the human's decision;
   write "recommend keep/drop", never "kept/dropped".

Never compute, read aloud, or quote a test-split number in this workflow.
The module discards the test split on purpose.
