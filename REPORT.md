# Bikeshare demand: methodology report

## Results

Daily Capital Bikeshare rides (`cnt`), with a temporal split: train = 2011,
validation = H1 2012 (182 days), test = H2 2012 (184 days). "Before" is the
unmodified starter (`reports/metrics_starter.json`, 348 training rows, 14
features). "After" adds `cnt_roll_28` (`reports/metrics.json`, from `make
all` at the final commit, 306 training rows, 15 features). The training
set shrinks because the longest lag window grew from 7 to 28 days.

| model | version | valid R^2 | valid MAE | test R^2 | test MAE |
|---|---|---|---|---|---|
| baseline_mean | before | -0.9318 | 2002.2 | -2.1441 | 2847.7 |
| baseline_mean | after | -0.6894 | 1877.3 | -1.7717 | 2659.6 |
| baseline_persistence | before | 0.4507 | 900.6 | 0.5099 | 840.1 |
| baseline_persistence | after | 0.4507 | 900.6 | 0.5099 | 840.1 |
| ridge | before | 0.6475 | 832.9 | 0.5833 | 947.1 |
| ridge | after | 0.6839 | 777.5 | 0.6239 | 879.3 |
| hgb | before | 0.2185 | 1296.6 | -0.4975 | 1972.6 |
| hgb | after | 0.2415 | 1296.6 | -0.46 | 1952.4 |

Notes on reading the table:

- **Persistence is identical before and after.** It predicts
  `cnt_lag_1`, and the validation and test rows are the same in both
  versions.
- **The mean baseline moves.** That is not the feature. Its training
  mean changed when 42 training rows left the set: 21 in January
  (the longer warm-up) and 21 from 2011-03-18 to 04-07 (the absent
  2011-03-10 now blanks 28 days instead of 7).
- **hgb's validation MAE is the same to one decimal.** That's a
  coincidence: unrounded it is 1296.581 vs 1296.577.

Every feature decision was made on validation. The test columns were
computed by `make train` at each commit, as the starter is built to do,
and no decision depended on them (see "Headline finding").

## Headline finding

A ridge regression on calendar, same-day weather and lagged-demand
features is the best of the four on R^2 (test **0.6239** vs persistence
**0.5099**). It does **not** beat "yesterday's count" on mean absolute
error (test MAE **879.3** vs **840.1**). Gradient boosting fails outright,
with a test R^2 of **-0.46**, worse than predicting yesterday. That is
because a tree trained only on 2011 cannot predict above the range it saw,
and ridership grew about 65% into 2012 (Exercise 4 shows the fix).
Nothing here is leakage: a model given `casual` and `registered` scores
R^2 1.000 (`python -m src.leakage_demo`), and the tests now catch that
column under any name.

**What my feature changed.** `cnt_roll_28` (mean rides over the 28 days
before *d*) raised ridge's validation R^2 from 0.6475 to 0.6839. Some of
that comparison mixes in a different set of training rows, so I
re-measured with the rows held fixed (`/ablation cnt_roll_28`): +0.0377
for ridge and +0.0486 for hgb, about four times the 0.01 noise floor for a
182-day validation set. **I kept it.**

**What the auditor said about this repo** (`reports/audit.jsonl`).
`leakage: none`, `temporal_ok`, `baseline_reported` and `reproducible`
were all true, with four findings:

- The README's test count was stale. I fixed it.
- The Exercise 4 output lists no persistence row. That is true, and it
  matters: see Exercise 4, where I compare against persistence by hand. I
  did not re-run `--final` just to add the row, because that would touch
  test a second time.
- `make train` always scores test. The required `metrics.json` needs test
  numbers, so I kept the starter behaviour and put the guard elsewhere:
  `src.ablation` and Exercise 4 never score test unless asked, and the
  only feature decision (DECISIONS.md, the "keep" line) cites validation
  numbers only.
- Same-day weather is an oracle for a day-ahead forecast. I agree;
  Exercise 5 depends on it.

## Exercise 4: target transformation

**Problem.** hgb cannot extrapolate. Trained on 2011 levels, it predicts
2011 levels.

**Fix.** Make the target relative to a recent level that already carries
the growth, so the tree only has to learn a scale it has seen. With
`python -m src.train --target-transform <name>` (validation only unless
`--final`), I tried three transforms, scoring hgb on validation:

| transform | what the tree predicts | hgb valid R^2 | hgb valid MAE | level-only valid R^2 |
|---|---|---|---|---|
| none | `cnt` | 0.2415 | 1296.6 | - |
| `log1p` | `log(1+cnt)` | 0.1805 | 1328.1 | -0.9891 |
| `ratio_roll28` | `cnt / cnt_roll_28` | 0.6754 | 732.4 | 0.5733 |
| `diff_roll7` | `cnt - cnt_roll_7` | **0.7139** | 737.4 | 0.5947 |

A log transform does not help: a tree is just as bounded in log space.
Both level-relative transforms work. I chose `diff_roll7` (best validation
R^2) and committed that choice before running `--final` once on test:

| diff_roll7, test | R^2 | MAE |
|---|---|---|
| hgb, transformed | 0.6331 | 830.1 |
| level-only (`cnt_roll_7` + training-mean deviation, no model) | 0.4793 | 870.9 |
| ridge, transformed target | 0.6228 | 880.3 |
| *for reference:* untransformed hgb / ridge / persistence (`metrics.json`) | -0.46 / 0.6239 / 0.5099 | 1952.4 / 879.3 / 840.1 |

**Is it defensible, or does it just win the comparison? Defensible as a
method, oversold as a win.**

- **It's a legitimate forecaster.** The inverse adds back `cnt_roll_7`,
  which uses only days before *d* and is covered by a leakage test. Every
  input is known at prediction time, and there is a principled reason for
  it (the trees cannot extrapolate a level).
- **But most of the gain is the level, not the model.** The same inverse
  applied to a constant already gets 0.5947 on validation.
- **The margins are thin.** On test, hgb beats untransformed ridge by
  0.009 R^2 and beats persistence by only 10 rides/day of MAE.
- **The choice was data-driven.** I picked the best of three transforms on
  182 days, so the margin over ridge is not evidence that boosting is
  better here.

The honest claim is narrow: with a level-relative target, gradient
boosting stops failing and roughly ties ridge. `make train` and
`metrics.json` remain untransformed.

## Exercise 5: ridge vs persistence

On test, ridge beats persistence by 0.114 R^2 (0.6239 vs 0.5099) but loses
by 39 rides/day of MAE (879.3 vs 840.1). `python -m src.ex5_compare` shows
why:

- **Ridge is biased.** Fit on 2011, it under-forecasts H2 2012 by **427
  rides/day on average** (bias -426.7). Persistence is essentially
  unbiased (+16.1).
- **Typical days favour persistence.** Its median miss is 506.5, against
  ridge's 806.5. Ridge is closer on only 79 of 184 test days.
- **Big misses favour ridge.** Its 90th-percentile miss is 1711.1 vs
  1960.2. It misses by more than 2,000 rides on 10 days vs 17, and RMSE
  is 1067.8 vs 1219.0. Persistence is worst on the days just after a
  shock: its largest miss is 2012-10-31, two days after Hurricane Sandy.

R^2 rewards ridge's fewer disasters; MAE punishes its bias. The bias is
large enough to outweigh the disasters.

**One more caveat.** Ridge's features include *observed* same-day weather,
which a day-ahead forecaster would only have as a forecast. Its edge on
the bad-weather days is therefore an upper bound.

**Verdict: I would not deploy this ridge over persistence.** For a
day-ahead count judged on MAE, it is worse and more complex. What it does
offer, fewer large misses, comes from information it would not have in
deployment. The version worth testing is ridge refit on all data through
the forecast date, which should remove most of the level bias, with
weather forecasts instead of observations. That needs a fresh holdout
this split no longer provides.

## Attribution

**Model: Claude Opus 5.5 (`claude-opus-5-5`, Anthropic-hosted, not a local
or open-weight model) via Claude Code, for every agent step, including
`agent/audit.py` (`AUDIT_MODEL=claude-opus-5-5`). I did not switch
models.** My laptop does not run Ollama. I used a git-ignored
`.claude/settings.local.json` to point the repo's Claude Code settings at
my Anthropic login instead of `localhost:11434`, as the starter README
allows. The agent work happened in two places:

- **Headless sessions inside the repo** (`claude -p`). These loaded this
  repo's `CLAUDE.md`, skills and hooks, so the hooks fired for real.
- **A driving Claude Code session one directory up**, which wrote most
  files directly. The repo's hooks did not run on those edits; `make
  test` was run after each one instead.

**Agent-generated (all code and prose in the diff from the starter):**

- `/new-feature` session: `tests/test_leakage.py::test_roll_28_is_mean_of_28_prior_calendar_days`
  and `test_roll_28_drops_windows_containing_the_gap`, `cnt_roll_28` and
  `WEEK` in `src/features.py` / `src/config.py`, and the docstring change
  in `test_only_expected_rows_are_dropped`.
- Fault-(c) session: `test_encode_weather_rejects_unseen_labels`.
- Fault-fix sessions: the fixes for faults (a), (b) and (c).
- `/ablation` sessions: the two "Ablate ..." lines in `docs/DECISIONS.md`.
- A headless session in Ketan's repo: the leakage-auditor report in
  `docs/AUDIT.md`. The verdict below it is mine.
- Driving session:
  - the raw-column classification in `src/config.py`;
  - `test_every_raw_column_is_classified`, `test_raw_file_is_the_pinned_copy`,
    `test_feature_set_is_exactly_the_declared_set` and
    `test_no_feature_set_reconstructs_the_target`;
  - `src/ablation.py` and `.claude/skills/ablation/SKILL.md`;
  - the Exercise 4 code in `src/train.py` (`TARGET_TRANSFORMS`,
    `TransformedTarget`, `score_transform`, `--target-transform` /
    `--final`);
  - `src/ex5_compare.py`;
  - `verify_raw` in `src/ingest.py`, `.claude/hooks/guard_raw.sh`, the
    PreToolUse entry in `.claude/settings.json`, and `.gitattributes`;
  - the edits to `CLAUDE.md`, `.gitignore` and `README.md`;
  - `docs/SEEDED_FAULTS.md`, `docs/AUDIT.md` and this report.

**My decisions.** Each one traces to a dated line in `docs/DECISIONS.md`.

- **What to predict, what counts as leakage, how to split, which
  baselines:** kept from the starter, and I agree with each of them (the
  five starter lines).
- **Which agent model to use:** Opus 5.5 rather than installing Ollama.
  This is recorded here, since it is about the setup, not the method.
- **Which feature to add: `cnt_roll_28`.** I chose it from three the agent
  proposed (the add line).
- **Whether to keep it: keep.** Also keep the starter's `cnt_roll_7`,
  even though hgb does better without it (the "Decision: keep" line).
- **Whether the answer is believable:**
  - Exercise 4 is a method fix but not a win (the two Exercise 4 lines).
  - Ridge is not worth deploying over persistence as built (the
    Exercise 5 line).

The agent proposed the Exercise 4 transform, the Exercise 5 verdict and the
keep/drop recommendations. I reviewed and accepted them.
