# Audit of a teammate's repo (Exercise 3)

**Repo audited:** https://github.com/km322/bikeshare-open (Ketan), at
commit `6dcc4f7` ("Check cnt_roll_7 on every complete-window row").
**Diff reviewed:** `git diff 44f7edb HEAD`, where `44f7edb` is his
unmodified-starter commit.

**How it was run.** I cloned the repo to a scratch directory. Before
running anything, I checked that his `.claude/settings.json`, hook and
`.claude/agents/leakage-auditor.md` were identical to the starter, and
that his `src/` diff opened no network, subprocess or delete paths. I
ran `make all` there with my pinned venv: exit 0, and his committed
`reports/metrics.json` was reproduced byte for byte. `make test`: 21
passed. Then a headless Claude Code session (Claude Opus 5.5) in his repo
root invoked **his** `leakage-auditor` subagent on that diff, and was
told to output the subagent's report verbatim. Run time was 101 s.

One Bash call from the auditor was denied by permissions: it tried to
build a worktree of the starter commit under `/tmp` to compare
pipelines. That did not block the audit; it used validation numbers from
his repo instead. The only edit to the text below is that the scratch
path is shortened to `<clone>`.

## The `/leakage-auditor` output, verbatim

## Audit of `git diff 44f7edb HEAD`

All five checks pass, and no finding changes a model score or reverses a model conclusion. One sentence in the docs now quotes a wrong number (see the end).

**How I checked:** I read `src/config.py`, the whole diff, and `src/features.py`, `src/train.py`, `src/ingest.py` and all three test files. `make all` reproduces the committed `reports/metrics.json` exactly. `make test` passes all 21 tests. `git status` is clean afterwards. I also ran a separate script at `/tmp/audit_scripts/check.py`.

**1. Target leakage: pass.**
- The features are the 15 columns listed in `metrics.json`. The feature list is built from `FORBIDDEN`, so `casual`, `registered`, `cnt`, `cnt_2d_bfr`, `instant`, `yr` and `days_since_2011` are all excluded.
- No feature equals `cnt`.
- A plain linear model fit on every row using all 15 features only reaches R^2 0.81. If some combination of features added up to `cnt`, this would be close to 1.0.
- The highest correlation with `cnt` is 0.825 (`cnt_roll_7`).

**2. Temporal ordering: pass.**
- In `src/features.py` lines 76-79, every lag and rolling feature shifts before it aggregates. The new `cnt_roll_28` is `shift(1).rolling(28)`.
- `temporal_split` (`src/train.py` lines 56-62) splits by date.
- The only fitted transformer is the `StandardScaler` inside the ridge pipeline, and it is fit on training rows only.
- The commit that adds `cnt_roll_28` justifies it with validation scores only. The mean-baseline test was moved from the test set to the validation set.

**3. Calendar gaps: pass.**
- All lags are built after `reindex_to_calendar`.
- `rolling(28)` keeps its default `min_periods=28`, so any window that includes the missing day 2011-03-10 comes out as NaN and is dropped.
- I confirmed that `cnt_lag_7` and `cnt_roll_7` are now true 7-day windows rather than tied to `LAG_WARMUP`, and that rows 2011-03-11 to 2011-04-07 are removed.

**4. Silent row loss: pass.**
- The one `dropna()` drops exactly 56 rows: 728 raw rows become 306 + 182 + 184 = 672. The allowed maximum is `LAG_WARMUP + LAG_WARMUP * missing_days` = 28 + 28 = 56, so the bound is tight.
- `encode_weather` raises an error on unknown labels instead of quietly turning them into NaN, and a new test covers that.

**5. Baselines: pass.**
- `metrics.json` reports the mean and persistence baselines next to ridge and hgb, on both validation and test.
- The documented gain from `cnt_roll_28` holds up. On the same rows, adding it raises validation R^2 from 0.6462 to 0.6839 for ridge (+0.038) and from 0.1929 to 0.2415 for hgb (+0.049).

**One wrong number in the docs (not a methodology fault):** `docs/DECISIONS.md` line 10 says persistence "is within 0.07 R^2 of ridge and beats it on MAE." The diff left that line unchanged, but after the `cnt_roll_28` change the test R^2 gap is 0.6239 − 0.5099 = 0.114. The MAE half is still true: persistence 840.1 against ridge 879.3. Ridge still loses to "yesterday's count" on test MAE, so the conclusion stands.
- **Fix:** change that line to "within 0.12 R^2 (0.510 vs 0.624) and beats it on MAE (840 vs 879)."

Relevant files:
- <clone>/src/features.py
- <clone>/src/train.py
- <clone>/reports/metrics.json
- <clone>/docs/DECISIONS.md

## My verdict (separate from the auditor's output)

**No finding changes a number that matters.**

- **Pipeline, tests and conclusions:** I agree with all five passes. I
  checked the two that matter most myself. The pipeline reproduces from a
  clean run, and the suite is green.
- **Same feature as mine:** Ketan chose the same feature I did
  (`cnt_roll_28`, with the 7-day windows separated from
  `LAG_WARMUP`). His 15 features and every number in his
  `metrics.json` match mine exactly, so I can confirm the auditor's
  ablation figures (+0.038 / +0.049 valid R^2 on equal rows) from my own
  `/ablation cnt_roll_28` run.
- **The one real finding, a stale number in prose:**
  `docs/DECISIONS.md` line 10 still says persistence is "within 0.07
  R^2 of ridge". After `cnt_roll_28` the test gap is 0.114
  (0.6239 vs 0.5099). I agree with it.
  - It changes a number in the write-up, not in any model or metric.
  - It does not change the conclusion that ridge loses to persistence on
    test MAE (879.3 vs 840.1).
  - My repo had the same stale line, inherited from the starter. I
    corrected it with an appended dated line, since the five starter
    lines must stay.
- **Not covered by these five checks:** the same-day weather inputs that
  the batch auditor raised on my repo. They apply to his too: same
  features. That issue is about how the result is framed (it is not
  leakage), so it is outside this auditor's remit.
