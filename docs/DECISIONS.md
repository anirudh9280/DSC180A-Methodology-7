# Decisions

One line per methodological decision: what you chose, why, and what it changed.
Agents append here via the `/new-feature` skill; you append here by hand.

- Exclude `casual`, `registered`: they sum to `cnt` exactly. Leaky model R^2 1.000 -> honest ridge 0.583.
- Exclude `cnt_2d_bfr`: a shipped 2-day lag we cannot test the alignment of. Build our own lags instead.
- Exclude `instant`, `yr`, `days_since_2011`: time indexes let a tree predict a flat constant past 2011 and a linear model extrapolate a trend we have no grounds for.
- Split 2011 / H1 2012 / H2 2012 by date. Test set touched once, after all feature work.
- Report mean and persistence baselines with every score. Persistence (test R^2 0.510, MAE 840) is within 0.07 R^2 of ridge and beats it on MAE; that caveat goes in the report.
- 2026-11-17 Add `cnt_roll_28` (mean cnt over the 28 calendar days before d, `.shift(1).rolling(28)`, no min_periods): every input is observed before d, and windows touching the absent 2011-03-10 are dropped. Valid R^2 ridge 0.6475 -> 0.6839, hgb 0.2185 -> 0.2415; LAG_WARMUP 7 -> 28 shrinks train 348 -> 306 rows. Weekly lags now use `WEEK = 7`, not LAG_WARMUP.
- 2026-11-17 Classify every raw column in `src/config.py` (raw feature / encoded / redundant calendar label / forbidden / date) and declare the derived features. `test_every_raw_column_is_classified` reads the CSV header, so an appended column fails; `test_feature_set_is_exactly_the_declared_set` and a least-squares identity check catch a leaked column under a new name. No metric change.
- 2026-11-17 Ablate `cnt_roll_28` (`src.ablation`, valid only, same 306 train / 182 valid rows both ways). Valid R^2 ridge 0.6839 -> 0.6462 without (dR2 +0.0377, MAE 777.5 -> 835.5), hgb 0.2415 -> 0.1929 without (dR2 +0.0486). The effect holds once row counts are held equal; the "without" figures in the add entry above (0.6475 / 0.2185) came from 348 train rows. Recommend keep.
- 2026-11-17 Ablate `cnt_roll_7` (`src.ablation`, valid only, same 306 train / 182 valid rows both ways). Valid R^2 ridge 0.6839 -> 0.6668 without (dR2 +0.0171, MAE 777.5 -> 791.4), hgb 0.2415 -> 0.2673 without (dR2 -0.0258, MAE 1296.6 -> 1267.7): hgb is *better* without it. Ridge gain is above the 0.01 noise floor but small, and the feature overlaps `cnt_lag_*` / `cnt_roll_28`. Recommend keep for ridge (the stronger model by 0.44 valid R^2); the hgb loss is a finding to report, not a reason to drop.
- 2026-11-17 Decision: keep `cnt_roll_28` (ridge +0.038 valid R^2 on equal rows, ~4x the noise floor) and keep `cnt_roll_7` unchanged from the starter (ridge +0.017; the hgb loss is reported, not tuned away). Feature set frozen here; no further feature iteration.
