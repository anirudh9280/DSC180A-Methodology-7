# Seeded faults

Exercise 2, plus the failing run that Exercise 1 asks to be recorded here.

**How these were produced.** Every block below is pasted from a real run;
the only edit is that my home directory is shortened to `~`. Each fault
was seeded by a headless Claude Code session (`claude -p`, Claude Opus 5.5)
started in the repo root and told to apply one exact Edit and then stop.
Seeding through the Edit tool, not by hand, is what makes the project's
`PostToolUse` hook (`.claude/hooks/check_leakage.sh`) fire. Its stderr is
taken from the session's hook events (`--include-hook-events`). The red
output is `make test` run straight after the seed. The fix is a second,
fresh session that is *not told what was broken*. Its prompt is only
"`make test` is failing or something in src/ is wrong. Find the cause and
fix it in src/. Do not edit tests/ or data/." **Turns** is Claude Code's
`num_turns` for that fix session.

Caveat on the turn counts: the fault is the only uncommitted change, so
`git diff` points straight at it. These counts are a lower bound on what a
fault buried in a larger diff would cost.

## Exercise 1: the `cnt_roll_28` test, failing before the feature existed

Commit `3c60cd5` holds the test alone. `make test` at that commit:

```text
python -m pytest tests/ -q
.........F.....                                                          [100%]
=================================== FAILURES ===================================
________________ test_roll_28_is_mean_of_28_prior_calendar_days ________________

feats =         dteday  instant  season  ...  cnt_lag_1 cnt_lag_7   cnt_roll_7
0   2011-01-10     10.0  WINTER  ...      822.0...    1787.0  1530.142857
713 2012-12-31    731.0  WINTER  ...     1796.0     920.0  1531.428571

[714 rows x 28 columns]

    def test_roll_28_is_mean_of_28_prior_calendar_days(feats):
        """cnt_roll_28 on 2011-06-15 must equal the mean of raw cnt over the 28
        calendar days strictly before it (2011-05-18 .. 2011-06-14). Expected is
        computed from load_raw(), not from feats, so it cannot share a bug with
        build_features. Unlike the roll_7 test there is no `if` guard: a gap in
        the window fails loudly instead of passing vacuously."""
        raw = load_raw().set_index("dteday")[TARGET]
        date = pd.Timestamp("2011-06-15")
        window = pd.date_range(date - pd.Timedelta(days=28), date - pd.Timedelta(days=1), freq="D")
        assert len(window) == 28
        assert window.isin(raw.index).all(), "window has a gap in the raw data; pick another date"
        expected = raw[window].sum() / 28
>       assert feats.set_index("dteday").cnt_roll_28[date] == pytest.approx(expected)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_leakage.py:72: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self =             instant  season      yr  ... cnt_lag_1 cnt_lag_7   cnt_roll_7
dteday                               ...    ... 1787.0  1530.142857
2012-12-31    731.0  WINTER  2012.0  ...    1796.0     920.0  1531.428571

[714 rows x 27 columns]
name = 'cnt_roll_28'

    @final
    def __getattr__(self, name: str):
        """
        After regular attribute access, try looking up the name
        This allows simpler access to columns for interactive use.
        """
        # Note: obj.x will always call obj.__getattribute__('x') prior to
        # calling obj.__getattr__('x').
        if (
            name not in self._internal_names_set
            and name not in self._metadata
            and name not in self._accessors
            and self._info_axis._can_hold_identifiers_and_holds_name(name)
        ):
            return self[name]
>       return object.__getattribute__(self, name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       AttributeError: 'DataFrame' object has no attribute 'cnt_roll_28'. Did you mean: 'cnt_roll_7'?

.venv/lib/python3.12/site-packages/pandas/core/generic.py:6194: AttributeError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_roll_28_is_mean_of_28_prior_calendar_days
1 failed, 14 passed in 1.35s
make: *** [test] Error 1
```

It fails for the right reason: the column does not exist yet. The window
checks inside the test (28 dates, all present in the raw data) passed
before the lookup, so the test is not failing on its own set-up.

## Fault (a): a forbidden column added to `feature_columns`

**Change** (seeded via Edit):

```diff
diff --git a/src/features.py b/src/features.py
index e3c1859..6aa4b84 100644
--- a/src/features.py
+++ b/src/features.py
@@ -84,7 +84,7 @@ def build_features(df: pd.DataFrame) -> pd.DataFrame:
 def feature_columns(df: pd.DataFrame) -> list[str]:
     """Numeric columns that are safe to feed the model."""
     numeric = df.select_dtypes(include="number").columns
-    return [c for c in numeric if c not in FORBIDDEN]
+    return [c for c in numeric if c not in FORBIDDEN] + ["casual"]
 
 
 if __name__ == "__main__":
```

**Hook stderr** (`PostToolUse:Edit`, exit 2, so it went back to the agent
as a blocking message):

```text
[PostToolUse:Edit exit_code=2]
Leakage tests failed after editing ~/Desktop/DSC_Capstone/DSC180A-Methodology-7/src/features.py:
        renamed in the CSV) sails through it. This compares what the model
        actually receives against the independent list in src/config.py."""
        actual = set(feature_columns(feats))
        declared = RAW_FEATURES | DERIVED_FEATURES
>       assert actual == declared, (
            f"undeclared features {sorted(actual - declared)}; "
            f"declared but missing {sorted(declared - actual)}"
        )
E       AssertionError: undeclared features ['casual']; declared but missing []
E       assert {'atemp', 'ca..._roll_7', ...} == {'atemp', 'cn...yofweek', ...}
E         
E         Extra items in the left set:
E         'casual'
E         Use -v to get more diff

tests/test_leakage.py:42: AssertionError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: fo...
FAILED tests/test_leakage.py::test_feature_set_is_exactly_the_declared_set - ...
2 failed, 9 passed in 0.59s
```

**`make test`:**

```text
python -m pytest tests/ -q
......FF...........                                                      [100%]
=================================== FAILURES ===================================
__________________________ test_no_forbidden_features __________________________

feats =         dteday  instant  season  ...  cnt_lag_7   cnt_roll_7  cnt_roll_28
0   2011-01-31     31.0  WINTER  ...     141....142857  4071.321429
671 2012-12-31    731.0  WINTER  ...      920.0  1531.428571  3969.428571

[672 rows x 29 columns]

    def test_no_forbidden_features(feats):
        """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
        construction, so this cannot discover a *new* leaky column. What it
        catches is someone editing feature_columns() or FORBIDDEN -- which is
        exactly the fault the Step 7 hook demonstrates."""
        cols = set(feature_columns(feats))
>       assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"
E       AssertionError: forbidden columns in feature set: {'casual'}
E       assert not ({'atemp', 'casual', 'cnt_lag_1', 'cnt_lag_7', 'cnt_roll_28', 'cnt_roll_7', ...} & {'casual', 'cnt', 'cnt_2d_bfr', 'days_since_2011', 'instant', 'registered', ...})

tests/test_leakage.py:32: AssertionError
_________________ test_feature_set_is_exactly_the_declared_set _________________

feats =         dteday  instant  season  ...  cnt_lag_7   cnt_roll_7  cnt_roll_28
0   2011-01-31     31.0  WINTER  ...     141....142857  4071.321429
671 2012-12-31    731.0  WINTER  ...      920.0  1531.428571  3969.428571

[672 rows x 29 columns]

    def test_feature_set_is_exactly_the_declared_set(feats):
        """test_no_forbidden_features checks names against FORBIDDEN, so a leaked
        column under a new name (`out["reg"] = out.registered`, or `casual`
        renamed in the CSV) sails through it. This compares what the model
        actually receives against the independent list in src/config.py."""
        actual = set(feature_columns(feats))
        declared = RAW_FEATURES | DERIVED_FEATURES
>       assert actual == declared, (
            f"undeclared features {sorted(actual - declared)}; "
            f"declared but missing {sorted(declared - actual)}"
        )
E       AssertionError: undeclared features ['casual']; declared but missing []
E       assert {'atemp', 'ca..._roll_7', ...} == {'atemp', 'cn...yofweek', ...}
E         
E         Extra items in the left set:
E         'casual'
E         Use -v to get more diff

tests/test_leakage.py:42: AssertionError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: fo...
FAILED tests/test_leakage.py::test_feature_set_is_exactly_the_declared_set - ...
2 failed, 17 passed in 1.51s
make: *** [test] Error 1
```

**Turns to fix: 6.** The fix session's tool calls were `git diff && make
test`, a single Edit removing `+ ["casual"]`, then `make test` (19
passed). The hook re-ran on that Edit and exited 0.

Two tests catch it. `test_no_forbidden_features` is the starter's
name-based gate. `test_feature_set_is_exactly_the_declared_set` is new; it
would also catch the same leak under a name that is *not* in `FORBIDDEN`.

## Fault (b): `.shift(1)` deleted from `cnt_roll_7`

**Change** (seeded via Edit):

```diff
diff --git a/src/features.py b/src/features.py
index e3c1859..a4e2572 100644
--- a/src/features.py
+++ b/src/features.py
@@ -73,7 +73,7 @@ def build_features(df: pd.DataFrame) -> pd.DataFrame:
     # If you add a longer lag, raise LAG_WARMUP in src/config.py.
     out["cnt_lag_1"] = out[TARGET].shift(1)
     out["cnt_lag_7"] = out[TARGET].shift(WEEK)
-    out["cnt_roll_7"] = out[TARGET].shift(1).rolling(WEEK).mean()
+    out["cnt_roll_7"] = out[TARGET].rolling(WEEK).mean()
     # No min_periods: a window touching the absent 2011-03-10 stays NaN and
     # is dropped, rather than becoming a 27-day mean labelled as 28.
     out["cnt_roll_28"] = out[TARGET].shift(1).rolling(LAG_WARMUP).mean()
```

**Hook stderr** (`PostToolUse:Edit`, exit 2):

```text
[PostToolUse:Edit exit_code=2]
Leakage tests failed after editing ~/Desktop/DSC_Capstone/DSC180A-Methodology-7/src/features.py:
[672 rows x 29 columns]

    def test_rolling_mean_excludes_today(feats):
        """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
        by_date = feats.set_index("dteday")[TARGET]
        date = feats.dteday.iloc[200]
        window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
        if all(d in by_date.index for d in window):
            expected = sum(by_date[d] for d in window) / 7
>           assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)
E           assert np.float64(4605.285714285715) == 4724.428571428572 ± 0.00472443
E             
E             comparison failed
E             Obtained: 4605.285714285715
E             Expected: 4724.428571428572 ± 0.00472443

tests/test_leakage.py:84: AssertionError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.fl...
1 failed, 10 passed in 0.35s
```

**`make test`:**

```text
python -m pytest tests/ -q
...........F.......                                                      [100%]
=================================== FAILURES ===================================
_______________________ test_rolling_mean_excludes_today _______________________

feats =         dteday  instant  season  ...  cnt_lag_7   cnt_roll_7  cnt_roll_28
0   2011-01-31     31.0  WINTER  ...     141....428571  4071.321429
671 2012-12-31    731.0  WINTER  ...      920.0  1789.857143  3969.428571

[672 rows x 29 columns]

    def test_rolling_mean_excludes_today(feats):
        """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
        by_date = feats.set_index("dteday")[TARGET]
        date = feats.dteday.iloc[200]
        window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
        if all(d in by_date.index for d in window):
            expected = sum(by_date[d] for d in window) / 7
>           assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)
E           assert np.float64(4605.285714285715) == 4724.428571428572 ± 0.00472443
E             
E             comparison failed
E             Obtained: 4605.285714285715
E             Expected: 4724.428571428572 ± 0.00472443

tests/test_leakage.py:84: AssertionError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.fl...
1 failed, 18 passed in 1.33s
make: *** [test] Error 1
```

**Turns to fix: 6.** The tool calls were the same as for (a): `git diff &&
make test`, one Edit restoring `.shift(1)`, then `make test` (19 passed).
The hook exited 0 on the fixing Edit.

The test only catches this because 2011-09-17 (`feats.iloc[200]` after the 28-day warm-up) has a
complete 7-day window. The starter's `if all(d in by_date.index ...)` guard
means the test would *pass vacuously* if row 200 ever landed within a week
after a gap. The new `cnt_roll_28` tests assert that their window is
complete instead of skipping, for that reason.

## Fault (c): the weather encoder's `raise` removed

### First seeding: nothing noticed

**Change** (seeded via Edit):

```diff
diff --git a/src/features.py b/src/features.py
index e3c1859..e82a39a 100644
--- a/src/features.py
+++ b/src/features.py
@@ -26,8 +26,6 @@ def encode_weather(series: pd.Series) -> pd.Series:
     """
     known = series.dropna()
     unknown = set(known.unique()) - set(WEATHER_ORDER)
-    if unknown:
-        raise ValueError(f"unmapped weather labels: {sorted(unknown)}")
     return series.map(WEATHER_ORDER).astype(float)
 
 
```

**Hook output**: the hook ran and exited 0 with empty stderr, so the agent
was told nothing:

```text
[PostToolUse:Edit exit_code=0]
```

**`make test`:** green.

```text
python -m pytest tests/ -q
...................                                                      [100%]
19 passed in 1.25s
```

This is the finding of the exercise. The raw file contains only `GOOD`,
`MISTY` and `BAD`, so every data-driven test reaches the encoder with
labels it knows, and the guard is never exercised. The fault is real
anyway. The day the source adds a label (`SNOW`, or a lower-case `bad`),
`.map()` returns NaN for it and `dropna()` deletes those days silently.
The starter's row-loss bound would catch that only if the loss went over
its 56-row allowance, and only on data that has not arrived yet.

**Fix, part 1: a test that can see it (7 turns).** A headless session was
told the guard was missing and asked for one test in
`tests/test_leakage.py` (the file the hook runs). The test had to fail on
the faulty `src/`, with no `src/` edits allowed. It wrote
`test_encode_weather_rejects_unseen_labels`, which feeds
`["GOOD", "STORMY", "BAD"]` straight to `encode_weather`. `make test`
against the still-faulty `src/`:

```text
python -m pytest tests/ -q
................F...                                                     [100%]
=================================== FAILURES ===================================
__________________ test_encode_weather_rejects_unseen_labels ___________________

    def test_encode_weather_rejects_unseen_labels():
        """The raw data only holds GOOD/MISTY/BAD, so no data-driven test can see
        this guard. Without it `.map()` turns a new label into NaN and the later
        `dropna()` deletes those days silently -- feed it one directly."""
        s = pd.Series(["GOOD", "STORMY", "BAD"])
        try:
            out = encode_weather(s)
        except ValueError as e:
            assert "STORMY" in str(e)
        else:
>           pytest.fail(f"unseen label 'STORMY' accepted silently; got {out.tolist()}")
E           Failed: unseen label 'STORMY' accepted silently; got [0.0, nan, 2.0]

tests/test_leakage.py:136: Failed
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_encode_weather_rejects_unseen_labels - Fai...
1 failed, 19 passed in 1.34s
make: *** [test] Error 1
```

**Fix, part 2: restore the guard (2 turns, same session resumed).** The
agent used `git restore src/features.py` from Bash, not an Edit, so the
hook did not fire on the fix. That is harmless here, but it is the same
blind spot as the fourth fault below: anything done through Bash is
invisible to an `Edit|Write` matcher. The test was committed (`Fault (c)
was invisible: ...`) before the re-seed below.

### Second seeding: same fault, stronger suite

**Change:** identical to the diff above.

**Hook stderr** (`PostToolUse:Edit`, exit 2):

```text
[PostToolUse:Edit exit_code=2]
Leakage tests failed after editing ~/Desktop/DSC_Capstone/DSC180A-Methodology-7/src/features.py:
=================================== FAILURES ===================================
__________________ test_encode_weather_rejects_unseen_labels ___________________

    def test_encode_weather_rejects_unseen_labels():
        """The raw data only holds GOOD/MISTY/BAD, so no data-driven test can see
        this guard. Without it `.map()` turns a new label into NaN and the later
        `dropna()` deletes those days silently -- feed it one directly."""
        s = pd.Series(["GOOD", "STORMY", "BAD"])
        try:
            out = encode_weather(s)
        except ValueError as e:
            assert "STORMY" in str(e)
        else:
>           pytest.fail(f"unseen label 'STORMY' accepted silently; got {out.tolist()}")
E           Failed: unseen label 'STORMY' accepted silently; got [0.0, nan, 2.0]

tests/test_leakage.py:136: Failed
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_encode_weather_rejects_unseen_labels - Fai...
1 failed, 11 passed in 0.35s
```

**`make test`:**

```text
python -m pytest tests/ -q
................F...                                                     [100%]
=================================== FAILURES ===================================
__________________ test_encode_weather_rejects_unseen_labels ___________________

    def test_encode_weather_rejects_unseen_labels():
        """The raw data only holds GOOD/MISTY/BAD, so no data-driven test can see
        this guard. Without it `.map()` turns a new label into NaN and the later
        `dropna()` deletes those days silently -- feed it one directly."""
        s = pd.Series(["GOOD", "STORMY", "BAD"])
        try:
            out = encode_weather(s)
        except ValueError as e:
            assert "STORMY" in str(e)
        else:
>           pytest.fail(f"unseen label 'STORMY' accepted silently; got {out.tolist()}")
E           Failed: unseen label 'STORMY' accepted silently; got [0.0, nan, 2.0]

tests/test_leakage.py:136: Failed
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_encode_weather_rejects_unseen_labels - Fai...
1 failed, 19 passed in 1.31s
make: *** [test] Error 1
```

**Turns to fix: 6.** The fix session first tried `git checkout --
src/features.py`, which the permission check blocked because it discards
uncommitted work. It then restored the two lines with an Edit; the hook
re-ran and exited 0, and `make test` showed 20 passed.

**Total for (c): 7 + 2 turns to make the fault detectable and fix it; 6
turns to fix it once it was detectable.**

## Fault 4: a `cp` into `data/raw/` from Bash, which the hook cannot see

**The fault.** `data/raw/bike_daily.csv` is overwritten with a copy where
2012-05-15 (a validation day) has 1,500 more `registered` rides and 1,500
more `cnt`. The edit keeps `casual + registered == cnt`, so it looks like
plausible data. Exactly one line differs (sha256 `c43b7001fdb08724...`):

```text
< 501,"2012-05-15","SPRING","2012","MAY","N","TUE",20.748349,35.597736,79.4583,9.875264,625,4490,5115,"Y","MISTY",500,6118
> 501,"2012-05-15","SPRING","2012","MAY","N","TUE",20.748349,35.597736,79.4583,9.875264,625,5990,6615,"Y","MISTY",500,6118
```

### What happened when the agent tried it

Three headless sessions, each told to run exactly one command.
`--allowedTools "Bash(cp:*)"` or `"Bash(python3:*)"` stood in for a human
clicking "allow". In all three, **the PostToolUse hook did not fire**:
its matcher is `Edit|Write` and the tool was `Bash`. The writes were
stopped by other layers:

1. `cp` from outside the repo was **denied**:

```text
$ cp <scratch>/patch/bike_daily.csv data/raw/bike_daily.csv
Permission to use Bash with command cp <scratch>/patch/bike_daily.csv data/raw/bike_daily.csv has been denied.
```

2. `cp` from inside the repo was **denied** too:

```text
$ cp tmp_patch/bike_daily.csv data/raw/bike_daily.csv
Permission to use Bash with command cp tmp_patch/bike_daily.csv data/raw/bike_daily.csv has been denied.
```

   A control run, `cp tmp_patch/bike_daily.csv tmp_patch/copy_test.csv`
   with the same flags, succeeded. So the denial is path-specific: the
   Claude Code build used here (late 2026) matches the
   `Write(./data/raw/**)` deny rule against the *target of a `cp`*. That
   is better than the lesson assumes, but it only works by parsing the
   command line.

3. The same copy as a Python one-liner, which hides the target from that
   parser, got past the deny rule. It was stopped by something else:

```text
$ python3 -c "import shutil; shutil.copy('tmp_patch/bike_daily.csv', 'data/raw/bike_daily.csv')"
Permission for this action was denied by the Claude Code auto mode classifier. Reason: [Undermining Oversight].
```

   That is the **auto-mode classifier** from my own user-level Claude Code
   settings, a model-based check on Anthropic's side. It is not part of
   this repo and does not exist in the course's Ollama setup, so the repo
   should not count on it. I did not try to get around it, and I did not
   do the overwrite myself to get the "after" numbers.

**What the overwrite would do.** I did it in a throwaway copy of the
repo (`rsync` to a scratch directory, then the `cp` there, at commit
`8816ba4`). The real `data/raw` was never written:

```text
python -m src.ingest
OK ingest: 728 rows, 2011-01-03 -> 2012-12-31
python -m src.features
OK features: 672 rows, 15 features -> features.parquet
  temp, atemp, hum, windspeed, dayofweek, is_weekend, is_holiday, is_workday, month_sin, month_cos, weather_severity, cnt_lag_1, cnt_lag_7, cnt_roll_7, cnt_roll_28
make all: exit 0
```

`make test` in the tampered copy: `20 passed`. Validation metrics, real
repo vs tampered copy:

```text
baseline_persistence valid {'r2': 0.4507, 'mae': 900.6} -> {'r2': 0.4447, 'mae': 900.6}
ridge valid {'r2': 0.6839, 'mae': 777.5} -> {'r2': 0.6811, 'mae': 780.9}
hgb valid {'r2': 0.2415, 'mae': 1296.6} -> {'r2': 0.235, 'mae': 1303.6}
```

So nothing in the starter catches it. The pipeline runs, every test
passes, and every validation number moves (and the test-split numbers
would too). The leakage identity test is no help, because the tampered
row still satisfies it.

### What would catch it

- **A hash check in `make data`** (implemented). Pin the SHA-256 of the
  raw file in `src/config.py` and refuse to proceed on a mismatch. This
  is the only guard that does not depend on *how* the bytes changed: a
  `cp`, a script, a redirect, a re-download of a changed mirror, or an
  appended column. It also gets a test, so `make test` goes red too.
- **A PreToolUse hook on Bash** (implemented). Refuse any command that
  names `data/raw` together with something that can write. It runs
  *before* the command, so it prevents the write instead of reporting it
  afterwards. It is a string match, so a script that builds the path at
  runtime gets past it; the hash is the backstop.
- **Filesystem permissions** (not implemented). `chmod a-w
  data/raw/bike_daily.csv` would stop most writes outright, but git does
  not preserve the write bit for others, so each clone would have to redo
  it. That's a setup step, not a property of the repo.
- **A Stop hook running `make test`** (not implemented as a hook). Since
  the hash is now a test, "done" would fail. The PostToolUse hook only
  runs `tests/test_leakage.py`, so it would not catch this on its own.

**The implementation diff:**

```diff
diff --git a/.claude/hooks/guard_raw.sh b/.claude/hooks/guard_raw.sh
new file mode 100755
index 0000000..964e961
--- /dev/null
+++ b/.claude/hooks/guard_raw.sh
@@ -0,0 +1,33 @@
+#!/usr/bin/env bash
+# PreToolUse hook on Bash: refuse commands that could write under data/raw/.
+#
+# The Write/Edit deny rules in settings.json cover the Write and Edit tools,
+# and Claude Code also matches them against the target of a plain `cp`/`mv`.
+# They cannot see a write made inside a script or one-liner
+# (python -c "shutil.copy(...)", a redirect, sed -i). The PostToolUse hook
+# only fires on Edit|Write under src/. This runs BEFORE the command; exit 2
+# blocks it and sends stderr to the agent.
+#
+# It is a string match, so it can be evaded (a script file that builds the
+# path at runtime). The SHA-256 pin checked by `make data` and
+# tests/test_contract.py is the backstop that does not depend on parsing.
+set -uo pipefail
+
+cmd=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))')
+
+case "$cmd" in
+  *data/raw*) ;;
+  *) exit 0 ;;
+esac
+
+# Anything that names data/raw and can write: file utilities, in-place
+# editors, redirects, or a language runtime that can open files.
+write_pat='(^|[;&|(`[:space:]])(cp|mv|rm|tee|truncate|dd|install|rsync|ln|touch|chmod|unzip|tar|curl|wget|sed[[:space:]]+-i|perl[[:space:]]+-[a-z]*i|python[0-9.]*|node|ruby|Rscript)([[:space:]]|$)|>'
+if printf '%s' "$cmd" | grep -Eq "$write_pat"; then
+  echo "Blocked by .claude/hooks/guard_raw.sh: this command names data/raw/ and can write to it:" >&2
+  echo "  $cmd" >&2
+  echo "data/raw/bike_daily.csv is immutable (sha256 pinned in src/config.py). Read it with" >&2
+  echo "cat/head/wc, or load it through src.ingest.load_raw(); changes to raw data go through a human." >&2
+  exit 2
+fi
+exit 0
diff --git a/.claude/settings.json b/.claude/settings.json
index 5c849d8..429db29 100644
--- a/.claude/settings.json
+++ b/.claude/settings.json
@@ -7,6 +7,19 @@
     "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
   },
   "hooks": {
+    "PreToolUse": [
+      {
+        "matcher": "Bash",
+        "hooks": [
+          {
+            "type": "command",
+            "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/guard_raw.sh",
+            "timeout": 10,
+            "statusMessage": "Checking for writes to data/raw..."
+          }
+        ]
+      }
+    ],
     "PostToolUse": [
       {
         "matcher": "Edit|Write",
diff --git a/.gitattributes b/.gitattributes
new file mode 100644
index 0000000..b0d6f8d
--- /dev/null
+++ b/.gitattributes
@@ -0,0 +1 @@
+data/raw/*.csv -text
diff --git a/src/config.py b/src/config.py
index 7c6e33a..9560870 100644
--- a/src/config.py
+++ b/src/config.py
@@ -20,6 +20,12 @@ DATA_URL = (
     "interpretable-ml-book/master/data/bike.csv"
 )
 
+# SHA-256 of the checked-in raw file. `make data` and tests/test_contract.py
+# refuse anything else: the Write/Edit deny rules and the PostToolUse hook do
+# not see a write made from Bash (a script, a redirect), and a value edit that
+# keeps casual + registered == cnt passes every other test.
+RAW_SHA256 = "a6bed1577f6e4de56dcfae6d8315985073cc0ad3446c8c99ed60c86a0b5ad9dc"
+
 TARGET = "cnt"
 
 # --- Three reasons a column is kept out of the feature set --------------
diff --git a/src/ingest.py b/src/ingest.py
index 6654075..571600f 100644
--- a/src/ingest.py
+++ b/src/ingest.py
@@ -2,12 +2,13 @@
 
 Run: python -m src.ingest
 """
+import hashlib
 import sys
 import urllib.request
 
 import pandas as pd
 
-from src.config import DATA_URL, RAW
+from src.config import DATA_URL, RAW, RAW_SHA256
 
 EXPECTED_MIN_ROWS = 700
 REQUIRED_COLS = {"dteday", "cnt", "casual", "registered", "temp", "hum", "windspeed"}
@@ -20,6 +21,22 @@ def download(force: bool = False) -> None:
     urllib.request.urlretrieve(DATA_URL, RAW)
 
 
+def raw_sha256() -> str:
+    return hashlib.sha256(RAW.read_bytes()).hexdigest()
+
+
+def verify_raw() -> None:
+    """The raw file is immutable. Any byte change -- an edited value, an
+    appended column, a re-download of a changed mirror -- stops the pipeline
+    here instead of shifting every metric downstream without a word."""
+    got = raw_sha256()
+    if got != RAW_SHA256:
+        raise SystemExit(
+            f"FAIL ingest: {RAW.name} sha256 {got[:16]}... != pinned {RAW_SHA256[:16]}...\n"
+            f"  The raw data changed. Restore it: git checkout -- {RAW.relative_to(RAW.parents[2])}"
+        )
+
+
 def load_raw() -> pd.DataFrame:
     df = pd.read_csv(RAW, parse_dates=["dteday"])
     missing = REQUIRED_COLS - set(df.columns)
@@ -32,5 +49,7 @@ def load_raw() -> pd.DataFrame:
 
 if __name__ == "__main__":
     download(force="--force" in sys.argv)
+    verify_raw()
     df = load_raw()
-    print(f"OK ingest: {len(df)} rows, {df.dteday.min().date()} -> {df.dteday.max().date()}")
+    print(f"OK ingest: {len(df)} rows, {df.dteday.min().date()} -> {df.dteday.max().date()}, "
+          f"sha256 {RAW_SHA256[:16]}")
diff --git a/tests/test_contract.py b/tests/test_contract.py
index c2c2176..1b62b36 100644
--- a/tests/test_contract.py
+++ b/tests/test_contract.py
@@ -9,12 +9,13 @@ from src.config import (
     FORBIDDEN,
     LAG_WARMUP,
     RAW,
+    RAW_SHA256,
     RAW_FEATURES,
     REDUNDANT_CALENDAR_COLS,
     TARGET,
 )
 from src.features import build_features, feature_columns, reindex_to_calendar
-from src.ingest import load_raw
+from src.ingest import load_raw, raw_sha256
 from src.train import MODELS, temporal_split
 
 
@@ -68,6 +69,15 @@ def test_every_raw_column_is_classified():
     )
 
 
+def test_raw_file_is_the_pinned_copy():
+    """data/raw/bike_daily.csv is immutable. A value edit that preserves
+    casual + registered == cnt passes every other test and still moves every
+    metric; only the bytes can tell."""
+    assert raw_sha256() == RAW_SHA256, (
+        "data/raw/bike_daily.csv changed; restore it with git checkout"
+    )
+
+
 def test_beats_the_mean_baseline(feats):
     """A model that cannot beat predicting the training mean is not a result."""
     cols = feature_columns(feats)
```

Plus `.gitattributes` (`data/raw/*.csv -text`), so a Windows checkout
with `core.autocrlf` cannot rewrite line endings and break the hash on a
clean clone.

**After the fix, the same tampered copy:**

```text
python -m src.ingest
FAIL ingest: bike_daily.csv sha256 c43b7001fdb08724... != pinned a6bed1577f6e4de5...
  The raw data changed. Restore it: git checkout -- data/raw/bike_daily.csv
make: *** [data] Error 1
make all exit 2
```

`make test` (the assertion and summary lines, cut from the full output):

```text
E       AssertionError: data/raw/bike_daily.csv changed; restore it with git checkout
FAILED tests/test_contract.py::test_raw_file_is_the_pinned_copy - AssertionEr...
1 failed, 20 passed in 1.31s
```

**The PreToolUse hook, live** (headless session, same flags; the source
file did not exist, so nothing could have been written even if every
guard had failed):

```text
[PreToolUse:Bash exit_code=2]
Blocked by .claude/hooks/guard_raw.sh: this command names data/raw/ and can write to it:
  cp tmp_patch/bike_daily.csv data/raw/bike_daily.csv
data/raw/bike_daily.csv is immutable (sha256 pinned in src/config.py). Read it with
cat/head/wc, or load it through src.ingest.load_raw(); changes to raw data go through a human.
```

```text
[PreToolUse:Bash exit_code=2]
Blocked by .claude/hooks/guard_raw.sh: this command names data/raw/ and can write to it:
  python3 -c "import shutil; shutil.copy('tmp_patch/bike_daily.csv', 'data/raw/bike_daily.csv')"
data/raw/bike_daily.csv is immutable (sha256 pinned in src/config.py). Read it with
cat/head/wc, or load it through src.ingest.load_raw(); changes to raw data go through a human.
```

Both runs: the tool result came back as an error, the hook's stderr went
to the agent, and `data/raw/bike_daily.csv` still hashes to
`a6bed1577f6e4de5...`.

