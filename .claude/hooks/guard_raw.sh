#!/usr/bin/env bash
# PreToolUse hook on Bash: refuse commands that could write under data/raw/.
#
# The Write/Edit deny rules in settings.json cover the Write and Edit tools,
# and Claude Code also matches them against the target of a plain `cp`/`mv`.
# They cannot see a write made inside a script or one-liner
# (python -c "shutil.copy(...)", a redirect, sed -i). The PostToolUse hook
# only fires on Edit|Write under src/. This runs BEFORE the command; exit 2
# blocks it and sends stderr to the agent.
#
# It is a string match, so it can be evaded (a script file that builds the
# path at runtime). The SHA-256 pin checked by `make data` and
# tests/test_contract.py is the backstop that does not depend on parsing.
set -uo pipefail

cmd=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))')

case "$cmd" in
  *data/raw*) ;;
  *) exit 0 ;;
esac

# Anything that names data/raw and can write: file utilities, in-place
# editors, redirects, or a language runtime that can open files.
write_pat='(^|[;&|(`[:space:]])(cp|mv|rm|tee|truncate|dd|install|rsync|ln|touch|chmod|unzip|tar|curl|wget|sed[[:space:]]+-i|perl[[:space:]]+-[a-z]*i|python[0-9.]*|node|ruby|Rscript)([[:space:]]|$)|>'
if printf '%s' "$cmd" | grep -Eq "$write_pat"; then
  echo "Blocked by .claude/hooks/guard_raw.sh: this command names data/raw/ and can write to it:" >&2
  echo "  $cmd" >&2
  echo "data/raw/bike_daily.csv is immutable (sha256 pinned in src/config.py). Read it with" >&2
  echo "cat/head/wc, or load it through src.ingest.load_raw(); changes to raw data go through a human." >&2
  exit 2
fi
exit 0
