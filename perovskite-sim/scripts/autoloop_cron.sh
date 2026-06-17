#!/bin/bash
# Daily autoloop SWEEP (dry-run). Senses parity vs SCAPS, writes gap->cause->lever
# proposals to a timestamped log + the ledger; COMMITS NOTHING. Safe to run unattended.
# Installed via crontab (see scripts/autoloop_cron.sh header / CLAUDE.md). Remove with
# `crontab -e` (delete the autoloop line) or `crontab -r` (clears all).
#
# Logs: ~/Library/Logs/autoloop/  (outside the repo + OneDrive, so no git/sync noise)
#   cron.log          one line per invocation (start/done/skip + rc)
#   sweep-<TS>.log    full stdout/stderr of each sweep (the proposal JSON is at the end)

REPO="/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab/perovskite-sim"
PY="/Users/shane/Applications/anaconda3/bin/python"
LOGDIR="$HOME/Library/Logs/autoloop"
LOCKDIR="$LOGDIR/.sweep.lock.d"

mkdir -p "$LOGDIR"

# Single-instance lock (macOS has no flock); mkdir is atomic. Skip if a run is active.
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "$(date '+%F %T') skip: previous sweep still running" >> "$LOGDIR/cron.log"
  exit 0
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT

TS="$(date '+%Y%m%d-%H%M%S')"
LOG="$LOGDIR/sweep-$TS.log"
echo "$(date '+%F %T') start sweep -> $LOG" >> "$LOGDIR/cron.log"

cd "$REPO" || { echo "$(date '+%F %T') FAIL: cannot cd $REPO" >> "$LOGDIR/cron.log"; exit 1; }
"$PY" scripts/autoloop_run.py --boulder > "$LOG" 2>&1
RC=$?

echo "$(date '+%F %T') done rc=$RC ($LOG)" >> "$LOGDIR/cron.log"
exit 0
