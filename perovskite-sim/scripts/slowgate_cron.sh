#!/bin/bash
# Weekly SLOW-GATE run: `pytest -m slow` on the current main working tree.
# Rationale: the slow suite is the ONLY gate covering ion-coupled full J-V
# sweeps; the 2026-05-29 e9.3 clamp regression (~150x sweep slowdown) sat
# invisible for 8 weeks precisely because this gate never ran between
# manual validation passes. This job makes that gap structurally impossible.
#
# Installed via launchd: ~/Library/LaunchAgents/com.shane.solarlab-slowgate.plist
# (Sunday 04:00). Remove with:
#   launchctl unload ~/Library/LaunchAgents/com.shane.solarlab-slowgate.plist
#   rm ~/Library/LaunchAgents/com.shane.solarlab-slowgate.plist
#
# Logs: ~/Library/Logs/solarlab-slowgate/  (outside the repo + OneDrive)
#   cron.log            one line per invocation (start/done/skip + summary)
#   slow-<TS>.log       full pytest output of each run

REPO="/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab/perovskite-sim"
CONDA_BIN="/Users/shane/Applications/anaconda3/bin"
# launchd gives a minimal PATH (/usr/bin:/bin); anaconda must lead.
export PATH="$CONDA_BIN:$PATH"
export OPENBLAS_NUM_THREADS=1
LOGDIR="$HOME/Library/Logs/solarlab-slowgate"
LOCKDIR="$LOGDIR/.slowgate.lock.d"

mkdir -p "$LOGDIR"

# Single-instance lock (mkdir is atomic on macOS).
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "$(date '+%F %T') skip: previous slow-gate run still active" >> "$LOGDIR/cron.log"
  exit 0
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT

TS="$(date '+%Y%m%d-%H%M%S')"
LOG="$LOGDIR/slow-$TS.log"
cd "$REPO" || { echo "$(date '+%F %T') ERROR: repo missing" >> "$LOGDIR/cron.log"; exit 1; }

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "$(date '+%F %T') start slow-gate at $COMMIT -> $LOG" >> "$LOGDIR/cron.log"

"$CONDA_BIN/python" -m pytest -m slow -q --no-cov > "$LOG" 2>&1
RC=$?

SUMMARY="$(grep -E '[0-9]+ (passed|failed)' "$LOG" | tail -1)"
echo "$(date '+%F %T') done rc=$RC at $COMMIT: ${SUMMARY:-no summary}" >> "$LOGDIR/cron.log"

# Loud marker file when the gate is red, so a red run is visible at a glance.
if [ $RC -ne 0 ]; then
  echo "$(date '+%F %T') commit=$COMMIT rc=$RC ${SUMMARY:-see $LOG}" >> "$LOGDIR/RED.log"
fi
exit $RC
