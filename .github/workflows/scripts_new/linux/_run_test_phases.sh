#!/bin/bash
# Shared helpers for the Linux CPU/CUDA test scripts.
#
# The test step runs several pytest "phases" (different markers / coverage modes).  We want every phase to run even
# if an earlier one fails -- so a single CI run surfaces *all* failures and still produces complete coverage data --
# but we must also make the *real* test failures easy to find.  Previously each phase was guarded with
# `|| TEST_EXIT=$?` and the script ended with `exit $TEST_EXIT` *after* the coverage report ran, so
# `gh run view --log-failed` (and the GitHub UI) showed the coverage output, making a plain test failure look like a
# coverage failure.  These helpers instead capture each phase, then re-print the pytest FAILED/ERROR lines at the
# very end and emit them as ::error annotations plus a job summary.

declare -a FAILED_PHASES=()

# run_phase <label> <cmd...>: run a phase, tee-ing its output to a per-phase log, recording failures in
# FAILED_PHASES.  Never aborts the caller (so later phases still run), even under `set -e`.
run_phase() {
  local label="$1"; shift
  local log="phase_${label//[^A-Za-z0-9]/_}.log"
  local rc
  echo "::group::pytest phase: ${label}"
  "$@" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  echo "::endgroup::"
  if [ "$rc" -ne 0 ]; then
    FAILED_PHASES+=("${label}|${log}")
  fi
}

# report_failed_phases: print a consolidated failure summary (the last thing in the step log), emit GitHub Actions
# error annotations + a job summary, and `exit 1` if any phase failed.  Returns 0 (no exit) when everything passed.
report_failed_phases() {
  set +x
  if [ "${#FAILED_PHASES[@]}" -eq 0 ]; then
    echo "All test phases passed."
    return 0
  fi
  echo
  echo "######################################################################"
  echo "#  TEST FAILURES in ${#FAILED_PHASES[@]} phase(s)"
  echo "######################################################################"
  echo "## Test failures" >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
  local entry label log line
  for entry in "${FAILED_PHASES[@]}"; do
    label="${entry%%|*}"
    log="${entry##*|}"
    echo
    echo "=== phase '${label}' ==="
    if grep -Eq '^(FAILED|ERROR) ' "$log"; then
      while IFS= read -r line; do
        echo "  $line"
        echo "::error title=tests (${label})::${line}"
        echo "- \`${line}\`" >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
      done < <(grep -E '^(FAILED|ERROR) ' "$log")
    else
      echo "  (no pytest summary -- phase crashed/timed out; tail of log:)"
      tail -n 20 "$log" | sed 's/^/    /'
      echo "::error title=tests (${label})::phase exited non-zero with no pytest summary (crash/timeout?)"
    fi
  done
  exit 1
}
