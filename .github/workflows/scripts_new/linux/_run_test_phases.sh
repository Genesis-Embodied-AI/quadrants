#!/bin/bash
# Shared helper for the Linux CPU/CUDA test scripts.
#
# The scripts run several pytest "phases" (different markers / coverage modes) under `set -e`, so the FIRST failing
# phase aborts the step.  That keeps attribution clear: the failing phase's pytest output (with its FAILED/ERROR
# summary) is the last thing in the step log, the red step is the test step itself (coverage is collected in a
# separate `if: always()` workflow step), and run_phase re-emits the pytest FAILED/ERROR lines as GitHub Actions
# ::error annotations + a job summary so they show up in the PR without opening the log.

# run_phase <label> <cmd...>: run one phase; on failure, emit ::error annotations and abort the step.
run_phase() {
  local label="$1"; shift
  local log="phase_${label//[^A-Za-z0-9]/_}.log"
  local rc
  echo "::group::pytest phase: ${label}"
  "$@" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  echo "::endgroup::"
  if [ "$rc" -eq 0 ]; then
    return 0
  fi
  set +x
  echo
  echo "######################################################################"
  echo "#  TEST FAILURE in phase: ${label}"
  echo "######################################################################"
  echo "## Test failure in phase \`${label}\`" >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
  if grep -Eq '^(FAILED|ERROR) ' "$log"; then
    while IFS= read -r line; do
      echo "  $line"
      echo "::error title=tests (${label})::${line}"
      echo "- \`${line}\`" >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
    done < <(grep -E '^(FAILED|ERROR) ' "$log")
  else
    echo "  (no pytest summary -- phase crashed/timed out; see phase output above)"
    echo "::error title=tests (${label})::phase exited non-zero with no pytest summary (crash/timeout?)"
  fi
  exit "$rc"
}
