# ARGUS collaboration protocol

Read GitHub Issue #1 and the assigned issue before editing. GitHub is the
source of truth. ChatGPT defines scope, acceptance criteria and scientific
constraints and reviews; Codex implements and validates the assigned scope.

- One issue → one dedicated branch (`codex/issue-<N>-<short-name>`) → one PR.
- Implement only the assigned issue; leave unrelated work for separate issues.
- Never merge automatically. Leave the PR open for explicit user approval.
- Preserve feature definitions, thresholds, losses, labels, evaluation splits,
  and paper-facing metrics unless the assigned issue explicitly requires change.
- Never fabricate datasets, metrics, model outputs, API responses, screenshots,
  or experiment results. Clearly label synthetic fixtures and placeholders;
  passing smoke tests is not scientific validation of ARGUS.
- Surface or explicitly count dataset, feature and API failures. Do not silently
  substitute zeros or labels that contaminate evaluation. Existing fallbacks
  must be documented; changes to their semantics need a scoped issue.
- Run `python -m pip install -e .`, `python -m pytest -q`, and a CPU inference
  import/smoke command. Install test dependencies with
  `python -m pip install -e '.[test]'` first when necessary.
- PR descriptions must link the issue, map changed/moved files to packages,
  report exact commands and results, list limitations, and explicitly state
  whether numerical/model behavior changed.
- ChatGPT review outcomes are ACCEPT, CHANGES REQUESTED, or scoped follow-ups
  on the PR; address them within the agreed issue scope.
