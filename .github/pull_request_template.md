## Scope

Describe one feature/fix only. If this PR changes a shared schema, state the compatibility impact explicitly.

## Verification

- [ ] `python -m compileall -q src tests`
- [ ] `ruff check ...` passes
- [ ] `mypy ...` passes for MVP data/state modules
- [ ] `python -m unittest discover -s tests/unit -v` passes
- [ ] I merged the latest `origin/dev` into this feature branch before final review
- [ ] README/config changes are included if runtime commands or contracts changed

## SDN integration impact

- [ ] No SDN runtime impact, or
- [ ] Requires the 2C-4S self-hosted integration test after merge to `dev`

## Evidence

Attach concise logs/screenshots only when they help verify role, telemetry, snapshot, migration or rollback behavior.
