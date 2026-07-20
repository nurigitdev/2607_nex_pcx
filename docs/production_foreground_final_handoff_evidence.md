# Slice 294: Foreground Final Handoff Evidence

This Slice adds a final foreground handoff checklist for the supervised
pre-CX go-live mode.

The checklist validates:

- foreground operation evidence
- foreground production launch evidence
- foreground production shutdown evidence
- foreground go-live summary evidence
- bounded foreground worker command plan
- guarded foreground worker runner evidence
- foreground app and worker supervisor evidence
- operator handoff bundle manifest

Command:

```bash
./.venv/bin/python scripts/summarize_foreground_final_handoff.py \
  --json-output artifacts/foreground_final_handoff.json \
  --markdown-output artifacts/foreground_final_handoff.md \
  --pretty
```

Expected result:

- Exit code: `0`
- Status: `warning`
- Reason: foreground operation intentionally accepts manual process
  supervision and does not provide automatic restart.
- Failed checks: `0`

Operator rule:

- `warning` is acceptable only when all required checks pass.
- `blocked` must stop handoff until the failed check is corrected.
- Service registration can remain deferred for this controlled foreground
  operation mode.
