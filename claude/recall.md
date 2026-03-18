---
name: recall
description: >
  Runs the agent memory retrieval ritual before starting work. Use this skill automatically
  before any non-trivial task — implementing a feature, fixing a bug, writing a migration,
  designing an API, or touching anything that has architectural implications. Trigger when
  the user says "let's", "can you", "add", "fix", "implement", "refactor", "build", or
  starts describing work to be done. Also trigger when the user explicitly says "/recall"
  or "recall memory". Do NOT trigger for simple questions, explanations, or single-line edits.
---

# Recall — Memory Retrieval Ritual

## What to do

1. Determine the query from the upcoming task description. Be specific — use the actual domain
   (e.g. "add authentication to workflow API" not just "authentication").

2. Run:
```bash
python3 ~/agent-memory/memory_cli.py recall \
  --query "<query>" \
  --project YOUR_PROJECT \
  --top-k 10 \
  --format markdown
```

3. Read the output. Present a concise summary — do not dump the raw output:
   - **Gotchas / Errors** relevant to this work (highest priority — mistakes are expensive)
   - **Active decisions** that constrain the approach
   - **Conventions** that must be followed
   - Omit anything clearly unrelated (the CLI already enforces a similarity floor of 0.4 and includes stale memories so relevant ones can resurface)

4. If the DB is unreachable, say so briefly and continue.

5. Then proceed with the task.
