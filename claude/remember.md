---
name: remember
description: >
  Stores the current session's learnings to the agent memory database. Trigger automatically
  when a significant piece of work is complete — a feature is built, a bug is fixed, a design
  decision was made, or important gotchas were discovered. Also trigger when the user says
  "done", "that's it", "thanks", "good session", "/remember", or when the conversation is
  clearly winding down. Do NOT trigger after trivial exchanges or single-question answers.
  When in doubt at session end, offer to run it.
---

# Remember — Memory Storage

## Store individual memories (use `add` with heredoc)

Store each distinct learning as a separate memory. Do NOT write large session transcripts.

Always use heredoc syntax to avoid shell interpolation of `$` and special characters:

```bash
python3 ~/agent-memory/memory_cli.py add \
  --project YOUR_PROJECT \
  --category <category> \
  --content - <<'MEMORY'
One concise fact or insight. Can safely contain $dollar signs, "quotes", and `backticks`.
MEMORY
```

For architectural decisions that should outlast long unrelated sprints, add `--longevity high`:
```bash
python3 ~/agent-memory/memory_cli.py add \
  --project YOUR_PROJECT \
  --category decision \
  --longevity high \
  --content - <<'MEMORY'
Core architectural fact unlikely to change for the life of this project.
MEMORY
```

Categories: `convention`, `gotcha`, `pattern`, `preference`, `decision`

Guidelines:
- One memory per fact. "S3 env vars are named X not Y" is one memory. "dotenv loads from cwd" is another.
- Use `decision` for architectural choices that constrain future work (e.g. "decided to use X over Y because Z").
- Use `--longevity high` for facts unlikely to change for the life of the project (security rules, core architecture decisions). Default (`normal`) for everything else. There is no `permanent` — even important memories should eventually fade if truly abandoned, preventing stale facts from accumulating.
- Be specific and actionable. Include file paths, env var names, concrete details.
- Skip anything vague, obvious, or session-specific.
- Check recall first to avoid duplicates — `add` auto-supersedes similar memories (>0.85 cosine).
