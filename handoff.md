# Handoff.md — Task Execution Protocol

## Purpose
This file serves as the communication channel between **Jovi (Planner)** and the **Execution Team** (Theo + Leon).

---

## Team Status

- **Toni (Claude)**: Primary executor — weekly limit exhausted (temporarily unavailable)
- **Leon**: Active coding agent — executing tasks until Toni returns
- **Theo**: Optimizing and coordinating

---

## Role Definitions

| Role | Responsibility |
|------|----------------|
| **Jovi** | Planner — defines goals, requirements, and priorities |
| **Theo** | Optimization & Execution Agent — analyzes, optimizes, validates, and coordinates |
| **Leon** | Coding agent — executes technical tasks |
| **Toni (Claude)** | Primary executor — was handling execution but limit exhausted |

---

## Task Format

```markdown
## Incoming Tasks

### Task #N
- **Priority**: [high/medium/low]
- **Objective**: [clear goal statement]
- **Requirements**:
  - [specific requirement 1]
  - [specific requirement 2]
- **Constraints**: [any limitations]
- **Expected Output**: [what success looks like]
- **Notes**: [any context or references]
```

---

## Status Tracking

| Status | Meaning |
|--------|---------|
| `pending` | Task received, not yet started |
| `in_progress` | Being executed by Leon |
| `review` | Awaiting Theo's validation |
| `completed` | Task finished and validated |
| `blocked` | Needs clarification or resource |

---

## Execution Rules

1. **No task starts until clearly defined** — if unclear, Theo asks Jovi for clarification
2. **Theo optimizes before assigning** — improve the approach before execution
3. **Leon executes with autonomy** — but flags edge cases or risks early
4. **Theo validates before completion** — check correctness and completeness
5. **Blockers get escalated immediately** — don't wait to surface issues

---

## Current Active Tasks

*None yet*

---

## Incoming Tasks

*Waiting for Jovi to add tasks*

---

## Completed Tasks

*None yet*