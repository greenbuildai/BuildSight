---
name: planning-tasks
description: Generates comprehensive implementation plans for multi-step tasks. Use when you have a spec or requirements, before touching code.
---

# Planning Implementation Tasks

## When to use this skill
- When you have a clear design or spec.
- Before starting any coding on a feature or new component.
- When the user asks for "writing plans" or implementation strategy.
- When transitioning from `brainstorming-ideas`.

## Workflow
1.  **Check Context**: Run in a dedicated worktree if available.
2.  **Define Granularity**: Break tasks into 2-5 minute steps.
3.  **Draft Implementation**: Write the plan to `docs/plans/YYYY-MM-DD-<feature-name>.md`.
4.  **Validate**: Offer execution options (Subagent-Driven or Parallel Session).

## Instructions

### 1. Plan Structure
Every plan **MUST** start with this header:
```markdown
# [Feature Name] Implementation Plan

> **For AI Agents:** REQUIRED SUB-SKILL: Use `executing-plans` (if available) to implement this task-by-task.

**Goal:** [One sentence describing what this builds]
**Architecture:** [2-3 sentences about approach]
**Tech Stack:** [Key technologies/libraries]
```

### 2. Task Granularity
Each step must be an atomic action (2-5 minutes):
*   Create failing test.
*   Run test (verify fail).
*   Create minimal implementation.
*   Run test (verify pass).
*   Commit.

### 3. Output Format
Structure tasks clearly:
```markdown
### Task N: [Component Name]

**Files:**
- Create: `exact/path/to/file.py`
- Modify: `exact/path/to/existing.py:123-145`
- Test: `tests/exact/path/to/test.py`

**Step 1: Write the failing test**
[Code snippet]

**Step 2: Verify Failure**
[Command & Expected Output]

**Step 3: Implementation**
[Code snippet]

**Step 4: Verify Success**
[Command & Expected Output]

**Step 5: Commit**
[Git command]
```

### 4. Implementation Rules
*   **Exact Paths:** Use absolute or relative paths meticulously.
*   **DRY/YAGNI:** Do not over-engineer.
*   **TDD:** Test-Driven Development is mandatory unless impossible.
*   **Commits:** Small, frequent, atomic commits.

### 5. Transition
After saving the plan, offer:
*   **Option 1: Subagent-Driven:** Use `subagent-driven-development` (if available).
*   **Option 2: Parallel Session:** Create a new session with `executing-plans` (if available).
