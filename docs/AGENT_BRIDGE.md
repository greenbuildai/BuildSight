# Gemini ↔ Claude Code Collaboration Bridge

## Shared Architecture
- **Gemini (Antigravity)**: Orchestrator, Planner, Reviewer. Creates tasks, writes execution plans, creates handoffs, and verifies handbacks.
- **Claude Code (CLI)**: Implementer. Executes terminal tools, implements features, runs tests, and creates handbacks.

## Source of Truth
- **Repository Root**: `e:\Company\Green Build AI\Prototypes\BuildSight`
- **Active Branch**: `main` (from initialized git)
- **State Control**: `TASK_STATE.json`
- **MCP Ecosystem**: `21st-dev`, `browsermcp`, `context7`, `github`, `netlify`, `pinecone`, `supabase-mcp-server`, `windows-cli`

## Handoff Protocol
1. **Gemini** updates `CLAUDE_HANDOFF.md` with the explicit implementation plan, required commands, and files to touch.
2. **Gemini** updates `TASK_STATE.json`, setting `current_task.status` to `"pending_claude"`.
3. **Claude Code** reads `CLAUDE_HANDOFF.md`, executes the requested commands/modifications inside the shared workspace.
4. **Claude Code** writes the execution report into `GEMINI_HANDBACK.md`.
5. **Claude Code** updates `TASK_STATE.json`, setting `status` to `"pending_gemini"`.
6. **Gemini** reads `GEMINI_HANDBACK.md` to verify the execution and proceeds to the next objective or loop.
