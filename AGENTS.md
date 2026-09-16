# AGENTS.md — regWatch

## Project Overview

RegWatch is a multi-agent regulatory compliance copilot for **DORA × EU AI Act**. Target architecture (per spec) is 5 layers: React UI → Observability → LangGraph orchestration → MCP tools → RAG corpus. **Current state:** layers 1-3 are implemented in Python under `agent/` (249 default tests); the CopilotKit chat scaffold is not yet wired to the graph. The `SETUP/` folder holds planning docs and is **out of scope** for code work — it and `docs/` are gitignored (kept locally, out of the remote).

## Tech Stack & Key Paths

- **Frontend**: React 19 + TypeScript + Vite 8 — `src/main.tsx` → `src/App.tsx`
- **Node backend**: `server.ts` (CopilotKit Runtime v2, port `8200`, `/api/copilotkit`), model `openai:gpt-5-mini`. Not yet connected to the Python graph.
- **Python agent** (`agent/`, one `uv` project, packages `rag`, `mcp_server`, `graph`):
  - `rag/` — EUR-Lex ingestion, structural chunking, Qdrant retrieval with parent-document return
  - `mcp_server/` — 4 regulatory tools over Streamable HTTP (port `8300`); pure sync functions taking the retriever as an argument, so they are consumed both over the wire and in-process
  - `graph/` — LangGraph orchestration: Retrieval → Analyst → Critic, bounded retry, two explicit refusals, demo CLI
- **Package managers**: pnpm (JS, strict policy below) + `uv` (Python 3.12)
- **Styling**: plain CSS + CSS variables, light/dark via `prefers-color-scheme`, modern CSS nesting

## Setup & Commands

```bash
sfw pnpm install          # install JS deps (policy-enforced, see below)
sfw pnpm dev              # Vite dev server (frontend)
sfw pnpm build            # tsc -b && vite build
sfw pnpm lint             # eslint .
sfw tsx server.ts         # CopilotKit runtime backend (separate terminal, port 8200)

sfw pnpm qdrant           # docker compose up -d qdrant
sfw pnpm ingest           # ingest the corpus (also :verify, :dry-run)
sfw pnpm mcp              # MCP server, port 8300 (also mcp:stdio)
sfw pnpm graph "<question>"          # ask the graph one question
sfw pnpm test:agent                  # 249 tests
sfw pnpm test:agent:integration      # needs a reachable Qdrant
```

**Every Python command must run inside WSL2** — the Windows→WSL2 `localhost` relay is broken on this machine, so Qdrant is unreachable from Windows-side Python:

```bash
wsl.exe -e bash -lc "export PATH=\"\$HOME/.local/bin:\$PATH\" && cd '/mnt/c/Users/Lordtoinou/Desktop/Projects/Business Online/Blockchain_Cie/Agents_IA/regWatch' && <command>"
```

No `dev:server` script exists — the Node backend is started manually via `tsx server.ts`.

## Code Style

**TypeScript** — `verbatimModuleSyntax`, `noUnusedLocals`, `noUnusedParameters` enforced. ESM (`"type": "module"`), imports carry `.ts`/`.tsx` extensions. React 19 with `React.StrictMode`, functional components only. CopilotKit v2 imports use the `/v2` subpath. CSS variables for theming, native nesting, no preprocessor.

**Python** — `from __future__ import annotations` everywhere, modern typing (`str | None`). Nodes take their collaborators as keyword arguments and `functools.partial` binds them at graph assembly, so dependency injection survives the framework and tests hand in fakes. Docstrings explain *why*, not *what*.

**Both** — code comments in English; docs, commits and reports in French. **Never reference specs in code** (no `# per SPEC §6.1`, no `# DC21`) — traceability lives in the documents. Commit style: `feat(scope) - lowercase english subject`.

**Tests** — pytest, TDD strict (failing test first, with the right error). `integration` and `network` markers are excluded by default via `addopts`. No JS tests, no CI yet.

## Supply-Chain Policy (pnpm-workspace.yaml)

The pnpm policy is **intentionally strict** — do not loosen it to unblock installs:

- `minimumReleaseAge: 10080` (7 days) + `minimumReleaseAgeStrict: true` — packages newer than 7 days are rejected.
- `trustPolicy: no-downgrade` — blocks cross-branch trust downgrades.
- `blockExoticSubdeps: true`, `strictDepBuilds: true`, `dangerouslyAllowAllBuilds: false`.
- `trustPolicyExclude` lists `vite`, `semver`, `pino`, `undici` — these are documented false positives from pnpm's migrated-CICD bug (#10202). Each has a justification comment. Do not remove without re-verifying.
- `allowBuilds`: `esbuild: true` (needed by vite), `@scarf/scarf: false` (telemetry, denied).

If an install fails due to policy, **read `pnpm-workspace.yaml` first** — the answer is usually already documented there.

Python deps are pinned exactly (`langgraph==1.2.11`, `langchain-mistralai==1.1.6`) in `agent/pyproject.toml`, with `agent/uv.lock` committed. Add a dependency by editing `pyproject.toml` and running `uv sync` — never by installing into the environment directly.

## Working Principles

### 1. Think Before Coding
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First
- No features beyond what was asked.
- No abstractions for single-use code.
- No speculative flexibility or configurability.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

### 3. Surgical Changes
- Don't touch adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

### 4. Goal-Driven Execution
- Transform tasks into verifiable goals with explicit success criteria.
- For multi-step tasks, produce a numbered plan with a verify step each.
- Loop until criteria are met.

## Meta — Keep This File Alive

After a correction or clarification reveals a missing or wrong durable rule:
- Broken rule → add the rule. Missing context → add the context. Wrong abstraction → document the right one.
- Update only if the change is durable or recurrent. Never auto-update without explicit user confirmation.
- If this file exceeds 150 lines, condense before adding any new rule.
