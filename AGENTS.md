# AGENTS.md — regWatch

## Project Overview

RegWatch is a multi-agent regulatory compliance copilot for **DORA × EU AI Act**. Target architecture (per spec) is 5 layers: React UI → Observability → LangGraph orchestration → MCP tools → RAG corpus. **Current state:** only the CopilotKit chat scaffold (frontend + Node runtime) exists. The `SETUP/` folder holds planning docs and is **out of scope** for code work — it is gitignored.

## Tech Stack & Key Paths

- **Frontend**: React 19 + TypeScript + Vite 8 — `src/main.tsx` → `src/App.tsx`
- **Backend**: Node HTTP server — `server.ts` (CopilotKit Runtime v2, port `8200`, `/api/copilotkit`)
- **AI layer**: `@copilotkit/react-core` + `@copilotkit/runtime` (v2 API), model `openai:gpt-5-mini`
- **Package manager**: pnpm (strict supply-chain policy, see below)
- **TypeScript**: project references — `tsconfig.app.json` (src + `server.ts`, DOM, bundler mode) / `tsconfig.node.json` (`vite.config.ts`, Node)
- **Styling**: plain CSS + CSS variables, light/dark via `prefers-color-scheme`, modern CSS nesting — `src/index.css`, `src/App.css`

## Setup & Commands

```bash
sfw pnpm install          # install deps (policy-enforced, see below)
sfw pnpm dev              # Vite dev server (frontend)
sfw pnpm build            # tsc -b && vite build
sfw pnpm lint             # eslint .
sfw pnpm preview          # preview built frontend
sfw tsx server.ts         # CopilotKit runtime backend (separate terminal, port 8200)
```

No `dev:server` script exists in `package.json` — backend is started manually via `tsx server.ts`. TODO: verify if a script should be added.

## Code Style

- TypeScript everywhere; `verbatimModuleSyntax`, `noUnusedLocals`, `noUnusedParameters` enforced.
- ESM (`"type": "module"`); imports use `.ts`/`.tsx` extensions (`allowImportingTsExtensions`).
- React 19 with `React.StrictMode`; functional components only.
- CopilotKit v2 imports use the `/v2` subpath (`@copilotkit/react-core/v2`, `@copilotkit/runtime/v2`).
- CSS: CSS variables for theming, nesting via native CSS, no preprocessor.
- No tests, no CI, no Docker yet.

## Supply-Chain Policy (pnpm-workspace.yaml)

The pnpm policy is **intentionally strict** — do not loosen it to unblock installs:

- `minimumReleaseAge: 10080` (7 days) + `minimumReleaseAgeStrict: true` — packages newer than 7 days are rejected.
- `trustPolicy: no-downgrade` — blocks cross-branch trust downgrades.
- `blockExoticSubdeps: true`, `strictDepBuilds: true`, `dangerouslyAllowAllBuilds: false`.
- `trustPolicyExclude` lists `vite`, `semver`, `pino`, `undici` — these are documented false positives from pnpm's migrated-CICD bug (#10202). Each has a justification comment. Do not remove without re-verifying.
- `allowBuilds`: `esbuild: true` (needed by vite), `@scarf/scarf: false` (telemetry, denied).

If an install fails due to policy, **read `pnpm-workspace.yaml` first** — the answer is usually already documented there.

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
