# AGENTS.md — project handoff

Read this first. It is written for any coding agent or CLI picking this repo up
cold. For depth, see [ARCHITECTURE.md](ARCHITECTURE.md), [SECURITY.md](SECURITY.md)
and [LEARNINGS.md](LEARNINGS.md); this file is the *operational* summary:
how to run it, where things live, and the traps that already cost hours.

## What this is

**GitHub Agent** — a GitHub assistant: an **MCP tool server** (`server.py`) +
a **LangGraph agent** (single, or supervisor + 3 specialists), **hybrid RAG**
over repos, **approval-gated writes**, **guardrails**, **usage metering**,
**evals**, and two front-ends (Streamlit UI, FastAPI/SSE API).

- Owner/GitHub: `Varshini0511` · Repo: `Varshini0511/github-agent-mcp` (**public**, branch `master`)
- Python 3.11, Windows 11, venv at `.\venv` (use `.\venv\Scripts\python.exe`)
- Purpose: a **learning + portfolio** project. Quality, explainability and honest
  documentation matter more than feature count.

## Commands

```powershell
# tests — offline, no network, no keys needed beyond .env/dummies (~10 s)
.\venv\Scripts\python.exe -m pytest -q            # expect: 163 passed, 1 skipped
.\venv\Scripts\python.exe -m ruff check .         # lint
.\venv\Scripts\python.exe -m ruff format .        # CI enforces `ruff format --check`

# run
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py   # UI  :8501
.\venv\Scripts\python.exe -m uvicorn api:app --port 8000      # API :8000/docs
.\venv\Scripts\python.exe chat.py [--multi]                   # terminal chat

# live (uses real API quota)
.\venv\Scripts\python.exe evals\run.py [case-id]              # golden set + LLM judge
.\venv\Scripts\python.exe scripts\compare_search.py           # dense vs hybrid, side by side
.\venv\Scripts\python.exe scripts\measure_retrieval.py        # objective recall numbers
```

`Settings` requires `GITHUB_TOKEN` and `GROQ_API_KEY` to *exist* (fail-fast);
tests never use them for real (CI sets dummies).

## Map: where to change what

| I want to… | Touch |
|---|---|
| Add / change a GitHub tool | `github_client.py` (HTTP) → `server.py` (`@mcp.tool` wrapper; the **docstring is what the model reads**) → if it writes: add to `WRITE_TOOLS` in `agent_core.py` and to the right set in `_SPECIALIST_TOOLS` in `multi_agent.py` → test in `tests/test_write_ops.py` |
| Change agent behaviour / prompt | `SYSTEM_PROMPT` in `agent_core.py`; specialist prompts in `multi_agent.py` |
| Add middleware | `assemble_agent` (agent_core) / `assemble_supervisor` (multi_agent) — **order matters** |
| Add a setting | `config.py` Field → `.env.example` → README config table |
| Guardrail rule | `guardrails.py` + `tests/test_guardrails.py` |
| Retrieval | `retrieval.py` (pure BM25/RRF) · `rag.py` (index + search) |
| HTTP endpoint | `api.py` + `tests/test_api.py` |
| UI | `streamlit_app.py` + `tests/test_ui.py` (Streamlit `AppTest`) |
| Quality evals | `evals/cases.yaml` |

### Module one-liners

`config` validated settings · `github_client` retrying REST client · `server` MCP tools (22) · `devops_tools`/`ci_logs`/`pr_diff`/`devops_schemas` CI + PR helpers ·
`agent_core` assemble/send/resume/stream_turn + guard & retry middleware · `multi_agent` supervisor ·
`guardrails` redaction + injection quarantine · `observability` usage/cost · `memory` long-term
memory tools · `reflection` optional answer critic · `ratelimit` 429 helpers · `runner` sync↔async
bridge · `rag`/`retrieval` index + hybrid search · `api` FastAPI · `streamlit_app` UI.

## Conventions

- **Test seam:** `agent_core.assemble_agent(model, tools, checkpointer)` builds the *real* graph.
  Tests pass `ScriptedChatModel` (`tests/fakes.py`) so tool loop, interrupts, guardrails,
  streaming and the supervisor run deterministically offline. Prefer extending these over
  mocking internals. Tests import helpers flat: `from fakes import ...`.
- **Offline by default.** Anything needing a network/key must skip or be opt-in
  (`RUN_EVALS=1`, `pytest.importorskip`).
- ruff: line length 100, rules `E,F,I,B,UP,SIM`. Flat module layout (no package dir).
- Comments explain **why**, not what. Don't add files nobody asked for.
- Errors returned by tools are written for an **LLM to read and act on**.
- Never overclaim in docs — measured numbers and known limitations are stated plainly.

## Gotchas that already cost real time

1. **Settings are cached at process start** (`@lru_cache`). After *any* `.env` edit,
   restart the servers. A stale server holding an expired token once looked exactly like
   "the new token doesn't work".
2. **The MCP server is a child process with a minimal environment.** It finds `.env` by
   *absolute path* (`config.py`). On Streamlit Cloud there is no `.env`, so `streamlit_app.py`
   writes one from `st.secrets` at startup.
3. **Async resources belong to one task.** `MCPAdapter` and the SQLite saver/store must be
   used by the task that opened them. Use `runner.AgentWorker` (one long-lived owning task;
   jobs run as child tasks). Never `runner.run(build_agent().__aenter__())` and use the agent
   later — it fails as "Connection closed" / "threads can only be started once".
4. **Never `git add -A` / `git add .`.** `.claude/launch.json` carries the owner's local
   edits. Stage explicit paths. `ride_smart/` and `browser_sessions/` (live login sessions!)
   are an unrelated project, gitignored — never commit or touch them.
5. **Don't write regex/escape-heavy code through shell heredocs** on Windows — backslashes
   get mangled (a `\b` became a literal backspace once). Use a file-write tool.
6. **Console is cp1252**; CLIs `reconfigure(stdout, utf-8)`. Keep script output ASCII-safe.
7. **Provider quirks:**
   - Groq `gpt-oss` rejects hallucinated tool args with HTTP 400 `tool_use_failed` →
     `retry_malformed_tool_call` re-prompts with what was wrong.
   - Gemini returns `content` as a **list of blocks** → always `content_to_text()`.
   - Gemini's SDK does **not** auto-retry 429s → `retry_on_rate_limit` parses the hint.
   - Embedding model is `models/gemini-embedding-001` (`text-embedding-004` 404s).
8. **Free-tier limits shape everything:** `gemini-3.6-flash` ≈ 5 req/min,
   `gemini-flash-lite-latest` ≈ 15/min, embeddings ≈ 100/min, Groq ≈ 8k tokens/min.
   One "add a feature" request is ~17–40 model calls.
9. `.streamlit/config.toml` disables the file watcher on purpose (`.pyc` writes triggered
   reruns mid-startup and raced the background thread).
10. LangChain 1.x: `create_agent` (not `create_react_agent`); `langchain.mcp` is beta.
    HITL resume payload is `Command(resume={"decisions": [{"type": "approve"}]})`.

## Configuration (names only — never commit values)

`GITHUB_TOKEN` (required) · `CHAT_MODEL` · `GOOGLE_API_KEY` / `GROQ_API_KEY` ·
`ALLOW_WRITES` · `GUARDRAILS_REDACT_PII` · `ENABLE_REFLECTION` · `RAG_HYBRID` ·
`AGENT_RUN_LIMIT` · `API_KEY` · `MODEL_PRICES_JSON` · `LANGSMITH_*` (optional tracing).
Full list with comments: `.env.example`.

Local `.env` today: model `google_genai:gemini-3.6-flash`, `ALLOW_WRITES=true`
(local only — keep it **false** on any shared/public deployment). The paid Gemini tier is
**not confirmed active** (AI Studio showed "No billing account" when last checked), so assume
free-tier limits.

## Current state

- Local `master` is ahead of `origin/master` (indexing hardening, AGENTS.md, DevOps agents):
  **not pushed**. Ask the owner before pushing.
- **DevOps agents** (`ci_triage`, `pr_reviewer`; read-only, structured `CIReport`/`PRReview`
  with quote grounding) are built and tested offline. They diagnose only — no auto-fix.
- **CI webhook** (`webhook.py` + `POST /webhooks/github` in `api.py`): a failed `workflow_run`
  can auto-trigger `ci_triage`. HMAC-verified (`GITHUB_WEBHOOK_SECRET`; 503 until set),
  de-duplicated per delivery, capped per commit (`CI_AUTO_TRIAGE_MAX_PER_COMMIT`), opt-in
  (`CI_AUTO_TRIAGE`, default off). Read-only; fixing still needs human-approved `code_writer`.
  `GET /webhooks/ci` is the activity feed. Offline-tested only (signed fake delivery + stand-in
  runner); never hit by a live GitHub delivery. 179 tests total.
- Backlog: ci_triage -> code_writer auto-fix handoff (still approval-gated) with a retry cap;
  DevOps/webhook evals on fixtures; live-delivery smoke test.
- **Verified offline only:** streaming, long-term memory and reflection were tested against a
  scripted fake model, not yet against a live provider. Docker image build is untested
  (daemon was off). **CI is green on GitHub** (run on `f2f6265`: lint + tests on 3.11/3.12).
- A Streamlit Community Cloud deployment exists (secrets set there).
- No `LICENSE` file yet.

### Open ideas / backlog

- `/health` + UI banner reporting **GitHub token validity / expiry** (proposed, not built).
- Persistent BM25 inverted index (currently rebuilt per query; fine to ~10⁵ chunks).
- Hybrid ranking for *prose* queries is unmeasured; lexical weight is 2.0 for code-shaped
  queries, 1.0 otherwise — don't tune without a metric.
- Incremental re-indexing; Postgres checkpointer/store; reranker.

## Related work (separate repos)

- **`Varshini0511/expense-tracker`** — the app this agent is used to modify (C# .NET API +
  Next.js UI; Supabase, Render, Vercel). In flight: a **policy-type dropdown** feature.
  Branch `add-policy-type` exists with partial commits (verify by reading it); no PR yet; the
  DB column `policy_type` was added by the owner. Vercel's *Root Directory* must be
  `ExpenseTracker/expense-ui`. **The owner wants this done by driving the GitHub Agent, not by
  hand-editing the repo.**
- **Portfolio site** — `C:\NextJS\Portfolio_NextJs\portfolio-app` (Next.js; content in
  `src/data/profile.ts`). Already references this project.

## Working with the owner

- They are **learning**: when asked, explain concepts in plain language with analogies, step by
  step, before showing code.
- They value **verified results** (show the output) and **honest limits** over reassurance.
- Cost-conscious; prefers prepaid / hard caps over open-ended billing.
- Confirm before anything outward-facing: pushing, making repos public, enabling writes,
  changing credentials.

## Safety rules for agents

- Never print, log or commit secrets. `.env`, `*.sqlite*`, `vector_store/`,
  `browser_sessions/` stay out of git.
- Keys that appeared in earlier chat transcripts (Groq / Google) should be treated as exposed —
  recommend rotating them if transcripts are shared.
- Don't enable `ALLOW_WRITES` on a public deployment; the approval gate is a backstop, not
  a substitute for least-privilege tokens.
