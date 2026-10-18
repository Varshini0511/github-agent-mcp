# GitHub Agent

[![CI](https://github.com/Varshini0511/github-agent-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/Varshini0511/github-agent-mcp/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![style](https://img.shields.io/badge/lint-ruff-informational)

A GitHub assistant built the way you would build one for production: an
**MCP tool server**, a **LangGraph agent** (single or supervisor + specialists),
**hybrid RAG** over repositories, **approval-gated writes**, **guardrails**,
**usage metering**, **evals**, and both a **web UI** and an **HTTP/SSE API**.

> It explains codebases with `path:line` citations, triages issues, and opens
> pull requests — and it is designed so that a hostile README, a pasted token,
> a rate-limited model or a confused loop can't turn into a bad write.

📐 [Architecture](ARCHITECTURE.md) · 🛡️ [Security model](SECURITY.md) · 🧠 [Concepts index](LEARNINGS.md)

## What's inside

| Capability | How it works | Code |
|---|---|---|
| **MCP server** | 22 GitHub tools over stdio, reusable by any MCP client | `server.py` |
| **Agent loop** | LangGraph model ⇄ tools; any chat model via one `CHAT_MODEL` string | `agent_core.py` |
| **Multi-agent** | Supervisor routes to `repo_qa` / `issue_triage` / `code_writer` / `ci_triage` / `pr_reviewer` (scoped tools, own prompts, structured outputs for the DevOps pair) | `multi_agent.py` |
| **Hybrid RAG** | Chunk → embed → FAISS **+ BM25**, fused with Reciprocal Rank Fusion; identifier-aware weighting | `rag.py`, `retrieval.py` |
| **Human-in-the-loop** | Writes freeze the graph at an `interrupt()`; approve/reject resumes it | `HumanInTheLoopMiddleware` |
| **Guardrails** | Secret/PII redaction both ways; prompt-injection quarantine of tool output | `guardrails.py` |
| **Memory** | Per-conversation (checkpointer) **and** cross-conversation (`remember`/`recall`) | `memory.py` |
| **Self-reflection** | Optional critic fact-checks answers against tool output, one bounded revision | `reflection.py` |
| **Observability** | Tokens, model/tool calls, latency, est. cost per turn; JSON log lines; optional LangSmith | `observability.py` |
| **Streaming** | Token + tool-activity events in the UI and over SSE | `agent_core.stream_turn` |
| **Reliability** | Backoff + jitter, rate-limit awareness, corrective retries, bounded loops | `github_client.py`, `agent_core.py` |
| **Evals** | Golden set, deterministic checks + LLM-as-judge, hallucination guard | `evals/` |
| **Tests** | 100+ offline tests, including the *real* agent graph against a scripted fake model | `tests/` |

## Quick start

```bash
git clone https://github.com/Varshini0511/github-agent-mcp && cd github-agent-mcp
python -m venv venv && venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -e ".[dev,rag,ui,api]"

cp .env.example .env     # then set GITHUB_TOKEN and a model key (GOOGLE_API_KEY or GROQ_API_KEY)
streamlit run streamlit_app.py
```

Writes are **off** by default. See [Configuration](#configuration) to enable them.

### Other ways to run it

```bash
python chat.py                 # terminal chat (add --multi for the supervisor team)
python agent.py "question"     # one-shot
uvicorn api:app --port 8000    # HTTP + SSE API  ->  http://localhost:8000/docs
docker compose up --build      # UI on :8501, API on :8000
```

### HTTP API

```bash
# one turn
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message": "What does Varshini0511/expense-tracker do?"}'

# streaming (Server-Sent Events: thread, tool_start, tool_end, token, final | interrupt)
curl -N localhost:8000/chat/stream -H 'content-type: application/json' \
  -d '{"message": "Where is CORS configured in Varshini0511/expense-tracker?"}'

# a write pauses with {"type": "interrupt", ...}; approve it explicitly
curl -s localhost:8000/resume -H 'content-type: application/json' \
  -d '{"thread_id": "<id from /chat>", "decisions": [{"type": "approve"}]}'
```

An API call alone can never mutate a repository: it needs `ALLOW_WRITES=true`
**and** a separate `/resume` approval. Set `API_KEY` and send `X-API-Key`;
without a key only loopback clients are accepted.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GITHUB_TOKEN` | — *(required)* | Fine-grained PAT. Read-only is enough to explore. |
| `CHAT_MODEL` | `groq:openai/gpt-oss-120b` | Any `provider:model` LangChain understands (e.g. `google_genai:gemini-flash-lite-latest`) |
| `GOOGLE_API_KEY` / `GROQ_API_KEY` | — | Key for the chosen provider / embeddings |
| `ALLOW_WRITES` | `false` | Master switch for every write tool |
| `GUARDRAILS_REDACT_PII` | `true` | Redact emails / phones / cards in tool output and answers |
| `ENABLE_REFLECTION` | `false` | Fact-check each answer; costs one extra model call |
| `RAG_HYBRID` | `true` | BM25 + dense fusion (set `false` for dense only) |
| `AGENT_RUN_LIMIT` | `25` | Max model calls per turn |
| `API_KEY` | — | Required `X-API-Key` for the HTTP API |
| `MODEL_PRICES_JSON` | — | Override the cost estimator, e.g. `{"gemini":[0.3,2.5]}` |

Full list with comments: [`.env.example`](.env.example).

**Enabling writes safely:** create a fine-grained PAT scoped to specific
repositories with *Contents*, *Pull requests* and *Issues* set to
read & write, then set `ALLOW_WRITES=true`. Every write still pauses for approval.

## Tools

| Group | Tools |
|---|---|
| Read | `list_repo_files`, `read_file`, `read_file_lines`, `list_issues`, `get_issue` |
| RAG | `index_repo`, `search_code` |
| CI/CD (read-only) | `get_ci_failure` (one call: failed job + shrunk log excerpt + history), `list_workflow_runs`, `get_workflow_run`, `get_job_log`, `list_check_runs` |
| Pull requests (read-only) | `list_pull_requests`, `get_pull_request`, `get_pr_diff` |
| Write 🔒 | `create_issue`, `comment_on_issue`, `add_labels`, `create_branch`, `write_file`, `edit_file`, `open_pull_request` |
| Agent-local | `write_todos` (progress checklist), `remember`, `recall` (long-term memory) |

🔒 = needs `ALLOW_WRITES=true` **and** human approval. There is intentionally
no tool to merge, delete, force-push or change repository settings.

## DevOps agents (CI triage + PR review)

In multi-agent mode two **read-only** specialists join the team:

- `ci_triage` diagnoses a failed workflow run and returns a structured `CIReport`
  (category, root cause, verbatim evidence, suggested fix, re-run advice, confidence).
- `pr_reviewer` reviews a pull request and returns a structured `PRReview`
  (risk, findings with file/line/quote, what was not reviewed, recommendation).

Handoff example: *"PR 12 is failing - why, and is the change risky?"* runs
`ci_triage` first and passes its diagnosis into `pr_reviewer`.

Every evidence/finding quote is checked against what the tools actually returned
and tagged `[verified]` or `[UNVERIFIED]`. Logs are shrunk to the failing step and
diffs are budgeted with real new-file line numbers, so this fits free-tier limits.

Limits (stated plainly): it **diagnoses, it does not fix**. Fixing still goes
through `code_writer` with your approval. Tested offline against fake GitHub
data only, not yet against a live provider. Token permissions needed in
addition to the above: *Actions: read*, *Checks: read*, *Pull requests: read*.

### Automatic triage on a failed build (webhook)

`POST /webhooks/github` lets a failed GitHub Actions run trigger `ci_triage`
**by itself**, so a diagnosis is waiting without anyone asking. It is built to
be safe by default:

- **Signed.** Every delivery is verified with an HMAC (`X-Hub-Signature-256`)
  against `GITHUB_WEBHOOK_SECRET`; the endpoint returns 503 until that secret is
  set, so an unsigned request can never trigger the agent.
- **Filtered.** Only *completed, failed* `workflow_run` events act; successes,
  in-progress runs and other event types are ignored.
- **De-duplicated.** GitHub retries and manual re-deliveries are processed once.
- **Capped.** `CI_AUTO_TRIAGE_MAX_PER_COMMIT` (default 2) stops a commit that
  keeps failing from draining your quota.
- **Opt-in and read-only.** `CI_AUTO_TRIAGE=false` by default only *records*
  failures; set it `true` to auto-diagnose. **Fixing is never automatic** — it
  still goes through `code_writer` with your approval.

`GET /webhooks/ci` returns the recent auto-triage feed (what fired, what was
diagnosed, what was capped). To go live you need a public URL for the API
(GitHub must reach it) and a webhook on the repo pointing at `/webhooks/github`.
Everything here is tested offline with a signed fake delivery and a stand-in
triage runner; it has not yet run against a live GitHub delivery.

## Observability

Every turn reports model calls, tokens, tool calls, latency and an estimated
cost (the UI sidebar, `usage` in API responses, and one JSON log line per turn):

```json
{"event": "turn", "thread_id": "api-3f9c…", "model_calls": 4, "input_tokens": 5120,
 "output_tokens": 640, "tool_calls": 3, "tools": ["read_file_lines", "search_code"],
 "latency_s": 6.82, "cost_usd": 0.0027}
```

For full traces, set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` — no code change.
Costs are **estimates at list price**; on a free tier the real bill is $0.

## Testing

```bash
pytest -q                          # offline: no network, no keys   (~10 s)
ruff check . && ruff format --check .

python evals/run.py                # LIVE: scored golden set (uses API quota)
python evals/run.py hallucination-guard
```

The offline suite is the interesting part: `tests/fakes.py` provides a
scripted chat model so the **real** LangGraph agent — tool loop, approval
interrupts, guardrails, streaming, usage metering, memory, reflection and the
supervisor — runs deterministically in CI. The live evals measure answer
quality, including a case that *fails* the agent if it invents code for a file
that does not exist.

## Project layout

```
server.py            MCP tool server (separate process)
github_client.py     resilient GitHub REST client
rag.py retrieval.py  index + hybrid search (BM25, RRF)
agent_core.py        agent assembly, send / resume / stream_turn, guard middleware
multi_agent.py       supervisor + specialists
guardrails.py        redaction + injection defence (pure functions)
observability.py     usage / cost / structured metrics
memory.py reflection.py
runner.py            sync ↔ async bridge (AsyncRunner, AgentWorker)
api.py               FastAPI + SSE        streamlit_app.py   web UI
evals/               golden set + LLM judge
tests/               offline suite (fake model drives the real graph)
```

## Deploying

* **Container:** `docker build -t github-agent .` (non-root, health-checked) or
  `docker compose up --build`.
* **Streamlit Community Cloud:** add the variables above as app *Secrets*; the
  app writes them to a `.env` for the MCP subprocess. Keep `ALLOW_WRITES=false`
  on any public instance — one token serves every visitor.

## Honest limitations

* Injection and PII detection are heuristics — approval + least-privilege are the real backstops ([SECURITY.md](SECURITY.md)).
* BM25 is rebuilt per query (fine for ~10⁵ chunks); a persistent inverted index is the next step.
* SQLite state and the in-process rate limiter are single-node; production would use Postgres + a gateway.
* Small free-tier models loop or mis-plan on large multi-file changes; a stronger model is a one-line `CHAT_MODEL` change.
* No incremental re-indexing; `index_repo` rebuilds the repo's index.

## License

No license has been chosen yet — add a `LICENSE` file before accepting contributions.
