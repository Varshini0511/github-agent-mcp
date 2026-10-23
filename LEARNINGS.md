# Concepts index

Every concept below is implemented in this repository — the right-hand column
says where, so each claim is checkable. Grouped by the problem it solves.

## Agents & orchestration

| Concept | In one line | Where |
|---|---|---|
| **Tool use / function calling** | The model *requests* tools described by name, docstring and typed args; the runtime executes them | `server.py` (`@mcp.tool`), `agent_core.assemble_agent` |
| **ReAct loop** | reason → act → observe → repeat until no tool is requested | LangGraph graph built by `create_agent` |
| **Model Context Protocol** | Tools live in a separate server any client can use, over JSON-RPC/stdio | `server.py`, `MCPAdapter` in `agent_core.build_agent` |
| **Supervisor / multi-agent** | A manager delegates to scoped specialists; each sees only the tools it needs | `multi_agent.assemble_supervisor` |
| **Agent-as-tool** | A whole agent wrapped so the manager sees it as one tool (`task in → text out`) | `multi_agent._wrap` |
| **Task decomposition** | An explicit checklist (`write_todos`) stops a weak model repeating finished work | `TodoListMiddleware` |
| **Agent handoff** | One specialist's output becomes the next one's input (CI diagnosis → PR review) | `multi_agent.SUPERVISOR_PROMPT` |
| **Structured output** | The final answer is a validated Pydantic object (via `ToolStrategy`), not free text | `devops_schemas.py`, `multi_agent._make_specialist` |
| **Event-driven agent** | A webhook, not a person, starts the run | `webhook.py`, `api.py` `/webhooks/github` |
| **Middleware pattern** | Cross-cutting behaviour wrapped around model / tool calls, composable and orderable | `agent_core.assemble_agent` |

## Memory & context

| Concept | In one line | Where |
|---|---|---|
| **Short-term memory** | Conversation state checkpointed per `thread_id`, survives restarts | `AsyncSqliteSaver` in `build_agent` |
| **Long-term memory** | Durable user facts shared across conversations, namespaced by `user_id`, via a Store | `memory.py` |
| **Context-window management** | Summarise old turns, keep recent ones verbatim | `SummarizationMiddleware` |
| **Human-in-the-loop** | Graph `interrupt()` freezes the run; a decision resumes it from the checkpoint | `HumanInTheLoopMiddleware`, `agent_core.resume` |

## Retrieval (RAG)

| Concept | In one line | Where |
|---|---|---|
| **Chunking + embeddings + vector search** | Split code, embed, retrieve by meaning with `path:line` citations | `rag.RepoIndex` |
| **Lexical search (BM25)** | Exact-term ranking with an identifier-aware tokenizer | `retrieval.BM25` |
| **Context shrinking** | A multi-megabyte CI log is cut to the failing step; a PR diff is budgeted and ordered by importance | `ci_logs.extract_failure`, `pr_diff.render_pr_diff` |
| **Hybrid retrieval** | Run both retrievers and fuse; each covers the other's blind spot | `rag.RepoIndex.search` |
| **Reciprocal Rank Fusion** | Combine rankings from incomparable scorers using ranks only | `retrieval.reciprocal_rank_fusion` |
| **Weighted fusion** | Trust the keyword leg more for code-shaped queries; breaks ties | `retrieval.looks_like_identifier` |

## Safety & trust

| Concept | In one line | Where |
|---|---|---|
| **Indirect prompt-injection defence** | Tool output = untrusted data: scrub, quarantine, instruct the model not to obey | `guardrails.quarantine_untrusted` |
| **Secret & PII redaction** | Credentials never reach the model or chat history; Luhn-checked card detection | `guardrails.py` |
| **Least privilege / defence in depth** | Hard switch + approval gate + no destructive tools | `server._guard`, `WRITE_TOOLS` |
| **Fail-open safety nets** | Reflection may never turn a good answer into an error | `reflection.critique` |
| **Grounding / claim verification** | Every quoted log or diff line is checked against the tool output and tagged `[verified]` or `[UNVERIFIED]` | `devops_schemas.quote_is_grounded` |
| **Webhook authentication (HMAC)** | A shared secret signs the body; verify with a constant-time compare | `webhook.verify_signature` |
| **Idempotency** | A redelivered event is handled once | `webhook.DeliveryLog` |
| **Circuit breaker / attempt cap** | An automatic loop gets a hard ceiling so it cannot burn quota | `webhook.AttemptLimiter` |
| **Safe-by-default automation** | Auto-triage is opt-in and read-only; fixing stays human-approved | `CI_AUTO_TRIAGE` |
| **Self-reflection** | A critic fact-checks the draft against tool evidence; one bounded revision | `reflection.py`, `agent_core.send` |

## Reliability

| Concept | In one line | Where |
|---|---|---|
| **Retry, backoff, jitter** | Transient failures retried without stampeding | `github_client._request` |
| **Rate-limit awareness** | Honour `Retry-After` / provider-suggested delays | `github_client`, `retry_on_rate_limit` |
| **Corrective retry** | Tell the model *what* it got wrong instead of resampling blindly | `retry_malformed_tool_call` |
| **Bounded loops** | Per-turn model-call cap | `ModelCallLimitMiddleware` |
| **LLM-readable errors** | Failures are returned as text a model can act on | `server.py` tool wrappers |

## Observability & operations

| Concept | In one line | Where |
|---|---|---|
| **Usage metering** | Tokens, model calls, tool calls, latency per turn via callbacks | `observability.UsageCallback` |
| **Cost estimation** | Token counts × overridable price table, clearly labelled an estimate | `observability.estimate_cost` |
| **Structured logging** | One JSON line per turn, ready for a log pipeline | `observability.log_turn` |
| **Tracing** | LangSmith traces with env vars only | README |
| **Streaming (SSE)** | Tokens and tool events as they happen | `agent_core.stream_turn`, `api.py` |
| **Graceful shutdown** | In-flight turns finish; MCP + DB handles close | `AgentWorker.close` |

## Quality & testing

| Concept | In one line | Where |
|---|---|---|
| **Evaluation harness** | Golden set scored by deterministic checks + LLM-as-judge | `evals/` |
| **Hallucination test** | A case that *fails* the agent if it invents a missing file | `evals/cases.yaml` |
| **Testing agents offline** | Scripted fake model drives the *real* graph | `tests/fakes.py`, `tests/test_agent_graph.py` |
| **HTTP-boundary mocking** | GitHub API faked with `responses` | `tests/test_github_client.py` |
| **UI testing** | Headless Streamlit `AppTest` | `tests/test_ui.py` |
| **CI** | Lint + format + tests on 3.11 / 3.12; live evals on demand | `.github/workflows/ci.yml` |

## Engineering practice

* **Separation of concerns** — HTTP client / tools / agent wiring / UI are distinct layers.
* **Validated, fail-fast configuration** — `pydantic-settings`, `SecretStr` (`config.py`).
* **Async ↔ sync bridging** — one owning task per agent (`runner.py`; the *why* is in
  [ARCHITECTURE.md](ARCHITECTURE.md)).
* **Containerisation** — `Dockerfile`, non-root user, health check, compose file.

## Things that went wrong, and what they taught

* *An agent that "worked" re-committed one file five times* → added an explicit
  progress checklist and a "never repeat a tool call" rule; weak models need
  structure, not just a bigger prompt.
* *"Connection closed" / "threads can only be started once" in the UI* → async
  resources must be used by the task that opened them; fixed by giving each
  agent one long-lived owning task.
* *The first hybrid-search test failed* → an exact-match hit that dense search
  missed **tied** with dense #1 under plain RRF; weighting the lexical leg for
  code-shaped queries fixed it. Tests found a real ranking flaw.
* *A phone-number regex chopped a 16-digit number in two* → added boundary
  look-arounds; redaction must never leave fragments behind.
* *CI went red on a push even though every test passed locally* → the project is
  installed as a package in CI, and new modules were missing from the
  `py-modules` list in `pyproject.toml`; they imported fine from the working
  directory. A test now fails whenever a packaged module imports an unpackaged
  one. "Works on my machine" is a real category of bug.
* *The free tier's limits shaped the design* — token-frugal tools
  (`read_file_lines`, `edit_file`), rate-limit-aware retries and a usage meter
  all exist because the constraint was real.
