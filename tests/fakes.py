"""Test doubles for exercising the REAL agent graph without a network.

``ScriptedChatModel`` is a chat model that replays a pre-written list of
AIMessages, one per model call. Because it implements the same interface the
framework expects (``bind_tools``, sync/async generate, streaming), the real
``create_agent`` graph -- routing, tool execution, middleware, checkpointer,
interrupts -- runs for real around it. Only the LLM itself is faked.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


def tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def ai_text(
    text: str, *, tin: int = 10, tout: int = 5, model: str = "fake-gemini-flash"
) -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": tin, "output_tokens": tout, "total_tokens": tin + tout},
        response_metadata={"model_name": model},
    )


def ai_tool(
    calls: list[dict], *, tin: int = 20, tout: int = 8, model: str = "fake-gemini-flash"
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=calls,
        usage_metadata={"input_tokens": tin, "output_tokens": tout, "total_tokens": tin + tout},
        response_metadata={"model_name": model},
    )


class ScriptedChatModel(BaseChatModel):
    """Replays ``script`` in order, one AIMessage per call."""

    script: list[Any]
    cursor: int = 0
    seen_messages: list[Any] = []  # what each call was shown (for assertions)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools, **kwargs):  # the agent calls this once at build
        return self

    def _next(self, messages) -> AIMessage:
        self.seen_messages.append(list(messages))
        if self.cursor >= len(self.script):
            raise AssertionError(
                f"ScriptedChatModel exhausted after {len(self.script)} calls -- "
                "the agent made more model calls than the test scripted."
            )
        msg = self.script[self.cursor]
        self.cursor += 1
        return msg

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self._next(messages)
        if msg.tool_calls:
            chunks = [
                {
                    "name": tc["name"],
                    "args": __import__("json").dumps(tc["args"]),
                    "id": tc["id"],
                    "index": i,
                }
                for i, tc in enumerate(msg.tool_calls)
            ]
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=chunks,
                    usage_metadata=msg.usage_metadata,
                    response_metadata=msg.response_metadata,
                )
            )
            return
        words = str(msg.content).split(" ")
        for i, word in enumerate(words):
            last = i == len(words) - 1
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=word + ("" if last else " "),
                    usage_metadata=msg.usage_metadata if last else None,
                    response_metadata=msg.response_metadata if last else {},
                )
            )

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in self._stream(messages, stop, run_manager, **kwargs):
            yield chunk
