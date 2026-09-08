"""Conditional query analysis, structured rewrite, and drift guards."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

from pydantic import BaseModel, Field, field_validator

QueryType = Literal["simple", "complex", "vague", "multi_intent"]

_ENTITY_RE = re.compile(
    r"(?:"
    r"[A-Za-z][A-Za-z0-9_+./-]{1,}"
    r"|「[^」]{1,40}」"
    r"|“[^”]{1,40}”"
    r"|《[^》]{1,40}》"
    r"|[\u4e00-\u9fff]{2,12}(?:能力|系统|框架|服务|协议|数据库|检索|任务)"
    r")"
)
_MULTI_INTENT_RE = re.compile(
    r"(分别|以及|和.+相比|差在哪|差什么|一方面|另一方面|既要|又要)"
)
_VAGUE_RE = re.compile(
    r"(怎么优化|如何优化|有什么帮助|怎么样|怎么提升|还能怎么|优化一下|讲讲|介绍一下)"
)
_SIMPLE_TECH_RE = re.compile(
    r"(?i)\b(sqlite|mysql|postgres|redis|docker|kubernetes|k8s|mcp|"
    r"rag|bm25|rrf|fastapi|python|idempotency|cron|feishu)\b"
    r"|用了什么|要求哪些|是什么"
)


class QueryRewriteResult(BaseModel):
    """Structured rewrite payload; free-form model text is never searched directly."""

    query_type: QueryType
    original_intent: str = Field(min_length=1, max_length=2_000)
    rewritten_queries: list[str] = Field(default_factory=list, max_length=3)
    must_preserve: list[str] = Field(default_factory=list, max_length=20)
    rewrite_needed: bool = False

    @field_validator("rewritten_queries")
    @classmethod
    def queries_must_be_nonempty(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if len(cleaned) > 3:
            raise ValueError("rewritten_queries must contain at most 3 items")
        return cleaned


class QueryRewriteTrace(BaseModel):
    """Privacy-safe rewrite telemetry (no full resume/JD bodies)."""

    query_type: QueryType
    rewrite_used: bool = False
    rewrite_query_count: int = 0
    rewrite_fallback: bool = False
    rewrite_latency_ms: float = 0.0


QueryRewriter = Callable[[str, QueryRewriteResult], Awaitable[QueryRewriteResult]]


class OpenAICompatibleQueryRewriter:
    """Rewrite via an OpenAI-compatible chat endpoint (e.g. ModelScope)."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        if not model.strip():
            raise ValueError("rewrite model must not be empty")
        if not api_key.strip():
            raise ValueError("rewrite api_key must not be empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        from openai import AsyncOpenAI

        self._model = model.strip()
        self._timeout_s = timeout_s
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
        )

    async def __call__(
        self, query: str, analysis: QueryRewriteResult
    ) -> QueryRewriteResult:
        import asyncio
        import json

        from openai import OpenAIError

        payload = {
            "query_type": analysis.query_type,
            "original_query": query,
            "must_preserve": analysis.must_preserve,
            "instructions": (
                "Return JSON only with keys: query_type, original_intent, "
                "rewritten_queries (1-3 search strings), must_preserve, rewrite_needed. "
                "Keep must_preserve entities verbatim. Do not invent new facts. "
                "rewritten_queries are retrieval strings, not answers."
            ),
        }
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You rewrite search queries for a private knowledge base. "
                                "Output strict JSON only."
                            ),
                        },
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                ),
                timeout=self._timeout_s,
            )
        except (OpenAIError, TimeoutError, asyncio.TimeoutError) as exc:
            raise RuntimeError(f"query rewrite LLM failed: {type(exc).__name__}") from exc
        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise RuntimeError("query rewrite LLM returned empty content")
        data = json.loads(content)
        return QueryRewriteResult.model_validate(
            {
                "query_type": data.get("query_type", analysis.query_type),
                "original_intent": data.get("original_intent") or analysis.original_intent,
                "rewritten_queries": data.get("rewritten_queries") or [],
                "must_preserve": data.get("must_preserve") or analysis.must_preserve,
                "rewrite_needed": True,
            }
        )


def extract_must_preserve(query: str) -> list[str]:
    """Pull likely entities that rewritten queries must keep."""

    found: list[str] = []
    seen: set[str] = set()
    for match in _ENTITY_RE.finditer(query):
        token = match.group(0).strip()
        key = token.casefold()
        if key in seen or len(token) < 2:
            continue
        seen.add(key)
        found.append(token)
        if len(found) >= 12:
            break
    return found


def analyze_query(query: str) -> QueryRewriteResult:
    """Lightweight deterministic query analysis without an LLM call."""

    text = query.strip()
    entities = extract_must_preserve(text)
    if not text:
        return QueryRewriteResult(
            query_type="simple",
            original_intent="",
            rewritten_queries=[],
            must_preserve=[],
            rewrite_needed=False,
        )
    if _MULTI_INTENT_RE.search(text) or text.count("？") + text.count("?") >= 2:
        query_type: QueryType = "multi_intent"
    elif _VAGUE_RE.search(text) and len(text) < 40:
        query_type = "vague"
    elif len(text) <= 24 and (_SIMPLE_TECH_RE.search(text) or len(entities) <= 2):
        query_type = "simple"
    elif len(text) <= 18 and not _VAGUE_RE.search(text):
        query_type = "simple"
    else:
        query_type = "complex"
    rewrite_needed = query_type != "simple"
    return QueryRewriteResult(
        query_type=query_type,
        original_intent=text,
        rewritten_queries=[],
        must_preserve=entities,
        rewrite_needed=rewrite_needed,
    )


def apply_rewrite_guard(
    original: str,
    result: QueryRewriteResult,
    *,
    max_queries: int = 3,
) -> QueryRewriteResult | None:
    """Reject drifted or invalid rewrites; caller should fallback to original."""

    if max_queries < 1 or max_queries > 3:
        return None
    queries = [item.strip() for item in result.rewritten_queries if item.strip()]
    if not queries or len(queries) > max_queries:
        return None
    if any(len(item) > 500 for item in queries):
        return None
    # Do not invent long novel claims: rewritten text must stay near original length budget.
    if any(len(item) > max(120, len(original) * 3) for item in queries):
        return None
    preserved = [token for token in result.must_preserve if token.strip()]
    for token in preserved:
        needle = token.casefold()
        if not any(needle in item.casefold() for item in queries):
            return None
    return result.model_copy(
        update={
            "rewritten_queries": queries[:max_queries],
            "must_preserve": preserved,
            "original_intent": result.original_intent.strip() or original.strip(),
        }
    )


async def deterministic_rewrite(
    query: str,
    analysis: QueryRewriteResult,
) -> QueryRewriteResult:
    """Offline-safe rewriter used by eval/CI when no LLM rewriter is configured."""

    text = query.strip()
    preserve = analysis.must_preserve or extract_must_preserve(text)
    rewritten: list[str] = []
    if analysis.query_type == "multi_intent":
        parts = re.split(r"[，,、；;]|以及|分别|和", text)
        for part in parts:
            cleaned = part.strip(" ？?。")
            if len(cleaned) >= 4:
                rewritten.append(cleaned)
        if len(rewritten) < 2:
            rewritten = [text, f"{text} 证据 出处", f"{text} 能力差距"]
    elif analysis.query_type == "vague":
        rewritten = [
            text,
            f"{text} 可靠性 恢复 幂等",
            f"{text} RAG 检索 工程化",
        ]
    else:
        rewritten = [text, f"{text} 实现 设计", f"{text} 评测 指标"]
    # Keep preserve tokens visible.
    if preserve:
        joined = " ".join(preserve[:4])
        rewritten = [f"{item} {joined}".strip() for item in rewritten]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in rewritten:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return analysis.model_copy(
        update={
            "rewritten_queries": deduped[:3],
            "must_preserve": preserve,
            "rewrite_needed": True,
        }
    )


async def prepare_search_queries(
    query: str,
    *,
    enabled: bool,
    max_rewritten_queries: int = 3,
    rewriter: QueryRewriter | None = None,
) -> tuple[list[str], QueryRewriteTrace]:
    """Return retrieval queries with fallback-to-original semantics.

    When rewrite is enabled for a non-simple query but no rewriter is injected,
    fall back to the original query (do not invent expansions).
    """

    import time

    started = time.perf_counter()
    analysis = analyze_query(query)
    if not enabled or not analysis.rewrite_needed:
        return (
            [query.strip()],
            QueryRewriteTrace(
                query_type=analysis.query_type,
                rewrite_latency_ms=round((time.perf_counter() - started) * 1_000, 3),
            ),
        )
    if rewriter is None:
        return (
            [query.strip()],
            QueryRewriteTrace(
                query_type=analysis.query_type,
                rewrite_used=False,
                rewrite_query_count=0,
                rewrite_fallback=True,
                rewrite_latency_ms=round((time.perf_counter() - started) * 1_000, 3),
            ),
        )

    try:
        drafted = await rewriter(query, analysis)
        guarded = apply_rewrite_guard(
            query, drafted, max_queries=max_rewritten_queries
        )
    except Exception:
        guarded = None
    latency = round((time.perf_counter() - started) * 1_000, 3)
    if guarded is None or not guarded.rewritten_queries:
        return (
            [query.strip()],
            QueryRewriteTrace(
                query_type=analysis.query_type,
                rewrite_used=False,
                rewrite_query_count=0,
                rewrite_fallback=True,
                rewrite_latency_ms=latency,
            ),
        )
    return (
        guarded.rewritten_queries,
        QueryRewriteTrace(
            query_type=guarded.query_type,
            rewrite_used=True,
            rewrite_query_count=len(guarded.rewritten_queries),
            rewrite_fallback=False,
            rewrite_latency_ms=latency,
        ),
    )


def assert_no_document_scope_expansion(
    original_allowed_sources: Sequence[str] | None,
) -> None:
    """Rewrite must never widen authorized document sets; callers keep filters."""

    _ = original_allowed_sources
