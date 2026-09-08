"""Unit tests for conditional query rewrite guards."""

from __future__ import annotations

from nanobot.knowledge.query_rewrite import (
    QueryRewriteResult,
    analyze_query,
    apply_rewrite_guard,
    deterministic_rewrite,
    prepare_search_queries,
)


def test_simple_query_skips_rewrite() -> None:
    analysis = analyze_query("用了什么 SQLite 数据库？")
    assert analysis.query_type == "simple"
    assert analysis.rewrite_needed is False


def test_vague_and_multi_intent_need_rewrite() -> None:
    vague = analyze_query("这个还能怎么优化？")
    multi = analyze_query("我的项目和这个 JD 在 Agent、RAG、工程能力上分别差什么？")
    assert vague.query_type == "vague"
    assert vague.rewrite_needed is True
    assert multi.query_type == "multi_intent"
    assert multi.rewrite_needed is True


def test_guard_rejects_missing_entity() -> None:
    original = "比较 OmniAgent 与 JD 的 RAG 差距"
    drifted = QueryRewriteResult(
        query_type="complex",
        original_intent=original,
        rewritten_queries=["完全无关的天气查询"],
        must_preserve=["RAG"],
        rewrite_needed=True,
    )
    assert apply_rewrite_guard(original, drifted) is None


def test_guard_rejects_empty_rewrites() -> None:
    original = "比较 RAG 差距"
    empty = QueryRewriteResult(
        query_type="complex",
        original_intent=original,
        rewritten_queries=[],
        must_preserve=["RAG"],
        rewrite_needed=True,
    )
    assert apply_rewrite_guard(original, empty) is None


async def test_prepare_search_queries_fallbacks_on_invalid_rewriter() -> None:
    async def bad_rewriter(query: str, analysis: QueryRewriteResult) -> QueryRewriteResult:
        raise TimeoutError("llm timeout")

    queries, trace = await prepare_search_queries(
        "这个项目对 Agent 岗有什么帮助？",
        enabled=True,
        rewriter=bad_rewriter,
    )
    assert queries == ["这个项目对 Agent 岗有什么帮助？"]
    assert trace.rewrite_fallback is True
    assert trace.rewrite_used is False


async def test_complex_query_uses_deterministic_rewrite() -> None:
    queries, trace = await prepare_search_queries(
        "这个项目对 Agent 岗有什么帮助？",
        enabled=True,
        rewriter=deterministic_rewrite,
    )
    assert trace.rewrite_used is True
    assert 1 <= len(queries) <= 3
    assert all(queries)


async def test_enabled_without_rewriter_falls_back_to_original() -> None:
    query = "这个项目对 Agent 岗有什么帮助？"
    queries, trace = await prepare_search_queries(query, enabled=True, rewriter=None)
    assert queries == [query]
    assert trace.rewrite_fallback is True
