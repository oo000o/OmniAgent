"""Manually labelled retrieval corpus for realistic ablation (not the CI 12-case fixture)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Category = Literal[
    "keyword",
    "semantic",
    "paraphrase",
    "compound",
    "vague",
    "hard_negative",
]


@dataclass(frozen=True, slots=True)
class LabeledRetrievalCase:
    case_id: str
    query: str
    category: Category
    relevant_sources: tuple[str, ...]


# Short single-chunk documents so source_name ≈ one stable evidence unit.
DOCUMENTS: dict[str, str] = {
    "idempotency.md": (
        "Idempotency keys prevent duplicate task creation when the model retries the same "
        "intent. Replaying the identical create request returns the existing task instead of "
        "inserting another side effect."
    ),
    "optimistic_lock.md": (
        "Optimistic locking uses expected_version on task updates. Stale concurrent writers "
        "receive a version conflict and must re-read before retrying."
    ),
    "checkpoint.md": (
        "Workflow checkpoints store stage, evidence ids, plan_revision, and real task or cron "
        "ids. After a crash, recovery rebuilds completed mappings from the authoritative store."
    ),
    "rrf.md": (
        "Reciprocal rank fusion merges BM25 lexical rankings with vector semantic rankings "
        "without requiring comparable raw scores."
    ),
    "bm25.md": (
        "SQLite FTS5 BM25 lexical retrieval finds exact technical terms in private documents "
        "and ranks by the bm25() function."
    ),
    "vector.md": (
        "Vector embeddings retrieve semantically related passages using cosine similarity over "
        "stored float vectors for each chunk."
    ),
    "citations.md": (
        "Stable citations expose source file names and character offsets as [K1] style evidence "
        "so answers remain inspectable."
    ),
    "hitl.md": (
        "Human-in-the-loop confirmation reads the original user message and binds plan_revision. "
        "Model claims that the user agreed are rejected without explicit confirmation."
    ),
    "replan.md": (
        "Bounded replanning revises rejected learning plans at most twice. Exhausted revisions "
        "fail with REPLAN_EXHAUSTED instead of looping forever."
    ),
    "feishu.md": (
        "The Feishu channel uses a WebSocket long connection without a public webhook. Cron jobs "
        "return reminders to the originating Feishu chat."
    ),
    "mcp.md": (
        "The task domain is isolated behind a stdio MCP server with Pydantic-validated tools for "
        "create, list, update, and cancel."
    ),
    "trace.md": (
        "RunStore traces record tool names, latency, error codes, and recovery flags without "
        "storing resume text, JD bodies, secrets, or full tool arguments."
    ),
    "eval.md": (
        "The offline evaluation baseline reports deterministic fixture pass rates including "
        "career workflow, observability, and fault-injection groups."
    ),
    "docker.md": (
        "Docker Compose packages the gateway with persistent workspace volumes, health checks, "
        "and non-root execution for self-hosted demos."
    ),
    # Hard-negative bait: shares keywords with reliability docs but does not answer recovery queries.
    "ui_theme.md": (
        "The WebUI theme uses CSS variables for light surfaces. Retry buttons in the settings "
        "panel only refresh the theme preview and never create tasks or checkpoints."
    ),
    "glossary_retry.md": (
        "In product marketing copy, 'retry' means asking the assistant another question. It is "
        "unrelated to idempotent task creation or checkpoint recovery after process crashes."
    ),
}

CASES: tuple[LabeledRetrievalCase, ...] = (
    # keyword
    LabeledRetrievalCase("k1", "JD 场景里用了什么幂等机制？", "keyword", ("idempotency.md",)),
    LabeledRetrievalCase("k2", "BM25 在哪里实现？", "keyword", ("bm25.md",)),
    LabeledRetrievalCase("k3", "MCP 任务服务怎么隔离？", "keyword", ("mcp.md",)),
    LabeledRetrievalCase("k4", "Feishu 是否需要公网 webhook？", "keyword", ("feishu.md",)),
    LabeledRetrievalCase("k5", "Docker Compose 有什么作用？", "keyword", ("docker.md",)),
    LabeledRetrievalCase("k6", "optimistic locking 解决什么？", "keyword", ("optimistic_lock.md",)),
    LabeledRetrievalCase("k7", "RRF 融合哪两路检索？", "keyword", ("rrf.md",)),
    LabeledRetrievalCase("k8", "RunStore trace 记什么？", "keyword", ("trace.md",)),
    # semantic
    LabeledRetrievalCase(
        "s1", "岗位如果重视失败恢复，这个项目哪里能对上？", "semantic", ("checkpoint.md",)
    ),
    LabeledRetrievalCase(
        "s2", "怎样避免模型重试把待办写两遍？", "semantic", ("idempotency.md",)
    ),
    LabeledRetrievalCase(
        "s3", "答案如何证明来自哪份资料？", "semantic", ("citations.md",)
    ),
    LabeledRetrievalCase(
        "s4", "计划写歪了会不会无限改下去？", "semantic", ("replan.md",)
    ),
    LabeledRetrievalCase(
        "s5", "用户口头说同意就能创建任务吗？", "semantic", ("hitl.md",)
    ),
    LabeledRetrievalCase(
        "s6", "关键词和语义结果怎么合成一个列表？", "semantic", ("rrf.md",)
    ),
    LabeledRetrievalCase(
        "s7", "向量检索大概怎么打分？", "semantic", ("vector.md",)
    ),
    LabeledRetrievalCase(
        "s8", "质量门禁离线怎么验？", "semantic", ("eval.md",)
    ),
    # paraphrase
    LabeledRetrievalCase(
        "p1", "重复调用的时候怎么避免创建两次任务？", "paraphrase", ("idempotency.md",)
    ),
    LabeledRetrievalCase(
        "p2", "并发更新被旧版本覆盖怎么办？", "paraphrase", ("optimistic_lock.md",)
    ),
    LabeledRetrievalCase(
        "p3", "工具成功了但进程挂了怎么接着干？", "paraphrase", ("checkpoint.md",)
    ),
    LabeledRetrievalCase(
        "p4", "引用编号从哪来的？", "paraphrase", ("citations.md",)
    ),
    LabeledRetrievalCase(
        "p5", "定时提醒打回哪个会话？", "paraphrase", ("feishu.md",)
    ),
    LabeledRetrievalCase(
        "p6", "日志会不会把简历原文打出来？", "paraphrase", ("trace.md",)
    ),
    # compound
    LabeledRetrievalCase(
        "c1",
        "我的 RAG 和工程可靠性分别体现在哪些文档？",
        "compound",
        ("rrf.md", "checkpoint.md"),
    ),
    LabeledRetrievalCase(
        "c2",
        "Agent 岗关心的工具边界和确认机制在哪？",
        "compound",
        ("mcp.md", "hitl.md"),
    ),
    LabeledRetrievalCase(
        "c3",
        "检索链路里词法与向量各负责什么？",
        "compound",
        ("bm25.md", "vector.md"),
    ),
    LabeledRetrievalCase(
        "c4",
        "交付和评测怎么证明不是口头 Demo？",
        "compound",
        ("docker.md", "eval.md"),
    ),
    LabeledRetrievalCase(
        "c5",
        "重规划上限和确认版本绑定如何一起约束副作用？",
        "compound",
        ("replan.md", "hitl.md"),
    ),
    # vague
    LabeledRetrievalCase("v1", "这个还能怎么优化？", "vague", ("eval.md", "rrf.md")),
    LabeledRetrievalCase("v2", "对 Agent 岗有什么帮助？", "vague", ("checkpoint.md", "mcp.md")),
    LabeledRetrievalCase("v3", "可靠性方面怎么样？", "vague", ("idempotency.md", "checkpoint.md")),
    LabeledRetrievalCase("v4", "工程化做到哪一步了？", "vague", ("docker.md", "eval.md")),
    # hard negatives: query about recovery; ui_theme/glossary share retry wording
    LabeledRetrievalCase(
        "h1",
        "checkpoint 写入失败后如何恢复且不重复创建任务？",
        "hard_negative",
        ("checkpoint.md", "idempotency.md"),
    ),
    LabeledRetrievalCase(
        "h2",
        "模型重试导致的重复副作用如何避免？",
        "hard_negative",
        ("idempotency.md",),
    ),
    LabeledRetrievalCase(
        "h3",
        "进程崩溃后怎样只执行尚未完成的 MCP 调用？",
        "hard_negative",
        ("checkpoint.md",),
    ),
    LabeledRetrievalCase(
        "h4",
        "旧 plan_revision 确认为什么必须拒绝？",
        "hard_negative",
        ("hitl.md", "replan.md"),
    ),
    LabeledRetrievalCase(
        "h5",
        "Trace 如何在可观测的同时保护隐私？",
        "hard_negative",
        ("trace.md",),
    ),
    LabeledRetrievalCase(
        "h6",
        "Feishu 长连接部署省掉了什么基础设施？",
        "hard_negative",
        ("feishu.md",),
    ),
    LabeledRetrievalCase(
        "h7",
        "Pydantic 在任务工具入口挡住了什么？",
        "hard_negative",
        ("mcp.md",),
    ),
    LabeledRetrievalCase(
        "h8",
        "BM25 适合抓哪类查询？",
        "hard_negative",
        ("bm25.md",),
    ),
)

assert 30 <= len(CASES) <= 50
assert set(DOCUMENTS).issuperset({source for case in CASES for source in case.relevant_sources})
