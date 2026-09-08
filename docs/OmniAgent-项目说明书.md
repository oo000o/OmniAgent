# OmniAgent 项目说明书

> **封版口径（求职 / Demo）**  
> 仓库：https://github.com/oo000o/OmniAgent  
> 基线提交：`0c2c20e`（CI 全绿：Linux 3.11 / 3.14、Windows 3.14、WebUI、TUI、Docker）  
> 离线评测：**92 / 92**  
> 适用岗位：AI Agent 应用开发 / Agent 开发工程师 / RAG·LLM 应用工程

本文只写仓库里**已实现、可核验**的能力；消融与 latency 数字来自本地评测环境，其中 live 实验使用真实模型调用；均**不是 Production SLA**。

---

## 1. 一句话介绍

**OmniAgent** 是面向个人与小团队的**自托管私有知识与任务自动化 Agent**：把本地文档问答、结构化任务、定时跟进和飞书 / WebUI 双端交互，连成一条**可追踪、可恢复、可评测**的工作流——而不是又一个只能聊天的 Bot。

---

## 2. 要解决的问题

| 痛点 | 常见做法的问题 | OmniAgent 的做法 |
| --- | --- | --- |
| 资料散落在 PDF / 笔记 / 聊天里 | 纯对话，无出处 | 向量检索 + BM25 降级 / 可选 Hybrid RRF + 稳定引用 `[K1]` |
| 模型说「已创建任务」 | 文本不是事务回执 | 任务写入独立 MCP + SQLite，回查后才推进状态 |
| 重试导致重复待办 | 无幂等 | 稳定幂等键 + 权威存储对账 |
| 计划写歪了无限改 | 死循环或直接放行 | 确定性 Verifier + **最多 2 次**有界重规划 |
| 飞书与网页各说各话 | 会话分裂 | 统一会话 + 共享任务 / 知识库 |
| 长任务中途崩溃 | 无法恢复或重复执行 | Checkpoint + 故障注入回归 |
| 「效果感觉还行」 | 无法复现 | **92/92** 确定性离线评测 + CI |

**主演示场景（求职准备）：**  
用户上传简历与岗位 JD / 学习计划 → Agent 检索证据、分析差距、生成计划 → **用户明确确认** → 经 MCP 创建真实任务并挂定时检查 → 到点回传原飞书会话 → WebUI 可查同一批任务与完整运行 Trace。

同一套能力也适用于课程资料整理、个人研究文档问答、小团队事项跟进。

---

## 3. 与上游 nanobot 的关系（简历必说清）

| 层级 | 来源 | 说明 |
| --- | --- | --- |
| Agent Runtime、渠道、基础工具 | 开源 [HKUDS/nanobot](https://github.com/HKUDS/nanobot) | 通用执行与扩展点 |
| 私有知识领域层、检索与引用 | **本项目** | BM25 / Vector / Hybrid RRF、证据绑定 |
| 可选 Query Rewrite / Cross-Encoder Rerank | **本项目** | 默认关闭；由消融决定 |
| 任务 MCP、幂等 / 乐观锁 / 确认 | **本项目** | 写操作工程约束 |
| 求职工作流状态机、有界重规划 | **本项目** | Human-in-the-loop |
| RunStore / Trace / 评测门禁 | **本项目** | 可观测与质量门禁 |
| 飞书长连接、Docker Compose 交付 | 上游能力 + **本项目配置与加固** | 端到端可验收产品 |

**表述原则：** 写「基于 nanobot Runtime 落地的应用产品」，不写「从零实现 Agent 框架」，不把未使用的 LangChain / LangGraph 写成「精通」。

---

## 4. 系统架构

```text
飞书（WebSocket 长连接） ─┐
                          ├─> Channel / Session ─> Agent Runtime ─> LLM（魔搭等兼容 API）
WebUI ────────────────────┘          │                  │
                                     │                  ├─> Knowledge Tools
                                     │                  │    ├─ SQLite FTS5 / BM25
                                     │                  │    ├─ Vector Search
                                     │                  │    ├─ Hybrid RRF（可选）
                                     │                  │    ├─ Conditional Rewrite（默认关）
                                     │                  │    └─ Cross-Encoder Rerank（默认关）
                                     │                  │
                                     │                  ├─> Task MCP Server（stdio）
                                     │                  │    └─ SQLite Tasks
                                     │                  │
                                     └─> Cron ──────────┴─> 按来源渠道回传

Runtime Events ─> Run Store（SQLite）─> WebUI 运行详情
```

**三类状态必须分清：**

1. **对话状态**：帮助模型理解语境，可丢失或重建。  
2. **工作流 Checkpoint**：阶段、证据、计划版本、真实 task / cron ID。  
3. **业务事实**：Task MCP 的 SQLite 与 Cron 存储——副作用是否成功以这里为准。

---

## 5. 核心能力说明

### 5.1 检索与引用（RAG）

完整可选链路：

```text
Query Analysis
→ Conditional Rewrite（仅 complex / vague / multi_intent；失败回退原 Query）
→ BM25 / Vector（Hybrid 模式下并行召回后经 RRF 融合）
→ Cross-Encoder Rerank（仅 RRF Top-N；打分异常时保留原检索排序）
→ Citation / 证据绑定
```

**封版默认（由 39 条人工标注消融决定，不为技术栈展示强行上 Hybrid）：**

| 配置项 | 默认值 | 原因（摘要） |
| --- | --- | --- |
| `retrievalMode` | **`vector`** | 各类别上 Vector ≥ Hybrid；keyword 上 Hybrid 还因 BM25 拉低 MRR / NDCG |
| `queryRewrite.enabled` | **`false`** | live 质量略降，触发时 P95 约 20–30s |
| `rerank.enabled` | **`false`** | 英文 MS-MARCO CE 在本语料上 Recall@3 从 0.7949 掉到 0.4744（live 消融） |
| `fallbackToLexical` | `true` | Embedding 不可用时可降级 BM25 |

- 返回稳定引用：文件名、字符区间、引用编号；工作流 Gap 必须绑定已保存证据。  
- 检索 Trace 记模式 / 成败 / 耗时 / 条数等，**不落查询正文与文档内容**。  
- LexicalOverlap Scorer 仅用于 deterministic 离线测试，以及构建时无法加载 `sentence-transformers` 时的占位 scorer；不作为默认或线上 Reranker。真实 CrossEncoder 在打分阶段异常时**保留原检索（RRF）排序**，不会切到 LexicalOverlap。  
- CI **不**默认开启 rewrite / rerank，保持 deterministic。

### 5.2 任务领域（MCP）

- 独立 stdio MCP：创建、查询、分页筛选、更新、取消。  
- **Pydantic** 校验工具参数；非法输入在入口拒绝。  
- **幂等键** 防重复创建；**乐观锁（expected_version）** 防丢失更新。  
- 高风险取消等操作要求**显式确认**（读取 runtime 原始用户消息，不信任模型转述）。

### 5.3 求职工作流与有界重规划

```text
documents_ready → evidence_retrieved → gaps_analyzed → plan_verifying
        ↺ 有界重规划（最多 2 次）→ awaiting_confirmation → confirmed
        → tasks_creating → tasks_created → scheduled → completed
```

- 确定性 Verifier 拒绝未覆盖 Gap、未知能力引用、重复标题等；超限 → `REPLAN_EXHAUSTED`。  
- 确认绑定具体 `plan_revision`；确认旧版本 → `VERSION_CONFLICT`。  
- 登记 task_id 前必须回查共享 TaskStore。

### 5.4 失败恢复

```text
MCP / Cron 已成功 → Checkpoint 写入失败 → 进程退出
    → 重启后按幂等键 / 来源读取权威回执
    → 重建已完成映射 → 只执行尚未完成的调用
```

同一恢复动作重复执行三次，仍只保留一条 Task 或一条 Cron（故障注入覆盖）。

### 5.5 跨端与定时

- 单用户可开启统一会话（WebUI ↔ 飞书）。  
- Cron 记录创建时的来源渠道与会话，触发后回传原飞书会话。  
- 飞书使用**长连接**，不要求公网 IP / Webhook。

### 5.6 可观测与安全

- RunStore 最小 Trace：`run_id → workflow_id → plan_revision → step → tool → result / error / retry`。  
- 聚合给出样本数、成功率、Avg / P95 延迟、工具成功率（仓库样本分位数，**非生产 SLA**）。  
- 知识文本视为不可信证据；工具可限制工作区；服务令牌鉴权。  
- Trace / 日志不落简历 / JD 原文、完整查询、完整工具参数或凭证。

---

## 6. 技术栈

| 类别 | 技术 |
| --- | --- |
| 语言与运行时 | Python 3.11+ |
| Agent / 渠道 | nanobot Runtime、飞书（lark-oapi 长连接）、WebUI（React / TypeScript） |
| 数据与检索 | SQLite、FTS5 / BM25、向量检索、RRF；可选 Rewrite / CrossEncoder |
| 工具与协议 | MCP（stdio）、Pydantic |
| 交付与质量 | Docker Compose、GitHub Actions、pytest、Ruff、BasedPyright |
| 模型接入 | ModelScope / OpenAI 兼容 API（配置化，密钥不进仓库） |

---

## 7. 工程验证与指标

### 7.1 确定性离线基线（CI 门禁）

| 验证项 | 结果 | 含义边界 |
| --- | --- | --- |
| 离线评测总计 | **92 / 92** | 工程夹具回归，非生产准确率 |
| 求职工作流 | 17（成功 9 / 边界 4 / 恢复 4） | 场景通过率 |
| 运行观测 | 8 | Trace / 聚合契约 |
| 故障注入与隐私 | 7 | 恢复与省略策略 |
| 检索夹具（12 查询） | BM25 / Vector / Hybrid RRF 的 Recall@3、MRR、NDCG@3 | 确定性向量夹具 |
| GitHub Actions | **全绿**（含 Windows 3.14） | CI **不含**全量 `pytest --cov`；覆盖率用本地 `pytest --cov`（`fail_under=75`） |
| Docker / 飞书 | 非 root、健康检查、长连接真实验收 | 需有效开发者应用凭证 |

```bash
python -m evaluation.run
# → artifacts/evaluation/latest.json
```

### 7.2 检索封版消融（39 条人工标注，本地 seal）

```bash
python -m evaluation.run_seal          # offline / deterministic
python -m evaluation.run_seal --live   # 真实 LLM Rewrite + CrossEncoder（需凭证）
```

**口径约定（简历只引用一套）：**

| 用途 | Artifact | 说明 |
| --- | --- | --- |
| **简历 / 面试对外唯一 Retrieval 数字** | `artifacts/evaluation/retrieval_ablation_live.json` | 39 条人工标注；默认 Vector：**Recall@3 79.5% / MRR 79.9% / NDCG@3 77.0%** |
| 默认 mode 决策（category 拆分） | offline seal / category 跑数 | 见下表；**不与上列整体数字混写** |
| Rewrite / Rerank 负向结论与 latency | 同 live artifact + `benchmark_live.json` | 仅说明「为何默认关闭」，不替代 Vector 主数字 |

offline 使用 deterministic fixture embedding；BM25 / Vector / Hybrid 的检索实现与 live 保持一致，但**数值口径独立，不直接等同于真实模型结果**。不要把 offline 的 78.2% 与 live 的 79.5% 在同一简历句子里混用。

**简历主表（live，39 条；Vector 为默认 Pipeline）：**

| Pipeline | Recall@3 | MRR | NDCG@3 |
| --- | ---: | ---: | ---: |
| BM25 | 0.2436 | 0.2564 | 0.2465 |
| **Vector（默认）** | **0.7949** | **0.7991** | **0.7701** |
| Hybrid + RRF | 0.7949 | 0.7735 | 0.7548 |
| + Rewrite | 0.7564 | 0.7222 | 0.7192 |
| + Rerank（ms-marco-MiniLM） | 0.4744 | 0.4530 | 0.4343 |
| Final | 0.4744 | 0.4530 | 0.4343 |

**按 category 的 BM25 / Vector / Hybrid（offline fixture；仅用于决定默认 mode）：**

| Category | n | 结论摘要 |
| --- | ---: | --- |
| keyword | 8 | Vector 全优；Hybrid MRR/NDCG 低于 Vector；BM25 单独可用但未超过 Vector |
| semantic / paraphrase / compound / vague | 8/6/5/4 | Vector 与 Hybrid **持平** |
| hard_negative | 8 | Vector 在 MRR / NDCG 上优于 Hybrid |

→ **默认 `retrievalMode=vector`**，Hybrid 保留为可选，不为展示而默认开启。

**Rewrite / Rerank latency（live；说明关闭原因，非主指标）：**

| Rewrite latency | count | P50 | P95 |
| --- | ---: | ---: | ---: |
| overall（39 条） | 39 | ~0 ms | ~22 s |
| **triggered-only** | **6** | **~22 s** | **~27 s** |

| Rerank latency（live CE） | P50 | P95 |
| --- | ---: | ---: |
| 单次 Top-N | ~45 ms | ~52 ms |

**Benchmark 范围：** 当前可靠性验证以 **92/92** 确定性离线评测、RunStore 运行聚合及故障注入用例为主；检索侧单独记录 Rewrite / Rerank latency。当前未将本地 Workflow 墙钟延迟作为 Production SLA 或核心求职指标。

### 7.3 CI 说明

- rewrite / rerank **默认关闭**，不依赖外部凭证 / 模型下载。  
- Windows 上真实进程类测试（exec platform、WebUI gateway smoke）**串行执行**，仍真实跑，不 skip。  
- 托管 runner 上不做全量 `pytest --cov`（历史 cancel / OOM）；严格 Linux leg 仍跑 ruff、basedpyright、全量测试与 92 评测。

---

## 8. 已覆盖的异常与边界（面试高频）

- 非法 JSON、空查询、数量超限 → 工具入口拒绝  
- 检索越界到未授权文档 → 按允许路径过滤  
- Gap 引用不存在的证据 → 拒绝状态迁移  
- 计划未覆盖缺失能力 → 有界重规划，不直接给用户确认  
- 重规划超过两次 → `REPLAN_EXHAUSTED`  
- 模型伪造「用户已确认」→ 拒绝（校验原始消息）  
- 确认旧 plan 版本 → `VERSION_CONFLICT`  
- 伪造 task_id → 回查 SQLite 后拒绝  
- 幂等重放 → 返回已有任务，不重复副作用  
- 创建中途重启 → 从已持久化任务恢复剩余清单  
- Embedding 失败 → 可降级 BM25  
- Rewrite / Rerank 超时或结构化失败 → 回退原 Query / RRF  
- 飞书附件路径被 LLM 改写空格 → 媒体目录内唯一规范化匹配  
- 飞书「先传文件后发文字」竞态 → 会话串行 + 短时合并 + 附件独立落盘  

---

## 9. 明确不做的范围（边界意识）

- Multi-Agent 协作平台、复杂多智能体编排  
- Memory / HyDE / Step-Back / Deep Mode 等未验证增强  
- Kubernetes / 大规模分布式 / 消息队列生产架构  
- 自动投递简历、爬取受保护招聘站、绕过验证码  
- 模型训练、SFT、RLHF  
- 把离线夹具指标包装成「生产用户成功率」  
- CI 全量 coverage 门禁（改为本地覆盖率）  
- 为刷绿改 expected value 或降低安全断言  

---

## 10. 快速启动（Docker，推荐）

```powershell
cd <项目根目录>   # 例如 D:\Omni\nanobot\source
Copy-Item .env.omniagent.example .env.omniagent
# 编辑：MODELSCOPE_API_KEY、FEISHU_*、NANOBOT_WEB_TOKEN

docker compose -f docker-compose.yml -f docker-compose.omniagent.yml build nanobot-gateway
docker compose -f docker-compose.yml -f docker-compose.omniagent.yml up -d nanobot-gateway
docker compose -f docker-compose.yml -f docker-compose.omniagent.yml ps
```

- WebUI：http://127.0.0.1:8765  
- 健康检查：http://127.0.0.1:18790/health  
- 日志：`docker compose -f docker-compose.yml -f docker-compose.omniagent.yml logs -f nanobot-gateway`

知识默认配置（示例，见 `examples/omniagent.config.example.json`）：

```json
"knowledge": {
  "retrievalMode": "vector",
  "queryRewrite": { "enabled": false },
  "rerank": { "enabled": false },
  "fallbackToLexical": true
}
```

详细权限与跨端说明见 `docs/omniagent-cross-channel-workflow.md`。

---

## 11. 演示验收清单（约 3 分钟）

1. WebUI 上传一份 PDF / Markdown 并入库。  
2. 提问并核对答案中的 `[K1]` 等引用。  
3. 飞书私聊：根据资料创建 1～2 个任务（或走确认后的工作流）。  
4. WebUI 列出任务，确认与飞书一致。  
5. 打开本次运行详情：耗时、工具、错误 / 重试（若有）。  
6. （可选）展示 `evaluation.run` / `evaluation.run_seal` 报告截图，并说明默认关闭 Rewrite / Rerank 的原因。

发布前自检：凭据未进 Git；飞书应用可用范围正确。

---

## 12. 简历 STAR 口径（定稿可用）

**S/T：** 面向私有资料分散、模型副作用不可直接信任的问题，设计 WebUI / 飞书双端的求职准备 Agent，将「简历 / JD 分析 → 计划确认 → 任务执行 → 定时跟进」变成可追踪流程。

**A：** 实现 SQLite FTS5/BM25、向量召回与可选 RRF，并把真实 chunk 引用绑定到 Gap；以 MCP 隔离任务领域，通过 Pydantic、幂等键、乐观锁、原始消息确认和副作用回查约束写操作；设计 Checkpoint 状态机与有界重规划；对 Rewrite / Rerank 做条件接入与消融，按指标决定默认关闭；将离线评测、pytest、Docker 与静态检查接入 GitHub Actions。

**R：** 确定性离线用例 **92/92**；构建 39 条人工标注检索集进行消融评测，live Vector 达 **Recall@3 79.5% / MRR 79.9% / NDCG@3 77.0%**（artifact：`retrieval_ablation_live.json`），结合 category 对比确定 Vector 为默认检索，并根据 live 实验结果关闭负收益 Rewrite / Rerank；Docker 与真实飞书链路验收，CI 全绿（含 Windows）。

**禁止写法：** 「已上线日活 X 万」「生产准确率提升 X%」「精通未使用的框架」「CI 全量 coverage 通过」「Rewrite / Rerank 已全面提升效果」；也不要把 offline category 跑数与 live 整体数字混写成同一组指标。

---

## 13. 相关文档索引

| 文档 | 内容 |
| --- | --- |
| `README.md` | 产品总览与快速启动 |
| `docs/omniagent-project.md` | 设计与验收摘要 |
| `docs/career-workflow-engineering.md` | 工作流状态机、恢复、面试追问与缺陷复盘 |
| `docs/omniagent-cross-channel-workflow.md` | 跨端与飞书交付 |
| `docs/omniagent-task-mcp.md` | 任务 MCP 设计 |
| `evaluation/README.md` | 92 基线与 seal 消融说明 |

---

## 14. 作者说明

本说明书覆盖并取代此前同名文档，描述当前**功能与 RAG 封版**后的可核验基线。指标与边界以代码、测试与 CI 为准；若后续基线升级，请同步更新第 7 节数字与默认配置表。
