# OmniAgent 项目说明书

> 版本口径：求职/演示封版基线（离线评测 92/92，GitHub Actions 全绿，Docker + 飞书真实验收通过）  
> 仓库：https://github.com/oo000o/OmniAgent  
> 适用岗位：AI Agent 应用开发 / Agent 开发工程师 / RAG·LLM 应用工程

---

## 1. 一句话介绍

**OmniAgent** 是面向个人与小团队的**自托管私有知识与任务自动化 Agent**：把本地文档问答、结构化任务、定时跟进和飞书/WebUI 双端交互，连成一条**可追踪、可恢复、可评测**的工作流——而不是又一个只能聊天的 Bot。

---

## 2. 要解决的问题

| 痛点 | 常见做法的问题 | OmniAgent 的做法 |
| --- | --- | --- |
| 资料散落在 PDF/笔记/聊天里 | 纯对话，无出处 | 混合检索 + 稳定引用 `[K1]` |
| 模型说「已创建任务」 | 文本不是事务回执 | 任务写入独立 MCP + SQLite，回查后才推进状态 |
| 重试导致重复待办 | 无幂等 | 稳定幂等键 + 权威存储对账 |
| 飞书与网页各说各话 | 会话分裂 | 统一会话 + 共享任务/知识库 |
| 长任务中途崩溃 | 无法恢复或重复执行 | Checkpoint + 故障注入回归 |
| 「效果感觉还行」 | 无法复现 | 92 条确定性离线基线 + CI |

**主演示场景（求职准备）：**  
用户上传简历与岗位 JD / 学习计划 → Agent 检索证据、分析差距、生成计划 → **用户明确确认** → 经 MCP 创建真实任务并挂定时检查 → 到点回传原飞书会话 → WebUI 可查同一批任务与完整运行 Trace。

同一套能力也适用于课程资料整理、个人研究文档问答、小团队事项跟进。

---

## 3. 与上游 nanobot 的关系（简历必说清）

| 层级 | 来源 | 说明 |
| --- | --- | --- |
| Agent Runtime、渠道、基础工具 | 开源 [HKUDS/nanobot](https://github.com/HKUDS/nanobot) | 通用执行与扩展点 |
| 私有知识领域层、混合检索与引用 | **本项目** | BM25 + 向量 + RRF、证据绑定 |
| 任务 MCP、幂等/乐观锁/确认 | **本项目** | 写操作工程约束 |
| 求职工作流状态机、有界重规划 | **本项目** | Human-in-the-loop |
| RunStore / Trace / 评测门禁 | **本项目** | 可观测与质量门禁 |
| 飞书长连接、Docker Compose 交付 | 上游能力 + **本项目配置与加固** | 端到端可验收产品 |

**表述原则：** 写「基于 nanobot Runtime 落地的应用产品」，不写「从零实现 Agent 框架」，不把未使用的 LangChain/LangGraph 写成「精通」。

---

## 4. 系统架构

```text
飞书（WebSocket 长连接） ─┐
                          ├─> Channel / Session ─> Agent Runtime ─> LLM（魔搭等兼容 API）
WebUI ────────────────────┘          │                  │
                                     │                  ├─> Knowledge Tools
                                     │                  │    ├─ SQLite FTS5 / BM25
                                     │                  │    ├─ Vector Search
                                     │                  │    └─ RRF + Citations
                                     │                  │
                                     │                  ├─> Task MCP Server（stdio）
                                     │                  │    └─ SQLite Tasks
                                     │                  │
                                     └─> Cron ──────────┴─> 按来源渠道回传

Runtime Events ─> Run Store（SQLite）─> WebUI 运行详情
```

**三类状态必须分清：**

1. **对话状态**：帮助模型理解语境，可丢失或重建。  
2. **工作流 Checkpoint**：阶段、证据、计划版本、真实 task/cron ID。  
3. **业务事实**：Task MCP 的 SQLite 与 Cron 存储——副作用是否成功以这里为准。

---

## 5. 核心能力说明

### 5.1 混合检索与引用（RAG）

- 文档解析与切分，索引落本地 SQLite。  
- **BM25（FTS5）** 与 **向量召回** 并行，经 **RRF** 融合排序。  
- 返回稳定引用：文件名、字符区间、引用编号；工作流 Gap 必须绑定已保存证据，禁止「凭记忆编差距」。  
- 检索结构化事件记录模式/成败/耗时/条数等，**不落查询正文与文档内容**。  
- Embedding 不可用时可降级 BM25，并标记 `retrieval_fallback`。

### 5.2 任务领域（MCP）

- 独立 stdio MCP：创建、查询、分页筛选、更新、取消。  
- **Pydantic** 校验工具参数；非法输入在入口拒绝。  
- **幂等键** 防重复创建；**乐观锁（expected_version）** 防丢失更新。  
- 高风险取消等操作要求**显式确认**（读取 runtime 原始用户消息，不信任模型转述）。

### 5.3 求职工作流与有界重规划

状态机（摘要）：

```text
documents_ready → evidence_retrieved → gaps_analyzed → plan_verifying
        ↺ 有界重规划（最多 2 次）→ awaiting_confirmation → confirmed
        → tasks_creating → tasks_created → scheduled → completed
```

- 确定性 Verifier 拒绝未覆盖 Gap、未知能力引用、重复标题等；超限以 `REPLAN_EXHAUSTED` 失败，不进死循环。  
- 确认绑定具体 `plan_revision`；确认旧版本 → `VERSION_CONFLICT`。  
- 登记 task_id 前必须回查共享 TaskStore（存在性、一对一、source 前缀等）。

### 5.4 失败恢复

```text
MCP/Cron 已成功 → Checkpoint 写入失败 → 进程退出
    → 重启后按幂等键/来源读取权威回执
    → 重建已完成映射 → 只执行尚未完成的调用
```

同一恢复动作重复执行三次，仍只保留一条 Task 或一条 Cron（由故障注入用例覆盖）。

### 5.5 跨端与定时

- 单用户部署可开启统一会话（WebUI ↔ 飞书）。  
- Cron 记录**创建时的来源渠道与会话**，触发后回传原飞书会话。  
- 飞书使用**长连接**，不要求公网 IP / Webhook。

### 5.6 可观测与安全

- RunStore 最小 Trace：`run_id → workflow_id → plan_revision → step → tool → result/error/retry`。  
- 聚合接口给出样本数、成功率、Avg/P95 延迟、工具成功率（仓库样本分位数，**非生产 SLA**）。  
- 知识文本视为不可信证据，不能覆盖系统指令；工具限制工作区；服务令牌鉴权；限制最大工具迭代。  
- Trace/日志不落简历/JD 原文、完整查询、完整工具参数或凭证。

---

## 6. 技术栈

| 类别 | 技术 |
| --- | --- |
| 语言与运行时 | Python 3.11+ |
| Agent / 渠道 | nanobot Runtime、飞书（lark-oapi 长连接）、WebUI（React/TypeScript） |
| 数据与检索 | SQLite、FTS5/BM25、向量检索、RRF |
| 工具与协议 | MCP（stdio）、Pydantic |
| 交付与质量 | Docker Compose、GitHub Actions、pytest、Ruff、BasedPyright |
| 模型接入 | ModelScope / OpenAI 兼容 API（配置化，密钥不进仓库） |

---

## 7. 工程验证与指标（可写进简历的口径）

| 验证项 | 结果 | 含义边界 |
| --- | --- | --- |
| 确定性离线评测 | **92 / 92** | 工程夹具回归，非生产准确率 |
| 其中求职工作流 | 17（成功 9 / 边界 4 / 恢复 4） | 场景通过率 |
| 运行观测 | 8 | Trace/聚合契约 |
| 故障注入与隐私 | 7 | 恢复与省略策略 |
| 检索夹具（12 查询） | 报告 BM25 / 向量 / RRF 的 Recall@3、MRR、NDCG@3 | 确定性向量夹具，非线上 Embedding 分数 |
| GitHub Actions | 全部门禁通过（含 3.11 / 3.14 / Windows、WebUI、TUI、Docker、评测） | CI **不含**全量 `pytest --cov`（托管 runner 资源限制；覆盖率用本地 `pytest --cov`，`fail_under=75`） |
| Docker | 非 root、健康检查、持久化、镜像内评测 | 自托管演示级 |
| 飞书 | 长连接收发、知识引用、任务与跨端一致 | 需有效开发者应用凭证 |

运行评测：

```bash
python -m evaluation.run
# 报告：artifacts/evaluation/latest.json
```

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
- 飞书附件路径被 LLM 改写空格 → 媒体目录内唯一规范化匹配（不做模糊跨目录搜索）  
- 飞书「先传文件后发文字」竞态 → 会话串行 + 短时合并 + 附件独立落盘路径  

---

## 9. 明确不做的范围（展示边界意识）

- Multi-Agent 协作平台、复杂多智能体编排框架  
- Kubernetes / 大规模分布式 / 消息队列生产架构  
- 自动投递简历、爬取受保护招聘站、绕过验证码  
- 模型训练、SFT、RLHF  
- 把离线夹具指标包装成「生产用户成功率」  
- CI 全量 coverage 门禁（改为本地覆盖率）  

---

## 10. 快速启动（Docker，推荐）

```powershell
cd <项目根目录>   # 例如 D:\Omni\nanobot\source
Copy-Item .env.omniagent.example .env.omniagent   # 若尚无
# 编辑填入：MODELSCOPE_API_KEY、FEISHU_*、NANOBOT_WEB_TOKEN

docker compose -f docker-compose.yml -f docker-compose.omniagent.yml build nanobot-gateway
docker compose -f docker-compose.yml -f docker-compose.omniagent.yml up -d nanobot-gateway
docker compose -f docker-compose.yml -f docker-compose.omniagent.yml ps
```

- WebUI：http://127.0.0.1:8765  
- 健康检查：http://127.0.0.1:18790/health  
- 日志：`docker compose -f docker-compose.yml -f docker-compose.omniagent.yml logs -f nanobot-gateway`

详细权限与跨端说明见 `docs/omniagent-cross-channel-workflow.md`。

---

## 11. 演示验收清单（面试 Demo 脚本）

建议 3 分钟路径：

1. WebUI 上传一份 PDF/Markdown 并入库。  
2. 提问并核对答案中的 `[K1]` 等引用。  
3. 飞书私聊：根据资料创建 1～2 个任务（或走确认后的工作流）。  
4. WebUI 列出任务，确认与飞书一致。  
5. 打开本次运行详情：耗时、工具、错误/重试（若有）。  
6. （可选）说明定时回传原会话；或展示 `evaluation.run` 报告截图。

发布前自检：凭据未进 Git；飞书应用可用范围正确。

---

## 12. 简历 STAR 口径（定稿可用）

**S/T：** 面向私有资料分散、模型副作用不可直接信任的问题，设计 WebUI/飞书双端的求职准备 Agent，将「简历/JD 分析 → 计划确认 → 任务执行 → 定时跟进」变成可追踪流程。

**A：** 实现 SQLite FTS5/BM25、向量召回与 RRF 融合，并把真实 chunk 引用绑定到 Gap；以 MCP 隔离任务领域，通过 Pydantic、幂等键、乐观锁、原始消息确认和副作用回查约束写操作；设计 Checkpoint 状态机与有界重规划；将离线评测、pytest、Docker 与静态检查接入 GitHub Actions。

**R：** 确定性离线用例 **92/92**（求职工作流 17/17、运行观测 8/8、故障注入 7/7）；Docker 与真实飞书链路完成验收；CI 全绿。检索夹具上另行报告 BM25/向量/RRF 指标（确定性夹具，非线上分数）。

**禁止写法：** 「已上线日活 X万」「生产准确率提升 X%」「精通未使用的框架」「CI 全量 coverage 通过」。

---

## 13. 相关文档索引

| 文档 | 内容 |
| --- | --- |
| `README.md` | 产品总览与快速启动 |
| `docs/omniagent-project.md` | 设计与验收摘要 |
| `docs/career-workflow-engineering.md` | 工作流状态机、恢复、面试追问与缺陷复盘 |
| `docs/omniagent-cross-channel-workflow.md` | 跨端与飞书交付 |
| `docs/omniagent-task-mcp.md` | 任务 MCP 设计 |
| `evaluation/README.md` | 评测说明 |

---

最终检索链路（封版增强）：

```text
Query Analysis → Conditional Rewrite → BM25 + Vector → RRF → Cross-Encoder Rerank → Citation
```

- Rewrite 仅对 complex / vague / multi_intent 开启；失败回退原 Query。  
- Rerank 仅对 RRF Top-N；失败回退 RRF。  
- 12 条 CI 夹具保留；另约 40 条人工标注消融语料见 `evaluation.run_seal`。  
- Ablation / Benchmark / Fault 聚合数字属于本地 evaluation，不是 Production SLA。

## 14. 作者说明

本说明书描述仓库中**已实现且可核验**的能力，供秋招项目介绍、面试深挖与 Demo 对齐使用。指标与边界以代码、测试与 CI 为准；后续若基线版本升级，请同步更新第 7 节数字与封版说明。
