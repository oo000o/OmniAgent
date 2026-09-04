# OmniAgent：私有知识与任务自动化 Agent

OmniAgent 是一个面向个人和小团队的自托管 AI 助手。用户可以在 WebUI 或飞书中
上传本地资料、基于证据提问、创建和更新任务，并设置定时检查。它解决的不是“再做
一个聊天机器人”，而是把分散在文档、聊天和待办中的信息连接成可追踪的工作流。

## 使用场景

以求职准备为例，用户将岗位 JD、学习计划和项目资料加入知识库，然后在飞书中说：

> 根据学习计划创建本周三个任务，每天 20:00 检查进度，并标出任务依据。

系统会检索带出处的资料片段，通过 MCP 服务创建结构化任务，保存定时计划，并在
到点后把检查结果发回原飞书会话。用户回到 WebUI 后仍能看到同一批任务、上下文和
完整运行记录。

这个流程也适用于课程资料整理、个人研究、项目文档问答和小团队事项跟进。它不依赖
企业内部数据；演示数据可以由用户自己的公开 JD、课程讲义和项目文档组成。

## 系统结构

```text
飞书（长连接） ─┐
                 ├─> Channel / Session ─> Agent Runtime ─> LLM
WebUI ───────────┘          │                  │
                            │                  ├─> Knowledge Tools
                            │                  │    ├─ SQLite FTS5 / BM25
                            │                  │    ├─ Vector Search
                            │                  │    └─ RRF + Citations
                            │                  │
                            │                  ├─> Task MCP Server
                            │                  │    └─ SQLite Tasks
                            │                  │
                            └─> Cron ──────────┴─> 原会话回传

Runtime Events ─> Run Store ─> WebUI 运行详情（耗时、Token、工具、重试、错误）
```

## 核心设计

- 混合检索：使用 SQLite FTS5/BM25 与向量相似度检索，通过 RRF 合并排名，并返回
  稳定的文件名、字符区间和引用编号。
- 工具边界：任务能力封装为独立 stdio MCP 服务，提供创建、查询、更新、取消工具；
  参数由 Pydantic 校验。
- 可靠写入：创建与变更支持幂等键，更新采用乐观锁，取消操作需要显式确认，降低
  模型重试和并发修改造成的重复或覆盖。
- 跨端连续性：单用户部署可开启统一会话；任务和知识状态独立持久化。Cron 保存任务
  来源渠道，到期后回传原飞书会话。
- 可观测性：运行事件写入 workspace 级 SQLite，WebUI 展示模型、状态、耗时、Token、
  工具调用、重试和错误详情。
- 安全边界：知识库文本被标记为不可信证据，不能覆盖系统指令；工具限制工作区范围，
  服务使用鉴权接口并限制最大工具迭代次数。

## 与原始 nanobot 的区别

nanobot 提供通用 Agent Runtime、渠道和基础工具；OmniAgent 在其扩展点上实现了一个
具体、可验收的应用产品：新增私有知识领域层、混合检索与引用、结构化任务 MCP、跨端
工作流、运行观测面板、可靠性策略和独立评测门禁。项目重点不是复刻框架，而是展示
如何基于通用 Runtime 完成业务建模、工程约束和端到端交付。

## 本地运行

1. 复制 `examples/omniagent.config.example.json` 到 nanobot 配置目录，并将
   `<PROJECT_ROOT>` 替换为项目绝对路径。
2. 设置 `FEISHU_APP_ID`、`FEISHU_APP_SECRET` 和 `FEISHU_OPEN_ID`。不要提交密钥。
3. 安装飞书依赖并启动网关：

   ```bash
   nanobot plugins enable feishu
   nanobot gateway
   ```

4. 也可以通过 Compose 构建包含飞书依赖的镜像。先复制环境变量模板并填入真实值：

   ```bash
   cp .env.omniagent.example .env.omniagent
   docker compose -f docker-compose.yml -f docker-compose.omniagent.yml build nanobot-gateway
   docker compose -f docker-compose.yml -f docker-compose.omniagent.yml up -d nanobot-gateway
   docker compose -f docker-compose.yml -f docker-compose.omniagent.yml ps
   docker compose -f docker-compose.yml -f docker-compose.omniagent.yml logs -f nanobot-gateway
   ```

   Windows PowerShell 将第一条命令替换为
   `Copy-Item .env.omniagent.example .env.omniagent`。该覆盖配置会把 WebUI 与健康检查
   绑定到容器网络、持久化 `workspace` 和任务数据库，并使用容器内路径启动任务 MCP。

飞书采用长连接模式，不要求公网 IP。具体权限和发布步骤见
[`omniagent-cross-channel-workflow.md`](./omniagent-cross-channel-workflow.md)。

## 可复现验证

运行离线评测：

```bash
python -m evaluation.run
```

报告保存在 `artifacts/evaluation/latest.json`。当前基线为 60 条确定性工程用例，覆盖
检索、引用、RRF、任务幂等和错误参数拒绝。它用于回归验证，不冒充生产流量或企业
真实数据。异步渠道重试、MCP 子进程、运行观测等能力由 pytest 集成测试覆盖。

GitHub Actions 会自动执行评测、保存 JSON artifact，并在 Docker 镜像内再次运行
打包后的 `omniagent-eval` 命令。

## 演示验收清单

- [x] WebUI 添加一份 PDF/Markdown，提问后答案包含可核对的 `[K1]` 引用。
- [x] 飞书创建三个任务，WebUI 能查询到相同任务。
- [x] 重放相同幂等键不会产生重复任务。
- [x] 定时检查触发后返回原飞书会话。
- [x] WebUI 能查看本次运行耗时、Token、工具、重试和错误。
- [x] `python -m evaluation.run` 达到 60/60。
- [x] Docker 镜像以非 root 运行，并在镜像内通过 60/60 容器评测。
- [ ] 仓库发布后由 GitHub Actions 再次通过 CI 门禁。

前五项中的真实飞书收发需要开发者应用凭证；离线测试使用受控替身，不把模拟结果
描述成线上验证。

## 简历口径（完成真实验收后使用）

**OmniAgent｜私有知识与任务自动化 Agent**  
技术栈：Python、FastAPI、Pydantic、SQLite/FTS5、RAG、MCP、Feishu、React、Docker、
GitHub Actions

- 设计并实现 WebUI/飞书双端 Agent，将混合检索、可追溯引用、结构化任务和定时回传
  串联为端到端工作流；使用 BM25、向量检索与 RRF 融合提升不同类型查询的召回稳定性。
- 将任务领域封装为独立 MCP 服务，通过 Pydantic 校验、幂等键、乐观锁和高风险操作
  确认机制约束模型写操作，并提供 SQLite 持久化及分页筛选。
- 建设运行观测与自动化质量门禁，记录耗时、Token、工具调用、重试和错误；建立 60 条
  可复现离线评测，并接入 pytest、Docker 与 GitHub Actions。

简历只能填写已经实跑并留存证据的指标。真实飞书链路和容器 CI 未通过前，不写
“已上线”“生产可用”或虚构业务提升比例。
