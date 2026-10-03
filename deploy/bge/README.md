# 远端 BGE 检索服务

默认选型为 `BAAI/bge-m3`（稠密 1024 维）与 `BAAI/bge-reranker-v2-m3`。服务位于 `/data4/sybai/globex`；2026-10-01发现原glodex所在/home只读并产生I/O后，服务运行环境改为本项目的`/data4/sybai/globex/service-runtime`，不重装或删除原Conda环境，不需要启动远端Docker daemon。依赖版本来自原环境快照，见[requirements-service.txt](requirements-service.txt)；权重仍复用原models目录。实际恢复进度见[环境记录](../../docs/改动记录/2026-10-01/BGE环境存储故障定位.md)。

## 管理服务

在本项目目录执行：

```sh
make bge-status
make bge-start
make bge-stop
```

仅管理该目录登记且身份匹配的进程。模型按一次开发/演示启停，不再默认长期保留：`make bge-start`开启远端两个模型及本机隧道，等待加载完成；用完执行`make bge-stop`关闭模型进程及隧道、释放本项目GPU占用。启动失败时收回本项目服务，不强制杀其他进程。

`make docker-up`先调用上述启动入口，再启动项目；`make docker-stop`先停止项目容器，再关闭远端模型和隧道。关闭网页不会自动停止项目，演示结束需执行停止命令；一轮使用期间复用模型，不每条消息重载。服务器重启后也不自动启动，不写开机服务、空闲定时器或新守护进程。

仅进程可见的 GPU 为物理 GPU 0，两模型以 FP16 共存，GPU 1 不加载模型。单请求超过 8192 token 或总批量超过 32768 token 明确拒绝，不静默截断；高负载时队列满返回 429。

## 本机与 Docker 连接

```sh
make bge-start
curl --fail http://127.0.0.1:18780/health
```

隧道只绑定本机回环地址，连接到远端 `127.0.0.1:18780`，使用已有 SSH 别名 `globex`，不改 SSH 配置。

本次部署生成的 `.env.bge` 包含独立服务密钥，权限为 0600，已在 Git 和 Docker 构建上下文中忽略。**不要提交、截图或粘贴其内容。** 它只包含检索相关配置，不覆盖 LLM 凭据。Docker Desktop 通过 `host.docker.internal:18780` 访问隧道；其它 Docker 平台需要单独核对宿主网络路由，不能直接开放公网监听。

客户端真实验证：

```sh
docker run --rm --env-file .env.bge \
  -e LLM_API_KEY=unused-for-retrieval-probe \
  -v "$PWD/scripts:/app/scripts:ro" \
  globex-backend python -B -m scripts.verify_bge_client
```

该命令只调用 embedding/rerank，不请求聊天模型。正式应用仍需自己的 LLM 配置；准备好 `.env` 后可按顺序加载两个文件：

```sh
docker compose --env-file .env --env-file .env.bge \
  -f docker/docker-compose.yaml up -d --build
```

直接使用Compose不会管理远端模型启停；日常应使用`make docker-up`/`make docker-stop`。只验证检索时使用`make bge-start`，验证完执行`make bge-stop`。

后续用户要求创建本机 Docker 环境后，已实际启动 `globex` 分组的 app、worker、frontend、Redis 和 Qdrant。查看使用 `make docker-ps`，停止使用 `make docker-stop`；记录见 [本机 Docker 部署](../../docs/改动记录/2026-09-30/本机Docker-globex环境部署.md)。

## 数据与索引边界

`.env.bge` 使用 `sybai_globex_products_bge_m3_20260930` 与 `sybai_globex_category_bge_m3_20260930` 新集合名，避免复用旧模型空间。语义回答缓存暂关闭，避免旧 embedding 缓存误命中；原缓存未删除。此次模型安装没有重建/清空现有向量库，没有修改订单、库存或买家记录。

后续 Docker 初始化已在新的 `globex_qdrant-data` 卷中建入 3700 商品向量及 230 知识片段（45 篇文档），没有覆盖原有其它项目的数据卷。

## 服务协议

- `GET /health`：加载完成后返回模型、版本、维度与窗口。
- `POST /v1/embeddings`：OpenAI 兼容 `model`/`input`，返回归一化向量与 index。
- `POST /v1/rerank`：`model`/`query`/`documents`/`top_n`，返回按分数降序的原始文档 index。分数为 logit，不是概率。
- 推理接口必须携带 `Authorization: Bearer <专用密钥>`；没有聊天接口。

日志和部署版本：远端 `logs/`、`models/manifest.json`、`service/environment-before.json`。下载失败的临时片段保留在本项目 logs，不参与模型加载。

单独停止本机隧道使用 `make bge-tunnel-stop`；它不会停止远端模型，不应代替演示收尾的`make bge-stop`。停止服务和隧道不会删除模型权重、日志、旧缓存或业务数据。按需启停的实测见[2026-10-02记录](../../docs/改动记录/2026-10-02/检索模型按需启停.md)。
