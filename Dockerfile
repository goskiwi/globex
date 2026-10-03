# ---- 构建阶段：uv 安装依赖到独立虚拟环境 ----
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# 先装依赖（利用层缓存），再拷源码
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY app ./app
# 测试与运行镜像使用同一份打包目录，不能被可写 data 卷遮住。
COPY data/catalog-v3.jsonl ./catalog/catalog-v3.jsonl

# ---- 测试阶段：开发依赖与测试只存在于容器，不在宿主机创建 .venv ----
FROM builder AS test
RUN apt-get update && apt-get install -y --no-install-recommends redis-server make && apt-get clean
RUN uv sync --frozen --no-install-project --extra optimization
COPY scripts ./scripts
COPY tests ./tests
COPY knowledge ./knowledge
COPY data ./data
COPY eval ./eval
COPY Dockerfile Makefile ./
COPY docker ./docker
COPY frontend/public ./frontend/public
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
CMD ["pytest", "-q"]

# ---- 运行阶段：slim 运行时，仅带 venv 与源码 ----
FROM python:3.11-slim

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1

COPY --from=builder /app/.venv /app/.venv
COPY app ./app
# 品类洞察知识库文档（启动时灌入 RAG 向量库）
COPY knowledge ./knowledge
# 版本化商品底座放在只读目录，不能放进会被命名卷遮蔽的 /app/data。
COPY --from=builder /app/catalog ./catalog

# 运行时数据目录（会话/偏好；向量库走 QDRANT_URL 指向 qdrant 服务）
RUN mkdir -p /app/data
ENV DATA_DIR=/app/data

EXPOSE 8000
CMD ["uvicorn", "app.presentation.server:app", "--host", "0.0.0.0", "--port", "8000"]
