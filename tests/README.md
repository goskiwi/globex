# 如何运行测试

使用 Docker，测试环境由 Makefile 提供合成配置，不读取 `.env` 或调用外部模型。

| 命令 | 场景 |
| --- | --- |
| `make` / `make test-core` | 日常选购业务修改：检索约束、偏好、上下文、子任务、推荐、交易审批、身份隔离和中断 |
| `make test-special` | 默认检查打包配置、观测验证、Prompt 发布、迁移和 Redis 重启恢复 |
| `make test-special TESTS="tests/test_semantic_memory.py tests/test_memory_upgrade.py"` | 按本次修改明确选择已有测试文件；可以传 pytest 节点 ID |
| `make test-backend` | 完整后端回归，包含所有专项和历史评测测试 |
| `make test-frontend` | 前端测试及构建 |
| `make test` | 完整后端、前端及前端构建 |

每个入口先构建测试镜像；未改变的依赖层使用 Docker 缓存。核心集合是常见业务边界，
并不代替修改模块的专项验证。跨模块修改和重要交付使用完整回归。
`pytest` 仍默认收集全部 tests，没有跳过规则、自动门禁或新的测试框架。

安全和交易保护仍保留：未经批准不写入、重复执行不重复扣库存、买家隔离、
错误事实不能交付等。合并用例时搬移独有断言，不只为减少数字而删保护。

目录测试验证有效性及必要业务覆盖，不冻结当前商品总数。
历史目录生成器和评测数据仍有调用者，其复现测试留在完整回归。
测试所需脱敏反例位于 `tests/fixtures`，原始证据仍留在 `eval/verification`。

`test_runtime_distribution.py` 的打包测试在构建后的 Docker 测试镜像内执行，
实际加载与运行镜像同源的 `/app/catalog`，不是读取 Dockerfile 字符串模拟成功。
