HARNESS_RUN_DIR ?= eval/verification/harness-local-$(shell date +%Y%m%d-%H%M%S)
.DEFAULT_GOAL := test-core
HARNESS_SUITE ?= eval/harness/v1/suite.json
HARNESS_STRATEGIES ?= current,candidate
HARNESS_PYTHON ?= docker run --rm $(if $(wildcard .env),--env-file .env,) -v "$(CURDIR)/eval/verification:/app/eval/verification" globex-backend-test python
GLOBEX_COMPOSE = docker compose -p globex --env-file .env --env-file .env.bge -f docker/docker-compose.yaml

# 仅选择已有用例，不改变 pytest 默认全量收集，也不连接真实服务。
BACKEND_TEST_RUN = docker run --rm --network none -e LLM_API_KEY=test-only -e EMBEDDING_API_KEY=test-only -e IDENTITY_HMAC_SECRET=synthetic-globex-test-secret-000000000000 globex-backend-test
CORE_TESTS = tests/test_catalog_sku_selection.py tests/test_pricing.py \
 tests/test_structured_preferences.py tests/test_preference_lifecycle.py \
 tests/test_native_memory_confirmation.py tests/test_langgraph_runtime.py \
 tests/test_context_boundaries.py tests/test_request_context.py \
 tests/test_agent_handoff.py tests/test_research_delivery_boundaries.py \
 tests/test_product_decision.py tests/test_product_delivery.py \
 tests/test_confirmation_service.py tests/test_confirmation_routes.py \
 tests/test_trade_store.py tests/test_identity.py tests/test_session_fencing.py \
 tests/test_execution_stop.py tests/test_shopping_flow.py
# 可覆盖，例如 make test-special TESTS=tests/test_queue_redis_restart.py
TESTS ?= tests/test_runtime_distribution.py tests/test_langfuse_verification.py \
 tests/test_prompt_registry.py tests/test_runtime_schema_migrations.py \
 tests/test_queue_redis_restart.py

.PHONY: docker-up docker-ps docker-stop
docker-up: bge-start
	$(GLOBEX_COMPOSE) up -d --build
	$(GLOBEX_COMPOSE) exec -T frontend nginx -s reload

docker-ps:
	$(GLOBEX_COMPOSE) ps

docker-stop:
	$(GLOBEX_COMPOSE) stop
	$(MAKE) bge-stop

.PHONY: test-backend-image test-core test-special test-backend test-frontend test harness-image eval-harness eval-harness-contracts eval-harness-dev eval-harness-release globex-ssh

test-backend-image:
	docker build --target test -t globex-backend-test .

test-core: test-backend-image
	$(BACKEND_TEST_RUN) pytest -q $(CORE_TESTS)

test-special: test-backend-image
	$(BACKEND_TEST_RUN) pytest -q $(TESTS)

test-backend: test-backend-image
	$(BACKEND_TEST_RUN) pytest -q

test-frontend:
	docker build --target build -t globex-frontend-test frontend
	docker run --rm --network none globex-frontend-test sh -c 'npm test -- --run && npm run build'

test: test-backend test-frontend

globex-ssh:
	./scripts/connect_globex.sh

.PHONY: bge-start bge-status bge-stop bge-tunnel bge-tunnel-stop
bge-start:
	sh scripts/bge_session.sh start

bge-status:
	sh scripts/bge_remote.sh status

bge-stop:
	sh scripts/bge_session.sh stop

bge-tunnel:
	sh scripts/bge_tunnel.sh start

bge-tunnel-stop:
	sh scripts/bge_tunnel.sh stop

# 每次 Harness 修改：安全契约 + 完整真实模型冒烟 + 同源码证据检查。
harness-image:
	docker build --target test -t globex-backend-test .

eval-harness: harness-image
	$(HARNESS_PYTHON) -m scripts.eval.harness run --suite $(HARNESS_SUITE) --profile smoke --strategies $(HARNESS_STRATEGIES) --output $(HARNESS_RUN_DIR)
	$(HARNESS_PYTHON) -m scripts.eval.harness verify $(HARNESS_RUN_DIR)

eval-harness-contracts: harness-image
	$(HARNESS_PYTHON) -m scripts.eval.harness run --suite $(HARNESS_SUITE) --profile contracts --output $(HARNESS_RUN_DIR)

eval-harness-dev: harness-image
	$(HARNESS_PYTHON) -m scripts.eval.harness run --suite $(HARNESS_SUITE) --profile dev --strategies $(HARNESS_STRATEGIES) --output $(HARNESS_RUN_DIR)

eval-harness-release: harness-image
	$(HARNESS_PYTHON) -m scripts.eval.harness run --suite $(HARNESS_SUITE) --profile release --strategies $(HARNESS_STRATEGIES) --output $(HARNESS_RUN_DIR) --require-benefit
	$(HARNESS_PYTHON) -m scripts.eval.harness verify $(HARNESS_RUN_DIR) --minimum release
