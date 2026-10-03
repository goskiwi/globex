# -*- coding: utf-8 -*-
"""发布形态与评测命令的回归契约。"""
from __future__ import annotations

from pathlib import Path
import re
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_packaged_catalog_loads_with_empty_mutable_data(tmp_path, monkeypatch):
    from app.infrastructure.persistence import seed_products
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # 在构建好的测试镜像中实际加载与运行镜像同源的目录，不检查 Dockerfile 文本。
    assert seed_products._CATALOG_FIXTURE.is_relative_to(PROJECT_ROOT / "catalog")
    products = seed_products.build_seed_products()
    assert products and any(product.skus for product in products)
    assert list(tmp_path.iterdir()) == []


def test_compose_passes_reranker_configuration_to_app_and_worker():
    compose = yaml.safe_load((PROJECT_ROOT / "docker/docker-compose.yaml").read_text())
    app = compose["services"]["app"]["environment"]
    worker = compose["services"]["worker"]["environment"]
    for key in ("RERANKER_BASE_URL", "RERANKER_MODEL"):
        # 校验实际服务映射中的变量绑定，允许 YAML anchor 和默认值写法变化。
        assert re.match(r"\$\{" + key + r"(?:[-:]|\})", app[key])
        assert worker[key] == app[key]


def test_empty_fallback_does_not_inject_another_provider_model():
    compose = yaml.safe_load((PROJECT_ROOT / "docker/docker-compose.yaml").read_text())
    for service in ("app", "worker"):
        assert compose["services"][service]["environment"]["LLM_FALLBACK_MODEL"] == "${LLM_FALLBACK_MODEL-}"


def test_compose_shares_context_and_independent_embedding_configuration():
    compose = yaml.safe_load((PROJECT_ROOT / "docker/docker-compose.yaml").read_text())
    required = {
        "CONTEXT_STRATEGY": "${CONTEXT_STRATEGY:-layered}",
        "CONTEXT_PRUNING_TIMING": "${CONTEXT_PRUNING_TIMING:-pressure}",
        "CONTEXT_PRODUCT_TOKENS": "${CONTEXT_PRODUCT_TOKENS:-6000}",
        "CONTEXT_TARGET_TOKENS": "${CONTEXT_TARGET_TOKENS:-48000}",
        "EMBEDDING_BASE_URL": "${EMBEDDING_BASE_URL-}",
        "EMBEDDING_API_KEY": "${EMBEDDING_API_KEY-}",
        "EMBEDDING_DIM": "${EMBEDDING_DIM:-1024}",
        "EMBEDDING_VERSION": "${EMBEDDING_VERSION-}",
    }
    for service in ("app", "worker"):
        environment = compose["services"][service]["environment"]
        for key, value in required.items():
            assert environment.get(key) == value, (service, key)


def test_local_stack_has_dedicated_project_and_loopback_ports():
    compose=yaml.safe_load((PROJECT_ROOT/'docker/docker-compose.yaml').read_text())
    assert compose['name']=='globex'
    assert set(compose['volumes'])=={'app-data','qdrant-data','redis-data'}
    for service in compose['services'].values():
        assert all(port.startswith('127.0.0.1:') for port in service.get('ports',[]))
    assert compose['services']['frontend']['depends_on']['app']['condition']=='service_healthy'


def test_empty_optional_embedding_configuration_falls_back_to_llm(monkeypatch,tmp_path):
    from app.infrastructure.settings import load_settings
    monkeypatch.setenv("LLM_BASE_URL","https://example.invalid/v1")
    monkeypatch.setenv("LLM_API_KEY","placeholder")
    monkeypatch.setenv("DATA_DIR",str(tmp_path))
    for key in ("EMBEDDING_BASE_URL","EMBEDDING_API_KEY"):
        monkeypatch.setenv(key,"")
    settings=load_settings()
    assert settings.embedding_base_url==settings.llm_base_url
    assert settings.embedding_api_key==settings.llm_api_key
