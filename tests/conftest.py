import pytest
from tests.sql_fixtures import engines
@pytest.fixture(autouse=True)
async def close_sql_test_engines():
    try:yield
    finally:
        while engines:await engines.pop().dispose()
