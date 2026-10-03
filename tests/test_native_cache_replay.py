"""缓存固定轨迹诊断必须保留真实失败，不把冻结答案当模型效果。"""
import asyncio
from dataclasses import replace
import pytest
from scripts.eval.harness import cache_replay
from tests.native_model_helpers import client_model, completion
from tests.test_prompt_cache import streaming
from tests.test_retrieval import _settings


@pytest.mark.parametrize('cancelled',[False,True])
async def test_replay_failure_or_cancellation_closes_model(tmp_path,monkeypatch,cancelled):
    entered=asyncio.Event()
    async def handler(request):
        entered.set()
        if cancelled:await asyncio.Event().wait()
        return streaming(completion())  # OK 不是预算/SKU JSON，不能判通过。
    model=await client_model(tmp_path,handler,stream=True)
    monkeypatch.setattr(cache_replay,'create_chat_model',lambda *args,**kwargs:model)
    task=asyncio.create_task(cache_replay.run_cache_replay(
        {'id':'failure','scenario':'budget','budgets':[300,180]},
        replace(_settings(tmp_path),context_prompt_layout='stable_prefix'),None,tmp_path,0,None,
        {'output_limit':128,'request_timeout_seconds':5}))
    try:
        await asyncio.wait_for(entered.wait(),2)
        if cancelled:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):await task
        else:
            row=await task
            assert not row['passed'] and not any(row['checks'].values())
            assert [item['assistant'] for item in row['transcript']]==['OK','OK']
            assert len(row['usage'])==2
        assert model.client.is_closed()
    finally:
        task.cancel()
        await asyncio.gather(task,return_exceptions=True)
        await model.aclose()


async def test_unknown_layout_is_rejected_before_model_construction(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('不得开始模型请求')
    monkeypatch.setattr(cache_replay,'create_chat_model',forbidden)
    with pytest.raises(ValueError,match='布局'):
        await cache_replay.run_cache_replay({'scenario':'budget'},
            replace(_settings(tmp_path),context_prompt_layout='typo'),None,tmp_path,0,None,{})
