"""专用 BGE HTTP 契约单测；GPU 加载另在远端真实验证。"""
from pathlib import Path
from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest
from scripts.bge_retrieval_service import create_app,Engine,EMBEDDING_MODEL,RERANKER_MODEL

KEY='test-service-key-that-is-not-a-real-secret'


class FakeEngine:
    embedding_limit=rerank_limit=8192
    def __init__(self,*args):pass
    def embed(self,texts):return [[1.0]+[0.0]*1023 for _ in texts],len(texts)*3
    def rerank(self,query,documents):return [float(i) for i in range(len(documents))],len(documents)*4


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path,KEY,engine_factory=FakeEngine)) as client:yield client


def test_health_and_authentication(client):
    assert client.get('/health').json()['embedding']==EMBEDDING_MODEL
    payload={'model':EMBEDDING_MODEL,'input':['测试']}
    assert client.post('/v1/embeddings',json=payload).status_code==401
    assert client.post('/v1/embeddings',json=payload,headers={'Authorization':'Bearer wrong'}).status_code==401


def test_embedding_contract_preserves_indexes_and_dimension(client):
    response=client.post('/v1/embeddings',json={'model':EMBEDDING_MODEL,'input':['背包','backpack']},headers={'Authorization':'Bearer '+KEY})
    assert response.status_code==200
    payload=response.json()
    assert [item['index'] for item in payload['data']]==[0,1]
    assert all(len(item['embedding'])==1024 for item in payload['data'])
    assert payload['usage']['prompt_tokens']==6


def test_rerank_returns_sorted_indexes_and_all_requested_scores(client):
    response=client.post('/v1/rerank',json={'model':RERANKER_MODEL,'query':'背包','documents':['a','b','c'],'top_n':3},headers={'Authorization':'Bearer '+KEY})
    assert response.status_code==200
    assert [item['index'] for item in response.json()['results']]==[2,1,0]
    assert response.json()['usage']['input_tokens']==12


@pytest.mark.parametrize('payload',[
    {'model':EMBEDDING_MODEL,'input':['text'],'dimensions':512},
    {'model':EMBEDDING_MODEL,'input':['text']*33},
    {'model':EMBEDDING_MODEL,'input':[]},
])
def test_invalid_requests_fail_without_silent_conversion(client,payload):
    assert client.post('/v1/embeddings',json=payload,headers={'Authorization':'Bearer '+KEY}).status_code==422


def test_wrong_model_and_top_n_are_rejected(client):
    headers={'Authorization':'Bearer '+KEY}
    assert client.post('/v1/embeddings',json={'model':'other','input':['test']},headers=headers).status_code==400
    assert client.post('/v1/rerank',json={'model':RERANKER_MODEL,'query':'test','documents':['a'],'top_n':2},headers=headers).status_code==400


def test_token_limits_are_explicit_not_truncation():
    with pytest.raises(HTTPException):Engine.check_lengths([8193],8192)
    with pytest.raises(HTTPException):Engine.check_lengths([8192]*5,8192)
    Engine.check_lengths([8192],8192)
