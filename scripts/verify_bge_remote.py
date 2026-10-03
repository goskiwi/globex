"""真实远端推理冒烟；只输出数值/状态，不输出密钥和完整向量。"""
import json
import math
from pathlib import Path
import time
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import importlib.metadata as metadata

ROOT=Path('/data4/sybai/globex')
BASE='http://127.0.0.1:18780'


def main():
    key=(ROOT/'service/api-key').read_text().strip()
    health=json.load(urlopen(BASE+'/health',timeout=5))
    def post(path,payload):
        request=Request(BASE+path,data=json.dumps(payload).encode(),
            headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        started=time.monotonic()
        with urlopen(request,timeout=60) as response:result=json.load(response)
        return result,round((time.monotonic()-started)*1000,2)
    texts=['想找一个适合旅行的防水背包','A waterproof backpack for travel and hiking.','A stainless steel electric kettle for boiling water.']
    embedded,embedding_ms=post('/v1/embeddings',{'model':'BAAI/bge-m3','input':texts})
    vectors=[item['embedding'] for item in embedded['data']]
    assert len(vectors)==3 and all(len(v)==1024 and all(math.isfinite(x) for x in v) for v in vectors)
    norms=[sum(x*x for x in v)**.5 for v in vectors]
    assert all(abs(norm-1)<.02 for norm in norms)
    similarities=[sum(a*b for a,b in zip(vectors[0],v)) for v in vectors[1:]]
    assert similarities[0]>similarities[1]
    ranked,reranker_ms=post('/v1/rerank',{'model':'BAAI/bge-reranker-v2-m3','query':texts[0],
        'documents':[texts[1],texts[2],'A facial moisturizer for dry skin.'],'top_n':3})
    assert len(ranked['results'])==3 and ranked['results'][0]['index']==0
    assert all(math.isfinite(row['relevance_score']) for row in ranked['results'])
    try:
        urlopen(Request(BASE+'/v1/embeddings',data=json.dumps({'model':'BAAI/bge-m3','input':['test']}).encode(),
            headers={'Content-Type':'application/json'}),timeout=10)
        raise AssertionError('未授权请求被放行')
    except HTTPError as error:assert error.code==401
    before=json.loads((ROOT/'service/environment-before.json').read_text())['packages']
    after={item.metadata['Name']:item.version for item in metadata.distributions() if item.metadata['Name']}
    assert before==after
    report={'status':'passed','health':health,'embedding_ms':embedding_ms,'reranker_ms':reranker_ms,
        'embedding_dimension':1024,'embedding_norms':norms,'query_document_cosines':similarities,
        'reranker_results':ranked['results'],'unauthorized_status':401,'conda_packages_unchanged':True}
    destination=ROOT/'logs'/('verification-'+str(time.time_ns())+'.json')
    with destination.open('x') as output:json.dump(report,output,ensure_ascii=False,indent=2)
    print(json.dumps({**report,'evidence_file':str(destination)},ensure_ascii=False))


if __name__=='__main__':main()
