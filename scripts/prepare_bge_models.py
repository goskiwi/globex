"""仅在指定项目目录准备公开 BGE 权重；已有用户缓存只读复用。"""
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path('/data4/sybai/globex')
OLD_CACHE = Path('/home/sybai/.cache/huggingface/hub')
REPOS = {'bge-m3':'BAAI/bge-m3','bge-reranker-v2-m3':'BAAI/bge-reranker-v2-m3'}


def main():
    if ROOT.is_symlink() or ROOT.resolve()!=ROOT or ROOT.stat().st_uid!=os.getuid():
        raise RuntimeError('目标目录不是当前用户拥有的预期目录')
    os.environ['HF_HOME']=str(ROOT/'cache/huggingface')
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
    api=HfApi(endpoint='https://huggingface.co',token=False)
    report={'models':{},'environment':sys.prefix}
    allowed={'config.json','config_sentence_transformers.json','sentence_bert_config.json',
             'modules.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json',
             'sentencepiece.bpe.model','1_Pooling/config.json'}
    for directory,repo in REPOS.items():
        info=api.model_info(repo,files_metadata=True,timeout=30)
        names={item.rfilename for item in info.siblings}
        weight='model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
        if weight not in names:raise RuntimeError('未找到预期单文件权重')
        selected=sorted((names & allowed)|{weight})
        destination=ROOT/'models'/directory
        destination.mkdir(parents=True,exist_ok=True)
        if destination.resolve()!=destination:raise RuntimeError('模型目录存在符号链接，未写入')
        metadata={'model':repo,'revision':info.sha,'files':selected}
        revision_file=destination/'globex-revision.json'
        if revision_file.exists():
            if json.loads(revision_file.read_text())!=metadata:raise RuntimeError('目录已有不同模型版本，未覆盖')
        else:
            if any(destination.iterdir()):raise RuntimeError('模型目录已有未知文件，未覆盖')
            with revision_file.open('x') as output:json.dump(metadata,output,indent=2)
        copied=downloaded=0
        for name in selected:
            target=destination/name
            cached=try_to_load_from_cache(repo,name,revision=info.sha,cache_dir=str(OLD_CACHE))
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                # 重跑只允许复用同一模型版本，不覆盖未知文件。
                if not revision_file.exists() or json.loads(revision_file.read_text()).get('revision')!=info.sha:
                    raise RuntimeError('目标包含未确认版本文件，未覆盖')
                continue
            if isinstance(cached,str) and Path(cached).is_file():
                with open(cached,'rb') as source, target.open('xb') as output:
                    shutil.copyfileobj(source,output,1024*1024)
                copied+=1
            else:
                hf_hub_download(repo,name,revision=info.sha,local_dir=str(destination),token=False,
                    endpoint='https://huggingface.co')
                downloaded+=1
            print(json.dumps({'model':repo,'file':name,'ready':True}),flush=True)
        report['models'][directory]={**metadata,'copied_from_existing_cache':copied,'downloaded':downloaded}
    with (ROOT/'models/manifest.json').open('w') as output:
        json.dump(report,output,indent=2)
    print(json.dumps({'status':'prepared','models':report['models']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
