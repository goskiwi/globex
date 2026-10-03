"""管理本项目唯一服务进程；仅匹配 PID、启动时间、用户、命令和工作目录后停止。"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

ROOT=Path('/data4/sybai/globex')
PYTHON=str(ROOT/'service-runtime/bin/python')
SCRIPT=ROOT/'service/bge_retrieval_service.py'
STATE=ROOT/'service/process.json'
PROC=Path('/proc')


def process_identity(pid):
    folder=PROC/str(pid)
    if not folder.exists():return None
    try:
        stat=(folder/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0] in {'Z','X'}:return None
        command=(folder/'cmdline').read_bytes().split(b'\0')
        if not command[0]:return None  # 退出中的进程可先清空命令，再成为 zombie。
        uid_line=next(line for line in (folder/'status').read_text().splitlines() if line.startswith('Uid:'))
        uid=int(uid_line.split()[1])
        cwd=(folder/'cwd').resolve(strict=True)
    except (FileNotFoundError,ProcessLookupError):return None
    if uid!=os.getuid() or command[:3]!=[PYTHON.encode(),b'-B',str(SCRIPT).encode()] or cwd!=ROOT:
        raise RuntimeError('PID 不属于此项目服务，拒绝操作')
    return stat[19]


def current():
    if not STATE.exists():return None
    state=json.loads(STATE.read_text())
    identity=process_identity(state['pid'])
    if identity is None:return None
    if identity!=state['start_ticks']:raise RuntimeError('PID 已被复用，拒绝操作')
    return state


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','status','stop']);args=parser.parse_args()
    if ROOT.resolve()!=ROOT or ROOT.stat().st_uid!=os.getuid():raise RuntimeError('项目目录不匹配')
    state=current()
    if args.action=='status':
        print(json.dumps({'running':bool(state),'process':state}));return
    if args.action=='stop':
        if state:
            os.kill(state['pid'],signal.SIGTERM)
            for _ in range(100):
                if process_identity(state['pid']) is None:break
                time.sleep(.2)
            else:raise RuntimeError('服务未停止，未强制结束任何进程')
        print(json.dumps({'running':False}));return
    if state:
        print(json.dumps({'already_running':True,'process':state}));return
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        probe.bind(('127.0.0.1',18780))
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','GLOBEX_RETRIEVAL_ROOT':str(ROOT),
        'HF_HOME':str(ROOT/'cache/huggingface'),'HF_XET_CACHE':str(ROOT/'cache/xet'),
        'XDG_CACHE_HOME':str(ROOT/'cache'),'TMPDIR':str(ROOT/'cache/tmp'),
        'TORCH_HOME':str(ROOT/'cache/torch'),'CUDA_CACHE_PATH':str(ROOT/'cache/cuda'),
        'TRITON_CACHE_DIR':str(ROOT/'cache/triton'),
        'CUDA_VISIBLE_DEVICES':'0','CUDA_DEVICE_ORDER':'PCI_BUS_ID',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1',
        'HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','TOKENIZERS_PARALLELISM':'false','BGE_DEVICE':'cuda:0','BGE_PORT':'18780'}
    with (ROOT/'logs/service.log').open('ab') as log:
        process=subprocess.Popen([PYTHON,'-B',str(SCRIPT)],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    identity=None
    for _ in range(20):
        if process.poll() is not None:raise RuntimeError('服务启动失败，请查看本项目 service.log')
        try:identity=process_identity(process.pid)
        except (RuntimeError,FileNotFoundError,PermissionError):pass
        if identity:break
        time.sleep(.1)
    if not identity:raise RuntimeError('启动后身份尚未核实，未登记或停止任何未知进程')
    state={'pid':process.pid,'start_ticks':identity,'port':18780,'started_at':time.time()}
    STATE.write_text(json.dumps(state,indent=2))
    print(json.dumps({'started':True,'process':state}))


if __name__=='__main__':main()
