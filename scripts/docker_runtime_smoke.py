"""在无外网的临时运行镜像内启动实际服务，仅核验进程/健康路由/本地数据库。"""
import json
import subprocess
import time
import httpx


def main():
    process=subprocess.Popen(['uvicorn','app.presentation.server:app','--host','127.0.0.1','--port','8000'])
    try:
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('服务提前退出')
            try:
                response=httpx.get('http://127.0.0.1:8000/health',timeout=2)
                response.raise_for_status()
                payload=response.json()
                assert payload['status']=='ok',payload
                assert payload['database']=='sqlite' and payload['trade_database']=='sqlite',payload
                print(json.dumps({'verified':'runtime_http_and_sqlite','health':payload,
                    'external_model_verified':False,'external_retrieval_verified':False},ensure_ascii=False),flush=True)
                return
            except httpx.TransportError:
                time.sleep(.25)
        raise TimeoutError('服务未在90秒内启动')
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill();process.wait()


if __name__=='__main__':main()
