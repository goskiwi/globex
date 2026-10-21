"""持久 AG-UI 运行压测：同一响应消费公开 SSE 至真实终态。"""
import os
import uuid
from locust import HttpUser, between, task
from scripts.run_stream import run_input, read_stream

class CommerceRunUser(HttpUser):
    wait_time=between(1,3)

    @task
    def submit_and_stream(self):
        body=run_input('locust-'+uuid.uuid4().hex,os.environ['GLOBEX_BUYER_ID'],
                       '找几个适合长途飞行的颈枕，要小众设计')
        with self.client.post('/commerce/ag-ui/run',json=body,
            headers={'Authorization':'Bearer '+os.environ['GLOBEX_API_TOKEN']},
            timeout=120,name='POST /commerce/ag-ui/run',catch_response=True) as response:
            if response.status_code!=200:
                response.failure(f'运行请求失败：{response.status_code}');return
            try:read_stream(response.text).require_success()
            except ValueError as error:response.failure(str(error))
