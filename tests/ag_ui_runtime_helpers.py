"""测试同正式页面的 AGUIRuntime，不保留旧的直接执行器。"""
import asyncio
import json
import tempfile
from pathlib import Path
from app.infrastructure.ag_ui_journal import AGUIJournal
from app.presentation.ag_ui_runtime import AGUIRuntime


async def run_frames(orchestrator, body, intent, confirmations=None):
    with tempfile.TemporaryDirectory(prefix='globex-ui-test-') as directory:
        runtime=AGUIRuntime(AGUIJournal(Path(directory)/'journal.db'),orchestrator,confirmations)
        await runtime.startup()
        try:
            await runtime.start(body,intent)
            cursor=0
            while True:
                events,status,last=await runtime.journal.events(body.run_id,intent.buyer_id,cursor)
                for item in events:
                    cursor=item['seq']
                    yield 'data: '+json.dumps(item['event'],ensure_ascii=False)+'\n\n'
                if status!='running' and cursor>=last:break
                await asyncio.sleep(.001)
        finally:
            await runtime.shutdown()


def unused_runtime():
    raise AssertionError('此测试仅检查目录/身份接口，不应开始 AG-UI 执行')
