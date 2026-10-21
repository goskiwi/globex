"""错误终态的统一投影；实时、日志恢复和历史均从同一事实生成。"""
from copy import deepcopy


def apply_run_error(state, message, code):
    cancelled = code == 'CANCELLED'
    interrupted = code == 'SERVER_RESTART'
    state['executionError'] = message
    state['errorCode'] = code
    state['status'] = 'cancelled' if cancelled else 'error'
    state['executionStatus'] = 'partial' if cancelled else 'failed'
    state['stopReason'] = state.get('stopReason') or {
        'CANCELLED':'cancelled', 'SERVER_RESTART':'server_restart',
        'PERSISTENCE_FAILED':'persistence_error',
    }.get(code, code)
    process = state.get('process')
    if isinstance(process, dict):
        process['status'] = 'cancelled' if cancelled else 'interrupted' if interrupted else 'failed'
        process['error'] = {'code':code, 'message':message}
        for step in process.get('steps', []):
            if step['status'] == 'running':
                step.update(status='cancelled' if cancelled else 'unconfirmed', summary='运行已结束，未取得完整执行结果')


def error_events(state, message, code):
    snapshot = deepcopy(state)
    apply_run_error(snapshot, message, code)
    return [{'type':'STATE_SNAPSHOT', 'snapshot':snapshot},
            {'type':'RUN_ERROR', 'message':message, 'code':code}]
