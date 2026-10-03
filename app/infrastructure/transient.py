"""只依据异常类型、HTTP 状态和供应商结构化错误码判定临时故障。"""
import httpx
import openai


def is_transient_error(error: BaseException) -> bool:
    if isinstance(error, (TimeoutError, ConnectionError, httpx.TimeoutException,
                          httpx.NetworkError, openai.APIConnectionError)):
        return True
    if isinstance(error, (httpx.HTTPStatusError, openai.APIStatusError)):
        status = error.response.status_code
        return status in {408, 429} or 500 <= status < 600
    if isinstance(error, openai.APIError):
        return error.code in {"Throttling.Concurrency", "rate_limit_exceeded", "server_error"}
    return False
