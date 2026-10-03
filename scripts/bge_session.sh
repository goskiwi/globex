#!/bin/sh
# 开发/演示期间使用远端模型；复用已有进程身份校验与本机隧道，不建立守护进程。
set -eu
bge_project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
case "${1:-status}" in
  status) exec sh "$bge_project_dir/scripts/bge_remote.sh" status ;;
  stop)
    sh "$bge_project_dir/scripts/bge_remote.sh" stop
    exec sh "$bge_project_dir/scripts/bge_tunnel.sh" stop
    ;;
  start) ;;
  *) echo '用法：bge_session.sh start|status|stop' >&2; exit 2 ;;
esac

sh "$bge_project_dir/scripts/bge_remote.sh" start
# 启动未完成时收回本项目服务和隧道；不遗留一个不可用的常驻模型进程。
cleanup_start() {
  sh "$bge_project_dir/scripts/bge_remote.sh" stop || true
  sh "$bge_project_dir/scripts/bge_tunnel.sh" stop || true
}
trap cleanup_start EXIT
trap 'exit 1' HUP INT TERM
sh "$bge_project_dir/scripts/bge_tunnel.sh" start
bge_attempt=0
while [ "$bge_attempt" -lt 120 ]; do
  if curl --fail --silent --show-error --max-time 3 http://127.0.0.1:18780/health; then
    trap - EXIT HUP INT TERM
    printf '\n检索模型已加载，本次使用完请执行 make bge-stop 或 make docker-stop。\n'
    exit 0
  fi
  bge_attempt=$((bge_attempt + 1))
  sleep 1
done
echo '检索模型加载超时，已请求关闭本项目服务和隧道。' >&2
exit 1
