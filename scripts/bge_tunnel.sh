#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
control_socket="$project_dir/.bge-tunnel.sock"
case "${1:-start}" in
  status) exec ssh -S "$control_socket" -O check globex ;;
  stop)
    if [ ! -e "$control_socket" ]; then
      echo '本项目 SSH 隧道未运行'; exit 0
    fi
    exec ssh -S "$control_socket" -O exit globex
    ;;
  start) ;;
  *) echo '用法：bge_tunnel.sh start|status|stop' >&2; exit 2 ;;
esac
if ssh -S "$control_socket" -O check globex >/dev/null 2>&1; then
  echo '本项目 SSH 隧道已运行'; exit 0
fi
if [ -e "$control_socket" ]; then
  echo '控制 socket 已存在但不可用，未覆盖；请先核对该文件。' >&2; exit 1
fi
# 不继承别名中另外配置的转发；发现时要求显式核对，不擅自打开其它端口。
if ssh -G globex 2>/dev/null | awk '/^(localforward|remoteforward|dynamicforward) / {found=1} END {exit found ? 0 : 1}'; then
  echo 'SSH 别名已有额外端口转发，请先核对配置。' >&2; exit 1
fi
exec ssh -fNT -M -S "$control_socket" -o ControlPersist=no -o BatchMode=yes \
  -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=yes -o ExitOnForwardFailure=yes -o ClearAllForwardings=no \
  -o PermitLocalCommand=no -o RemoteCommand=none \
  -L 127.0.0.1:18780:127.0.0.1:18780 globex
