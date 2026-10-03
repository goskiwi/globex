#!/bin/sh
set -eu
action="${1:-status}"
case "$action" in
  start|status|stop) ;;
  *) echo '用法：bge_remote.sh start|status|stop' >&2; exit 2 ;;
esac
exec ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 -o StrictHostKeyChecking=yes -o ClearAllForwardings=yes \
  -o PermitLocalCommand=no -o RemoteCommand=none globex \
  "/data4/sybai/globex/service-runtime/bin/python -B /data4/sybai/globex/service/bge_service_control.py $action"
