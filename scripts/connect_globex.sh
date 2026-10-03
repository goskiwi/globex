#!/bin/sh
set -eu

# 主机、用户、端口和私钥统一复用 ~/.ssh/config，项目内不复制连接凭据。
exec ssh globex
