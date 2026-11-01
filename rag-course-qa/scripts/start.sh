#!/usr/bin/env bash
# Linux / macOS / 云服务器一键启动脚本
set -e
cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  echo "[1/3] 创建虚拟环境..."
  python3 -m venv .venv
fi

echo "[2/3] 安装依赖..."
./.venv/bin/pip install -q -r requirements.txt

if [ ! -f ".env" ]; then
  echo "未找到 .env，已从 .env.example 复制，请填写密钥后重新启动"
  cp .env.example .env
  exit 1
fi

echo "[3/3] 启动服务 http://0.0.0.0:8000"
./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
