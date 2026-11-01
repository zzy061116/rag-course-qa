# Windows 一键启动脚本（PowerShell）
# 用法： .\scripts\start.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv")) {
    Write-Host "[1/3] 创建虚拟环境..." -ForegroundColor Cyan
    python -m venv .venv
}

Write-Host "[2/3] 安装依赖..." -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Write-Host "未找到 .env，已从 .env.example 复制，请填写密钥后重新启动" -ForegroundColor Yellow
    Copy-Item .env.example .env
    exit 1
}

Write-Host "[3/3] 启动服务 http://localhost:8000" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
