@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo 请先按照 README.md 安装 Python 依赖。
    pause
    exit /b 1
)
echo 浏览器访问 http://127.0.0.1:8765
".venv\Scripts\python.exe" py_main.py workbench serve
pause
