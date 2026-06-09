@echo off
chcp 65001 >nul
REM ============================================================
REM 农作物识别系统 - Windows 本地启动脚本
REM 用法: deploy.bat [--port 7860]
REM ============================================================

set PORT=7860

REM 解析参数
:parse_args
if "%~1"=="" goto start
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto parse_args
)
shift
goto parse_args

:start
echo ==========================================
echo   🌾 农作物识别系统 - 本地启动
echo ==========================================
echo   端口: %PORT%
echo ==========================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

REM 检查依赖
echo 📦 检查依赖...
pip show torch >nul 2>&1
if errorlevel 1 (
    echo ⚠️ 未安装 PyTorch，正在安装...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

pip show gradio >nul 2>&1
if errorlevel 1 (
    echo ⚠️ 未安装 Gradio，正在安装...
    pip install -r requirements.txt
)

REM 检查模型
echo.
echo 🔍 检查模型...
if exist "saved_models\clip\clip-vit-large-patch14-336-v2\best.pth" (
    echo   ✅ 找到 CLIP 微调模型
) else if exist "saved_models\clip\hot_reload\best.pth" (
    echo   ✅ 找到热加载模型
) else (
    echo   ⚠️ 未找到模型文件
    echo   请将模型放到 saved_models\clip\ 目录
)

REM 启动服务
echo.
echo 🚀 启动服务...
echo   访问地址: http://localhost:%PORT%
echo.
python app.py --port %PORT%

pause
