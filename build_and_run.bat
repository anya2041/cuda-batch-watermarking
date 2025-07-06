@echo off
echo ≡⚙️ Compiling CUDA project...

nvcc src\main.cu -o watermark.out -Iinclude -Xcompiler "/EHsc" -allow-unsupported-compiler

if %errorlevel% neq 0 (
    echo ❌ Compilation failed.
    pause
    exit /b %errorlevel%
)

echo ▶ Running project...
watermark.out

echo ✅ Done.
pause
