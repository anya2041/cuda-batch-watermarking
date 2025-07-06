@echo off
echo ≡🔥 Compiling CUDA project...
nvcc -o watermark.out src\main.cu

IF ERRORLEVEL 1 (
    echo ⚠️ Compilation failed.
    pause
    exit /b
)

echo 🚀 Running project...
watermark.out

echo ✅ Done.
start output\watermarked.pgm
pause
