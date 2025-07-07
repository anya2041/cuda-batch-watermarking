@echo off
echo ≡🔥 Compiling CUDA batch watermarking project...
nvcc -std=c++17 -O2 -o watermark.exe src\main.cu

IF ERRORLEVEL 1 (
    echo  Compilation failed.
    pause
    exit /b
)

echo  Running project on all .pgm files in input\ ...
watermark.exe

IF ERRORLEVEL 1 (
    echo  Runtime error during execution.
    pause
    exit /b
)

echo Done. Output images saved to output\ folder.

REM Optional: Open the first few outputs in default image viewer
start output\img_000_wm.pgm
start output\img_001_wm.pgm
start output\img_002_wm.pgm

pause
