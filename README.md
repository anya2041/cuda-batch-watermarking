"""
# CUDA Batch Watermarking

This project demonstrates a GPU-accelerated watermarking process applied to multiple grayscale images using CUDA.

## How to Run

### Requirements
- CUDA 12.4+
- NVIDIA GPU
- Visual Studio 2019/2022 (for Windows)

### Build & Run (Windows)
1. Open Developer Command Prompt for Visual Studio
2. Navigate to project directory
```
cd C:\Users\YourName\Projects\cuda-batch-watermarking
```
3. Run:
```
build_and_run.bat
```

### Build & Run (Linux/WSL)
```
make
./watermark.out
```

## Arguments
- Watermark image: `watermark.pgm`
- Input images: in `input/`
- Output: saved in `output/`
- Logs: saved in `logs/`

## Sample Output
See `output/` and `logs/` folders after run.

## Author
Anya H.
"""


# 2. Makefile

"""
# CUDA Makefile

TARGET = watermark.out
NVCC = nvcc
SRC = src/main.cu
INCLUDES = -Iinclude

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) -o $(TARGET) $(INCLUDES)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET) output/*.pgm logs/*
"""


# 3. build_and_run.bat (Windows)

"""
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
"""


# 4. src/main.cu (CUDA C++ logic for watermarking)
# You already have this file — if not, I will generate it separately.


# 5. include/
# Any custom headers go here. Example: include/watermarking.h (optional)


# 6. input/
# Place 5-10 grayscale images (e.g. `.pgm`) here for testing


# 7. output/
# Output watermarked images will appear here


# 8. logs/
# Logs printed during watermarking (optional, e.g. timestamps)


# 9. watermark.pgm
# A small grayscale image you want to apply as watermark on top of all images


