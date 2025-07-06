# Makefile for CUDA Batch Watermarking
PROJECT_NAME = watermark
SRC_DIR = src
INCLUDE_DIR = include

NVCC = nvcc
CFLAGS = -I$(INCLUDE_DIR) -allow-unsupported-compiler

SRC = $(SRC_DIR)/main.cu
OUT = $(PROJECT_NAME).exe

all: build

build:
	$(NVCC) $(CFLAGS) -o $(OUT) $(SRC)

run: build
	./$(OUT)

clean:
	rm -f $(OUT)


#HOW TO USE THIS
# make           # compiles
# make run       # compiles + runs
# make clean     # deletes the executable
