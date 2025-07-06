TARGET = watermark.out
SRC = src/main.cu

$(TARGET): $(SRC)
	nvcc -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
