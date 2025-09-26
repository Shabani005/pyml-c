# ---- Project Settings ----
NAME    := pyml
SRC     := cml.c
CFLAGS  := -O2 -fPIC
LDFLAGS := -lm
CC      := gcc

# Detect platform for Python extension file
ifeq ($(OS),Windows_NT)
    EXT := pyd
else
    EXT := so
endif

TARGET := $(NAME).$(EXT)

# ---- Rules ----

# Build Python package (requires `build` module)
build: $(TARGET)
	python -m build

# Install build & release tools
full:
	pip install build twine

# Compile the extension library
$(TARGET): $(SRC) cml.h
	$(CC) -shared -o $@ $(CFLAGS) $(SRC) $(LDFLAGS)

# Clean artifacts
.PHONY: clean
clean:
	rm -f $(TARGET) *.o
	rm -rf build dist *.egg-info
