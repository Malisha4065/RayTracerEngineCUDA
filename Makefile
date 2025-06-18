# Compiler
NVCC = nvcc

# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

# Target executable
TARGET = $(BINDIR)/raytracer

# Source files
CUDA_SOURCES = $(wildcard $(SRCDIR)/*.cu)
OBJECTS = $(CUDA_SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)

# Compiler flags
NVCC_FLAGS = -std=c++11 -O3 -arch=sm_50
INCLUDES = -I$(INCDIR)
LIBS = -lSDL2 -lcurand -lm

# Create directories if they don't exist
$(shell mkdir -p $(OBJDIR) $(BINDIR))

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $@ $(LIBS)

# Compile CUDA source files to object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Rebuild everything
rebuild: clean all

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install libsdl2-dev

.PHONY: all clean rebuild install-deps