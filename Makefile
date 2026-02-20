# Makefile for Pareidolia - Spectral Voice Hallucination Plugin for disting NT
#
# Dual build targets:
#   make hardware - Build ARM .o for disting NT hardware
#   make test     - Build native .dylib/.so for desktop testing in VCV Rack nt_emu
#   make clean    - Clean build artifacts
#   make size     - Show .text section size (must be under 64KB)

# Project configuration
PROJECT = pareidolia
NT_API_PATH ?= distingNT_API

# Version from git
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "v1.0.0-dev")

# Source files
SOURCES = pareidolia.cpp

# Include paths
INCLUDES = -I. -I$(NT_API_PATH)/include

# Common defines
DEFINES_COMMON = -DPAREIDOLIA_VERSION=\"$(VERSION)\"
DEFINES_HARDWARE = $(DEFINES_COMMON)
DEFINES_TEST = $(DEFINES_COMMON) -DNT_EMU_DEBUG

# Common compiler flags
CXXFLAGS_COMMON = -std=c++11 -Wall -Wextra -Wno-unused-parameter -fno-rtti -fno-exceptions

# Hardware build (ARM Cortex-M7)
CXX_ARM = arm-none-eabi-g++
CXXFLAGS_ARM = $(CXXFLAGS_COMMON) $(DEFINES_HARDWARE) \
	-mcpu=cortex-m7 \
	-mfpu=fpv5-d16 \
	-mfloat-abi=hard \
	-mthumb \
	-Os \
	-fPIC

# Desktop test build
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CXX_TEST = clang++
    DYLIB_EXT = dylib
else
    CXX_TEST = g++
    DYLIB_EXT = so
endif

CXXFLAGS_TEST = $(CXXFLAGS_COMMON) $(DEFINES_TEST) -O2 -fPIC

# Output directories
PLUGINS_DIR = plugins
BUILD_DIR = build

# Targets
.PHONY: all hardware test clean size

all: hardware test

# Hardware target - ARM .o for disting NT
hardware: $(PLUGINS_DIR)/$(PROJECT).o

$(PLUGINS_DIR)/$(PROJECT).o: $(SOURCES) | $(PLUGINS_DIR)
	$(CXX_ARM) $(CXXFLAGS_ARM) $(INCLUDES) -c $(SOURCES) -o $@
	@echo "Hardware build complete: $@"
	@ls -lh $@

# Test target - native .dylib/.so for VCV Rack nt_emu
test: $(PLUGINS_DIR)/$(PROJECT).$(DYLIB_EXT)

$(PLUGINS_DIR)/$(PROJECT).$(DYLIB_EXT): $(SOURCES) | $(PLUGINS_DIR)
ifeq ($(UNAME_S),Darwin)
	$(CXX_TEST) $(CXXFLAGS_TEST) $(INCLUDES) -dynamiclib -undefined dynamic_lookup $(SOURCES) -o $@
else
	$(CXX_TEST) $(CXXFLAGS_TEST) $(INCLUDES) -shared $(SOURCES) -o $@
endif
	@echo "Desktop test build complete: $@"
	@ls -lh $@

# Create output directory
$(PLUGINS_DIR):
	mkdir -p $(PLUGINS_DIR)

# Clean build artifacts
clean:
	rm -rf $(PLUGINS_DIR) $(BUILD_DIR)
	@echo "Build artifacts cleaned"

# Show .text section size (hardware build must exist)
size: hardware
	@echo "Section sizes for $(PLUGINS_DIR)/$(PROJECT).o:"
	@arm-none-eabi-size $(PLUGINS_DIR)/$(PROJECT).o
	@echo ""
	@echo ".text size breakdown:"
	@arm-none-eabi-objdump -h $(PLUGINS_DIR)/$(PROJECT).o | grep -E '\.text|\.rodata'
	@echo ""
	@echo "Target: .text + .rodata < 65536 bytes (64KB)"
