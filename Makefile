# Makefile for Python project using Ruff

# Python interpreter to use
PYTHON := python3

# Ruff command
RUFF := ruff

# Source directory
SRC_DIR := src

# Test directory
TEST_DIR := tests

# Examples directory
EXAMPLES_DIR := examples

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make lint      : Run Ruff linter"
	@echo "  make format    : Run Ruff formatter"
	@echo "  make check     : Run Ruff linter and formatter in check mode"
	@echo "  make all       : Run all Ruff checks and formatting"

.PHONY: lint
lint:
	$(RUFF) check $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)

.PHONY: format
format:
	$(RUFF) format $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)

.PHONY: check
check:
	$(RUFF) check $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	$(RUFF) format --check $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)

.PHONY: all
all: lint format