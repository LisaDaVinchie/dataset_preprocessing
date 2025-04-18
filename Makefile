BASE_DIR := $(shell pwd)
PYTHON := $(shell which python3)

DATA_DIR := $(BASE_DIR)/data
SRC_DIR := $(BASE_DIR)/src
TEST_DIR := $(BASE_DIR)/tests

RAW_DATA_DIR := $(DATA_DIR)/raw
PROCESSED_DATA_DIR := $(DATA_DIR)/processed



.PHONY: config convert test help

test:
	@echo "Running tests in $(TEST_DIR)"
	$(PYTHON) -m unittest discover -s $(TEST_DIR) -p "*test.py"

