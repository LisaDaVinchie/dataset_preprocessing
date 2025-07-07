BASE_DIR := $(shell pwd)
PYTHON := $(shell which python3)

DATA_DIR := $(BASE_DIR)/data
SRC_DIR := $(BASE_DIR)/src
TEST_DIR := $(BASE_DIR)/tests

RAW_DIR_NAME := raw
PROCESSED_DIR_NAME := processed
ORIGINAL_NANMASKS_DIR_NAME := original_nanmasks

COMB_NAME := combined
INFRARED_NAME := infrared
MICROWAVE_NAME := microwave

COMB_DIR := $(DATA_DIR)/$(COMB_NAME)
IR_DIR := $(DATA_DIR)/$(INFRARED_NAME)
MW_DIR := $(DATA_DIR)/$(MICROWAVE_NAME)
PROCESSED_DATA_EXT = .pt

DATASET_DIR := $(DATA_DIR)/datasets
SPECS_DIR := $(DATA_DIR)/specs

DATASET_BASENAME := dataset
DATASET_FILE_EXT := .pt

SPECS_BASENAME := dataset_specs
SPECS_FILE_EXT := .json

IDX := $(shell find "$(DATASET_DIR)" -type f -name "$(DATASET_BASENAME)_*$(DATASET_FILE_EXT)" | \
    sed 's|.*_\([0-9]*\)\$(DATASET_FILE_EXT)|\1|' | \
    sort -n | tail -1)
NEXT_IDX = $(shell echo $$(($(IDX) + 1)))

CURRENT_DATASET_PATH := $(DATASET_DIR)/$(DATASET_BASENAME)_$(IDX)$(DATASET_FILE_EXT)
NEXT_DATASET_PATH := $(DATASET_DIR)/$(DATASET_BASENAME)_$(NEXT_IDX)$(DATASET_FILE_EXT)

CURRENT_SPECS_PATH := $(SPECS_DIR)/$(SPECS_BASENAME)_$(IDX)$(SPECS_FILE_EXT)
NEXT_SPECS_PATH := $(SPECS_DIR)/$(SPECS_BASENAME)_$(NEXT_IDX)$(SPECS_FILE_EXT)

PATHS_FILE := $(SRC_DIR)/paths.json
PARAMS_FILE := $(SRC_DIR)/params.json

.PHONY: config download convert cut test help mask

config:
	@mkdir -p $(COMB_DIR)
	@mkdir -p $(COMB_DIR)/$(RAW_DIR_NAME)
	@mkdir -p $(COMB_DIR)/$(PROCESSED_DIR_NAME)
	@mkdir -p $(COMB_DIR)/$(ORIGINAL_NANMASKS_DIR_NAME)

	@mkdir -p $(IR_DIR)
	@mkdir -p $(IR_DIR)/$(RAW_DIR_NAME)
	@mkdir -p $(IR_DIR)/$(PROCESSED_DIR_NAME)
	@mkdir -p $(IR_DIR)/$(ORIGINAL_NANMASKS_DIR_NAME)

	@mkdir -p $(MW_DIR)
	@mkdir -p $(MW_DIR)/$(RAW_DIR_NAME)
	@mkdir -p $(MW_DIR)/$(PROCESSED_DIR_NAME)
	@mkdir -p $(MW_DIR)/$(ORIGINAL_NANMASKS_DIR_NAME)

	@mkdir -p $(DATASET_DIR)
	@mkdir -p $(SPECS_DIR)

	@echo "Storing paths to json..."
	@echo "{" > $(PATHS_FILE)
	@echo "    \"$(COMB_NAME)\": {" >> $(PATHS_FILE)
	@echo "     	\"raw_data_dir\": \"$(COMB_DIR)/$(RAW_DIR_NAME)/\"," >> $(PATHS_FILE)
	@echo "     	\"processed_data_dir\": \"$(COMB_DIR)/$(PROCESSED_DIR_NAME)/\"," >> $(PATHS_FILE)
	@echo "     	\"original_nanmasks_dir\": \"$(COMB_DIR)/$(ORIGINAL_NANMASKS_DIR_NAME)/\"" >> $(PATHS_FILE)
	@echo "    }," >> $(PATHS_FILE)
	@echo "    \"$(INFRARED_NAME)\": {" >> $(PATHS_FILE)
	@echo "     	\"raw_data_dir\": \"$(IR_DIR)/$(RAW_DIR_NAME)/\"," >> $(PATHS_FILE)
	@echo "     	\"processed_data_dir\": \"$(IR_DIR)/$(PROCESSED_DIR_NAME)/\"," >> $(PATHS_FILE)
	@echo "     	\"original_nanmasks_dir\": \"$(IR_DIR)/$(ORIGINAL_NANMASKS_DIR_NAME)/\"" >> $(PATHS_FILE)
	@echo "	}," >> $(PATHS_FILE)
	@echo "    \"$(MICROWAVE_NAME)\": {" >> $(PATHS_FILE)
	@echo "     	\"raw_data_dir\": \"$(MW_DIR)/$(RAW_DIR_NAME)/\"," >> $(PATHS_FILE)
	@echo "     	\"processed_data_dir\": \"$(MW_DIR)/$(PROCESSED_DIR_NAME)/\"," >> $(PATHS_FILE)
	@echo "     	\"original_nanmasks_dir\": \"$(MW_DIR)/$(ORIGINAL_NANMASKS_DIR_NAME)/\"" >> $(PATHS_FILE)
	@echo "	}," >> $(PATHS_FILE)
	@echo "    \"processed_data_ext\": \"$(PROCESSED_DATA_EXT)\"," >> $(PATHS_FILE)
	@echo "    \"dataset\": {" >> $(PATHS_FILE)
	@echo "     	\"current_dataset_path\": \"$(CURRENT_DATASET_PATH)\"," >> $(PATHS_FILE)
	@echo "     	\"next_dataset_path\": \"$(NEXT_DATASET_PATH)\"," >> $(PATHS_FILE)
	@echo "     	\"current_specs_path\": \"$(CURRENT_SPECS_PATH)\"," >> $(PATHS_FILE)
	@echo "     	\"next_specs_path\": \"$(NEXT_SPECS_PATH)\"" >> $(PATHS_FILE)
	@echo "    }" >> $(PATHS_FILE)
	@echo "}" >> $(PATHS_FILE)

download: config
	$(PYTHON) $(SRC_DIR)/download.py --paths $(PATHS_FILE) --params $(PARAMS_FILE)

convert: config
	$(PYTHON) $(SRC_DIR)/netcdf_to_torch.py --paths $(PATHS_FILE) --params $(PARAMS_FILE)

mask: config
	$(PYTHON) $(SRC_DIR)/generate_nan_masks.py --paths $(PATHS_FILE) --params $(PARAMS_FILE)

cut: config
	$(PYTHON) $(SRC_DIR)/generate_dataset_SST.py --paths $(PATHS_FILE) --params $(PARAMS_FILE)

test:
	@echo "Running tests in $(TEST_DIR)"
	$(PYTHON) -m unittest discover -s $(TEST_DIR) -p "*test.py"