BASE_DIR := $(shell pwd)
PYTHON := $(shell which python3)

DATA_DIR := $(BASE_DIR)/data
SRC_DIR := $(BASE_DIR)/src
TEST_DIR := $(BASE_DIR)/tests

RAW_DATA_DIR := $(DATA_DIR)/raw
PROCESSED_DATA_DIR := $(DATA_DIR)/processed
ORIGINAL_NANMASKS_DIR := $(DATA_DIR)/original_nanmasks

BIOCHEMISTRY_DIR_NAME := biochemistry
TEMPERATURE_DIR_NAME := temperature
ENSEMBLEPHYSICS_DIR_NAME := ensemble_physics
PROCESSED_DATA_EXT = .pt

DATASET_DIR := $(DATA_DIR)/datasets
SPECS_DIR := $(DATA_DIR)/specs
NANMASKS_DIR := $(DATA_DIR)/nanmasks
MINMAX_DIR := $(DATA_DIR)/minmax

DATASET_BASENAME := dataset
DATASET_FILE_EXT := .pt

SPECS_BASENAME := dataset_specs
SPECS_FILE_EXT := .json

NANMASKS_BASENAME := nans_mask
NANMASKS_FILE_EXT := .pt

MINMAX_BASENAME := minmax
MINMAX_FILE_EXT := .pt

IDX := $(shell find "$(DATASET_DIR)" -type f -name "$(DATASET_BASENAME)_*$(DATASET_FILE_EXT)" | \
    sed 's|.*_\([0-9]*\)\$(DATASET_FILE_EXT)|\1|' | \
    sort -n | tail -1)
NEXT_IDX = $(shell echo $$(($(IDX) + 1)))

CURRENT_DATASET_PATH := $(DATASET_DIR)/$(DATASET_BASENAME)_$(IDX)$(DATASET_FILE_EXT)
NEXT_DATASET_PATH := $(DATASET_DIR)/$(DATASET_BASENAME)_$(NEXT_IDX)$(DATASET_FILE_EXT)

CURRENT_SPECS_PATH := $(SPECS_DIR)/$(SPECS_BASENAME)_$(IDX)$(SPECS_FILE_EXT)
NEXT_SPECS_PATH := $(SPECS_DIR)/$(SPECS_BASENAME)_$(NEXT_IDX)$(SPECS_FILE_EXT)

CURRENT_NANMASKS_PATH := $(NANMASKS_DIR)/$(NANMASKS_BASENAME)_$(IDX)$(NANMASKS_FILE_EXT)
NEXT_NANMASKS_PATH := $(NANMASKS_DIR)/$(NANMASKS_BASENAME)_$(NEXT_IDX)$(NANMASKS_FILE_EXT)

CURRENT_MINMAX_PATH := $(MINMAX_DIR)/$(MINMAX_BASENAME)_$(IDX)$(MINMAX_FILE_EXT)
NEXT_MINMAX_PATH := $(MINMAX_DIR)/$(MINMAX_BASENAME)_$(NEXT_IDX)$(MINMAX_FILE_EXT)

PATHS_FILE := $(SRC_DIR)/paths.json
PARAMS_FILE := $(SRC_DIR)/params.json

.PHONY: config download convert cut test help

config:
	@echo "Storing paths to json..."
	@echo "{" > $(PATHS_FILE)
	@echo "    \"$(BIOCHEMISTRY_DIR_NAME)\": {" >> $(PATHS_FILE)
	@echo "     	\"raw_data_dir\": \"$(RAW_DATA_DIR)/$(BIOCHEMISTRY_DIR_NAME)/\"," >> $(PATHS_FILE)
	@echo "     	\"processed_data_dir\": \"$(PROCESSED_DATA_DIR)/$(BIOCHEMISTRY_DIR_NAME)/\"," >> $(PATHS_FILE)
	@echo "     	\"nan_masks_dir\": \"$(ORIGINAL_NANMASKS_DIR)/$(BIOCHEMISTRY_DIR_NAME)/\"" >> $(PATHS_FILE)
	@echo "    }," >> $(PATHS_FILE)
	@echo "    \"$(TEMPERATURE_DIR_NAME)\": {" >> $(PATHS_FILE)
	@echo "     	\"raw_data_dir\": \"$(RAW_DATA_DIR)/$(TEMPERATURE_DIR_NAME)/\"," >> $(PATHS_FILE)
	@echo "     	\"processed_data_dir\": \"$(PROCESSED_DATA_DIR)/$(TEMPERATURE_DIR_NAME)/\"," >> $(PATHS_FILE)
	@echo "     	\"nan_masks_dir\": \"$(ORIGINAL_NANMASKS_DIR)/$(TEMPERATURE_DIR_NAME)/\"" >> $(PATHS_FILE)
	@echo "    }," >> $(PATHS_FILE)
	@echo "    \"$(ENSEMBLEPHYSICS_DIR_NAME)\": {" >> $(PATHS_FILE)
	@echo "     	\"raw_data_dir\": \"$(RAW_DATA_DIR)/$(ENSEMBLEPHYSICS_DIR_NAME)/\"," >> $(PATHS_FILE)
	@echo "     	\"processed_data_dir\": \"$(PROCESSED_DATA_DIR)/$(ENSEMBLEPHYSICS_DIR_NAME)/\"," >> $(PATHS_FILE)
	@echo "     	\"nan_masks_dir\": \"$(ORIGINAL_NANMASKS_DIR)/$(ENSEMBLEPHYSICS_DIR_NAME)/\"" >> $(PATHS_FILE)
	@echo "    }," >> $(PATHS_FILE)
	@echo "    \"processed_data_ext\": \"$(PROCESSED_DATA_EXT)\"," >> $(PATHS_FILE)
	@echo "    \"dataset\": {" >> $(PATHS_FILE)
	@echo "     	\"current_dataset_path\": \"$(CURRENT_DATASET_PATH)\"," >> $(PATHS_FILE)
	@echo "     	\"next_dataset_path\": \"$(NEXT_DATASET_PATH)\"," >> $(PATHS_FILE)
	@echo "     	\"current_specs_path\": \"$(CURRENT_SPECS_PATH)\"," >> $(PATHS_FILE)
	@echo "     	\"next_specs_path\": \"$(NEXT_SPECS_PATH)\"," >> $(PATHS_FILE)
	@echo "     	\"current_nanmasks_path\": \"$(CURRENT_NANMASKS_PATH)\"," >> $(PATHS_FILE)
	@echo "     	\"next_nanmasks_path\": \"$(NEXT_NANMASKS_PATH)\"," >> $(PATHS_FILE)
	@echo "     	\"current_minmax_path\": \"$(CURRENT_MINMAX_PATH)\"," >> $(PATHS_FILE)
	@echo "     	\"next_minmax_path\": \"$(NEXT_MINMAX_PATH)\"" >> $(PATHS_FILE)
	@echo "    }" >> $(PATHS_FILE)
	@echo "}" >> $(PATHS_FILE)

download: config
	$(PYTHON) $(SRC_DIR)/download.py --paths $(PATHS_FILE) --params $(PARAMS_FILE)

convert: config
	$(PYTHON) $(SRC_DIR)/netcdf_to_torch.py --paths $(PATHS_FILE) --params $(PARAMS_FILE)

cut: config
	$(PYTHON) $(SRC_DIR)/generate_dataset.py --paths $(PATHS_FILE) --params $(PARAMS_FILE)

test:
	@echo "Running tests in $(TEST_DIR)"
	$(PYTHON) -m unittest discover -s $(TEST_DIR) -p "*test.py"