BASE_DIR := $(shell pwd)
PYTHON := $(shell which python3)

DATA_DIR := $(BASE_DIR)/data
SRC_DIR := $(BASE_DIR)/src
TEST_DIR := $(BASE_DIR)/tests

RAW_DATA_DIR := $(DATA_DIR)/raw
PROCESSED_DATA_DIR := $(DATA_DIR)/processed

BIOCHEMISTRY_DIR_NAME := biochemistry
TEMPERATURE_DIR_NAME := temperature
ENSEMBLEPHYSICS_DIR_NAME := ensemble_physics

PROCESSED_DATA_EXT = .pt

PATHS_FILE := $(SRC_DIR)/paths.json
PARAMS_FILE := $(SRC_DIR)/params.json

.PHONY: config download convert cut test help

config:
	@echo "Storing paths to json..."
	@echo "{" > $(PATHS_FILE)
	@echo "    \"$(BIOCHEMISTRY_DIR_NAME)\": {" >> $(PATHS_FILE)
	@echo "     	\"raw_data_dir\": \"$(RAW_DATA_DIR)/$(BIOCHEMISTRY_DIR_NAME)/\"," >> $(PATHS_FILE)
	@echo "     	\"processed_data_dir\": \"$(PROCESSED_DATA_DIR)/$(BIOCHEMISTRY_DIR_NAME)/\"" >> $(PATHS_FILE)
	@echo "    }," >> $(PATHS_FILE)
	@echo "		\"$(TEMPERATURE_DIR_NAME)\": {" >> $(PATHS_FILE)
	@echo "     	\"raw_data_dir\": \"$(RAW_DATA_DIR)/$(TEMPERATURE_DIR_NAME)/\"," >> $(PATHS_FILE)
	@echo "     	\"processed_data_dir\": \"$(PROCESSED_DATA_DIR)/$(TEMPERATURE_DIR_NAME)/\"" >> $(PATHS_FILE)
	@echo "    }," >> $(PATHS_FILE)
	@echo "    \"$(ENSEMBLEPHYSICS_DIR_NAME)\": {" >> $(PATHS_FILE)
	@echo "     	\"raw_data_dir\": \"$(RAW_DATA_DIR)/$(ENSEMBLEPHYSICS_DIR_NAME)/\"," >> $(PATHS_FILE)
	@echo "     	\"processed_data_dir\": \"$(PROCESSED_DATA_DIR)/$(ENSEMBLEPHYSICS_DIR_NAME)/\"" >> $(PATHS_FILE)
	@echo "    }," >> $(PATHS_FILE)
	@echo "		\"processed_data_ext\": \"$(PROCESSED_DATA_EXT)\"" >> $(PATHS_FILE)
	@echo "}" >> $(PATHS_FILE)

download: config
	$(PYTHON) $(SRC_DIR)/download.py --paths $(PATHS_FILE) --params $(PARAMS_FILE)

convert: config
	$(PYTHON) $(SRC_DIR)/netcdf_to_torch.py --paths $(PATHS_FILE) --params $(PARAMS_FILE)

cut:
	$(PYTHON) $(SRC_DIR)/cut.py --paths $(PATHS_FILE) --params $(PARAMS_FILE)

test:
	@echo "Running tests in $(TEST_DIR)"
	$(PYTHON) -m unittest discover -s $(TEST_DIR) -p "*test.py"