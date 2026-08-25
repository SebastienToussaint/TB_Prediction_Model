ifeq ($(OS),Windows_NT)
    VENV := .venv
    BIN  := $(VENV)/Scripts
    PY   := python
else
    VENV := .venv
    BIN  := $(VENV)/bin
    PY   := python3
endif

CHECK_VENV = test -d $(VENV) || $(PY) -m venv $(VENV)
CLEAN = rm -rf $(VENV) __pycache__ .pytest_cache

.PHONY: venv install image_collection test clean

help: ## (Show help message)
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

venv: ## (Create virtual environment if it doesn't exist)
	@$(CHECK_VENV)

install: venv ## (Installs dependencies and initiate virtual environment)
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -r requirements-pytorch.txt

fresh_install: venv
	$(BIN)/pip uninstall -y -r requirements.txt
	$(BIN)/pip uninstall -y -r requirements-pytorch.txt
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -r requirements-pytorch.txt


clean: ## (Clean project. Removes all dependencies and virtual environment)
	$(CLEAN)

clean_win:
	$(BIN)/pip uninstall -y -r requirements.txt
	$(BIN)/pip uninstall -y -r requirements-pytorch.txt

## TB model scripts:

## Data Processing scripts
dicom_pid_extraction: venv ## (extract PIDs from all .dcm files in target directory and write to file)
	@$(BIN)/python data_processing/dicom_pid_extraction.py $(ARGS)

file_to_dcm: venv ## (Recursively search through a directory and rename all files with no extension found within a DICOMDIR tagged folder to *.dcm)
	@$(BIN)/python data_processing/file_to_dcm.py $(ARGS)

file_collection: venv ## (Recursively search through given directory and copy all files (or only files with given suffix) to an output folder)
	@$(BIN)/python data_processing/file_collection.py $(ARGS)

standardize_images: venv ## (Resize all images in target directory to given dimensions)
	@$(BIN)/python data_processing/standardize_images.py $(ARGS)

standardize_filenames: venv ## (Standardize names of all files in a given directory)
	@$(BIN)/python data_processing/standardize_filenames.py $(ARGS)

quality_control: venv ## (Recursively check extensions and dimensions of image files, as well as check for duplicate images)
	@$(BIN)/python data_processing/quality_control.py $(ARGS)

extract_dcm_image: venv ## (Extract images from DICOM files, convert to PNG and save with PID as filename)
	@$(BIN)/python data_processing/extract_dcm_image.py $(ARGS)

change_extension: venv ## Change all files with a given extension to another (NOTE: put . in new extension ex: .png)
	@$(BIN)/python data_processing/change_extension.py $(ARGS)

generate_custom_metadata: venv ## Generate standard metadata file with only image names (for when metadata not available)
	@$(BIN)/python data_processing/generate_custom_metadata.py $(ARGS)

metadata_to_csv: venv ## Transform Shenzhen or Montgomery dataset metadata into standard csv form
	@$(BIN)/python data_processing/metadata_to_csv.py $(ARGS)

# Data analysis scripts
create_charts: venv ## Creates the requested charts of the metadata
	@$(BIN)/python data_analysis/create_charts.py $(ARGS)

## Testing Scripts
apply_json_annotations: venv ## (Apply annotations to their applicable images. Currently only applies to Shenzhen Dataset file/json structure. Example dataset is given under test_data)
	@$(BIN)/python testing/apply_json_annotations.py $(ARGS)

tta_testing: venv ## (Create new images by applying multiple transforms to existing images. Note: Currently only applies transformation to .png images)
	@$(BIN)/python testing/tta_testing.py $(ARGS)

testy: venv ## Test
	@$(BIN)/python training/dataloader_tests.py $(ARGS)
