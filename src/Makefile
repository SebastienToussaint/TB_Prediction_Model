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

clean: ## (Clean project. Removes all dependencies and virtual environment)
	$(CLEAN)

## TB model scripts:
dicom_pid_extraction: venv ## (extract PIDs from all .dcm files in target directory and write to file)
	@$(BIN)/python data_processing/dicom_pid_extraction.py $(ARGS)

file_to_dcm: venv ## (Recursively search through a directory and rename all files with no extension found within a DICOMDIR tagged folder to *.dcm)
	@$(BIN)/python data_processing/file_to_dcm.py $(ARGS)

image_collection: venv ## (Recursively search through given directory and copy all .dcm, .img, .jpeg and .png files found to an output file)
	@$(BIN)/python data_processing/image_collection.py $(ARGS)

apply_json_annotations: venv ## (Apply annotations to their applicable images. Currently only applies to Shenzhen Dataset file/json structure. Example dataset is given under test_data)
	@$(BIN)/python testing/apply_json_annotations.py $(ARGS)

tta_testing: venv ## (Create new images by applying multiple transforms to existing images. Note: Currently only applies transformation to .png images)
	@$(BIN)/python testing/tta_testing.py $(ARGS)

standardize_images: venv ## (Resize all images in target directory to given dimensions)
	@$(BIN)/python data_processing/standardize_images.py $(ARGS)





