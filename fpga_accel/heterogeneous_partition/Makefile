PYTHON ?= python3

.PHONY: test analyze

test:
	$(PYTHON) -m unittest discover -s tests -v

analyze:
	@test -n "$(OUTPUT_DIR)" || { echo "set OUTPUT_DIR=/new/artifact/directory" >&2; exit 2; }
	$(PYTHON) analyze_partition.py --output-dir "$(OUTPUT_DIR)"
