PYTHON ?= python3

.PHONY: test inspect evaluate

test:
	$(PYTHON) -m unittest discover -s tests -v

inspect:
	@test -n "$(CAPTURE)" || { echo "set CAPTURE=/path/to/capture.bin" >&2; exit 2; }
	$(PYTHON) parse_capture.py "$(CAPTURE)"

evaluate:
	@test -n "$(MODEL)" || { echo "set MODEL=/path/to/model.gguf" >&2; exit 2; }
	@test -n "$(CAPTURE)" || { echo "set CAPTURE=/path/to/capture.bin" >&2; exit 2; }
	@test -n "$(OUTPUT_DIR)" || { echo "set OUTPUT_DIR=/new/artifact/directory" >&2; exit 2; }
	$(PYTHON) evaluate_runtime_ffn.py --model "$(MODEL)" --capture "$(CAPTURE)" --output-dir "$(OUTPUT_DIR)"
