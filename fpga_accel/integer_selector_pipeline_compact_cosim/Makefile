# Dedicated bounded RTL co-simulation work area for the accepted compact
# integer Selector pipeline (`integer_selector_pipeline_compact`).
#
# The accepted kernel trees and fixture payloads are referenced read-only.

REPO_ROOT    := $(abspath $(CURDIR)/../..)
FIXTURE_DIR  := $(REPO_ROOT)/artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures
CASE_LIST    := $(CURDIR)/cases/rtl_cosim_cases.txt
ARTIFACT_DIR := $(REPO_ROOT)/artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim

.PHONY: help validate host cosim audit report all clean-build

help:
	@echo "make validate  - resolve the five cases against the accepted manifest"
	@echo "make host      - focused host g++ / ASan test of the dedicated testbench"
	@echo "make cosim     - csynth_design + cosim_design (verilog / xsim / trace none)"
	@echo "make audit     - protected-file SHA before/after verification"
	@echo "make report    - archive small summaries + build JSON/MD reports"
	@echo "make all       - validate + host + cosim + audit + report"

validate:
	python3 tools/validate_cases.py --fixtures "$(FIXTURE_DIR)" \
	  --cases "$(CASE_LIST)" \
	  --out "$(ARTIFACT_DIR)/cases/rtl_cosim_cases_resolved.json" \
	  > "$(ARTIFACT_DIR)/logs/00_case_resolution.log"

host:
	bash scripts/run_host_test.sh

cosim:
	bash scripts/run_cosim.sh

audit:
	python3 tools/protected_sha_audit.py snapshot \
	  --out "$(ARTIFACT_DIR)/protected_sha_after.json"
	python3 tools/protected_sha_audit.py verify \
	  --before "$(ARTIFACT_DIR)/protected_sha_before.json" \
	  --after "$(ARTIFACT_DIR)/protected_sha_after.json" \
	  --out "$(ARTIFACT_DIR)/protected_sha_verify.json"

report:
	python3 tools/build_report.py

all: validate host cosim audit report

clean-build:
	rm -rf /tmp/robobrain_integer_selector_compact_cosim \
	       /tmp/robobrain_integer_selector_compact_cosim_host
