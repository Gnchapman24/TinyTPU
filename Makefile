# TinyTPU build and verification entry points.
#
# Every target here runs in WSL (see AGENTS.md §5a). Windows-side steps (Vivado, board
# programming, host driver) are listed by `make help`.
#
# The OSS CAD Suite's bin/ is put first on PATH for recipes only, the same way tools/env.sh
# does it. Its own Python is never used; Python tools run from the uv project (.venv).

SHELL         := /bin/bash
.SHELLFLAGS   := -eu -o pipefail -c
.DEFAULT_GOAL := help

include tools/versions.env

OSS_CAD_SUITE ?= $(HOME)/tools/oss-cad-suite
export PATH   := $(OSS_CAD_SUITE)/bin:$(PATH)

# Array sizes for simulation (AGENTS.md §5, tier 1)
SIM_N      ?= 2 4
SIM_FULL_N ?= 8
# Optional pytest -k filter, e.g. `make sim K=uart`
K          ?=

PKG_SRCS  := $(sort $(wildcard rtl/*_pkg.sv))
MOD_SRCS  := $(sort $(filter-out $(PKG_SRCS),$(wildcard rtl/*.sv)))
RTL_SRCS  := $(PKG_SRCS) $(MOD_SRCS)
SBY_FILES := $(sort $(wildcard formal/*.sby))
TB_FILES  := $(sort $(wildcard tb/test_*.py))

UV_RUN      := uv run --locked
PYTEST_ARGS := -q $(if $(K),-k "$(K)")

.PHONY: help check-oss check-uv lint sim sim-full _sim formal docs docs-serve all clean

help:
	@echo "TinyTPU make targets (run in WSL):"
	@echo "  make lint        Verilator -Wall and yosys-slang check of every rtl/ module"
	@echo "  make sim         cocotb + Verilator tests at N = $(SIM_N)   (K=<filter> to select tests)"
	@echo "  make sim-full    cocotb + Verilator tests at N = $(SIM_FULL_N)"
	@echo "  make formal      SymbiYosys on every formal/*.sby"
	@echo "  make docs        generate RTL docs and run mkdocs build --strict"
	@echo "  make docs-serve  live docs preview"
	@echo "  make all         lint + sim + formal + docs"
	@echo "  make clean       remove simulation, formal, and docs outputs"
	@echo ""
	@echo "Windows-side steps (PowerShell, not make):"
	@echo "  Vivado 2025.2 TCL (synthesis, reports, bitstream)  -> syn/*.tcl (planned, tasks A2.4, A5.2)"
	@echo "  Host driver                                        -> host/ via uv (see tools/setup.md)"

# --- environment checks ----------------------------------------------------------------------

check-oss:
	@test -x "$(OSS_CAD_SUITE)/bin/yosys" || { \
	  echo "OSS CAD Suite not found at $(OSS_CAD_SUITE) (see tools/setup.md)" >&2; exit 1; }
	@installed="$$(cat "$(OSS_CAD_SUITE)/VERSION" 2>/dev/null || true)"; \
	if [ "$$installed" != "$(OSS_CAD_SUITE_RELEASE)" ]; then \
	  echo "OSS CAD Suite release '$$installed' does not match the pin '$(OSS_CAD_SUITE_RELEASE)' in tools/versions.env" >&2; \
	  exit 1; \
	fi
	@case "$$(command -v verilator)" in \
	  "$(OSS_CAD_SUITE)"/*) ;; \
	  *) echo "verilator resolves to $$(command -v verilator), not the OSS CAD Suite" >&2; exit 1 ;; \
	esac

check-uv:
	@command -v uv >/dev/null || { echo "uv not found (see tools/setup.md)" >&2; exit 1; }

# --- tier 0: lint -----------------------------------------------------------------------------

lint: check-oss
	@if [ -z "$(MOD_SRCS)" ]; then echo "lint: no modules in rtl/ yet"; exit 0; fi; \
	for f in $(MOD_SRCS); do \
	  top="$$(basename "$$f" .sv)"; \
	  echo "lint: $$top (verilator)"; \
	  verilator --lint-only -Wall -DASSERTIONS --top-module "$$top" $(RTL_SRCS); \
	  echo "lint: $$top (yosys-slang)"; \
	  yosys -q -m slang -p "read_slang -DASSERTIONS $(RTL_SRCS) --top $$top; hierarchy -check"; \
	done

# --- tier 1: cocotb + Verilator ---------------------------------------------------------------
# Test runners read the array size from TINYTPU_N and build into sim_build/<name>_N<n>.

sim:
	@$(MAKE) --no-print-directory _sim NS="$(SIM_N)"

sim-full:
	@$(MAKE) --no-print-directory _sim NS="$(SIM_FULL_N)"

_sim: check-oss check-uv
	@if [ -z "$(TB_FILES)" ]; then echo "sim: no cocotb tests in tb/ yet"; exit 0; fi; \
	for n in $(NS); do \
	  echo "sim: N=$$n"; \
	  TINYTPU_N=$$n $(UV_RUN) pytest tb $(PYTEST_ARGS); \
	done

# --- tier 2: SymbiYosys -----------------------------------------------------------------------
# Runs every .sby file, then fails if any of them failed. Run directories are formal/<name>*/.

formal: check-oss
	@if [ -z "$(SBY_FILES)" ]; then echo "formal: no .sby files in formal/ yet"; exit 0; fi; \
	failed=""; \
	cd formal; \
	for f in $(notdir $(SBY_FILES)); do \
	  echo "formal: $$f"; \
	  sby -f "$$f" || failed="$$failed $$f"; \
	done; \
	if [ -n "$$failed" ]; then echo "formal: FAILED:$$failed" >&2; exit 1; fi; \
	echo "formal: all passed"

# --- docs -------------------------------------------------------------------------------------

docs: check-uv
	@if [ ! -f mkdocs.yml ]; then echo "docs: mkdocs.yml not present yet (task A0.5)"; exit 0; fi; \
	if [ -f docs/gen_rtl_docs.py ]; then $(UV_RUN) python docs/gen_rtl_docs.py; fi; \
	$(UV_RUN) mkdocs build --strict

docs-serve: check-uv
	@if [ ! -f mkdocs.yml ]; then echo "docs: mkdocs.yml not present yet (task A0.5)"; exit 1; fi; \
	if [ -f docs/gen_rtl_docs.py ]; then $(UV_RUN) python docs/gen_rtl_docs.py; fi; \
	$(UV_RUN) mkdocs serve

# --- aggregate / housekeeping -----------------------------------------------------------------

all: lint sim formal docs

clean:
	rm -rf sim_build site docs/rtl results.xml
	find tb -name sim_build -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find tb -name results.xml -delete 2>/dev/null || true
	find formal -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
