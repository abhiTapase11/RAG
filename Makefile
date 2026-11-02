# doc-rag - make targets
#
#   make setup           create .venv and install dependencies
#   make ingest-samples  index the two bundled sample documents
#   make app             run the Streamlit UI
#   make eval            run the retrieval / answer evaluation
#   make test            run the test suite
#   make demo            mock-mode smoke run of the whole pipeline
#   make reset           drop the index (keeps the source files)
#   make clean           remove caches and the index
#   make distclean       also remove the virtualenv
#
# No GNU make on Windows? ./make.ps1 <target> runs the same commands.

PY ?= python
VENV := .venv

ifeq ($(OS),Windows_NT)
VPY := $(VENV)/Scripts/python.exe
else
VPY := $(VENV)/bin/python
endif

.PHONY: help setup ingest-samples app eval test demo reset clean distclean

help:
	@echo "targets: setup ingest-samples app eval test demo reset clean distclean"

setup:
	$(PY) -m venv $(VENV)
	$(VPY) -m pip install --upgrade pip setuptools wheel
	-$(VPY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(VPY) -m pip install -r requirements.txt
	@echo ""
	@echo "Done. Copy .env.example to .env to configure (optional - MOCK_LLM=1 needs no key)."

ingest-samples:
	$(VPY) scripts/ingest_samples.py

app:
	$(VPY) -m streamlit run app.py

eval:
	$(VPY) scripts/eval.py --ingest-if-empty

test:
	$(VPY) -m pytest

demo:
	$(VPY) scripts/demo.py --mock

reset:
	$(VPY) -c "import sys; sys.path.insert(0, 'src'); from docrag.store import DocStore; s = DocStore(); s.reset(); print('index reset')"

clean:
	$(VPY) scripts/clean.py

distclean: clean
	$(PY) scripts/clean.py --venv
