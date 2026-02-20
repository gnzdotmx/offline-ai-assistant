# Offline AI Assistant - Makefile
# Usage: make [target]
#   make          # show help
#   make install  # create venv + install deps
#   make run      # run app (optionally: make run MODELS_DIR=/path/to/models)
#
# On Windows without make: use "Option B: Manual setup" in README, or WSL.

PYTHON     ?= python3
VENV_DIR   ?= venv
# Prefer venv bin (Unix); fallback to Scripts (Windows) in run target
VENV_PY    := $(VENV_DIR)/bin/python
VENV_PIP   := $(VENV_DIR)/bin/pip
# Optional: custom models folder (shared with other projects). Pass at run time: make run MODELS_DIR=/path
MODELS_DIR ?=

.PHONY: help venv install run clone clean

help:
	@echo "Offline AI Assistant - targets:"
	@echo "  make install   Create virtualenv and install dependencies"
	@echo "  make run      Run the application"
	@echo "  make venv     Create virtualenv only"
	@echo "  make clone    Clone repo (for first-time setup)"
	@echo "  make clean    Remove venv and cache"
	@echo ""
	@echo "Optional at run time:"
	@echo "  make run MODELS_DIR=/path/to/your/models   Use existing models folder (shared with other projects)"
	@echo ""
	@echo "Environment (optional):"
	@echo "  OFFLINE_AI_MODELS_DIR  Same as MODELS_DIR; default: ~/.config/ai-offline-assistant/models"

venv:
	@echo "Creating virtualenv in $(VENV_DIR)..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Activate with: source $(VENV_DIR)/bin/activate  (Windows: $(VENV_DIR)\\Scripts\\activate)"

install: venv
	@echo "Installing dependencies..."
	$(VENV_PIP) install -r requirements.txt
	@echo "Done. Run with: make run"

run:
	@if [ -x "$(VENV_PY)" ]; then \
		[ -n "$(MODELS_DIR)" ] && export OFFLINE_AI_MODELS_DIR="$(MODELS_DIR)"; \
		$(VENV_PY) -m offline_ai_assistant.app_ui; \
	elif [ -x "$(VENV_DIR)/Scripts/python.exe" ]; then \
		$(VENV_DIR)/Scripts/python.exe -m offline_ai_assistant.app_ui; \
	else \
		[ -n "$(MODELS_DIR)" ] && export OFFLINE_AI_MODELS_DIR="$(MODELS_DIR)"; \
		$(PYTHON) -m offline_ai_assistant.app_ui; \
	fi

clone:
	git clone https://github.com/gnzdotmx/offline-ai-assistant.git
	cd offline-ai-assistant && $(MAKE) help

clean:
	rm -rf $(VENV_DIR)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned venv and cache."
