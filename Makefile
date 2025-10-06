# Makefile
.PHONY: install dev test clean

PYTHON := python3.11
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements/dev.txt

dev:
	docker-compose -f docker/docker-compose.yml up -d postgres redis chromadb
	# The following line will be uncommented once app/main.py is created
	# $(PYTHON_VENV) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

test:
	$(PYTHON_VENV) -m pytest tests/ -v --cov=app --cov-report=html

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	docker-compose -f docker/docker-compose.yml down -v