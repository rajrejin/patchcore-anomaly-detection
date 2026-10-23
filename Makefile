# Convenience wrapper only: every target maps to a plain command, so make is optional.
PY ?= python

setup:
	$(PY) -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps

test:
	$(PY) -m pytest --cov --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .
	mypy

full:
	mvinspect verify-data
	mvinspect train --all
	mvinspect evaluate --all
	mvinspect postprocess --all
	mvinspect cascade --all
	mvinspect report

serve:
	mvinspect serve

app:
	streamlit run app/streamlit_app.py

docker-up:
	docker compose up --build
