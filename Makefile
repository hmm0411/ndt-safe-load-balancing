.PHONY: compile lint typecheck test ci smoke clean-sdn

compile:
	python -m compileall -q src tests

lint:
	ruff check src/schemas src/telemetry src/orchestrator/app.py src/orchestrator/current_state.py tests/unit

typecheck:
	mypy --ignore-missing-imports src/schemas src/telemetry src/orchestrator/current_state.py

test:
	python -m unittest discover -s tests/unit -v

ci: compile lint typecheck test

smoke:
	./scripts/ci/smoke_test.sh

clean-sdn:
	./scripts/cleanup.sh
