.PHONY: compile lint typecheck test ci smoke clean-sdn

compile:
	python3 -m compileall -q src tests

lint:
	python3 -m ruff check src/schemas src/telemetry src/orchestrator/app.py src/orchestrator/current_state.py tests/unit
	
typecheck:
	python3 -m mypy --ignore-missing-imports src/schemas src/telemetry src/orchestrator/current_state.py

test:
	python3 -m unittest discover -s tests/unit -v

ci: compile lint typecheck test

smoke:
	./scripts/ci/smoke_test.sh

clean-sdn:
	./scripts/cleanup.sh
