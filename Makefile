.PHONY: install test profile-torch profile-nsys clean

install:
	pip install -e ".[dev]"

test:
	pytest -v tests/

profile-torch:
	python profiling/run_torch_profiler.py

profile-nsys:
	bash scripts/profile_nsys.sh

clean:
	rm -rf build dist *.egg-info .pytest_cache profiling_output/
