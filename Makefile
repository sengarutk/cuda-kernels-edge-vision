PYTHON ?= python
NCU ?= ncu
export PYTHONPATH := $(CURDIR)

.PHONY: all install test bench ncu torch-profile report figures paper check clean

all: test bench ncu report figures paper

install:
	pip install -r requirements.txt
	pip install --no-deps -e .

test:
	$(PYTHON) -m pytest

bench:
	$(PYTHON) benchmarks/bench_preprocess.py
	$(PYTHON) benchmarks/bench_filter.py
	$(PYTHON) benchmarks/bench_pipelines.py

ncu:
	NCU="$(NCU)" PYTHON="$(PYTHON)" bash scripts/profile_ncu.sh

torch-profile:
	$(PYTHON) profiling/run_torch_profiler.py

report:
	$(PYTHON) scripts/render_report.py

figures:
	$(PYTHON) paper/generate_figures.py

paper: report figures
	bash paper/build_paper.sh

check:
	$(PYTHON) scripts/render_report.py --check

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov paper/build
