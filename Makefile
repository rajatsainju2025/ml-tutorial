.PHONY: install lint format test lab train sample clean

install:            ## editable install with every extra
	pip install -e ".[all]"
	pre-commit install

lint:               ## static checks
	ruff check .
	ruff format --check src tests scripts nanogpt

format:             ## auto-fix lint + format
	ruff check --fix .
	ruff format src tests scripts nanogpt

test:               ## unit + smoke tests (CPU, < 1 min)
	pytest

lab:                ## open the notebooks
	jupyter lab notebooks

train:              ## train GPT-2 on Tiny Shakespeare (GPU strongly recommended)
	cd nanogpt && python train.py

sample:             ## sample from OpenAI's GPT-2 weights through our model.py
	cd nanogpt && python sample.py

clean:              ## remove caches and generated outputs
	find . -name __pycache__ -o -name .ipynb_checkpoints -o -name .pytest_cache -o -name .ruff_cache | xargs rm -rf
	rm -rf outputs notebooks/**/outputs
