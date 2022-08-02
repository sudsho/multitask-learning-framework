.PHONY: install test lint serve nlp vision clean

install:
	pip install -r requirements.txt

test:
	pytest -q

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

nlp:
	python examples/run_nlp.py --config configs/nlp.yaml

vision:
	python examples/run_vision.py --config configs/vision.yaml

clean:
	rm -rf __pycache__ .pytest_cache mlruns
	find . -name "*.pyc" -delete
