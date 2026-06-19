.PHONY: install data train test run compile smoke clean

install:
	pip install -r requirements.txt

data:
	python data/generate_synthetic_data.py

train:
	python -c "from src.pipeline import run_full_pipeline; run_full_pipeline()"

test:
	pytest tests/ -v

compile:
	python -m py_compile app.py src/*.py data/*.py tests/*.py

smoke:
	streamlit run app.py --server.headless true --server.port 8502 &
	sleep 8
	curl -sf http://localhost:8502 > /dev/null && echo "Streamlit OK" || echo "Streamlit FAILED"
	pkill -f "streamlit run app.py" || true

run:
	streamlit run app.py

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
	rm -f outputs/recommendations.csv outputs/backtest_results.csv
	rm -f models/*.joblib models/*.json
