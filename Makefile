run:
	python src/lab1/main.py

install: requirements.txt
	pip install -r requirements.txt

install-text-processing: requirements_text_processing.txt
	pip install -r requirements_text_processing.txt
	python -m spacy download ru_core_news_sm

install-text-classification: requirements_text_classification.txt
	pip install -r requirements_text_classification.txt
	python -m spacy download en_core_web_sm

install-dev: requirements.txt requirements_dev.txt 
	make install
	pip install -r requirements_dev.txt
	pip install -e .

install-dev-ju: requirements.txt 
	make install-dev
	pip install -r requirements_dev_ju.txt

install-dev-ju-nvim: requirements.txt
	make install-dev-ju
	pip install -r requirements_dev_ju_nvim.txt

regenerate-requirements: src
	pigar generate src

# Starts jupynium at localhost:18888, with socket opened for nvim listening at
# localhost:18898. You can attach to it `nvim --listen localhost:18898 <notebookFile>`.
start-jupynium:
	jupynium --notebook_URL localhost:18888 --nvim_listen_addr localhost:18898

mine-earthquakes-data: src/lab1/mine_earthquake_data.py
	python $^
