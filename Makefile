run:
	python src/lab1/main.py

install: requirements.txt
	pip install -r requirements.txt

install-dev: requirements.txt requirements_dev.txt 
	make install
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

