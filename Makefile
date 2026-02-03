.PHONY: install install-docling install-vector run-scrape run-prep-data scrape clean help milvus-start milvus-stop ingest

VENV := .venv
UV := uv
PYTHON := $(VENV)/bin/python

# Milvus Docker settings
MILVUS_CONTAINER := harel-milvus
MILVUS_DATA_DIR := $(PWD)/milvus_data

help:
	@echo "Harel Insurance Data Pipeline"
	@echo ""
	@echo "Usage:"
	@echo "  make install         Install scraper dependencies"
	@echo "  make install-docling Install docling for data preparation"
	@echo "  make install-vector  Install vector database dependencies"
	@echo "  make run-scrape      Run the scraper (downloads documents)"
	@echo "  make run-prep-data   Run data preparation (converts to markdown)"
	@echo "  make milvus-start    Start Milvus vector database (Docker)"
	@echo "  make milvus-stop     Stop Milvus vector database"
	@echo "  make ingest          Insert data into vector database"
	@echo "  make clean           Remove downloaded and prepared data"
	@echo ""

install:
	$(UV) pip install -r requirements.txt

install-docling:
	$(UV) pip install docling

run-scrape: install
	$(PYTHON) -m scraper.main

scrape: run-scrape

run-prep-data: install-docling
	$(PYTHON) -m data_prep.main

# Run with custom options
# Usage: make run-scrape-custom ARGS="-c 30 -o ./data"
run-scrape-custom: install
	$(PYTHON) -m scraper.main $(ARGS)

# Usage: make run-prep-custom ARGS="-w 8 -o ./data_prepared"
run-prep-custom: install-docling
	$(PYTHON) -m data_prep.main $(ARGS)

clean:
	rm -rf data/
	rm -rf data_prepared/
	@echo "Cleaned data and data_prepared directories"

# Vector database targets
install-vector:
	$(UV) pip install pymilvus FlagEmbedding tiktoken tqdm

milvus-start:
	@mkdir -p $(MILVUS_DATA_DIR)
	@if [ "$$(docker ps -q -f name=$(MILVUS_CONTAINER))" ]; then \
		echo "Milvus is already running"; \
	else \
		echo "Starting Milvus..."; \
		docker run -d --name $(MILVUS_CONTAINER) \
			-p 19530:19530 \
			-p 9091:9091 \
			-v $(MILVUS_DATA_DIR):/var/lib/milvus \
			milvusdb/milvus:v2.4.4 \
			milvus run standalone; \
		echo "Waiting for Milvus to start..."; \
		sleep 10; \
		echo "Milvus started on localhost:19530"; \
	fi

milvus-stop:
	@if [ "$$(docker ps -q -f name=$(MILVUS_CONTAINER))" ]; then \
		echo "Stopping Milvus..."; \
		docker stop $(MILVUS_CONTAINER); \
		docker rm $(MILVUS_CONTAINER); \
		echo "Milvus stopped"; \
	else \
		echo "Milvus is not running"; \
	fi

ingest: install-vector
	$(PYTHON) -m vector_db.ingest
