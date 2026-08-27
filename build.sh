#!/bin/bash
set -e
pip install -r requirements.txt
export SENTENCE_TRANSFORMERS_HOME=./models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
