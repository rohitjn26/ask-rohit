#!/bin/bash
set -e
pip install -r requirements.txt
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); SentenceTransformer('paraphrase-MiniLM-L6-v2')"
