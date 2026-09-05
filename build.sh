#!/bin/bash
set -e
pip install -r requirements.txt
python -c "
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
ef = ONNXMiniLM_L6_V2()
ef(['warmup'])
print('ONNX embedding model pre-download complete.')
"

python -c "
from flashrank import Ranker
Ranker(model_name='ms-marco-MiniLM-L-12-v2')
print('Flashrank re-ranker model pre-download complete.')
"
