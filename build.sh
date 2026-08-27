#!/bin/bash
set -e
pip install -r requirements.txt
python -c "
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
ef = ONNXMiniLM_L6_V2()
ef(['warmup'])
print('ONNX model pre-download complete.')
"
