#!/usr/bin/env bash
# setup.sh — Install all dependencies for the IR challenge solution
# Run once before executing any submission script.

set -e

echo "=== Installing Python dependencies ==="
pip install --upgrade pip

pip install \
    rank-bm25 \
    lightgbm \
    sentence-transformers \
    faiss-cpu \
    pandas \
    numpy \
    scikit-learn \
    tqdm \
    nltk \
    ir-measures \
    pyterrier

echo ""
echo "=== Downloading NLTK data ==="
python -c "
import nltk
for pkg in ['stopwords','punkt','punkt_tab','wordnet','omw-1.4']:
    nltk.download(pkg, quiet=False)
"

echo ""
echo "=== Pre-downloading sentence-transformer models ==="
python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
print('Downloading bi-encoder...')
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('Downloading cross-encoder...')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print('Done.')
"

echo ""
echo "=== Verifying FAISS ==="
python -c "import faiss; print('FAISS OK, version:', faiss.__version__)"

echo ""
echo "=== All dependencies installed! ==="
echo ""
echo "Next steps:"
echo "  1. Place corpus.jsonl, test_queries.csv, train_queries.csv,"
echo "     train_qrels.csv in the ./data/ directory"
echo "  2. Run submissions in order:"
echo "     python src/01_bm25_baseline.py    # ~30-60 min first run (builds index)"
echo "     python src/02_query_expansion.py  # fast (reuses index)"
echo "     python src/03_dense_hybrid.py     # ~2-4 hrs first run (builds FAISS)"
echo "     python src/04_ltr_reranker.py     # ~1 hr (trains LambdaMART)"
echo "     python src/05_ensemble.py         # final submission"
echo ""
echo "  Submissions are written to ./submissions/"