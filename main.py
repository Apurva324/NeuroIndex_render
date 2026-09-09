#!/usr/bin/env python3
"""
VectorDB entrypoint. Logic now lives in separate modules (see config.py,
distances.py, bruteforce.py, kdtree.py, hnsw.py, vectordb.py, storage.py,
ollama_client.py, text_utils.py, demo_data.py, server.py) instead of one
770-line file.
"""
from server import run

if __name__ == "__main__":
    run()
