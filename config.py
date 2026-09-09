"""
Central config. Every value can be overridden with an env var so you can tune
HNSW without touching code, e.g.:

    HNSW_M=32 HNSW_EF_CONSTRUCTION=400 HNSW_EF_SEARCH=100 python3 main.py
"""
import os

DIMS = int(os.environ.get("VECTORDB_DIMS", 384))

# --- HNSW tuning ---
HNSW_M = int(os.environ.get("HNSW_M", 16))                          # max neighbors/node/layer
HNSW_EF_CONSTRUCTION = int(os.environ.get("HNSW_EF_CONSTRUCTION", 200))  # build-time beam width
HNSW_EF_SEARCH = int(os.environ.get("HNSW_EF_SEARCH", 50))           # query-time beam width

# --- persistence ---
DATA_DIR = os.environ.get(
    "VECTORDB_DATA_DIR",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data"
    )
)

SQLITE_DB_FILE = os.path.join(
    DATA_DIR,
    "neuroindex.db"
)

# --- server ---
HOST = os.environ.get("VECTORDB_HOST", "0.0.0.0")
PORT = int(
    os.environ.get(
        "VECTORDB_PORT",
        os.environ.get("PORT", 10000)
    )
)


def as_dict():
    return {
        "dims": DIMS,
        "hnswM": HNSW_M,
        "hnswEfConstruction": HNSW_EF_CONSTRUCTION,
        "hnswEfSearch": HNSW_EF_SEARCH,
        "sqliteDatabase": SQLITE_DB_FILE,
    }
