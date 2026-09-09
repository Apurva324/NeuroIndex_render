from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from bm25 import BM25
from bruteforce import BruteForce
from hnsw import HNSW
from storage import SQLiteStorage


def to_vector(values: List[float]) -> np.ndarray:
    """Convert a plain list into the numpy array used internally by the
    indexes. Call this once when a vector enters the system (insert/load),
    not per comparison."""
    return np.asarray(values, dtype=np.float64)


def cosine_distance(a, b) -> float:
    """
    Cosine distance = 1 - cosine similarity.

    Returns a value where:
        0.0 = identical
        larger values = less similar
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(
            f"Vector dimension mismatch: {a.shape[0]} != {b.shape[0]}"
        )

    norm_a = float(np.dot(a, a))
    norm_b = float(np.dot(b, b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0

    similarity = float(np.dot(a, b)) / (
        norm_a ** 0.5 * norm_b ** 0.5
    )

    # Numerical safety.
    similarity = max(-1.0, min(1.0, similarity))

    return 1.0 - similarity


def euclidean_distance(a, b) -> float:
    """Euclidean distance."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(
            f"Vector dimension mismatch: {a.shape[0]} != {b.shape[0]}"
        )

    return float(np.linalg.norm(a - b))


def manhattan_distance(a, b) -> float:
    """Manhattan distance."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(
            f"Vector dimension mismatch: {a.shape[0]} != {b.shape[0]}"
        )

    return float(np.sum(np.abs(a - b)))


def get_distance(metric: str):
    """Return the configured distance function."""

    metric = metric.lower()

    if metric == "cosine":
        return cosine_distance

    if metric in ("euclidean", "l2"):
        return euclidean_distance

    if metric in ("manhattan", "l1"):
        return manhattan_distance

    raise ValueError(
        f"Unsupported distance metric: {metric}"
    )


# ======================================================================
# VectorDB
# ======================================================================

class VectorDB:
    """
    Persistent vector database backed by SQLite.

    Supports:
    - Vector insertion
    - Vector deletion
    - Exact BruteForce KNN
    - HNSW KNN
    - SQLite persistence
    """

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        state_file: Optional[str] = None,
        autosave: Optional[bool] = None,
    ):
        self.dim = dim
        self.metric = metric

        # Kept for compatibility with previous API.
        self.state_file = state_file
        self.autosave = autosave

        self.store = SQLiteStorage(
            config.SQLITE_DB_FILE
        )

        self.vectors: Dict[str, Dict[str, Any]] = {}
        self._dist = get_distance(metric)
        self._bf = BruteForce()
        self._hnsw = HNSW(
            m=getattr(config, "HNSW_M", 16),
            ef_construction=getattr(config, "HNSW_EF_CONSTRUCTION", 200),
            ef_search=getattr(config, "HNSW_EF_SEARCH", 50),
        )

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load vectors from SQLite and build the live indexes once."""

        try:
            records = self.store.load_vectors()

            for record in records:
                vector_id = str(record["id"])

                self.vectors[vector_id] = {
                    "id": vector_id,
                    "vector": record["vector"],
                    "metadata": record.get(
                        "metadata",
                        {},
                    ),
                }

                self._index_insert(vector_id)

        except Exception:
            self.vectors = {}

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:

        if len(vector) != self.dim:
            raise ValueError(
                f"Vector dimension mismatch: "
                f"expected {self.dim}, got {len(vector)}"
            )

        vector_id = str(vector_id)

        self.vectors[vector_id] = {
            "id": vector_id,
            "vector": vector,
            "metadata": metadata or {},
        }

        self.store.save_vector(
            vector_id,
            vector,
            metadata or {},
        )

        self._index_insert(vector_id)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, vector_id: str) -> bool:

        vector_id = str(vector_id)

        if vector_id not in self.vectors:
            return False

        del self.vectors[vector_id]

        try:
            self.store.delete_vector(vector_id)
        except Exception:
            pass

        self._bf.remove(vector_id)
        self._hnsw.remove(vector_id)

        return True

    # ------------------------------------------------------------------
    # Get
    # ------------------------------------------------------------------

    def get(
        self,
        vector_id: str,
    ) -> Optional[Dict[str, Any]]:

        return self.vectors.get(
            str(vector_id)
        )

    # ------------------------------------------------------------------
    # Count
    # ------------------------------------------------------------------

    def count(self) -> int:
        return len(self.vectors)

    # ------------------------------------------------------------------
    # Items
    # ------------------------------------------------------------------

    def items(self):
        return self.vectors.items()

    def _index_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert the public/storage vector shape to the index shape.

        BruteForce and HNSW both expect the vector under ``emb``.
        Keep ``vector`` as the public/storage field so existing APIs remain
        unchanged. ``emb`` is a numpy array converted once here (at insert
        time) rather than re-converted from a list on every distance call
        made during search - see the PERFORMANCE NOTE above cosine_distance.
        """
        metadata = item.get("metadata") or {}
        return {
            "id": str(item["id"]),
            "emb": to_vector(item.get("vector", [])),
            "metadata": metadata,
            "category": metadata.get("category", ""),
        }

    def _index_insert(self, vector_id: str) -> None:
        """Push one vector into both live indexes. Called once per
        insert/load - this is what lets search() below skip rebuilding."""
        item = self.vectors[vector_id]
        idx_item = self._index_item(item)
        self._bf.insert(idx_item)
        self._hnsw.insert(idx_item, self._dist)

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: List[float],
        k: int = 5,
        max_dist: float = 0.7,
    ) -> List[Tuple[float, Dict[str, Any]]]:

        if not self.vectors:
            return []

        if len(query_vector) != self.dim:
            raise ValueError(
                f"Query vector dimension mismatch: "
                f"expected {self.dim}, got {len(query_vector)}"
            )

        query_vector = to_vector(query_vector)

        # --------------------------------------------------------------
        # Both indexes are already built and kept in sync by
        # insert()/delete()/_load() above - just query whichever fits
        # the current size. No rebuild here (that used to be the entire
        # cost of every search call once past 100 vectors).
        # --------------------------------------------------------------

        if len(self.vectors) < 100:
            results = self._bf.knn(
                query_vector,
                k,
                self._dist,
            )
        else:
            results = self._hnsw.knn(
                query_vector,
                k,
                dist=self._dist,
            )

        # --------------------------------------------------------------
        # Convert IDs back to documents
        # --------------------------------------------------------------

        output = []

        for distance, vector_id in results:

            if distance > max_dist:
                continue

            vector_id = str(vector_id)

            item = self.vectors.get(
                vector_id
            )

            if item is None:
                continue

            output.append(
                (
                    distance,
                    item,
                )
            )

        return output


# ======================================================================
# DocumentDB
# ======================================================================

class DocumentDB:
    """
    Persistent document database for NeuroIndex RAG.

    Retrieval modes:

        Vector Search
             +
        BM25 Search
             ↓
          RRF Fusion
             ↓
        Hybrid Search
    """

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        state_file: Optional[str] = None,
        autosave: Optional[bool] = None,
    ):

        self.dim = dim
        self.metric = metric

        self.state_file = state_file
        self.autosave = autosave

        self.store = SQLiteStorage(
            config.SQLITE_DB_FILE
        )

        self.documents: Dict[
            str,
            Dict[str, Any]
        ] = {}

        
        self._dist = get_distance(metric)
        self._bf = BruteForce()
        self._hnsw = HNSW(
            m=getattr(config, "HNSW_M", 16),
            ef_construction=getattr(config, "HNSW_EF_CONSTRUCTION", 200),
            ef_search=getattr(config, "HNSW_EF_SEARCH", 50),
        )

        self._load()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load(self) -> None:

        try:

            records = self.store.load_documents()

            for record in records:

                document_id = str(
                    record["id"]
                )

                self.documents[document_id] = {
                    "id": document_id,
                    "title": record.get(
                        "title",
                        "",
                    ),
                    "text": record.get(
                        "text",
                        "",
                    ),
                    "embedding": record.get(
                        "embedding",
                        [],
                    ),
                    "metadata": record.get(
                        "metadata",
                        {},
                    ),
                }

                self._index_insert(document_id)

        except Exception:

            self.documents = {}

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(
        self,
        document_id: str,
        title: str,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:

        if len(embedding) != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected {self.dim}, got {len(embedding)}"
            )

        document_id = str(
            document_id
        )

        document = {
            "id": document_id,
            "title": title,
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
        }

        self.documents[
            document_id
        ] = document

        self.store.save_document(
            document_id,
            title,
            text,
            embedding,
            metadata or {},
        )

        self._index_insert(document_id)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(
        self,
        document_id: str,
    ) -> bool:

        document_id = str(
            document_id
        )

        if document_id not in self.documents:
            return False

        del self.documents[
            document_id
        ]

        try:
            self.store.delete_document(
                document_id
            )
        except Exception:
            pass

        self._bf.remove(document_id)
        self._hnsw.remove(document_id)

        return True

    # ------------------------------------------------------------------
    # Get
    # ------------------------------------------------------------------

    def get(
        self,
        document_id: str,
    ) -> Optional[Dict[str, Any]]:

        return self.documents.get(
            str(document_id)
        )

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_documents(
        self,
    ) -> List[Dict[str, Any]]:

        return list(
            self.documents.values()
        )

    # ------------------------------------------------------------------
    # Count
    # ------------------------------------------------------------------

    def count(self) -> int:
        return len(self.documents)

    def _index_item(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a stored document to the shape required by the indexes.

        The document API keeps its embedding under ``embedding`` while
        BruteForce/HNSW require the index vector under ``emb``. ``emb`` is
        a numpy array converted once here, not re-converted from a list
        on every distance call - see the PERFORMANCE NOTE above
        cosine_distance.
        """
        metadata = document.get("metadata") or {}
        return {
            "id": str(document["id"]),
            "emb": to_vector(document.get("embedding", [])),
            "metadata": metadata,
            "category": metadata.get("category", ""),
        }

    def _index_insert(self, document_id: str) -> None:
        """Push one document into both live indexes. Called once per
        insert/load - lets search_vector() below skip rebuilding."""
        document = self.documents[document_id]
        idx_item = self._index_item(document)
        self._bf.insert(idx_item)
        self._hnsw.insert(idx_item, self._dist)

    # ==================================================================
    # VECTOR SEARCH
    # ==================================================================

    def search_vector(
        self,
        query_vector: List[float],
        k: int = 5,
        max_dist: float = 0.7,
    ) -> List[Tuple[float, Dict[str, Any]]]:

        if not self.documents:
            return []

        if len(query_vector) != self.dim:
            raise ValueError(
                f"Query vector dimension mismatch: "
                f"expected {self.dim}, got {len(query_vector)}"
            )

        query_vector = to_vector(query_vector)

        # --------------------------------------------------------------
        # Both indexes are already built and kept in sync by
        # insert()/delete()/_load() above - just query whichever fits
        # the current size. No rebuild here (that used to cost ~630ms
        # per search at only 150 documents - see the benchmark).
        # --------------------------------------------------------------

        if len(self.documents) < 100:
            results = self._bf.knn(
                query_vector,
                k,
                self._dist,
            )
        else:
            results = self._hnsw.knn(
                query_vector,
                k,
                dist=self._dist,
            )

        output = []

        for distance, document_id in results:

            if distance > max_dist:
                continue

            document_id = str(
                document_id
            )

            document = self.documents.get(
                document_id
            )

            if document is None:
                continue

            output.append(
                (
                    distance,
                    document,
                )
            )

        return output

    # ==================================================================
    # BM25 SEARCH
    # ==================================================================

    def search_bm25(
        self,
        query_text: str,
        k: int = 5,
    ) -> List[Tuple[float, Dict[str, Any]]]:

        if not self.documents:
            return []

        if not query_text:
            return []

        documents = list(
            self.documents.values()
        )

        # Keep a stable ID mapping because BM25
        # returns integer positions.
        document_ids = [
            str(document["id"])
            for document in documents
        ]

        texts = [
            document.get(
                "text",
                "",
            )
            for document in documents
        ]

        bm25 = BM25(
            texts
        )

        results = bm25.search(
            query_text,
            top_k=k,
        )

        output = []

        for score, index in results:

            if index < 0:
                continue

            if index >= len(
                document_ids
            ):
                continue

            document_id = document_ids[
                index
            ]

            document = self.documents.get(
                document_id
            )

            if document is None:
                continue

            output.append(
                (
                    score,
                    document,
                )
            )

        return output

    # ==================================================================
    # HYBRID RRF SEARCH
    # ==================================================================

    def search(
        self,
        query_vector: List[float],
        k: int = 5,
        query_text: Optional[str] = None,
        max_dist: float = 0.7,
    ) -> List[Tuple[float, Dict[str, Any]]]:

        if not self.documents:
            return []

        # --------------------------------------------------------------
        # Vector ranking
        # --------------------------------------------------------------

        vector_results = self.search_vector(
            query_vector,
            k=k,
            max_dist=max_dist,
        )

        # --------------------------------------------------------------
        # BM25 ranking
        # --------------------------------------------------------------

        bm25_results = []

        if query_text:

            bm25_results = self.search_bm25(
                query_text,
                k=k,
            )

        # --------------------------------------------------------------
        # Reciprocal Rank Fusion
        # --------------------------------------------------------------

        rrf_k = 60

        rrf_scores: Dict[
            str,
            float
        ] = {}

        # Vector contribution.
        for rank, (_, document) in enumerate(
            vector_results,
            start=1,
        ):

            document_id = str(
                document["id"]
            )

            rrf_scores[
                document_id
            ] = (
                rrf_scores.get(
                    document_id,
                    0.0,
                )
                + 1.0 / (
                    rrf_k + rank
                )
            )

        # BM25 contribution.
        for rank, (_, document) in enumerate(
            bm25_results,
            start=1,
        ):

            document_id = str(
                document["id"]
            )

            rrf_scores[
                document_id
            ] = (
                rrf_scores.get(
                    document_id,
                    0.0,
                )
                + 1.0 / (
                    rrf_k + rank
                )
            )

        # --------------------------------------------------------------
        # Sort
        # --------------------------------------------------------------

        ranked = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        output = []

        for document_id, score in ranked[:k]:

            document = self.documents.get(
                document_id
            )

            if document is None:
                continue

            output.append(
                (
                    score,
                    document,
                )
            )

        return output

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(
        self,
    ) -> Dict[str, Any]:

        return {
            "documentCount": len(
                self.documents
            ),
            "dimension": self.dim,
            "metric": self.metric,
        }
