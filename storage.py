"""
SQLite persistence layer for NeuroIndex.

SQLite stores:
    - vector metadata
    - vector embeddings
    - document chunks
    - document embeddings

The actual vector indexes (HNSW, KD-Tree, Brute Force) remain in memory.
SQLite is responsible for persistent storage.
"""

import json
import os
import sqlite3
import threading


class SQLiteStorage:
    def __init__(self, db_path):
        self.db_path = db_path
        self.lock = threading.Lock()

        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False
        )

        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

        self._create_tables()
        self._migrate()

    def _create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    metadata TEXT NOT NULL,
                    category TEXT,
                    embedding TEXT NOT NULL
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)

    def _migrate(self):
        """
        Fixes two compatibility issues found in older DB files:

          1. `id` used to be INTEGER PRIMARY KEY, but every caller in
             vectordb.py/server.py passes string ids (str(data["id"])).
             Non-numeric ids throw "datatype mismatch" on insert.
          2. `documents` used to have no `metadata` column, so
             save_document(..., metadata) had nowhere to write it -
             that's your TypeError.

        Both are fixed by rebuilding the table in place, copying existing
        rows across. Safe to run on every startup - it's a no-op once a
        DB is already on the new schema.
        """
        self._migrate_vectors_table()
        self._migrate_documents_table()

    def _migrate_vectors_table(self):
        info = self.conn.execute("PRAGMA table_info(vectors)").fetchall()
        id_type = next((c[2] for c in info if c[1] == "id"), "TEXT").upper()
        if id_type == "TEXT":
            return

        self.conn.executescript("""
            ALTER TABLE vectors RENAME TO vectors_old;
            CREATE TABLE vectors (
                id TEXT PRIMARY KEY,
                metadata TEXT NOT NULL,
                category TEXT,
                embedding TEXT NOT NULL
            );
            INSERT INTO vectors (id, metadata, category, embedding)
                SELECT CAST(id AS TEXT), metadata, category, embedding FROM vectors_old;
            DROP TABLE vectors_old;
        """)

    def _migrate_documents_table(self):
        info = self.conn.execute("PRAGMA table_info(documents)").fetchall()
        cols = {c[1]: c for c in info}
        id_type = cols["id"][2].upper() if "id" in cols else "TEXT"
        has_metadata = "metadata" in cols

        if id_type == "TEXT" and has_metadata:
            return

        insert_cols = "id, title, text, embedding" + (", metadata" if has_metadata else "")

        self.conn.executescript(f"""
            ALTER TABLE documents RENAME TO documents_old;
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{{}}'
            );
            INSERT INTO documents ({insert_cols})
                SELECT CAST(id AS TEXT), title, text, embedding{", metadata" if has_metadata else ""}
                FROM documents_old;
            DROP TABLE documents_old;
        """)

    @staticmethod
    def _load_metadata(raw, legacy_category=None):
        """vectors.metadata / documents.metadata used to hold a plain string
        (old schema, pre-dating metadata-dict support). New code stores a
        JSON-encoded dict. Handle both so old rows don't crash the load."""
        if raw is None:
            raw = "{}"
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        meta = {"legacy_metadata": raw}
        if legacy_category:
            meta["category"] = legacy_category
        return meta

    # ---------------------------------------------------------
    # VECTOR METHODS
    # ---------------------------------------------------------

    def save_vector(self, vector_id, vector, metadata=None):
        metadata = metadata or {}
        with self.lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO vectors
                (id, metadata, category, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(vector_id),
                    json.dumps(metadata),
                    metadata.get("category", ""),
                    json.dumps(vector)
                )
            )

            self.conn.commit()

    def delete_vector(self, vector_id):
        with self.lock:
            self.conn.execute(
                "DELETE FROM vectors WHERE id = ?",
                (str(vector_id),)
            )

            self.conn.commit()

    def load_vectors(self):
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT id, metadata, category, embedding
                FROM vectors
                ORDER BY id
                """
            ).fetchall()

        vectors = []

        for row in rows:
            vectors.append({
                "id": row[0],
                "vector": json.loads(row[3]),
                "metadata": self._load_metadata(row[1], row[2]),
            })

        return vectors

    def vector_count(self):
        with self.lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM vectors"
            ).fetchone()

        return row[0]

    # ---------------------------------------------------------
    # DOCUMENT METHODS
    # ---------------------------------------------------------

    def save_document(self, document_id, title, text, embedding, metadata=None):
        metadata = metadata or {}
        with self.lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO documents
                (id, title, text, embedding, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(document_id),
                    title,
                    text,
                    json.dumps(embedding),
                    json.dumps(metadata),
                )
            )

            self.conn.commit()

    def delete_document(self, document_id):
        with self.lock:
            self.conn.execute(
                "DELETE FROM documents WHERE id = ?",
                (str(document_id),)
            )

            self.conn.commit()

    def load_documents(self):
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT id, title, text, embedding, metadata
                FROM documents
                ORDER BY id
                """
            ).fetchall()

        documents = []

        for row in rows:
            documents.append({
                "id": row[0],
                "title": row[1],
                "text": row[2],
                "embedding": json.loads(row[3]),
                "metadata": self._load_metadata(row[4]),
            })

        return documents

    def document_count(self):
        with self.lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()

        return row[0]

    def close(self):
        with self.lock:
            self.conn.close()