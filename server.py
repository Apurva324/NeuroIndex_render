from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import config
from demo_data import load_demo
from evaluation import compare_results, evaluate_methods, evaluate_methods_by_category
from client import AIClient
from fastembed import TextEmbedding
from vectordb import DocumentDB, VectorDB


# ======================================================================
# Configuration
# ======================================================================

# NOTE: previously hardcoded here (HOST = "0.0.0.0", PORT = 8080),
# duplicating and overriding config.py's env-var-driven HOST/PORT for no
# reason -- config.py already supports VECTORDB_HOST/VECTORDB_PORT, this
# was just never wired up. Fixed to actually use it, which is what makes
# it possible to run this behind Hugging Face Spaces (which requires
# listening on port 7860) without editing source per-deployment.
HOST = config.HOST
PORT = config.PORT

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = BASE_DIR / "index.html"

# Groq is used for LLM generation.
ai = AIClient()

# FastEmbed is used for local text embeddings.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)

# BAAI/bge-small-en-v1.5 produces 384-dimensional embeddings.
# config.DIMS is the project's vector-database dimension setting.
EMBEDDING_DIM = getattr(
    config,
    "DIMS",
    384,
)


# ======================================================================
# Databases
# ======================================================================

vectorDB = VectorDB(
    dim=EMBEDDING_DIM,
    metric="cosine",
)

docDB = DocumentDB(
    dim=EMBEDDING_DIM,
    metric="cosine",
)


# ======================================================================
# Helpers
# ======================================================================

def json_response(
    handler,
    data,
    status=200,
):
    body = json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    handler.send_response(status)

    handler.send_header(
        "Content-Type",
        "application/json; charset=utf-8",
    )

    handler.send_header(
        "Content-Length",
        str(len(body)),
    )

    handler.send_header(
        "Access-Control-Allow-Origin",
        "*",
    )

    handler.end_headers()

    handler.wfile.write(body)


def read_json_body(handler):

    length = int(
        handler.headers.get(
            "Content-Length",
            0,
        )
    )

    if length == 0:
        return {}

    body = handler.rfile.read(length)

    if not body:
        return {}

    return json.loads(
        body.decode("utf-8")
    )


def embed_text(text: str):
    embedding = next(
        embedding_model.embed([text])
    )
    return embedding.tolist()


def serialize_results(results):

    output = []

    for score, document in results:

        output.append(
            {
                "score": float(score),
                "id": document.get("id"),
                "title": document.get(
                    "title",
                    "",
                ),
                "text": document.get(
                    "text",
                    "",
                ),
                "metadata": document.get(
                    "metadata",
                    {},
                ),
            }
        )

    return output


def load_evaluation_queries():

    eval_file = (
        BASE_DIR /
        "eval_queries.json"
    )

    if not eval_file.exists():

        raise FileNotFoundError(
            f"Evaluation file not found: {eval_file}"
        )

    with open(
        eval_file,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ======================================================================
# HTTP Handler
# ======================================================================

class RequestHandler(
    BaseHTTPRequestHandler
):

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_message(
        self,
        format,
        *args,
    ):

        print(
            "%s - %s"
            % (
                self.address_string(),
                format % args,
            )
        )

    # ------------------------------------------------------------------
    # OPTIONS
    # ------------------------------------------------------------------

    def do_OPTIONS(self):

        self.send_response(200)

        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, DELETE, OPTIONS",
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type",
        )

        self.end_headers()

    # ==================================================================
    # GET
    # ==================================================================

    def do_GET(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path

        query = parse_qs(
            parsed.query
        )

        # --------------------------------------------------------------
        # Homepage
        # --------------------------------------------------------------

        if path == "/":

            try:

                with open(
                    INDEX_HTML_PATH,
                    "rb",
                ) as f:

                    body = f.read()

                self.send_response(200)

                self.send_header(
                    "Content-Type",
                    "text/html; charset=utf-8",
                )

                self.send_header(
                    "Content-Length",
                    str(len(body)),
                )

                self.end_headers()

                self.wfile.write(body)

            except FileNotFoundError:

                json_response(
                    self,
                    {
                        "error": "index.html not found"
                    },
                    404,
                )

            return

        # --------------------------------------------------------------
        # Vector search
        # --------------------------------------------------------------

        if path == "/search":

            try:

                query_vector = json.loads(
                    query.get(
                        "vector",
                        ["[]"],
                    )[0]
                )

                k = int(
                    query.get(
                        "k",
                        ["5"],
                    )[0]
                )

                results = vectorDB.search(
                    query_vector,
                    k=k,
                )

                json_response(
                    self,
                    {
                        "results":
                            serialize_results(
                                results
                            )
                    },
                )

            except Exception as e:

                json_response(
                    self,
                    {
                        "error": str(e)
                    },
                    400,
                )

            return

        # --------------------------------------------------------------
        # Items
        # --------------------------------------------------------------

        if path == "/items":

            json_response(
                self,
                {
                    "items": [
                        {
                            "id": item["id"],
                            "metadata": item.get(
                                "metadata",
                                {},
                            ),
                        }
                        for item in
                        vectorDB.vectors.values()
                    ]
                },
            )

            return

        # --------------------------------------------------------------
        # Benchmark
        # --------------------------------------------------------------

        if path == "/benchmark":

            json_response(
                self,
                {
                    "message":
                        "Benchmark endpoint available."
                },
            )

            return

        # --------------------------------------------------------------
        # Recall benchmark
        # --------------------------------------------------------------

        if path == "/benchmark/recall":

            json_response(
                self,
                {
                    "message":
                        "Recall benchmark endpoint available."
                },
            )

            return

        # --------------------------------------------------------------
        # HNSW info
        # --------------------------------------------------------------

        if path == "/hnsw-info":

            json_response(
                self,
                {
                    "documentCount":
                        docDB.count(),
                    "dimension":
                        docDB.dim,
                    "metric":
                        docDB.metric,
                },
            )

            return

        # --------------------------------------------------------------
        # Config
        # --------------------------------------------------------------

        if path == "/config":

            try:

                config_data = (
                    config.as_dict()
                )

            except Exception:

                config_data = {
                    "sqliteDatabase":
                        getattr(
                            config,
                            "SQLITE_DB_FILE",
                            None,
                        )
                }

            json_response(
                self,
                config_data,
            )

            return

        # --------------------------------------------------------------
        # Documents
        # --------------------------------------------------------------

        if path == "/doc/list":

            json_response(
                self,
                {
                    "documents":
                        docDB.list_documents()
                },
            )

            return

        # --------------------------------------------------------------
        # Status
        # --------------------------------------------------------------

        if path == "/status":

            json_response(
                self,
                {
                    "status": "ok",
                    "vectors":
                        vectorDB.count(),
                    "documents":
                        docDB.count(),
                },
            )

            return

        # --------------------------------------------------------------
        # Stats
        # --------------------------------------------------------------

        if path == "/stats":

            json_response(
                self,
                {
                    "vectors":
                        vectorDB.count(),
                    "documents":
                        docDB.count(),
                    "dimension":
                        docDB.dim,
                    "metric":
                        docDB.metric,
                    "embeddingModel":
                        EMBEDDING_MODEL,
                },
            )

            return

        # --------------------------------------------------------------
        # Not found
        # --------------------------------------------------------------

        json_response(
            self,
            {
                "error": "Not found"
            },
            404,
        )

    # ==================================================================
    # POST
    # ==================================================================

    def do_POST(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path

        # --------------------------------------------------------------
        # Insert vector
        # --------------------------------------------------------------

        if path == "/insert":

            try:

                data = read_json_body(
                    self
                )

                vector_id = str(
                    data["id"]
                )

                vector = data["vector"]

                metadata = data.get(
                    "metadata",
                    {},
                )

                vectorDB.insert(
                    vector_id,
                    vector,
                    metadata,
                )

                json_response(
                    self,
                    {
                        "message":
                            "Vector inserted",
                        "id":
                            vector_id,
                    },
                )

            except Exception as e:

                json_response(
                    self,
                    {
                        "error": str(e)
                    },
                    400,
                )

            return

        # --------------------------------------------------------------
        # Insert document
        # --------------------------------------------------------------

        if path == "/doc/insert":

            try:

                data = read_json_body(
                    self
                )

                document_id = str(
                    data["id"]
                )

                title = data.get(
                    "title",
                    "",
                )

                text = data.get(
                    "text",
                    "",
                )

                embedding = data.get(
                    "embedding"
                )

                metadata = data.get(
                    "metadata",
                    {},
                )

                if embedding is None:

                    embedding = embed_text(
                        text
                    )

                docDB.insert(
                    document_id,
                    title,
                    text,
                    embedding,
                    metadata,
                )

                json_response(
                    self,
                    {
                        "message":
                            "Document inserted",
                        "id":
                            document_id,
                    },
                )

            except Exception as e:

                json_response(
                    self,
                    {
                        "error": str(e)
                    },
                    400,
                )

            return

        # ==============================================================
        # DOCUMENT SEARCH
        # ==============================================================

        if path == "/doc/search":

            try:

                data = read_json_body(
                    self
                )

                question = data.get(
                    "query",
                    data.get(
                        "question",
                        "",
                    ),
                )

                k = int(
                    data.get(
                        "k",
                        5,
                    )
                )

                if not question:

                    raise ValueError(
                        "query/question is required"
                    )

                query_embedding = (
                    embed_text(
                        question
                    )
                )

                hits = docDB.search(
                    query_embedding,
                    k=k,
                    query_text=question,
                )

                json_response(
                    self,
                    {
                        "query":
                            question,
                        "results":
                            serialize_results(
                                hits
                            ),
                        "method":
                            "hybrid_rrf",
                    },
                )

            except Exception as e:

                json_response(
                    self,
                    {
                        "error": str(e)
                    },
                    400,
                )

            return

        # ==============================================================
        # RAG ASK
        # ==============================================================

        if path == "/doc/ask":

            try:

                data = read_json_body(
                    self
                )

                question = data.get(
                    "question",
                    data.get(
                        "query",
                        "",
                    ),
                )

                k = int(
                    data.get(
                        "k",
                        5,
                    )
                )

                if not question:

                    raise ValueError(
                        "question is required"
                    )

                # ------------------------------------------------------
                # Embedding
                # ------------------------------------------------------

                query_embedding = (
                    embed_text(
                        question
                    )
                )

                # ------------------------------------------------------
                # Hybrid retrieval
                # ------------------------------------------------------

                hits = docDB.search(
                    query_embedding,
                    k=k,
                    query_text=question,
                )

                # ------------------------------------------------------
                # Sources
                # ------------------------------------------------------

                contexts = []

                for i, (_, document) in enumerate(
                    hits,
                    start=1,
                ):

                    contexts.append(
                        {
                            "source": i,
                            "id":
                                document.get(
                                    "id"
                                ),
                            "title":
                                document.get(
                                    "title",
                                    "",
                                ),
                            "text":
                                document.get(
                                    "text",
                                    "",
                                ),
                        }
                    )

                # ------------------------------------------------------
                # Context text
                # ------------------------------------------------------

                context_text = ""

                for item in contexts:

                    context_text += (
                        f"[SOURCE {item['source']}]\n"
                        f"Title: {item['title']}\n"
                        f"Content: {item['text']}\n\n"
                    )

                # ------------------------------------------------------
                # RAG prompt
                # ------------------------------------------------------

                prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY the provided
sources.

If the sources do not contain enough information,
say that the information is not available in the
provided sources.

Cite sources using [SOURCE N] after the statements
they support.

Do not invent source numbers.

User question:
{question}

Retrieved sources:
{context_text}

Answer:
"""

                # ------------------------------------------------------
                # Generate
                # ------------------------------------------------------

                start = time.perf_counter()

                answer = ai.generate(
                    prompt
                )

                latency_ms = (
                    time.perf_counter()
                    - start
                ) * 1000

                # ------------------------------------------------------
                # Response
                # ------------------------------------------------------

                json_response(
                    self,
                    {
                        "question":
                            question,
                        "answer":
                            answer,
                        "sources":
                            contexts,
                        "contexts":
                            contexts,
                        "latencyMs":
                            round(
                                latency_ms,
                                3,
                            ),
                        "method":
                            "hybrid_rrf",
                    },
                )

            except Exception as e:

                json_response(
                    self,
                    {
                        "error": str(e)
                    },
                    400,
                )

            return

        # ==============================================================
        # THREE-WAY EVALUATION
        # ==============================================================

        if path == "/doc/evaluate":

            try:

                data = read_json_body(
                    self
                )

                k = int(
                    data.get(
                        "k",
                        5,
                    )
                )

                queries = (
                    load_evaluation_queries()
                )

                # ------------------------------------------------------
                # VECTOR
                # ------------------------------------------------------

                def vector_search(
                    question,
                    top_k,
                ):

                    embedding = (
                        embed_text(
                            question
                        )
                    )

                    return docDB.search_vector(
                        embedding,
                        k=top_k,
                    )

                # ------------------------------------------------------
                # BM25
                # ------------------------------------------------------

                def bm25_search(
                    question,
                    top_k,
                ):

                    return docDB.search_bm25(
                        question,
                        k=top_k,
                    )

                # ------------------------------------------------------
                # HYBRID RRF
                # ------------------------------------------------------

                def hybrid_search(
                    question,
                    top_k,
                ):

                    embedding = (
                        embed_text(
                            question
                        )
                    )

                    return docDB.search(
                        embedding,
                        k=top_k,
                        query_text=question,
                    )

                # ------------------------------------------------------
                # Methods
                # ------------------------------------------------------

                methods = {
                    "vector":
                        vector_search,
                    "bm25":
                        bm25_search,
                    "hybrid":
                        hybrid_search,
                }

                # ------------------------------------------------------
                # Evaluation
                # ------------------------------------------------------

                results = evaluate_methods(
                    methods,
                    queries,
                    k=k,
                )

                comparison = (
                    compare_results(
                        results
                    )
                )

                by_category = (
                    evaluate_methods_by_category(
                        methods,
                        queries,
                        k=k,
                    )
                )

                json_response(
                    self,
                    {
                        "results":
                            results,
                        "comparison":
                            comparison,
                        "byCategory":
                            by_category,
                        "k":
                            k,
                        "queries":
                            len(queries),
                        "documentCount":
                            docDB.count(),
                    },
                )

            except Exception as e:

                json_response(
                    self,
                    {
                        "error": str(e)
                    },
                    400,
                )

            return

        # --------------------------------------------------------------
        # Unknown POST
        # --------------------------------------------------------------

        json_response(
            self,
            {
                "error": "Not found"
            },
            404,
        )

    # ==================================================================
    # DELETE
    # ==================================================================

    def do_DELETE(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path

        # --------------------------------------------------------------
        # Delete vector
        # --------------------------------------------------------------

        if path.startswith(
            "/delete/"
        ):

            vector_id = path.split(
                "/delete/",
                1,
            )[1]

            deleted = vectorDB.delete(
                vector_id
            )

            json_response(
                self,
                {
                    "deleted":
                        deleted,
                    "id":
                        vector_id,
                },
            )

            return

        # --------------------------------------------------------------
        # Delete document
        # --------------------------------------------------------------

        if path.startswith(
            "/doc/delete/"
        ):

            document_id = path.split(
                "/doc/delete/",
                1,
            )[1]

            deleted = docDB.delete(
                document_id
            )

            json_response(
                self,
                {
                    "deleted":
                        deleted,
                    "id":
                        document_id,
                },
            )

            return

        # --------------------------------------------------------------
        # Not found
        # --------------------------------------------------------------

        json_response(
            self,
            {
                "error": "Not found"
            },
            404,
        )


# ======================================================================
# Server startup
# ======================================================================

def run():
    # NOTE: VectorDB already loads its persisted vectors from SQLite
    # automatically inside __init__ (via the private _load() method) at
    # module-import time, above. This function used to also call the
    # public vectorDB.load(), which was never a real method -- it always
    # raised AttributeError, silently caught below, and printed a
    # misleading "could not load vectors" warning on every startup even
    # though loading had already succeeded. Removed; nothing was lost.

    # Populate the demo vectors if VectorDB is empty.
    load_demo(vectorDB)

    server = HTTPServer(
        (
            HOST,
            PORT,
        ),
        RequestHandler,
    )

    print()
    print("=" * 60)
    print("NeuroIndex server")
    print("=" * 60)
    print(
        f"Server:     http://localhost:{PORT}"
    )
    print(
        f"Database:   {config.SQLITE_DB_FILE}"
    )
    print(
        f"Embedding:  {EMBEDDING_MODEL}"
    )
    print(
        f"Dimension:  {EMBEDDING_DIM}"
    )
    print(
        f"Documents:  {docDB.count()}"
    )
    print("=" * 60)
    print()

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print(
            "\nShutting down NeuroIndex..."
        )

    finally:

        server.server_close()


# Keep backwards compatibility if
# something calls run_server().
def run_server():
    run()


if __name__ == "__main__":
    run()

