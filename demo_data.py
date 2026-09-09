import math


DEMO_ITEMS = [
    ("Linked List", "cs", "nodes connected by pointers"),
    ("Binary Search Tree", "cs", "O(log n) search and insert"),
    ("Dynamic Programming", "cs", "memoization overlapping subproblems"),
    ("Graph BFS and DFS", "cs", "breadth and depth first traversal"),
    ("Hash Table", "cs", "O(1) lookup with collision chaining"),
    ("Calculus", "math", "derivatives integrals and limits"),
    ("Linear Algebra", "math", "matrices eigenvalues eigenvectors"),
    ("Probability", "math", "distributions random variables Bayes theorem"),
    ("Number Theory", "math", "primes modular arithmetic RSA cryptography"),
    ("Combinatorics", "math", "permutations combinations generating functions"),
    ("Supervised Learning", "ml", "classification regression labeled data"),
    ("Unsupervised Learning", "ml", "clustering dimensionality reduction"),
    ("Logistic Regression", "ml", "binary classification sigmoid probability"),
    ("Decision Trees", "ml", "recursive splitting classification regression"),
    ("Neural Networks", "ml", "layers neurons backpropagation deep learning"),
    ("KD-Tree", "search", "multidimensional binary search tree"),
    ("BM25", "search", "lexical ranking term frequency inverse document frequency"),
    ("Vector Search", "search", "nearest neighbor similarity embeddings"),
    ("Hybrid Search", "search", "combining lexical and vector retrieval"),
    ("Reciprocal Rank Fusion", "search", "combining ranked search results"),
]


def make_embedding(text, dims=768):
    """
    Create a deterministic 768-dimensional visualization/demo vector.

    These are demo vectors only. They are not Ollama embeddings.
    """

    vector = [0.0] * dims

    # Deterministic hash so the same text always produces the same vector.
    seed = 2166136261

    for ch in text:
        seed ^= ord(ch)
        seed = (seed * 16777619) & 0xffffffff

    # Spread the text across several dimensions.
    for i, ch in enumerate(text):
        idx = (seed + i * 37) % dims
        value = ((ord(ch) * 31 + seed + i * 17) % 1000) / 1000.0

        # Keep values centered around zero.
        vector[idx] += value - 0.5

    # Add deterministic broad distribution.
    for i in range(min(64, dims)):
        idx = (seed + i * 97) % dims
        vector[idx] += math.sin(seed * 0.000001 + i * 0.7) * 0.5

    # Normalize.
    norm = math.sqrt(sum(x * x for x in vector))

    if norm == 0:
        return vector

    return [x / norm for x in vector]


def _vector_count(db):
    if hasattr(db, "count"):
        return db.count()

    if hasattr(db, "size"):
        return db.size()

    return 0


def load_demo(db):
    """
    Load the 20 demo vectors into the current VectorDB.

    Compatible with the current VectorDB API:

        db.insert(vector_id, vector, metadata)
    """

    if _vector_count(db) > 0:
        print("Demo vectors already present.")
        return

    print("=" * 60)
    print("Loading NeuroIndex demo vectors")
    print("=" * 60)

    dims = getattr(db, "dim", None)

    if dims is None:
        dims = getattr(db, "dims", 768)

    inserted = 0

    for index, (title, category, description) in enumerate(
        DEMO_ITEMS,
        start=1,
    ):
        try:
            text = f"{title}: {description}"

            embedding = make_embedding(
                text,
                dims=dims,
            )

            metadata = {
                "title": title,
                "description": description,
                "category": category,
            }

            db.insert(
                str(index),
                embedding,
                metadata,
            )

            inserted += 1

        except Exception as e:
            print(f"[ERROR] {title}: {e}")

    print("-" * 60)
    print(f"Inserted {inserted}/{len(DEMO_ITEMS)} demo vectors")
    print(f"Vector count: {_vector_count(db)}")
    print("=" * 60)