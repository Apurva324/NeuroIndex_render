import time
from unittest import result


def precision_at_k(retrieved, relevant, k):
    """
    Precision@K

    Measures how many of the retrieved documents
    are relevant.
    """

    retrieved = retrieved[:k]

    if not retrieved:
        return 0.0

    relevant_count = sum(
        1
        for doc_id in retrieved
        if doc_id in relevant
    )

    return relevant_count / len(retrieved)


def recall_at_k(retrieved, relevant, k):
    """
    Recall@K

    Measures how many of the relevant documents
    were successfully retrieved.
    """

    if not relevant:
        return 0.0

    retrieved = retrieved[:k]

    found = sum(
        1
        for doc_id in retrieved
        if doc_id in relevant
    )

    return found / len(relevant)


def reciprocal_rank(retrieved, relevant):
    """
    Reciprocal Rank.

    Returns:
        1 / rank of the first relevant document
    """

    for rank, doc_id in enumerate(
        retrieved,
        start=1,
    ):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def evaluate_search(
    search_fn,
    queries,
    k=5,
):
    """
    Evaluate one retrieval method.

    search_fn(question, k)
        -> list of dictionaries:
           [{"id": 1}, {"id": 5}, ...]

    queries:
        [
            {
                "question": "...",
                "relevant_ids": [1]
            }
        ]
    """

    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []
    latencies = []

    for item in queries:

        question = item["question"]

        relevant = set(
            str(x) for x in item["relevant_ids"]
        )

        start = time.perf_counter()

        results = search_fn(
            question,
            k,
        )

        latency = (
            time.perf_counter()
            - start
        ) * 1000

        latencies.append(
            latency
        )

        retrieved = [
            document["id"]
            for _, document in results
        ]

        precision_scores.append(
            precision_at_k(
                retrieved,
                relevant,
                k,
            )
        )

        recall_scores.append(
            recall_at_k(
                retrieved,
                relevant,
                k,
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                retrieved,
                relevant,
            )
        )

    n = len(queries)

    if n == 0:
        return {
            "queries": 0,
            "k": k,
            "precisionAtK": 0.0,
            "recallAtK": 0.0,
            "mrr": 0.0,
            "avgLatencyMs": 0.0,
        }

    return {
        "queries": n,
        "k": k,

        "precisionAtK": round(
            sum(precision_scores)
            / n,
            4,
        ),

        "recallAtK": round(
            sum(recall_scores)
            / n,
            4,
        ),

        "mrr": round(
            sum(reciprocal_ranks)
            / n,
            4,
        ),

        "avgLatencyMs": round(
            sum(latencies)
            / n,
            3,
        ),
    }


def evaluate_methods(
    methods,
    queries,
    k=5,
):
    """
    Evaluate multiple retrieval methods.

    methods:

        {
            "vector": vector_search,
            "bm25": bm25_search,
            "hybrid": hybrid_search,
        }

    Each function must have:

        search_fn(question, k)

    Returns:

        {
            "vector": {...},
            "bm25": {...},
            "hybrid": {...}
        }
    """

    results = {}

    for name, search_fn in methods.items():

        results[name] = evaluate_search(
            search_fn,
            queries,
            k,
        )

    return results


def evaluate_methods_by_category(methods, queries, k=5):
    """
    Same as evaluate_methods, but grouped by each query's "category"
    field (e.g. keyword / paraphrase / scenario). Queries without a
    category are bucketed under "uncategorized".

    Returns:

        {
            "keyword":    {"vector": {...}, "bm25": {...}, "hybrid": {...}},
            "paraphrase": {"vector": {...}, "bm25": {...}, "hybrid": {...}},
            "scenario":   {"vector": {...}, "bm25": {...}, "hybrid": {...}},
        }

    This is what actually shows whether vector/hybrid search earns its
    keep - on "keyword" queries every method looks great, the real
    signal is in "paraphrase" and "scenario".
    """

    categories = sorted({
        q.get("category", "uncategorized") for q in queries
    })

    by_category = {}

    for category in categories:
        category_queries = [
            q for q in queries
            if q.get("category", "uncategorized") == category
        ]

        by_category[category] = evaluate_methods(
            methods,
            category_queries,
            k=k,
        )

    return by_category


def improvement(
    baseline,
    improved,
):
    """
    Calculate percentage improvement
    from a baseline metric to an improved metric.
    """

    if baseline == 0:
        return 0.0

    return round(
        (
            (improved - baseline)
            / baseline
        ) * 100,
        2,
    )


def compare_results(results):
    """
    Generate a compact comparison summary.
    """

    summary = {}

    if "vector" in results:
        summary["vector"] = results[
            "vector"
        ]

    if "bm25" in results:
        summary["bm25"] = results[
            "bm25"
        ]

    if "hybrid" in results:
        summary["hybrid"] = results[
            "hybrid"
        ]

    # -----------------------------------------------------
    # Hybrid improvement over vector
    # -----------------------------------------------------

    if (
        "vector" in results
        and "hybrid" in results
    ):

        vector = results[
            "vector"
        ]

        hybrid = results[
            "hybrid"
        ]

        summary["hybridVsVector"] = {
            "precisionImprovementPercent":
                improvement(
                    vector[
                        "precisionAtK"
                    ],
                    hybrid[
                        "precisionAtK"
                    ],
                ),

            "recallImprovementPercent":
                improvement(
                    vector[
                        "recallAtK"
                    ],
                    hybrid[
                        "recallAtK"
                    ],
                ),

            "mrrImprovementPercent":
                improvement(
                    vector["mrr"],
                    hybrid["mrr"],
                ),

            "latencyChangePercent":
                improvement(
                    vector[
                        "avgLatencyMs"
                    ],
                    hybrid[
                        "avgLatencyMs"
                    ],
                ),
        }

    return summary