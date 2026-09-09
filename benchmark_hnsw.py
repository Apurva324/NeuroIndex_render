"""
NeuroIndex - HNSW vs Brute Force Benchmark

This benchmark is completely standalone.
It does NOT modify the SQLite database or NeuroIndex indexes.

Measures:
- Recall@K
- Average latency
- HNSW speedup
- HNSW performance at different ef_search values
- Performance at different dataset sizes

PERFORMANCE NOTE:
Vectors are stored as pre-normalized numpy arrays instead of plain Python
lists, and cosine_distance() is a single numpy dot product instead of a
pure-Python loop recomputing both norms every call. Same HNSW algorithm,
same recall/latency behavior - just ~35-40x faster to build, so this
benchmark actually finishes at 5k/10k vectors in reasonable time.
"""

import random
import time
import statistics

import numpy as np

from bruteforce import BruteForce
from hnsw import HNSW


# ============================================================
# Configuration
# ============================================================

DIM = 768

DATASET_SIZES = [
    1000,
    5000,
    10000,
    20000,
    # 100000,  # ~35-60 min for the build alone on a laptop CPU - uncomment if you have time to spare
]

NUM_QUERIES = 50
K = 5

EF_SEARCH_VALUES = [
    100,
    200,
    400,
]

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 400

SEED = 42


# ============================================================
# Distance function
# ============================================================

def cosine_distance(a, b):
    """
    Cosine distance = 1 - cosine similarity.

    Assumes a and b are already unit-normalized numpy arrays (see
    random_vector() below), so this is just 1 - dot(a, b) - no norm
    computation needed at query time. Lower distance = more similar.
    """

    return 1.0 - float(np.dot(a, b))


# ============================================================
# Random vector generation
# ============================================================

def random_vector(rng, dim=DIM):
    """
    Generate a random unit-normalized vector as a numpy array.

    Normalizing once here (instead of inside the distance function on
    every comparison) is what makes cosine_distance() a plain dot
    product - this is the main speedup vs the original implementation.
    """

    v = np.array([rng.uniform(-1.0, 1.0) for _ in range(dim)])
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return v
    return v / norm


def generate_dataset(size, rng):
    """
    Generate synthetic vector records.

    Format matches NeuroIndex's internal index format:
        {
            "id": ...,
            "emb": [...]   (numpy array here, not a plain list)
        }
    """

    return [
        {
            "id": str(i),
            "emb": random_vector(rng),
        }
        for i in range(size)
    ]


# ============================================================
# Recall
# ============================================================

def calculate_recall(brute_results, hnsw_results):
    """
    Calculate Recall@K.

    Brute force is treated as the exact ground truth.
    """

    ground_truth = {
        doc_id
        for _, doc_id in brute_results
    }

    retrieved = {
        doc_id
        for _, doc_id in hnsw_results
    }

    if not ground_truth:
        return 0.0

    return len(
        ground_truth.intersection(retrieved)
    ) / len(ground_truth)


# ============================================================
# Benchmark
# ============================================================

def build_once(dataset_size):
    """
    Build the dataset, Brute Force index, and ONE HNSW graph for this
    dataset size. ef_search does not affect graph structure - only how
    wide the beam is at query time - so the graph only needs to be built
    once per dataset size and reused across every ef_search value.
    (The previous version rebuilt the HNSW graph from scratch for every
    ef_search value - 3x more build work than necessary.)
    """

    print()
    print("=" * 70)
    print(f"Dataset: {dataset_size:,} vectors")
    print("=" * 70)

    rng = random.Random(SEED + dataset_size)

    print("Generating vectors...")
    items = generate_dataset(dataset_size, rng)

    brute = BruteForce()
    for item in items:
        brute.insert(item)

    print(f"Building HNSW (M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION})...")

    hnsw = HNSW(
        m=HNSW_M,
        ef_construction=HNSW_EF_CONSTRUCTION,
        ef_search=EF_SEARCH_VALUES[0],
        seed=SEED,
    )

    start = time.perf_counter()

    for i, item in enumerate(items, start=1):
        hnsw.insert(item, cosine_distance)
        if i % max(1, dataset_size // 10) == 0:
            elapsed = time.perf_counter() - start
            rate = i / elapsed
            remaining = (dataset_size - i) / rate
            print(
                f"  {i:,}/{dataset_size:,} inserted "
                f"({elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining)"
            )

    build_time = (time.perf_counter() - start) * 1000
    print(f"HNSW build time: {build_time:.2f} ms")

    # Same queries reused across every ef_search value for a fair comparison.
    queries = [random_vector(rng) for _ in range(NUM_QUERIES)]

    # Warm-up.
    for query in queries[:3]:
        brute.knn(query, K, cosine_distance)
        hnsw.knn(query, K, ef=EF_SEARCH_VALUES[0], dist=cosine_distance)

    # Ground-truth brute force results and latency (ef_search-independent).
    brute_latencies = []
    brute_results_all = []

    for query in queries:
        start = time.perf_counter()
        results = brute.knn(query, K, cosine_distance)
        elapsed = (time.perf_counter() - start) * 1000
        brute_latencies.append(elapsed)
        brute_results_all.append(results)

    brute_avg = statistics.mean(brute_latencies)

    return {
        "dataset_size": dataset_size,
        "hnsw": hnsw,
        "queries": queries,
        "brute_avg": brute_avg,
        "brute_results_all": brute_results_all,
        "build_time_ms": build_time,
    }


def benchmark(built, ef_search):
    """
    Run one ef_search sweep against an already-built HNSW graph.
    """

    dataset_size = built["dataset_size"]
    hnsw = built["hnsw"]
    queries = built["queries"]
    brute_avg = built["brute_avg"]
    brute_results_all = built["brute_results_all"]

    print()
    print(f"--- ef_search: {ef_search} ---")

    hnsw_latencies = []
    hnsw_results_all = []

    for query in queries:

        start = time.perf_counter()

        results = hnsw.knn(
            query,
            K,
            ef=ef_search,
            dist=cosine_distance,
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        hnsw_latencies.append(elapsed)
        hnsw_results_all.append(results)

    hnsw_avg = statistics.mean(hnsw_latencies)

    recalls = [
        calculate_recall(brute_results, hnsw_results)
        for brute_results, hnsw_results in zip(
            brute_results_all, hnsw_results_all
        )
    ]

    recall = statistics.mean(recalls)

    speedup = brute_avg / hnsw_avg if hnsw_avg > 0 else 0.0

    print(f"Brute Force latency : {brute_avg:.3f} ms")
    print(f"HNSW latency        : {hnsw_avg:.3f} ms")
    print(f"HNSW Recall@{K}       : {recall * 100:.2f}%")
    print(f"HNSW speedup        : {speedup:.2f}x")

    return {
        "dataset_size": dataset_size,
        "ef_search": ef_search,
        "recall": recall,
        "brute_latency_ms": brute_avg,
        "hnsw_latency_ms": hnsw_avg,
        "speedup": speedup,
        "build_time_ms": built["build_time_ms"],
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║          NeuroIndex HNSW Benchmark                        ║")
    print("╚════════════════════════════════════════════════════════════╝")

    print()
    print(f"Vector dimension : {DIM}")
    print(f"Queries          : {NUM_QUERIES}")
    print(f"K                : {K}")
    print(f"HNSW M           : {HNSW_M}")
    print(
        f"EF construction  : {HNSW_EF_CONSTRUCTION}"
    )

    all_results = []

    # --------------------------------------------------------
    # Run benchmarks
    # --------------------------------------------------------

    for dataset_size in DATASET_SIZES:

        built = build_once(dataset_size)

        for ef_search in EF_SEARCH_VALUES:

            result = benchmark(
                built,
                ef_search,
            )

            all_results.append(result)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    print()

    header = (
        f"{'Vectors':>10} "
        f"{'ef':>6} "
        f"{'Recall@5':>10} "
        f"{'Brute(ms)':>12} "
        f"{'HNSW(ms)':>12} "
        f"{'Speedup':>10}"
    )

    print(header)
    print("-" * len(header))

    for result in all_results:

        print(
            f"{result['dataset_size']:>10,} "
            f"{result['ef_search']:>6} "
            f"{result['recall'] * 100:>9.2f}% "
            f"{result['brute_latency_ms']:>12.3f} "
            f"{result['hnsw_latency_ms']:>12.3f} "
            f"{result['speedup']:>9.2f}x"
        )

    print()
    print("=" * 100)

    # --------------------------------------------------------
    # Best configuration
    # --------------------------------------------------------

    best = max(
        all_results,
        key=lambda x: (
            x["recall"],
            x["speedup"],
        ),
    )

    print()
    print("Best configuration:")
    print(
        f"  Dataset       : {best['dataset_size']:,} vectors"
    )
    print(
        f"  ef_search     : {best['ef_search']}"
    )
    print(
        f"  Recall@{K}      : {best['recall'] * 100:.2f}%"
    )
    print(
        f"  Brute latency : {best['brute_latency_ms']:.3f} ms"
    )
    print(
        f"  HNSW latency  : {best['hnsw_latency_ms']:.3f} ms"
    )
    print(
        f"  Speedup       : {best['speedup']:.2f}x"
    )
    print()


if __name__ == "__main__":
    main()