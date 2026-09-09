class BruteForce:
    """Exact KNN, O(n). Ground truth for recall measurement."""

    def __init__(self):
        self.items = []

    def insert(self, v):
        self.items.append(v)

    def knn(self, q, k, dist, filter_fn=None):
        pool = self.items if filter_fn is None else [v for v in self.items if filter_fn(v)]
        r = sorted((dist(q, v["emb"]), v["id"]) for v in pool)
        return r[:k]

    def remove(self, vid):
        self.items = [v for v in self.items if v["id"] != vid]

    def size(self):
        return len(self.items)
