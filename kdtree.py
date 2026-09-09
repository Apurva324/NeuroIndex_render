import heapq


class KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item):
        self.item = item
        self.left = None
        self.right = None


class KDTree:
    def __init__(self, dims):
        self.root = None
        self.dims = dims

    def insert(self, v):
        def ins(n, d):
            if n is None:
                return KDNode(v)
            ax = d % self.dims
            if v["emb"][ax] < n.item["emb"][ax]:
                n.left = ins(n.left, d + 1)
            else:
                n.right = ins(n.right, d + 1)
            return n
        self.root = ins(self.root, 0)

    def knn(self, q, k, dist, filter_fn=None):
        heap = []  # max-heap via negated distance: (-dist, id)

        def rec(n, d):
            if n is None:
                return
            if filter_fn is None or filter_fn(n.item):
                dn = dist(q, n.item["emb"])
                if len(heap) < k or dn < -heap[0][0]:
                    heapq.heappush(heap, (-dn, n.item["id"]))
                    if len(heap) > k:
                        heapq.heappop(heap)
            ax = d % self.dims
            diff = q[ax] - n.item["emb"][ax]
            closer, farther = (n.left, n.right) if diff < 0 else (n.right, n.left)
            rec(closer, d + 1)
            # keep exploring the far branch whenever the heap isn't full yet
            # (matters a lot once filter_fn is rejecting candidates) or the
            # splitting hyperplane is closer than our current worst match.
            if len(heap) < k or abs(diff) < -heap[0][0]:
                rec(farther, d + 1)

        rec(self.root, 0)
        return sorted((-d, i) for d, i in heap)

    def rebuild(self, items):
        self.root = None
        for v in items:
            self.insert(v)
