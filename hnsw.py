import heapq
import math
import random


class HNSW:
    """Hierarchical Navigable Small World graph.

    Configurable:
        M
        ef_construction
        ef_search

    The public API is intentionally kept compatible with NeuroIndex.
    """

    def __init__(self, m=16, ef_construction=200, ef_search=50, seed=42):
        self.M = m
        self.M0 = 2 * m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.mL = 1.0 / math.log(m)
        self.rng = random.Random(seed)

        # id -> {
        #     item,
        #     maxLyr,
        #     nbrs
        # }
        self.G = {}

        self.topLayer = -1
        self.entryPt = -1

    # ============================================================
    # Random level
    # ============================================================

    def _rand_level(self):
        u = self.rng.random()

        while u <= 0.0:
            u = self.rng.random()

        return int(
            math.floor(
                -math.log(u) * self.mL
            )
        )

    # ============================================================
    # Layer search
    # ============================================================

    def _search_layer(
        self,
        q,
        ep,
        ef,
        lyr,
        dist,
        filter_fn=None,
    ):
        """Beam search on a single layer."""

        if ep not in self.G:
            return []

        # Candidate min-heap.
        candidates = []

        # Result max-heap using negative distance.
        results = []

        visited = {ep}

        d0 = dist(
            q,
            self.G[ep]["item"]["emb"],
        )

        heapq.heappush(
            candidates,
            (d0, ep),
        )

        if (
            filter_fn is None
            or filter_fn(self.G[ep]["item"])
        ):
            heapq.heappush(
                results,
                (-d0, ep),
            )

        while candidates:

            current_dist, current_id = heapq.heappop(
                candidates
            )

            # Once the closest unexplored candidate is
            # worse than our worst result, we can stop.
            if len(results) >= ef:
                worst_dist = -results[0][0]

                if current_dist > worst_dist:
                    break

            node = self.G.get(current_id)

            if node is None:
                continue

            if lyr >= len(node["nbrs"]):
                continue

            for neighbor_id in node["nbrs"][lyr]:

                if neighbor_id in visited:
                    continue

                neighbor = self.G.get(neighbor_id)

                if neighbor is None:
                    continue

                visited.add(neighbor_id)

                neighbor_dist = dist(
                    q,
                    neighbor["item"]["emb"],
                )

                # Determine whether this node is
                # worth exploring further.
                if len(results) < ef:
                    should_add = True
                else:
                    worst_dist = -results[0][0]
                    should_add = neighbor_dist < worst_dist

                if not should_add:
                    continue

                heapq.heappush(
                    candidates,
                    (
                        neighbor_dist,
                        neighbor_id,
                    ),
                )

                if (
                    filter_fn is None
                    or filter_fn(neighbor["item"])
                ):
                    heapq.heappush(
                        results,
                        (
                            -neighbor_dist,
                            neighbor_id,
                        ),
                    )

                    if len(results) > ef:
                        heapq.heappop(results)

        return sorted(
            (
                -distance,
                node_id,
            )
            for distance, node_id in results
        )

    # ============================================================
    # Neighbor selection
    # ============================================================

    @staticmethod
    def _select_nbrs(candidates, max_m):
        return [
            node_id
            for _, node_id in candidates[:max_m]
        ]

    # ============================================================
    # Insert
    # ============================================================

    def insert(self, item, dist):

        vid = item["id"]

        # --------------------------------------------------------
        # New node
        # --------------------------------------------------------

        level = self._rand_level()

        self.G[vid] = {
            "item": item,
            "maxLyr": level,
            "nbrs": [
                []
                for _ in range(level + 1)
            ],
        }

        # First node.
        if self.entryPt == -1:

            self.entryPt = vid
            self.topLayer = level

            return

        ep = self.entryPt

        # --------------------------------------------------------
        # Greedy search through upper layers
        # --------------------------------------------------------

        for layer in range(
            self.topLayer,
            level,
            -1,
        ):

            if layer >= len(
                self.G[ep]["nbrs"]
            ):
                continue

            result = self._search_layer(
                item["emb"],
                ep,
                1,
                layer,
                dist,
            )

            if result:
                ep = result[0][1]

        # --------------------------------------------------------
        # Construction search
        # --------------------------------------------------------

        for layer in range(
            min(self.topLayer, level),
            -1,
            -1,
        ):

            candidates = self._search_layer(
                item["emb"],
                ep,
                self.ef_construction,
                layer,
                dist,
            )

            max_m = (
                self.M0
                if layer == 0
                else self.M
            )

            selected = self._select_nbrs(
                candidates,
                max_m,
            )

            self.G[vid]["nbrs"][layer] = selected

            # ----------------------------------------------------
            # Add bidirectional connections
            # ----------------------------------------------------

            for neighbor_id in selected:

                neighbor = self.G.get(
                    neighbor_id
                )

                if neighbor is None:
                    continue

                if layer >= len(
                    neighbor["nbrs"]
                ):
                    neighbor["nbrs"].extend(
                        []
                        for _ in range(
                            layer
                            + 1
                            - len(
                                neighbor["nbrs"]
                            )
                        )
                    )

                connections = neighbor[
                    "nbrs"
                ][layer]

                if vid not in connections:
                    connections.append(vid)

                # ------------------------------------------------
                # Prune neighbor connections
                # ------------------------------------------------

                if len(connections) > max_m:

                    scored = []

                    neighbor_embedding = (
                        neighbor["item"]["emb"]
                    )

                    for candidate_id in connections:

                        candidate = self.G.get(
                            candidate_id
                        )

                        if candidate is None:
                            continue

                        candidate_dist = dist(
                            neighbor_embedding,
                            candidate["item"]["emb"],
                        )

                        scored.append(
                            (
                                candidate_dist,
                                candidate_id,
                            )
                        )

                    scored.sort(
                        key=lambda x: x[0]
                    )

                    neighbor["nbrs"][layer] = [
                        candidate_id
                        for _, candidate_id
                        in scored[:max_m]
                    ]

            if candidates:
                ep = candidates[0][1]

        # --------------------------------------------------------
        # New highest layer
        # --------------------------------------------------------

        if level > self.topLayer:

            self.topLayer = level
            self.entryPt = vid

    # ============================================================
    # KNN
    # ============================================================

    def knn(
        self,
        q,
        k,
        ef=None,
        dist=None,
        filter_fn=None,
    ):

        if self.entryPt == -1:
            return []

        if dist is None:
            raise ValueError(
                "dist function is required"
            )

        ef = (
            self.ef_search
            if ef is None
            else ef
        )

        # Filtered search needs a larger beam.
        if filter_fn is not None:
            ef = max(
                ef,
                k * 8,
            )

        ef = max(
            ef,
            k,
        )

        ep = self.entryPt

        # --------------------------------------------------------
        # Greedy upper-layer traversal
        # --------------------------------------------------------

        for layer in range(
            self.topLayer,
            0,
            -1,
        ):

            if layer >= len(
                self.G[ep]["nbrs"]
            ):
                continue

            result = self._search_layer(
                q,
                ep,
                1,
                layer,
                dist,
            )

            if result:
                ep = result[0][1]

        # --------------------------------------------------------
        # Full bottom-layer search
        # --------------------------------------------------------

        result = self._search_layer(
            q,
            ep,
            ef,
            0,
            dist,
            filter_fn,
        )

        return result[:k]

    # ============================================================
    # Remove
    # ============================================================

    def remove(self, vid):

        if vid not in self.G:
            return

        # Remove references from neighbors.
        for node in self.G.values():

            node["nbrs"] = [
                [
                    neighbor_id
                    for neighbor_id in layer
                    if neighbor_id != vid
                ]
                for layer in node["nbrs"]
            ]

        # Remove node.
        del self.G[vid]

        # Rebuild entry point if necessary.
        if self.entryPt == vid:

            if not self.G:

                self.entryPt = -1
                self.topLayer = -1

            else:

                best_id = None
                best_level = -1

                for node_id, node in self.G.items():

                    if node["maxLyr"] > best_level:
                        best_level = node["maxLyr"]
                        best_id = node_id

                self.entryPt = best_id
                self.topLayer = best_level

    # ============================================================
    # Information
    # ============================================================

    def get_info(self):

        max_layer = max(
            self.topLayer + 1,
            1,
        )

        nodes_per_layer = [
            0
            for _ in range(max_layer)
        ]

        edges_per_layer = [
            0
            for _ in range(max_layer)
        ]

        nodes = []
        edges = []

        for vid, node in self.G.items():

            item = node["item"]

            nodes.append(
                {
                    "id": vid,
                    "metadata": item.get(
                        "metadata",
                        {},
                    ),
                    "category": item.get(
                        "category",
                        None,
                    ),
                    "maxLyr": node["maxLyr"],
                }
            )

            for layer in range(
                min(
                    node["maxLyr"],
                    max_layer - 1,
                )
                + 1
            ):

                nodes_per_layer[layer] += 1

                if layer >= len(
                    node["nbrs"]
                ):
                    continue

                for neighbor_id in node[
                    "nbrs"
                ][layer]:

                    # Avoid counting edges twice.
                    if vid < neighbor_id:

                        edges_per_layer[
                            layer
                        ] += 1

                        edges.append(
                            {
                                "src": vid,
                                "dst": neighbor_id,
                                "lyr": layer,
                            }
                        )

        return {
            "topLayer": self.topLayer,
            "nodeCount": len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges,
            "M": self.M,
            "efConstruction": self.ef_construction,
            "efSearch": self.ef_search,
        }

    # ============================================================
    # Size
    # ============================================================

    def size(self):
        return len(self.G)

